#!/usr/bin/env python3
"""
Hailo NPU Benchmarking Script
Measures FPS, latency, CPU, RAM, temperature, and power during video inference
"""

import time
import psutil
import csv
import os
import numpy as np
import cv2
import subprocess
import glob


# --- Performance Monitor ---
class PerformanceMonitor:
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.cpu_log = []
        self.ram_log = []
        self.fps_log = []
        self.temp_log = []
        self.power_log = []
        self.latency_log = []
        self.cpu_count = psutil.cpu_count()
        self.proc.cpu_percent(interval=None)
        self.prev_energy = None
        self.prev_time = None

    def get_cpu_temperature(self):
        try:
            files = glob.glob('/sys/class/thermal/thermal_zone*/temp')
            if files:
                with open(files[0], 'r') as f:
                    return int(f.read().strip()) / 1000.0
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for entries in temps.values():
                    if entries:
                        return entries[0].current
            res = subprocess.run(['vcgencmd', 'measure_temp'],
                                 capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and 'temp=' in res.stdout:
                return float(res.stdout.split('=')[1].replace("'C", ""))
        except Exception:
            pass
        return None

    def get_power_consumption(self):
        try:
            with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as f:
                energy = int(f.read().strip()) / 1e6
            now = time.perf_counter()
            if self.prev_energy is not None:
                p = (energy - self.prev_energy) / (now - self.prev_time)
                self.prev_energy, self.prev_time = energy, now
                return p
        except Exception:
            pass
        cpu_pct = psutil.cpu_percent()
        temp = self.get_cpu_temperature() or 0
        base, load, tf = 2.5, cpu_pct / 100.0, max(0, (temp - 40) / 40.0)
        return base + 4.0 * load + 1.0 * tf

    def log_performance(self, inference_time):
        t = time.perf_counter()
        self.cpu_log.append((t, self.proc.cpu_percent(interval=None) / self.cpu_count))
        self.ram_log.append((t, psutil.virtual_memory().percent))
        self.fps_log.append((t, 1.0 / inference_time if inference_time > 0 else 0))
        self.temp_log.append((t, self.get_cpu_temperature()))
        self.power_log.append((t, self.get_power_consumption()))
        self.latency_log.append((t, inference_time * 1000))


# --- Hailo Inference ---
try:
    from hailo_platform import (HEF, VDevice, ConfigureParams, InputVStreamParams,
                                OutputVStreamParams, InferVStreams, FormatType)


    def load_hailo_model(hef_path):
        device = VDevice()
        hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(hef, interface=hef.get_default_interface())
        network_groups = device.configure(hef, configure_params)
        network_group = network_groups[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group,
                                                                           format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group,
                                                                             format_type=FormatType.FLOAT32)
        return (device, network_group, input_vstreams_params, output_vstreams_params)


    def infer_hailo_model(hailo_objects, input_data):
        if hailo_objects is None:
            return None
        device, network_group, input_vstreams_params, output_vstreams_params = hailo_objects
        try:
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                if input_data.shape[1] == 3:
                    input_data = input_data.transpose(0, 2, 3, 1)
                input_dict = {}
                for input_info in infer_pipeline.input_vstream_infos:
                    input_dict[input_info.name] = input_data
                with network_group.activate():
                    results = infer_pipeline.infer(input_dict)
                return list(results.values())
        except Exception as e:
            print(f"Inference error: {e}")
            return None

except ImportError as e:
    print(f"HailoRT not available: {e}")
    load_hailo_model = infer_hailo_model = lambda *args: None


# --- Benchmarking Function ---
def benchmark_hailo_model(hef_path, video_path, input_shape=(640, 640), output_csv="hailo_benchmark.csv", iterations=5):
    print(f"\n--- Benchmarking Hailo NPU | Model: {hef_path} | Video: {video_path} ---")

    # Load model
    hailo_model = load_hailo_model(hef_path)
    if not hailo_model:
        print("Failed to load Hailo model.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Init monitor
    monitor = PerformanceMonitor()
    inference_times = []
    total_frames = 0
    peak_ram = peak_temp = 0

    # Benchmark loop
    total_start = time.perf_counter()
    for i in range(iterations):
        print(f"  Iteration {i + 1}/{iterations}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Preprocess
            img = cv2.cvtColor(cv2.resize(frame, input_shape), cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(img, 0).astype(np.float32)
            # Inference
            start = time.perf_counter()
            infer_hailo_model(hailo_model, inp)
            inf_time = time.perf_counter() - start
            # Log
            inference_times.append(inf_time)
            monitor.log_performance(inf_time)
            # Stats
            ram_mb = monitor.proc.memory_info().rss / 1024 / 1024
            peak_ram = max(peak_ram, ram_mb)
            temp_c = monitor.get_cpu_temperature() or 0
            peak_temp = max(peak_temp, temp_c)
            total_frames += 1

    cap.release()
    total_time = time.perf_counter() - total_start

    # Results
    avg_latency_ms = np.mean(inference_times) * 1000
    p95_latency_ms = np.percentile(inference_times, 95) * 1000
    overall_fps = total_frames / total_time
    avg_power = np.mean([p for _, p in monitor.power_log if p is not None]) if any(monitor.power_log) else None

    results = {
        'model_name': 'Hailo_NPU',
        'hef_file': os.path.basename(hef_path),
        'iterations': iterations,
        'overall_fps': overall_fps,
        'avg_latency_ms': avg_latency_ms,
        'p95_latency_ms': p95_latency_ms,
        'peak_ram_mb': peak_ram,
        'peak_temp_c': peak_temp,
        'avg_power_w': avg_power,
        'total_frames': total_frames,
        'total_time_s': total_time
    }

    # Save to CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"\n✅ Benchmark complete!")
    print(f"Overall FPS: {overall_fps:.2f}")
    print(f"Avg Latency: {avg_latency_ms:.2f} ms")
    print(f"Peak RAM: {peak_ram:.1f} MB")
    print(f"Peak Temp: {peak_temp:.1f}°C")
    print(f"Avg Power: {avg_power:.2f} W")
    print(f"Results saved to {output_csv}")


# --- Main ---
if __name__ == "__main__":
    BENCHMARK_CONFIG = {
        "hef_path": "models/best.hef",  # Update path
        "video_path": "safety_glasses_on.mov",  # Update path
        "input_shape": (640, 640),
        "output_csv": "hailo_benchmark_results.csv",
        "iterations": 5
    }
    benchmark_hailo_model(**BENCHMARK_CONFIG)
