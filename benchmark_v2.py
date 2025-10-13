import time
import psutil
import csv
import os
import numpy as np
import cv2
import subprocess
import glob


class PerformanceMonitor:
    """A helper class to track CPU, RAM, FPS, temperature, power, and latency usage over time."""

    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.cpu_log = []
        self.ram_log = []
        self.fps_log = []
        self.temp_log = []
        self.power_log = []
        self.latency_log = []  # NEW: Track inference latency per frame
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
        """Estimate power in watts via RAPL or Pi model."""
        # RAPL (Intel/AMD)
        try:
            with open('/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as f:
                energy = int(f.read().strip()) / 1e6
            now = time.perf_counter()
            if self.prev_energy is not None:
                p = (energy - self.prev_energy) / (now - self.prev_time)
                self.prev_energy, self.prev_time = energy, now
                return p
            self.prev_energy, self.prev_time = energy, now
        except Exception:
            pass
        # Raspberry Pi estimate
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
        self.latency_log.append((t, inference_time * 1000))  # NEW: Store latency in milliseconds


def benchmark_video_with_detailed_logging(model_name, model_infer_func,
                                          video_path, model_input_shape,
                                          output_csv, iterations=1):
    print(f"\n--- Benchmarking {model_name} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    monitor = PerformanceMonitor()
    times, frames = [], 0
    peak_ram, peak_temp = 0, 0

    total_start = time.perf_counter()
    for i in range(iterations):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = cv2.cvtColor(cv2.resize(frame, model_input_shape), cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(img, 0).astype(np.float32)
            if model_name in ["ONNX", "OpenVINO", "OpenVINO_INT8"]:
                inp = inp.transpose(0, 3, 1, 2)

            start = time.perf_counter()
            model_infer_func(inp)
            end = time.perf_counter()

            inf_time = end - start
            times.append(inf_time)
            monitor.log_performance(inf_time)

            rss = monitor.proc.memory_info().rss / 1024 / 1024
            peak_ram = max(peak_ram, rss)
            temp = monitor.get_cpu_temperature() or 0
            peak_temp = max(peak_temp, temp)
            frames += 1

    total_end = time.perf_counter()
    cap.release()

    overall_fps = frames / (total_end - total_start)
    avg_inf_fps = frames / sum(times)
    avg_latency_ms = np.mean(times) * 1000  # NEW: Average latency in ms
    p50_latency_ms = np.percentile(times, 50) * 1000  # NEW: Median latency
    p95_latency_ms = np.percentile(times, 95) * 1000  # NEW: 95th percentile latency

    avg_power = np.mean([p for _, p in monitor.power_log if p is not None]) \
        if any(p for _, p in monitor.power_log) else None

    results = {
        'model_name': model_name,
        'iterations': iterations,
        'overall_fps': overall_fps,
        'avg_inference_fps': avg_inf_fps,
        'avg_latency_ms': avg_latency_ms,  # NEW
        'p50_latency_ms': p50_latency_ms,  # NEW
        'p95_latency_ms': p95_latency_ms,  # NEW
        'peak_ram_mb': peak_ram,
        'peak_temp_c': peak_temp,
        'avg_power_w': avg_power,
        'cpu_log': monitor.cpu_log,
        'fps_log': monitor.fps_log,
        'ram_log': monitor.ram_log,
        'temp_log': monitor.temp_log,
        'power_log': monitor.power_log,
        'latency_log': monitor.latency_log,  # NEW
    }

    exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not exists:
            w.writeheader()
        w.writerow(results)

    print(f"Saved {model_name} results. Peak temp {peak_temp:.1f}°C, "
          f"Avg power {avg_power:.2f}W, Avg latency {avg_latency_ms:.2f}ms")


# --- Model loaders & inference functions ---
import tflite_runtime.interpreter as tflite


def load_tflite_model(path):
    it = tflite.Interpreter(model_path=path);
    it.allocate_tensors();
    return it


def infer_tflite_model(it, x):
    id0 = it.get_input_details()[0]['index']
    it.set_tensor(id0, x);
    it.invoke()
    return [it.get_tensor(o['index']) for o in it.get_output_details()]


try:
    import ncnn


    def load_ncnn_model(param_path, bin_path):
        net = ncnn.Net()
        net.load_param(param_path)
        net.load_model(bin_path)
        return net


    def infer_ncnn_model(net, input_data):
        mat = ncnn.Mat(input_data[0])
        ex = net.create_extractor()
        ex.input("in0", mat)
        ret, output = ex.extract("out0")
        return output
except ImportError:
    load_ncnn_model = None
    infer_ncnn_model = None

import torch


def load_torchscript_model(path):
    m = torch.jit.load(path, map_location='cpu');
    m.eval();
    return m


def infer_torchscript_model(m, x):
    t = torch.from_numpy(x)
    with torch.no_grad(): y = m(t)
    return y.numpy() if hasattr(y, 'numpy') else y


import onnxruntime as ort


def load_onnx_model(p):
    s = ort.InferenceSession(p);
    return s, s.get_inputs()[0].name


def infer_onnx_model(s, n, x): return s.run(None, {n: x})


from openvino.runtime import Core


def load_openvino_model(x): c = Core(); m = c.read_model(model=x); return c.compile_model(m, device_name="CPU")


def infer_openvino_model(cm, x):
    r = cm.create_infer_request();
    r.infer({cm.input(0).any_name: x});
    return r.get_output_tensor().data


if __name__ == "__main__":
    VIDEO, SHAPE, CSV = "safety_glasses_on.mov", (640, 640), "pi5_benchmark.csv"
    ITERS = 5

    # ONNX
    #s,n=load_onnx_model("models/best.onnx")
   # benchmark_video_with_detailed_logging("ONNX", lambda x: infer_onnx_model(s,n,x), VIDEO, SHAPE, CSV, ITERS)

    # OpenVINO FP32
   # ov=load_openvino_model("models/best_openvino_model/best.xml")
   # benchmark_video_with_detailed_logging("OpenVINO", lambda x: infer_openvino_model(ov,x), VIDEO, SHAPE, CSV, ITERS)

    # OpenVINO INT8
 #   ov8=load_openvino_model("models/best_int8_openvino_model/best.xml")
#    benchmark_video_with_detailed_logging("OpenVINO_INT8", lambda x: infer_openvino_model(ov8,x), VIDEO, SHAPE, CSV, ITERS)

    # TorchScript
    #ts=load_torchscript_model("models/best.torchscript")
    #benchmark_video_with_detailed_logging("TorchScript",
    #     lambda x: infer_torchscript_model(ts, x.transpose(0,3,1,2) if x.shape[-1]==3 else x),
       #  VIDEO, SHAPE, CSV, ITERS)

    # NCNN
    if load_ncnn_model:
        net = load_ncnn_model("models/best_ncnn_model/model.ncnn.param", "models/best_ncnn_model/model.ncnn.bin")
        benchmark_video_with_detailed_logging("NCNN", lambda x: infer_ncnn_model(net,x.transpose(0, 3, 1, 2) if x.shape[-1] == 3 else x),VIDEO, SHAPE, CSV, ITERS)


