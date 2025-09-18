import time
import psutil
import csv
import os
import numpy as np
import cv2

import time
import psutil
import csv
import os
import numpy as np
import cv2


class PerformanceMonitor:
    """A helper class to track CPU, RAM, and FPS usage over time."""

    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.cpu_log = []
        self.ram_log = []  # New: For RAM percentage over time
        self.fps_log = []
        self.cpu_count = psutil.cpu_count()
        self.proc.cpu_percent(interval=None)  # Initialize

    def log_performance(self, inference_time):
        """Logs CPU, RAM, and instantaneous FPS for the last frame."""
        timestamp = time.perf_counter()

        # Log CPU
        cpu_pct = self.proc.cpu_percent(interval=None) / self.cpu_count
        self.cpu_log.append((timestamp, cpu_pct))

        # Log RAM
        ram_pct = psutil.virtual_memory().percent
        self.ram_log.append((timestamp, ram_pct))

        # Log FPS
        fps_for_frame = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_log.append((timestamp, fps_for_frame))


def benchmark_video_with_detailed_logging(model_name, model_infer_func, video_path, model_input_shape, output_csv,
                                          iterations=1):
    print(f"\n--- Benchmarking {model_name} | Video: {video_path} | Iterations: {iterations} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    monitor = PerformanceMonitor()
    all_inference_times = []
    total_frame_count = 0
    peak_ram_mb = 0  # New: To track peak memory

    start_total_time = time.perf_counter()

    for i in range(iterations):
        print(f"  Starting iteration {i + 1}/{iterations}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Preprocess frame
            input_image = cv2.resize(frame, model_input_shape)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(input_image, axis=0).astype(np.float32)
            if model_name in ["ONNX", "OpenVINO", "OpenVINO_INT8"]:
                input_tensor = input_tensor.transpose(0, 3, 1, 2)

            # Run inference and log performance
            infer_start_time = time.perf_counter()
            model_infer_func(input_tensor)
            infer_end_time = time.perf_counter()

            inference_time = infer_end_time - infer_start_time
            all_inference_times.append(inference_time)
            monitor.log_performance(inference_time)

            # Update peak RAM
            current_ram = monitor.proc.memory_info().rss / (1024 * 1024)  # in MB
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram

            total_frame_count += 1

    end_total_time = time.perf_counter()
    cap.release()

    total_processing_time = end_total_time - start_total_time
    overall_fps = total_frame_count / total_processing_time
    avg_inference_fps = total_frame_count / sum(all_inference_times)

    results = {
        'model_name': model_name,
        'iterations': iterations,
        'overall_fps': overall_fps,
        'avg_inference_fps': avg_inference_fps,
        'peak_ram_mb': peak_ram_mb,
        'cpu_log': monitor.cpu_log,
        'fps_log': monitor.fps_log,
        'ram_log': monitor.ram_log,  # New
    }

    # Save to CSV
    fieldnames = list(results.keys())
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Processing complete for {model_name}. Results saved to {output_csv}")
# --- CSV saving utility ---


# --- TFLite ---
import tflite_runtime.interpreter as tflite

def load_tflite_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def infer_tflite_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return [interpreter.get_tensor(out['index']) for out in output_details]

# --- ONNX ---
import onnxruntime as ort

def load_onnx_model(path):
    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name
    return session, input_name

def infer_onnx_model(session, input_name, input_data):
    return session.run(None, {input_name: input_data})

# --- OpenVINO ---
from openvino.runtime import Core

def load_openvino_model(xml_path):
    core = Core()
    model = core.read_model(model=xml_path)
    compiled_model = core.compile_model(model, device_name="CPU")
    return compiled_model

def infer_openvino_model(compiled_model, input_data):
    infer_request = compiled_model.create_infer_request()
    input_tensor_name = compiled_model.input(0).any_name
    infer_request.infer({input_tensor_name: input_data})
    output = infer_request.get_output_tensor()
    return output.data

if __name__ == "__main__":
    # --- Configuration ---
    VIDEO_PATH = "safety_glasses_on.mov"
    MODEL_INPUT_SHAPE = (640, 640)
    OUTPUT_CSV = "full_video_benchmark_pi5.csv"
    ITERATIONS = 1

    # --- ONNX Benchmark (FP32) ---
    if 'load_onnx_model' in globals() and load_onnx_model:
        onnx_session, onnx_input_name = load_onnx_model("models/best_onnx.onnx")
        benchmark_video_with_detailed_logging("ONNX", lambda x: infer_onnx_model(onnx_session, onnx_input_name, x), VIDEO_PATH, MODEL_INPUT_SHAPE, OUTPUT_CSV, iterations=ITERATIONS)

    # --- OpenVINO Benchmark (FP32) ---
    if 'load_openvino_model' in globals() and load_openvino_model:
        openvino_fp32_model = load_openvino_model("models/best_openvino_model/best.xml")
        benchmark_video_with_detailed_logging("OpenVINO", lambda x: infer_openvino_model(openvino_fp32_model, x), VIDEO_PATH, MODEL_INPUT_SHAPE, OUTPUT_CSV, iterations=ITERATIONS)

    # --- OpenVINO Benchmark (INT8) - NEW ---
    if 'load_openvino_model' in globals() and load_openvino_model:
        # Assumes you have an INT8 model exported in this path
        openvino_int8_model = load_openvino_model("models/best_int8_openvino_model/best.xml")
        benchmark_video_with_detailed_logging("OpenVINO_INT8", lambda x: infer_openvino_model(openvino_int8_model, x), VIDEO_PATH, MODEL_INPUT_SHAPE, OUTPUT_CSV, iterations=ITERATIONS)

        # --- TFLite Benchmark ---
    if 'load_tflite_model' in globals() and load_tflite_model:
        tflite_interpreter = load_tflite_model("models/best_saved_model_tflite/best_float32.tflite")
        # TFLite expects NHWC input shape: (1, 640, 640, 3)
        benchmark_video_with_detailed_logging(
            "TFLite",
            lambda x: infer_tflite_model(tflite_interpreter, x),
            VIDEO_PATH,
            MODEL_INPUT_SHAPE,
            OUTPUT_CSV,
            iterations=ITERATIONS
        )