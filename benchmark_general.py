import time
import psutil
import platform

# Function to measure CPU, RAM %, and Inference time
# input: model_inference_function - function to run inference
# input: input_data - sample input for model
# output: dict with fps, cpu%, ram%
def benchmark_model(model_inference_function, input_data, iterations=100):
    cpu_usage = []
    ram_usage = []
    inference_times = []

    for _ in range(iterations):
        start_cpu = psutil.cpu_percent(interval=None)
        start_ram = psutil.virtual_memory().percent
        start_time = time.time()
        model_inference_function(input_data)
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_ram = psutil.virtual_memory().percent

        inference_times.append(end_time - start_time)
        cpu_usage.append(end_cpu - start_cpu)
        ram_usage.append(end_ram - start_ram)

    fps = iterations / sum(inference_times)
    return {
        'fps': fps,
        'cpu_percent': sum(cpu_usage)/len(cpu_usage),
        'ram_percent': sum(ram_usage)/len(ram_usage)
    }

# Template function placeholders for each Framework

def run_tflite_model(input_data):
    # Load and inference TFLite model
    # TODO: implement
    pass

def run_onnx_model(input_data):
    # Load and inference ONNX model
    # TODO: implement
    pass

def run_pytorch_mobile_model(input_data):
    # Load and inference PyTorch Mobile model
    # TODO: implement
    pass

# Example usage: 
# input_data = load_sample_image()  # Load or preprocess your input image
# print(benchmark_model(run_tflite_model, input_data))
