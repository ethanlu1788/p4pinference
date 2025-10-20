# p4pinference
### Benchmarking and Verification for Edge AI Inference Frameworks

This repository contains scripts, utilities, and notebooks used to **benchmark and verify YOLOv11-based PPE detection models** across multiple inference frameworks. The goal is to evaluate **runtime performance, accuracy consistency, and framework interoperability** on edge devices such as the **Raspberry Pi 4, Pi 5,** and **Hailo-8L**.

---

## Overview

`p4pinference` provides a unified structure for testing exported models across frameworks like **ONNX**, **OpenVINO**, **NCNN**, and **Hailo**.  
It ensures detection consistency using a validation dataset (`val/`) and supports flexible benchmarking through commented script sections.

- Loads the framework-specific model.
- Performs frame-by-frame inference.
- Calculates FPS, latency, and accuracy.
- Logs results for reproducibility and comparison.

---

## Project Structure

| File / Folder        | Description                                                                               |
|----------------------|-------------------------------------------------------------------------------------------|
| `benchmark_hailo.py` | Benchmarks on Hailo-8L accelerators.                                                      |
| `benchmark.py`       | Unified runner that executes multiple frameworks and compares performance.                |
| `model_val.ipynb`    | **Validation mode script.** Notebook to validate the different frameworks.                |
| `val/`               | **Validation dataset directory. Required** to verify model correctness across frameworks. |
| `requirements.txt`   | Python dependencies per framework.                                                        |
| `README.md`          | Project documentation (this file).                                                        |

---

## Validation Dataset

The **`val/`** folder is useful for verifying inference consistency and checking model accuracy using the **`model_val.ipynb`**.  
Each framework script uses these files to validate output predictions against ground truth annotations.

Expected structure:
val/
<pre> <code>val/
┣ images/
┃ ┣ img_0001.jpg
┃ ┗ img_0002.jpg
┗ labels/
┣ img_0001.txt
┗ img_0002.txt </code> </pre>
---

## Usage

You can benchmark individual frameworks or all at once.
### Benchmark all except Hailo (benchmark.py)
This script can run all included frameworks sequentially, or you can comment/uncomment specific framework calls inside to run them one by one.
### Benchmark Hailo-8L
Hailo benchmarks run separately due to unique hardware and SDK requirements.

---

## Benchmark Metrics

Each inference run records the following data:

| Metric           | Description                               |
|------------------|-------------------------------------------|
| **FPS**          | Frames per second (throughput).          |
| **Latency (ms)** | Average time per frame.                   |
| **CPU / RAM Usage** | Captures resource efficiency.            |
| **Power Usage (Watts)** | Measures electrical power consumption during inference. |
| **Temperature (Pi Only)** | Monitors thermal stability during extended tests. |

Results are logged automatically in the directory, name is governed in benchmark.py or benchmark_hailo.py.

## Environment Setup

Create virtual environment and activate with Python version 3.11
then
<pre> <code> pip install requirements.txt </code> </pre>