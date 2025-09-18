from ultralytics import YOLO

# Load your trained YOLOv11n model
# This should be your original .pt file
model = YOLO('../models/best.pt')

# Export the model directly to TFLite format
# This will create 'yolov11n_float32.tflite'
model.export(format='openvino', int8=True)

print("Successfully exported to yolov11n_float32.tflite")