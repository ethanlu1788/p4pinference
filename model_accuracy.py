import os
import random
from ultralytics import YOLO
from typing import List, Dict

class FunctionalValidator:
    """
    An interactive tool to visually compare the outputs of different model
    frameworks on a random subset of validation images.
    """
    def __init__(self, model_paths: Dict[str, str], val_images_dir: str):
        """
        Initializes the validator by loading all specified models.

        Args:
            model_paths (Dict[str, str]): A dictionary where keys are framework
                names (e.g., "ONNX") and values are paths to the model files.
            val_images_dir (str): Path to the directory containing validation images.
        """
        print("Loading models... This may take a moment.")
        self.models = {name: YOLO(path) for name, path in model_paths.items()}
        self.val_images_dir = val_images_dir
        print("All models loaded successfully.")

    def _choose_images(self, n: int) -> List[str]:
        """Randomly selects 'n' image paths from the validation directory."""
        try:
            all_images = [os.path.join(self.val_images_dir, f) for f in os.listdir(self.val_images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            return random.sample(all_images, min(n, len(all_images)))
        except FileNotFoundError:
            print(f"Error: Validation image directory not found at '{self.val_images_dir}'")
            return []

    def run_validation_session(self, num_images_to_test: int = 3):
        """
        Starts an interactive session to cycle through frameworks and view predictions.

        Args:
            num_images_to_test (int): The number of random images to test in this session.
        """
        image_paths = self._choose_images(num_images_to_test)
        if not image_paths:
            return

        print(f"\nSelected {len(image_paths)} images for this validation session:")
        for img_path in image_paths:
            print(f"  - {os.path.basename(img_path)}")

        frameworks = list(self.models.keys())
        current_framework_index = 0

        while True:
            framework_name = frameworks[current_framework_index]
            model = self.models[framework_name]

            print("\n" + "="*50)
            print(f"--- Testing Framework: {framework_name} ---")
            print("="*50)

            # Run inference on the chosen images with the current model
            print("Displaying prediction results... (Press any key on an image window to continue)")
            for img_path in image_paths:
                results = model.predict(source=img_path)
                for result in results:
                    print(f"  -> Result for {os.path.basename(result.path)}:")
                    result.show()

            # Interactive menu
            print("\n--- Options ---")
            user_input = input("Enter 'n' for next model, 'p' for previous, or 'q' to quit: ").lower()

            if user_input == 'n':
                current_framework_index = (current_framework_index + 1) % len(frameworks)
            elif user_input == 'p':
                current_framework_index = (current_framework_index - 1 + len(frameworks)) % len(frameworks)
            elif user_input == 'q':
                print("Exiting functional validator.")
                break
            else:
                print("Invalid input. Staying on the current model.")

if __name__ == '__main__':
    # --- Configuration ---
    # 1. Define the paths to all the models you want to compare.
    #    The keys (e.g., "ONNX_FP32") will be used as labels.
    MODEL_PATHS = {
        "PyTorch_FP32": "models/best.pt",
        "ONNX_FP32": "models/best_onnx.onnx",
        "OpenVINO_FP32": "models/best_openvino_model",
        "OpenVINO_INT8": "models/best_int8_openvino_model"
    }

    # 2. Set the path to your validation images directory.
    VALIDATION_IMAGES_DIR = "val/images"

    # 3. Set how many random images you want to test per session.
    NUM_IMAGES = 3

    # --- Run the Validator ---
    validator = FunctionalValidator(MODEL_PATHS, VALIDATION_IMAGES_DIR)
    validator.run_validation_session(num_images_to_test=NUM_IMAGES)
