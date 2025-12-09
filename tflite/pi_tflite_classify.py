import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")
LABELS_PATH = "labels.txt"

def load_labels(filename):
    """Load labels from a text file, one label per line."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def set_input_tensor(interpreter, image):
    """Set the input tensor for the interpreter from a PIL RGB image."""
    input_details = interpreter.get_input_details()[0]
    height = input_details["shape"][1]
    width = input_details["shape"][2]
    input_index = input_details["index"]

    # Resize to model expected size
    image = image.resize((width, height), Image.BILINEAR)

    # Convert to numpy, add batch dimension
    input_data = np.expand_dims(image, axis=0)

    # If model expects float, normalise to [0, 1]
    if input_details["dtype"] == np.float32:
        input_data = np.float32(input_data) / 255.0

    interpreter.set_tensor(input_index, input_data)

def classify_image(interpreter, top_k=3):
    """Run inference and return top_k (label_index, score) pairs."""
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])
    output_data = np.squeeze(output_data)

    # Handle quantized models (int8 / uint8) by de-quantising
    if output_details["dtype"] != np.float32:
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            output_data = scale * (output_data - zero_point)

    # Get top_k results
    top_k_indices = np.argsort(-output_data)[:top_k]
    results = [(i, float(output_data[i])) for i in top_k_indices]
    return results

def capture_image_from_camera(filename="capture.jpg", preview_seconds=2):
    """Capture a single image from PiCamera and save to a file."""
    camera = PiCamera()
    try:
        camera.resolution = (640, 480)
        camera.start_preview()
        time.sleep(preview_seconds)  # let camera adjust exposure
        camera.capture(filename)
        camera.stop_preview()
    finally:
        camera.close()
    return filename

def main():
    print("Loading TFLite model...")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    labels = load_labels(LABELS_PATH)
    print(f"Loaded {len(labels)} labels.")

    print("Capturing image from PiCamera...")
    image_path = capture_image_from_camera("capture.jpg", preview_seconds=2)
    print(f"Image saved to {image_path}")

    # Open image as PIL RGB
    image = Image.open(image_path).convert("RGB")

    # Preprocess and run inference
    print("Running inference...")
    set_input_tensor(interpreter, image)

    start_time = time.time()
    results = classify_image(interpreter, top_k=5)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print(f"Inference time: {elapsed_ms:.2f} ms")
    print("Top predictions:")

    for idx, score in results:
        label = labels[idx] if idx < len(labels) else f"Label {idx}"
        print(f"  {label:20s} : {score:.4f}")

if __name__ == "__main__":
    main()
