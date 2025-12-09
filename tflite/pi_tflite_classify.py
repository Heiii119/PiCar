#!/usr/bin/env python3
"""
Real-time TFLite image classification on Raspberry Pi using PiCamera2.

- Shows a live preview window.
- Overlays the top predicted label and confidence on the video.
- Prints top-3 predictions to the console once per second.
"""

import os
import time

import cv2
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# ---------------------------------------------------------------------
# Paths: model & labels (kept next to this script in the tflite folder)
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_labels(path):
    """Load labels from a text file (one label per line)."""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                labels.append(line)
    return labels

def set_input_tensor(interpreter, image):
    """
    Resize and copy a PIL Image into the TFLite interpreter input tensor.

    Handles both uint8 and float32 input types.
    """
    input_details = interpreter.get_input_details()[0]
    height, width = input_details["shape"][1], input_details["shape"][2]

    # Ensure RGB and resize to model input size
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)

    input_data = np.array(image)

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    input_data = np.expand_dims(input_data, axis=0)

    # Adjust type according to input tensor type
    if input_details["dtype"] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:  # typically float32
        input_data = input_data.astype(np.float32) / 255.0

    interpreter.set_tensor(input_details["index"], input_data)

def classify_image(interpreter, top_k=3):
    """
    Run inference and return top_k (class_index, score) pairs.

    Handles quantized outputs correctly using scale and zero_point if present.
    """
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])[0]

    # Dequantize if needed
    if output_details["dtype"] == np.uint8:
        scale, zero_point = output_details.get("quantization", (1.0, 0))
        scores = (output_data.astype(np.float32) - zero_point) * scale
    else:
        scores = output_data.astype(np.float32)

    # Get indices of top k scores
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_k_indices]

# ---------------------------------------------------------------------
# Main real-time loop
# ---------------------------------------------------------------------

def main():
    # 1. Load model and labels
    print("Loading TFLite model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    labels = load_labels(LABELS_PATH)
    print(f"Loaded {len(labels)} labels.")

    # 2. Set up PiCamera2 for preview
    camera = Picamera2()
    # Use a reasonable preview configuration (you can tweak resolution if desired)
    preview_config = camera.create_preview_configuration()
    camera.configure(preview_config)
    camera.start()
    print("Camera started.")
    print("Press 'q' in the preview window or Ctrl+C in the terminal to stop.")

    last_print_time = 0.0
    print_interval = 1.0  # seconds between console logs

    try:
        while True:
            # 3. Capture a frame as a NumPy array (RGB)
            frame = camera.capture_array()  # shape (H, W, 3), RGB

            # Convert to PIL Image for TFLite preprocessing
            image = Image.fromarray(frame)

            # 4. Run inference
            set_input_tensor(interpreter, image)
            results = classify_image(interpreter, top_k=3)

            if not results:
                # No result: just show frame without text
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("PiCar TFLite Real-Time Preview", frame_bgr)
            else:
                # Top result
                top_id, top_score = results[0]
                if 0 <= top_id < len(labels):
                    top_label = labels[top_id]
                else:
                    top_label = f"Class {top_id}"

                label_text = f"{top_label}: {top_score:.2f}"

                # 5. Console logging (top-3 once per second)
                now = time.time()
                if now - last_print_time >= print_interval:
                    print("Top predictions:")
                    for class_id, score in results:
                        if 0 <= class_id < len(labels):
                            name = labels[class_id]
                        else:
                            name = f"Class {class_id}"
                        print(f"  {name:30s}  {score:.3f}")
                    print("-" * 40)
                    last_print_time = now

                # 6. Draw overlay text on the frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    frame_bgr,
                    label_text,
                    (10, 30),  # position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,              # font scale
                    (0, 255, 0),      # color (B, G, R) = green
                    2,                # thickness
                    cv2.LINE_AA
                )

                # 7. Show frame
                cv2.imshow("PiCar TFLite Real-Time Preview", frame_bgr)

            # 8. Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key 'q' pressed.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C).")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera and windows closed. Program finished.")

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
