#!/usr/bin/env python3
"""
pi_tflite_traffic_signs.py

Use a Teachable Machine TensorFlow Lite model on a Raspberry Pi
to recognise traffic signs and show the top prediction.

This version uses Picamera2 instead of cv2.VideoCapture,
which works reliably with the official Raspberry Pi Camera Module
on modern Raspberry Pi OS.

Expected files in the same folder:
  - model.tflite  (Teachable Machine export, TensorFlow Lite)
  - labels.txt    (one label per line, in the order of model outputs)

Example labels.txt (simplified, no numbers at the front):
  background
  stop
  person
  slow
  uturn
"""

import argparse
import time
from pathlib import Path

import numpy as np
import cv2

# Picamera2 for Raspberry Pi Camera Module
from picamera2 import Picamera2

# Try to import TFLite interpreter from tflite_runtime (lightweight) first,
# then fall back to tensorflow.lite if needed.
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime.interpreter")
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        print("Using tensorflow.lite.Interpreter")
    except ImportError as e:
        raise ImportError(
            "Could not import tflite_runtime or tensorflow.lite.\n"
            "Install one of them, e.g.:\n"
            "  pip3 install tflite-runtime\n"
        ) from e

# ------------------------ Utility: Load labels ------------------------ #

def load_labels(labels_path: str):
    """
    Load labels from labels.txt (one label per line).

    If a line starts with '0 ' or '1 ' (Teachable Machine style),
    strip the leading index and space, just in case.
    """
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label = line.strip()
            if not label:
                continue
            parts = label.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                label = parts[1]
            labels[i] = label
    return labels

# ------------------------ Utility: Preprocess frame ------------------------ #

def preprocess_frame_bgr(frame_bgr: np.ndarray, input_size):
    """
    Resize and convert a BGR OpenCV frame to the model's input shape.

    input_size: (height, width)
    Returns a numpy array ready to feed into the TFLite interpreter.
    """
    h, w = input_size
    frame_resized = cv2.resize(frame_bgr, (w, h))
    # Convert BGR (OpenCV) to RGB for the model
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
    return input_data

# ------------------------ Action mapping (behaviour) ------------------------ #

def handle_prediction(label: str, score: float):
    """
    Decide what to do based on the predicted label and score.

    For now, it only prints actions. You can connect this to your
    motor control / line-following code later if you want.
    """

    CONFIDENCE_THRESHOLD = 0.6
    if score < CONFIDENCE_THRESHOLD:
        print(f"Low confidence ({score:.2f}) - treating as background/none.")
        return

    label_lc = label.lower()

    if label_lc == "stop":
        print("ACTION: STOP car (detected STOP sign)")
    elif label_lc in ("tl_red", "red", "traffic_red"):
        print("ACTION: RED traffic light -> STOP")
    elif label_lc in ("tl_yellow", "yellow", "traffic_yellow"):
        print("ACTION: YELLOW traffic light -> SLOW / PREPARE TO STOP")
    elif label_lc in ("tl_green", "green", "traffic_green"):
        print("ACTION: GREEN traffic light -> GO")
    elif label_lc == "slow":
        print("ACTION: SLOW DOWN")
    elif label_lc == "uturn":
        print("ACTION: U-TURN")
    elif label_lc in ("person", "human"):
        print("ACTION: PERSON detected -> STOP or SLOW for safety")
    elif label_lc == "animal":
        print("ACTION: ANIMAL detected -> STOP or SLOW for safety")
    elif label_lc == "background":
        print("No sign detected (background).")
    else:
        print(f"No specific action mapped for label '{label_lc}'.")

# ------------------------ Main inference loop using Picamera2 ------------------------ #

def run_inference(
    model_path: str,
    labels_path: str,
    display: bool = True,
):
    # Load labels
    labels = load_labels(labels_path)
    print("Loaded labels:", labels)

    # Load TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input & output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
    _, input_h, input_w, _ = input_shape
    print(f"Model input shape: {input_shape}")

    # ---------------- Picamera2 setup ---------------- #
    picam = Picamera2()
    # Use a reasonable preview size; we'll resize to model size anyway
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # small warm-up

    print("Camera started. Press 'q' in the window or Ctrl+C in the terminal to quit.")

    try:
        while True:
            # Capture RGB frame from Picamera2
            frame_rgb = picam.capture_array()  # shape (H, W, 3), RGB order

            # Convert to BGR for OpenCV drawing
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Preprocess for model
            input_data = preprocess_frame_bgr(frame_bgr, (input_h, input_w))

            interpreter.set_tensor(input_details[0]["index"], input_data)
            t0 = time.time()
            interpreter.invoke()
            infer_time = (time.time() - t0) * 1000.0  # ms

            output_data = interpreter.get_tensor(output_details[0]["index"])[0]
            top_index = int(np.argmax(output_data))
            top_score = float(output_data[top_index])
            label = labels.get(top_index, f"class_{top_index}")

            print(f"Prediction: {label} ({top_score:.2f})  |  Inference time: {infer_time:.1f} ms")

            handle_prediction(label, top_score)

            if display:
                text = f"{label}: {top_score:.2f}"
                cv2.putText(
                    frame_bgr,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Traffic Sign Detection (Picamera2)", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C).")

    finally:
        picam.stop()
        if display:
            cv2.destroyAllWindows()

# ------------------------ CLI entry point ------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi TFLite traffic sign detection (Teachable Machine model, Picamera2)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.tflite",
        help="Path to TFLite model file (default: model.tflite in current folder).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Path to labels file (default: labels.txt in current folder).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window display (useful via SSH).",
    )

    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser())
    labels_path = str(Path(args.labels).expanduser())
    display = not args.no_display

    run_inference(
        model_path=model_path,
        labels_path=labels_path,
        display=display,
    )

if __name__ == "__main__":
    main()
