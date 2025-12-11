#!/usr/bin/env python3
"""
pi_tflite_traffic_signs.py

Use a Teachable Machine TensorFlow Lite model on a Raspberry Pi
to recognise traffic signs and show the top prediction.

Expected files in the same folder:
  - model.tflite  (Teachable Machine export, TensorFlow Lite)
  - labels.txt    (one label per line, in the order of model outputs)

Recommended labels for your project:
  background
  stop
  person
  animal
  tl_red
  tl_yellow
  tl_green
  slow
  uturn
"""

import argparse
import time
from pathlib import Path

import numpy as np
import cv2

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

    Returns a dict: {class_index: label_string}
    where class_index is 0-based.
    """
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label = line.strip()
            if label:
                labels[i] = label
    return labels

# ------------------------ Utility: Preprocess frame ------------------------ #

def preprocess_frame(frame: np.ndarray, input_size):
    """
    Resize and convert a BGR OpenCV frame to the model's input shape.

    input_size: (height, width)
    Returns a numpy array ready to feed into the TFLite interpreter.
    """
    h, w = input_size
    # Teachable Machine models are usually 224x224 or 192x192, RGB uint8
    frame_resized = cv2.resize(frame, (w, h))
    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Add batch dimension: (1, h, w, 3)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
    return input_data

# ------------------------ Action mapping (behaviour) ------------------------ #

def handle_prediction(label: str, score: float):
    """
    Decide what to do based on the predicted label and score.

    For now, it only prints actions. You can connect this to your
    motor control / line-following code (e.g. in line.py).
    """

    # Only react if we're confident enough
    CONFIDENCE_THRESHOLD = 0.6
    if score < CONFIDENCE_THRESHOLD:
        print(f"Low confidence ({score:.2f}) - treating as background/none.")
        return

    # Normalise label to lower-case just in case
    label_lc = label.lower()

    # You can customise this mapping for your car:
    if label_lc == "stop":
        print("ACTION: STOP car (detected STOP sign)")
        # TODO: call your motor control code to stop

    elif label_lc in ("tl_red", "red", "traffic_red"):
        print("ACTION: Treat as RED traffic light -> STOP")
        # TODO: integrate with your traffic light logic

    elif label_lc in ("tl_yellow", "yellow", "traffic_yellow"):
        print("ACTION: YELLOW traffic light -> PREPARE TO STOP / SLOW DOWN")
        # TODO: maybe slow down

    elif label_lc in ("tl_green", "green", "traffic_green"):
        print("ACTION: GREEN traffic light -> GO")
        # TODO: resume/continue

    elif label_lc == "slow":
        print("ACTION: SLOW DOWN")
        # TODO: reduce motor speed

    elif label_lc == "uturn":
        print("ACTION: U-TURN")
        # TODO: implement U-turn behaviour

    elif label_lc in ("person", "human"):
        print("ACTION: PERSON detected -> STOP or SLOW for safety")
        # TODO: stop / slow car

    elif label_lc == "animal":
        print("ACTION: ANIMAL detected -> STOP or SLOW for safety")
        # TODO: stop / slow car

    elif label_lc == "background":
        # Nothing special
        print("No sign detected (background).")

    else:
        # Unknown label - just print it
        print(f"No specific action mapped for label '{label_lc}'.")

# ------------------------ Main inference loop ------------------------ #

def run_inference(
    model_path: str,
    labels_path: str,
    camera_index: int = 0,
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

    # Assume a single input tensor: (1, h, w, 3)
    input_shape = input_details[0]["shape"]
    _, input_h, input_w, _ = input_shape
    print(f"Model input shape: {input_shape}")

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Press 'q' in the window or Ctrl+C in the terminal to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            # Preprocess frame for model input
            input_data = preprocess_frame(frame, (input_h, input_w))

            # Feed into interpreter
            interpreter.set_tensor(input_details[0]["index"], input_data)
            t0 = time.time()
            interpreter.invoke()
            infer_time = (time.time() - t0) * 1000.0  # ms

            # Get output
            output_data = interpreter.get_tensor(output_details[0]["index"])[0]  # shape: (num_classes,)

            # Teachable Machine TFLite models often output float32 probabilities
            # but check output_details[0]["dtype"] if needed.
            # Find top prediction
            top_index = int(np.argmax(output_data))
            top_score = float(output_data[top_index])

            # Look up label
            label = labels.get(top_index, f"class_{top_index}")

            # Print to terminal
            print(f"Prediction: {label} ({top_score:.2f})  |  Inference time: {infer_time:.1f} ms")

            # Call action mapping
            handle_prediction(label, top_score)

            # Show on screen (for debugging / teaching)
            if display:
                # Draw label & confidence on frame
                text = f"{label}: {top_score:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Traffic Sign Detection", frame)

                # Quit with 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

# ------------------------ CLI entry point ------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi TFLite traffic sign detection (Teachable Machine model)."
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
        "--camera",
        type=int,
        default=0,
        help="Camera index for OpenCV VideoCapture (default: 0).",
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
        camera_index=args.camera,
        display=display,
    )

if __name__ == "__main__":
    main()
