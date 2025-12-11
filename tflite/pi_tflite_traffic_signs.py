#!/usr/bin/env python3
"""
Real-time TFLite traffic sign classification on Raspberry Pi using PiCamera2.

- Shows a live preview window.
- Overlays the top predicted label and confidence on the video.
- Prints top-3 predictions to the console once per second.
- Calls handle_prediction() to map labels to actions (STOP / SLOW / UTURN, etc.).
- Uses different thresholds for background vs. sign classes.
"""

import os
import time

import cv2
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# ---------------------------------------------------------------------
# Paths: model & labels
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")

# ---------------------------------------------------------------------
# Per-class threshold settings
# ---------------------------------------------------------------------

BACKGROUND_LABEL = "background"   # must match the label name in labels.txt
BACKGROUND_THRESHOLD = 0.90       # only accept background if score >= 0.95
SIGN_THRESHOLD = 0.30             # accept any non-background sign if score >= 0.40

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_labels(path):
    """Load labels from a text file (one label per line)."""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # If line is like "0 background", drop the leading number.
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                line = parts[1]
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
# Behaviour mapping: what to do for each label
# ---------------------------------------------------------------------

def handle_prediction(label, score):
    """
    Map predicted label + confidence to actions.
    Thresholding is handled BEFORE calling this function.
    """
    l = label.lower()

    if l == "stop":
        print(f"ACTION: STOP car  (score={score:.2f})")
    elif l == "slow":
        print(f"ACTION: SLOW DOWN  (score={score:.2f})")
    elif l == "uturn":
        print(f"ACTION: U-TURN  (score={score:.2f})")
    elif l == "person":
        print(f"ACTION: PERSON detected -> STOP/SLOW for safety (score={score:.2f})")
    elif l == "background":
        print(f"No sign detected (background). (score={score:.2f})")
    else:
        print(f"No specific action mapped for '{label}' (score={score:.2f})")

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
    print(f"Loaded {len(labels)} labels:")
    print(labels)
    print(f"BACKGROUND_THRESHOLD = {BACKGROUND_THRESHOLD}, SIGN_THRESHOLD = {SIGN_THRESHOLD}")

    # 2. Set up PiCamera2 for preview
    camera = Picamera2()
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

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            chosen_label = None
            chosen_score = None

            if results:
                # ----- Decide using different thresholds for background vs signs -----
                background_score = None
                best_sign_label = None
                best_sign_score = -1.0

                for class_id, score in results:
                    # Map index to label
                    if 0 <= class_id < len(labels):
                        label = labels[class_id]
                    else:
                        label = f"Class {class_id}"

                    if label.lower() == BACKGROUND_LABEL.lower():
                        background_score = score
                    else:
                        # Track the best non-background (sign) label
                        if score > best_sign_score:
                            best_sign_score = score
                            best_sign_label = label

                # Rule 1: prefer a sign if any sign >= SIGN_THRESHOLD
                if best_sign_label is not None and best_sign_score >= SIGN_THRESHOLD:
                    chosen_label = best_sign_label
                    chosen_score = best_sign_score
                # Rule 2: otherwise, accept background only if very confident
                elif background_score is not None and background_score >= BACKGROUND_THRESHOLD:
                    chosen_label = BACKGROUND_LABEL
                    chosen_score = background_score
                # Rule 3: nothing passes thresholds -> no confident detection
                else:
                    print("No confident detection (all scores below thresholds).")

                # ----- Display & behaviour if we chose something -----
                if chosen_label is not None:
                    label_text = f"{chosen_label}: {chosen_score:.2f}"

                    # Call behaviour function
                    handle_prediction(chosen_label, chosen_score)

                    # Console logging (top-3 once per second)
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

                    # Draw overlay text on the frame
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
            cv2.imshow("PiCar TFLite Traffic Signs", frame_bgr)

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
