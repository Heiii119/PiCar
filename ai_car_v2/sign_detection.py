# sign_detection.py
import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class SignDetector:

    def __init__(self,
                 model_path="model.tflite",
                 labels_path="labels.txt",
                 conf_threshold=0.75):

        self.model_path = model_path
        self.labels_path = labels_path
        self.conf_threshold = conf_threshold

        self.interpreter = None
        self.input_index = None
        self.output_index = None
        self.in_h = 224
        self.in_w = 224
        self.class_names = []

        self._load_labels()
        self._load_model()

    # ======================================================
    # LOAD LABELS  (Teachable Machine: "0 name" per line)
    # ======================================================
    def _load_labels(self):
        if not os.path.exists(self.labels_path):
            print("❌ SignDetector: labels.txt not found")
            return

        with open(self.labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split off the leading index: "0 stop" -> "stop"
                parts = line.split(" ", 1)
                name = parts[1] if len(parts) == 2 else parts[0]
                self.class_names.append(name)

        print(f"✅ SignDetector: {len(self.class_names)} labels "
              f"-> {self.class_names}")

    # ======================================================
    # LOAD TFLITE MODEL
    # ======================================================
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print("❌ SignDetector: model.tflite not found")
            return

        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]

            self.input_index = inp["index"]
            self.output_index = out["index"]

            # input shape is [1, H, W, 3] for Teachable Machine
            self.in_h = inp["shape"][1]
            self.in_w = inp["shape"][2]

            print(f"✅ SignDetector: TFLite loaded | input={inp['shape']} "
                  f"| dtype={inp['dtype']}")

        except Exception as e:
            print("❌ SignDetector: Failed to load model:", e)
            self.interpreter = None

    # ======================================================
    # SIGN DETECTION
    # ======================================================
    def detect(self, frame):
        """
        Returns: (label or None, confidence float)
        """
        if self.interpreter is None or not self.class_names:
            return None, 0.0

        try:
            # 1) Resize to model input size
            img = cv2.resize(frame, (self.in_w, self.in_h))

            # 2) BGR -> RGB  (Teachable Machine trains on RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 3) Normalize to -1..1  (Teachable Machine standard!)
            img = (img.astype(np.float32) / 127.5) - 1.0

            # 4) Add batch dim -> [1, H, W, 3]  (NHWC, no transpose)
            blob = np.expand_dims(img, axis=0)

            # 5) Inference
            self.interpreter.set_tensor(self.input_index, blob)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_index)[0]

            # Teachable Machine output is ALREADY softmax probabilities
            class_id = int(np.argmax(output))
            confidence = float(output[class_id])
            label = self.class_names[class_id]

            if confidence < self.conf_threshold:
                return None, confidence

            return label, confidence

        except Exception as e:
            print("SignDetector detect error:", e)
            return None, 0.0
