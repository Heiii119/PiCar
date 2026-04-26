# sign_detection.py
import cv2
import numpy as np
import os


class SignDetector:

    def __init__(self,
                 model_path="model.onnx",
                 conf_threshold=0.75):

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.net = None
        self.class_names = None
        self.label_map = None

        self._load_model()

    # =========================================
    # LOAD ONNX MODEL
    # =========================================
    def _load_model(self):

        if not os.path.exists(self.model_path):
            print("SignDetector: model not found -> disabled")
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)

            # ⚠ IMPORTANT:
            # MUST match training class order exactly
            self.class_names = [
                "background",
                "stop",
                "person",
                "slow",
                "Uturn",
                "go"
            ]

            # Map raw labels to controller labels
            self.label_map = {
                "stop": "stop",
                "person": "stop",     # treat person as stop
                "slow": "slow",
                "Uturn": "uturn",
                "go": "go",
                "background": None
            }

            print("SignDetector: ONNX model loaded")

        except Exception as e:
            print("SignDetector: model load failed:", e)
            self.net = None

    # =========================================
    # DETECT SIGN
    # =========================================
    def detect(self, frame):
        """
        Returns:
            (label, confidence)
            label = None if no valid detection
        """

        if self.net is None:
            return None, 0.0

        try:
            h, w, _ = frame.shape

            # ---- Center Crop ----
            crop_fraction = 0.6
            cw = int(w * crop_fraction)
            ch = int(h * crop_fraction)
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2

            crop = frame[y1:y1 + ch, x1:x1 + cw]

            # ---- Resize to model input ----
            img = cv2.resize(crop, (224, 224))

            # Convert BGR → RGB (very important for TF models)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize
            img = img.astype(np.float32) / 255.0

            # Convert HWC → NCHW
            blob = np.transpose(img, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            self.net.setInput(blob)
            output = self.net.forward()[0]

            class_id = int(np.argmax(output))
            confidence = float(output[class_id])

            raw_label = self.class_names[class_id]
            mapped_label = self.label_map.get(raw_label, None)

            if mapped_label is None:
                return None, 0.0

            if confidence < self.conf_threshold:
                return None, confidence

            return mapped_label, confidence

        except Exception as e:
            print("SignDetector detect error:", e)
            return None, 0.0
