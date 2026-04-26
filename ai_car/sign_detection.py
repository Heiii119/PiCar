# sign_detection.py

import os
import cv2
import numpy as np


class SignDetector:

    def __init__(self,
                 model_path="model.onnx",
                 conf_threshold=0.75):

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.net = None

        # MUST match training order exactly
        self.class_names = [
            "background",
            "stop",
            "person",
            "slow",
            "Uturn",
            "go"
        ]

        self._load_model()

    # ======================================================
    # LOAD ONNX MODEL
    # ======================================================

    def _load_model(self):

        if not os.path.exists(self.model_path):
            print("❌ SignDetector: ONNX model not found")
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            print("✅ SignDetector: ONNX model loaded")

        except Exception as e:
            print("❌ SignDetector: Failed to load model:", e)
            self.net = None

    # ======================================================
    # SIGN DETECTION (NHWC FIX FOR TF ONNX)
    # ======================================================

    def detect(self, frame):
        """
        Returns:
            label (str or None)
            confidence (float)
        """

        if self.net is None:
            return None, 0.0

        try:
            # Resize to model input size
            img = cv2.resize(frame, (224, 224))

            # Create blob (NCHW)
            blob = cv2.dnn.blobFromImage(
                img,
                scalefactor=1/255.0,
                size=(224, 224),
                swapRB=False,
                crop=False
            )

            # ✅ VERY IMPORTANT:
            # Convert NCHW -> NHWC for TensorFlow-exported ONNX
            blob = blob.transpose(0, 2, 3, 1)

            self.net.setInput(blob)
            output = self.net.forward()[0]

            class_id = int(np.argmax(output))
            confidence = float(output[class_id])

            label = self.class_names[class_id]

            # Confidence filter
            if confidence < self.conf_threshold:
                return None, confidence

            return label, confidence

        except Exception as e:
            print("SignDetector detect error:", e)
            return None, 0.0
