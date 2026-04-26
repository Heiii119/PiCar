# sign_detection.py

import os
import cv2
import numpy as np


class SignDetector:

    def __init__(self,
                 model_path="model.onnx",
                 conf_threshold=0.75,
                 inference_interval=4):

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.inference_interval = inference_interval

        self.net = None

        # Class order MUST match training
        self.class_names = [
            "background",
            "stop",
            "person",
            "slow",
            "Uturn",
            "go"
        ]

        # Frame counter
        self.frame_count = 0

        # Store last detection result
        self.last_label = None
        self.last_conf = 0.0

        self._load_model()

    # ======================================================
    # LOAD MODEL
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
    # DETECT (runs every N frames)
    # ======================================================

    def detect(self, frame):

        if self.net is None:
            return None, 0.0

        self.frame_count += 1

        # ✅ Only run inference every N frames
        if self.frame_count % self.inference_interval != 0:
            return self.last_label, self.last_conf

        try:
            img = cv2.resize(frame, (224, 224))

            blob = cv2.dnn.blobFromImage(
                img,
                scalefactor=1/255.0,
                size=(224, 224),
                swapRB=False,
                crop=False
            )

            # NCHW -> NHWC (required for your model)
            blob = blob.transpose(0, 2, 3, 1)

            self.net.setInput(blob)
            output = self.net.forward()[0]

            class_id = int(np.argmax(output))
            confidence = float(output[class_id])
            label = self.class_names[class_id]

            if confidence >= self.conf_threshold:
                self.last_label = label
                self.last_conf = confidence
            else:
                self.last_label = None
                self.last_conf = confidence

            return self.last_label, self.last_conf

        except Exception as e:
            print("SignDetector detect error:", e)
            return self.last_label, self.last_conf
