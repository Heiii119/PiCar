# sign_detection.py
import os
import cv2
import numpy as np
import onnxruntime as ort


class SignDetector:

    def __init__(self,
                 model_path="model.onnx",
                 conf_threshold=0.75):

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.session = None
        self.input_name = None
        self.input_layout = None   # "NHWC" or "NCHW"

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
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )

            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            shape = inp.shape   # e.g. [1,224,224,3] or [1,3,224,224]

            # Auto-detect layout: if dim[1]==3 it's NCHW, else NHWC
            if len(shape) == 4 and shape[1] == 3:
                self.input_layout = "NCHW"
            else:
                self.input_layout = "NHWC"

            print(f"✅ SignDetector: ONNX loaded | input={shape} "
                  f"| layout={self.input_layout}")

        except Exception as e:
            print("❌ SignDetector: Failed to load model:", e)
            self.session = None

    # ======================================================
    # SOFTMAX (safe for both logits and probabilities)
    # ======================================================
    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ======================================================
    # SIGN DETECTION
    # ======================================================
    def detect(self, frame):
        """
        Returns:
            label (str or None)
            confidence (float)
        """
        if self.session is None:
            return None, 0.0

        try:
            # 1) Resize
            img = cv2.resize(frame, (224, 224))

            # 2) BGR -> RGB  (TF/Keras models train on RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 3) Normalize to 0..1
            img = img.astype(np.float32) / 255.0

            # 4) Arrange to the layout the model actually expects
            if self.input_layout == "NCHW":
                # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))

            # 5) Add batch dimension
            blob = np.expand_dims(img, axis=0)

            # 6) Inference
            output = self.session.run(
                None, {self.input_name: blob}
            )[0][0]

            # 7) Normalize to probabilities (handles logits AND softmax)
            probs = self._softmax(output)

            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            label = self.class_names[class_id]

            if confidence < self.conf_threshold:
                return None, confidence

            return label, confidence

        except Exception as e:
            print("SignDetector detect error:", e)
            return None, 0.0
