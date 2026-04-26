import os
import cv2
import time
import numpy as np
import threading
from flask import Flask, Response, request, redirect
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "model.tflite"

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

CURRENT_LABEL = "None"
CURRENT_CONF = 0.0
MODE = "MANUAL"

# =========================================================
# LOAD CORAL MODEL
# =========================================================

interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# =========================================================
# PiCar
# =========================================================

class PiCar:

    def __init__(self):
        self.auto_mode = False
        self.camera = cv2.VideoCapture(0)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.inference_loop, daemon=True).start()

    # ==========================
    # CAMERA THREAD
    # ==========================
    def capture_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame

    # ==========================
    # INFERENCE THREAD
    # ==========================
    def inference_loop(self):
        global CURRENT_LABEL, CURRENT_CONF, MODE

        while self.running:

            if not self.auto_mode:
                MODE = "MANUAL"
                time.sleep(0.05)
                continue

            MODE = "AUTO"

            with self.lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            input_tensor = self.preprocess(frame)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            preds = interpreter.get_tensor(output_details[0]['index'])

            class_id = np.argmax(preds)
            confidence = float(preds[0][class_id])

            if confidence > CONF_THRESHOLD:
                CURRENT_LABEL = CLASS_NAMES[class_id]
                CURRENT_CONF = round(confidence, 2)
            else:
                CURRENT_LABEL = "None"
                CURRENT_CONF = 0.0

    # ==========================
    # PREPROCESS (INT8 MODEL)
    # ==========================
    def preprocess(self, frame):
        _, height, width, _ = input_shape
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.uint8)
        return frame

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
car = PiCar()

@app.route("/")
def index():
    return """
    <h1>RC Car - Coral EdgeTPU</h1>
    <img src='/video' width='800'>
    <form method='post' action='/cmd'>
        <button name='a' value='auto'>AUTO</button>
        <button name='a' value='manual'>MANUAL</button>
    </form>
    """

@app.route("/cmd", methods=["POST"])
def cmd():
    if request.form.get("a") == "auto":
        car.auto_mode = True
    else:
        car.auto_mode = False
    return redirect("/")

@app.route("/video")
def video():

    def gen():
        global MODE, CURRENT_LABEL, CURRENT_CONF

        while True:
            frame = car.get_frame()
            if frame is None:
                continue

            cv2.putText(frame, f"Mode: {MODE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            cv2.putText(frame,
                        f"Sign: {CURRENT_LABEL}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

            cv2.putText(frame,
                        f"Confidence: {CURRENT_CONF}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, threaded=True)
