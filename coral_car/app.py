import cv2
import numpy as np
import threading
from flask import Flask, Response, request, redirect

# ===== Coral =====
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# =========================================================
# GLOBAL STATUS
# =========================================================

MODE = "MANUAL"
CURRENT_LABEL = "None"
CURRENT_CONF = 0.0

MODEL_PATH = "model.tflite"

# 👇 Put your label names here (important!)
LABELS = {
    0: "STOP",
    1: "LEFT",
    2: "RIGHT"
}

# =========================================================
# LOAD CORAL MODEL
# =========================================================

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# =========================================================
# PiCar Class (Example Structure)
# =========================================================

class PiCar:

    def __init__(self):
        self.auto_mode = False
        self.camera = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame

    def detect_sign(self, frame):
        global CURRENT_LABEL, CURRENT_CONF

        h, w = input_shape[1], input_shape[2]

        resized = cv2.resize(frame, (w, h))
        resized = np.expand_dims(resized, axis=0)

        common.set_input(interpreter, resized)
        interpreter.invoke()

        output = interpreter.get_output_details()[0]
        results = interpreter.get_tensor(output['index'])

        class_id = np.argmax(results)
        confidence = float(results[0][class_id])

        if confidence > 0.7:
            CURRENT_LABEL = LABELS.get(class_id, "Unknown")
            CURRENT_CONF = round(confidence, 2)
        else:
            CURRENT_LABEL = "None"
            CURRENT_CONF = 0.0

    def loop(self):
        global MODE

        while True:
            if self.auto_mode:
                MODE = "AUTO"

                frame = self.get_frame()
                if frame is None:
                    continue

                self.detect_sign(frame)

                # 👉 Add line following + motor logic here

            else:
                MODE = "MANUAL"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
car = PiCar()

threading.Thread(target=car.loop, daemon=True).start()


@app.route("/")
def index():
    return """
    <h1>RC Car - Coral AI</h1>
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

            # Overlay Mode
            cv2.putText(frame,
                        f"Mode: {MODE}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

            # Overlay Road Sign + Confidence
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
    app.run(host="0.0.0.0", port=9090)
