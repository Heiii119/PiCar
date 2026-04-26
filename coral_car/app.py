import os
import cv2
import time
import numpy as np
import threading
from flask import Flask, Response, request, redirect

# ===== Coral =====
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_edgetpu.tflite")

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

CURRENT_LABEL = "None"
CURRENT_CONF = 0.0
MODE = "MANUAL"

PWM_CONFIG = {
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 410,
    "THROTTLE_SLOW_PWM": 400,
    "THROTTLE_STOPPED_PWM": 393,
}

# =========================================================
# LOAD CORAL MODEL
# =========================================================

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_size = common.input_size(interpreter)

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

        # Start threads
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.inference_loop, daemon=True).start()

    # =====================================================
    # CAMERA THREAD (runs at full speed)
    # =====================================================

    def capture_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            with self.lock:
                self.latest_frame = frame

    # =====================================================
    # CORAL INFERENCE THREAD
    # =====================================================

    def inference_loop(self):
        global CURRENT_LABEL, CURRENT_CONF, MODE

        while self.running:

            if not self.auto_mode:
                MODE = "MANUAL"
                self.set_throttle(PWM_CONFIG["THROTTLE_STOPPED_PWM"])
                time.sleep(0.05)
                continue

            MODE = "AUTO"

            with self.lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            input_tensor = self.preprocess(frame)

            common.set_input(interpreter, input_tensor)
            interpreter.invoke()

            output = interpreter.get_output_details()[0]
            preds = interpreter.get_tensor(output['index'])

            class_id = np.argmax(preds)
            confidence = float(preds[0][class_id])

            if confidence > CONF_THRESHOLD:
                label = CLASS_NAMES[class_id]
                CURRENT_LABEL = label
                CURRENT_CONF = round(confidence, 2)
                self.react_to_sign(label)
            else:
                CURRENT_LABEL = "None"
                CURRENT_CONF = 0.0

            time.sleep(0.01)  # tiny delay for stability

    # =====================================================
    # PREPROCESS (FOR INT8 MODEL)
    # =====================================================

    def preprocess(self, frame):
        width, height = input_size
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.uint8)
        return frame

    # =====================================================
    # MOTOR REACTION
    # =====================================================

    def react_to_sign(self, label):

        if label == "stop":
            self.set_throttle(PWM_CONFIG["THROTTLE_STOPPED_PWM"])

        elif label == "slow":
            self.set_throttle(PWM_CONFIG["THROTTLE_SLOW_PWM"])

        elif label == "go":
            self.set_throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])

        elif label == "Uturn":
            self.set_steering(PWM_CONFIG["STEERING_LEFT_PWM"])

        elif label == "person":
            self.set_throttle(PWM_CONFIG["THROTTLE_STOPPED_PWM"])

    # Replace these with PCA9685 control
    def set_steering(self, pwm):
        print("Steering:", pwm)

    def set_throttle(self, pwm):
        print("Throttle:", pwm)

    # =====================================================
    # STREAM FRAME ACCESS
    # =====================================================

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
    app.run(host="0.0.0.0", port=9090, threaded=True)# =========================================================
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
