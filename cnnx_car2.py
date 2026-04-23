#!/usr/bin/env python3

import cv2
import time
import threading
import numpy as np
from flask import Flask, Response, render_template_string, request

import board
import busio
from adafruit_pca9685 import PCA9685

# =========================================
# CONFIG
# =========================================

MODEL_PATH = "model.onnx"
DEVICE = 0
CONF_THRESHOLD = 0.6
LINE_THRESHOLD = 120

CLASS_NAMES = ["background", "go", "stop", "slow", "person", "Uturn"]

STEERING_MIN = 280
STEERING_MAX = 480
STEERING_CENTER = 380

THROTTLE_FORWARD = 410
THROTTLE_SLOW = 400
THROTTLE_STOPPED = 393

MODE = "MANUAL"
AUTOPILOT_RUNNING = False
E_STOP = False

values = {
    "steering": STEERING_CENTER,
    "throttle": THROTTLE_STOPPED
}

# =========================================
# PCA9685
# =========================================

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c, address=0x40)
pca.frequency = 60

STEER_CH = 1
THROTTLE_CH = 0

def set_pwm(channel, pwm):
    pwm = int(np.clip(pwm, 0, 4095))
    duty = int((pwm / 4095.0) * 65535)
    pca.channels[channel].duty_cycle = duty

def apply_control():
    set_pwm(STEER_CH, values["steering"])
    set_pwm(THROTTLE_CH, values["throttle"])

# =========================================
# MODEL
# =========================================

net = None

def init_model():
    global net
    print("✅ Loading ONNX model (CPU)...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✅ Model loaded successfully")

# =========================================
# LINE FOLLOW
# =========================================

def line_follow(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.6):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(mask)

    if moments["m00"] > 1000:
        cx = int(moments["m10"] / moments["m00"])
        error = cx - (w // 2)
        normalized_error = error / (w//2)

        steer = STEERING_CENTER - int(normalized_error * 120)
        values["steering"] = max(STEERING_MIN, min(STEERING_MAX, steer))
        values["throttle"] = THROTTLE_FORWARD - 5
    else:
        values["throttle"] = THROTTLE_STOPPED

# =========================================
# CAMERA THREAD
# =========================================

frame_counter = 0
CURRENT_LABEL = "None"
_latest_jpeg = None
_latest_lock = threading.Lock()

def camera_worker():
    global frame_counter, CURRENT_LABEL, _latest_jpeg

    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    else:
        print("✅ Camera opened successfully")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_counter += 1

            # ✅ WORKING ON 4TH FRAME
            if frame_counter % 4 == 0:
                if MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and not E_STOP:

                    img = cv2.resize(frame, (224, 224))

                    blob = cv2.dnn.blobFromImage(
                        img,
                        scalefactor=1.0/255.0,
                        size=(224, 224),
                        mean=(0, 0, 0),
                        swapRB=True,
                        crop=False
                    )

                    net.setInput(blob)
                    output = net.forward()[0]

                    class_id = int(np.argmax(output))
                    confidence = float(output[class_id])
                    label = CLASS_NAMES[class_id]
                    CURRENT_LABEL = label

                    if confidence > CONF_THRESHOLD:

                        if label in ["stop", "person"]:
                            values["throttle"] = THROTTLE_STOPPED

                        elif label == "go":
                            values["throttle"] = THROTTLE_FORWARD

                        elif label == "slow":
                            values["throttle"] = THROTTLE_SLOW

                        elif label == "background":
                            line_follow(frame)

                        elif label == "Uturn":
                            values["steering"] = STEERING_MAX
                            values["throttle"] = THROTTLE_FORWARD

                    apply_control()

            cv2.putText(frame, f"Mode: {MODE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"Label: {CURRENT_LABEL}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            ret, jpeg = cv2.imencode(".jpg", frame)
            if ret:
                with _latest_lock:
                    _latest_jpeg = jpeg.tobytes()

        except Exception as e:
            print("⚠ Camera thread error:", e)
            time.sleep(0.1)

# =========================================
# FLASK
# =========================================

app = Flask(__name__)

HTML = """
<html>
    <body>
        <h1>PiCar Dashboard</h1>
        <img src="/video" width="640"/>
        <br><br>

        <form method="post" action="/cmd">
            <button name="action" value="auto">AUTOPILOT</button>
            <button name="action" value="manual">MANUAL</button>
            <button name="action" value="stop">STOP</button>
        </form>
    </body>
</html>
"""

@app.route("/")
def index():
    return HTML

@app.route("/cmd", methods=["POST"])
def cmd():
    global MODE, AUTOPILOT_RUNNING, E_STOP

    action = request.form.get("action")

    if action == "auto":
        MODE = "AUTOPILOT"
        AUTOPILOT_RUNNING = True
        E_STOP = False

    elif action == "manual":
        MODE = "MANUAL"
        AUTOPILOT_RUNNING = False

    elif action == "stop":
        E_STOP = True
        values["throttle"] = THROTTLE_STOPPED
        apply_control()

    return ("", 204)

@app.route("/video")
def video():
    def gen():
        while True:
            with _latest_lock:
                frame = _latest_jpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    init_model()
    threading.Thread(target=camera_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=6000)
