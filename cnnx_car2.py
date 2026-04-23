#!/usr/bin/env python3

import cv2
import time
import threading
import numpy as np
from flask import Flask, Response

from smbus2 import SMBus

# =========================
# CONFIG
# =========================
DEVICE = 0
PORT = 6088
MODEL_PATH = "model.onnx"

CAMERA_FPS = 10
CAMERA_DT = 1.0 / CAMERA_FPS
CONTROL_DT = 1.0 / 60.0

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.7

LINE_THRESHOLD = 100

# =========================
# PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 1

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED = 370
THROTTLE_FORWARD = 415
THROTTLE_SLOW = 400

STEERING_CENTER = 380
STEERING_MIN = 300
STEERING_MAX = 480

# =========================
# GLOBAL STATE
# =========================
MODE = "AUTOPILOT"
AUTOPILOT_RUNNING = True
E_STOP = False
CURRENT_LABEL = "None"

values = {
    "steering": STEERING_CENTER,
    "throttle": THROTTLE_STOPPED,
}

_latest_jpeg = None
_latest_lock = threading.Lock()

# =========================
# PCA9685 DRIVER
# =========================
class PCA9685:
    MODE1 = 0x00
    PRESCALE = 0xFE
    RESTART = 0x80
    SLEEP = 0x10

    def __init__(self, busnum, address=0x40, freq=60):
        self.bus = SMBus(busnum)
        self.address = address
        self.set_pwm_freq(freq)

    def write8(self, reg, val):
        self.bus.write_byte_data(self.address, reg, val)

    def read8(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq):
        prescaleval = int(25000000.0 / (4096 * freq) - 1)
        oldmode = self.read8(self.MODE1)
        self.write8(self.MODE1, oldmode | self.SLEEP)
        self.write8(self.PRESCALE, prescaleval)
        self.write8(self.MODE1, oldmode)
        time.sleep(0.005)
        self.write8(self.MODE1, oldmode | self.RESTART)

    def set_pwm_12bit(self, channel, value):
        value = max(0, min(4095, int(value)))
        base = 0x06 + 4 * channel
        self.write8(base + 2, value & 0xFF)
        self.write8(base + 3, (value >> 8) & 0xFF)

# =========================
# MODEL
# =========================
net = None

def init_model():
    global net
    print("✅ Loading ONNX model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✅ Model loaded")

# =========================
# CENTROID + CURVATURE
# =========================
def centroid_line_detection(frame):

    h, w = frame.shape[:2]
    roi = frame[int(h*0.6):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    ys, xs = np.where(mask > 0)

    if len(xs) < 500:
        values["throttle"] = THROTTLE_STOPPED
        return

    # --- Weighted centroid (bottom rows more important)
    weights = ys + 1
    cx = int(np.average(xs, weights=weights))

    error = (cx - w//2) / (w//2)

    # --- Curvature approximation
    curvature = np.polyfit(ys, xs, 1)[0] / w

    # --- Steering control
    steer = STEERING_CENTER - int(error * 120)
    steer = max(STEERING_MIN, min(STEERING_MAX, steer))

    # --- Curve slowdown
    curve_factor = min(1.0, abs(curvature) * 5.0)
    throttle = int(np.interp(curve_factor,
                              [0,1],
                              [THROTTLE_FORWARD, THROTTLE_SLOW]))

    values["steering"] = steer
    values["throttle"] = throttle

# =========================
# CAMERA THREAD (10 FPS)
# =========================
def camera_worker():
    global CURRENT_LABEL, _latest_jpeg

    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_counter = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1

        if frame_counter % 4 == 0:
            img = cv2.resize(frame, (224,224))
            blob = cv2.dnn.blobFromImage(img, 1/255.0,
                                         (224,224),
                                         swapRB=True)

            net.setInput(blob)
            output = net.forward()[0]

            class_id = int(np.argmax(output))
            confidence = float(output[class_id])
            label = CLASS_NAMES[class_id]
            CURRENT_LABEL = label

            if confidence > CONF_THRESHOLD:
                if label in ["stop","person"]:
                    values["throttle"] = THROTTLE_STOPPED
                elif label == "slow":
                    values["throttle"] = THROTTLE_SLOW
                elif label == "go":
                    values["throttle"] = THROTTLE_FORWARD
                elif label == "Uturn":
                    values["steering"] = STEERING_MAX
                else:
                    centroid_line_detection(frame)
        else:
            centroid_line_detection(frame)

        cv2.putText(frame, f"Label: {CURRENT_LABEL}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        ret, jpeg = cv2.imencode(".jpg", frame)
        if ret:
            with _latest_lock:
                _latest_jpeg = jpeg.tobytes()

        # --- enforce 10 FPS
        elapsed = time.time() - start
        sleep = CAMERA_DT - elapsed
        if sleep > 0:
            time.sleep(sleep)

# =========================
# CONTROL LOOP (60 Hz)
# =========================
def control_loop():
    pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)

    while True:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
        pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])
        time.sleep(CONTROL_DT)

# =========================
# FLASK
# =========================
app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
    <body>
        <h1>PiCar Dashboard</h1>
        <img src="/video" width="640"/>
    </body>
    </html>
    """

@app.route("/video")
def video():
    def generate():
        while True:
            with _latest_lock:
                frame = _latest_jpeg
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.03)

    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    init_model()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
