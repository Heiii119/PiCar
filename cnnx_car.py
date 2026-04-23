#!/usr/bin/env python3

from flask import Flask, jsonify
import time
import threading
import cv2
import numpy as np
from smbus2 import SMBus

# =========================
# CONFIG
# =========================
DEVICE = "/dev/video0"
PORT = 6088

MODEL_PATH = "model.onnx"

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

MODE = "MANUAL"
AUTOPILOT_RUNNING = False
E_STOP = False
CURRENT_LABEL = "None"

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
THROTTLE_SLOW = 405
THROTTLE_REVERSE = 305

STEERING_CENTER = 380
STEERING_MIN = 305
STEERING_MAX = 480

CONTROL_DT = 1.0 / 60.0
LINE_THRESHOLD = 100

app = Flask(__name__)

values = {
    "throttle": THROTTLE_STOPPED,
    "steering": STEERING_CENTER,
}

# =========================
# PCA9685
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
# ONNX MODEL INITIALIZATION
# =========================
net = None

def init_model():
    global net
    print("✅ Loading ONNX model (CPU)...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# =========================
# LINE CALIBRATION
# =========================
def calibrate_line(frame):
    global LINE_THRESHOLD
    roi = frame[int(frame.shape[0]*0.6):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LINE_THRESHOLD = int(np.mean(gray) * 0.8)

# =========================
# ADVANCED LINE FOLLOW
# =========================
def line_follow(frame):

    h, w = frame.shape[:2]
    roi = frame[int(h*0.6):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray, LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(gray_mask, hsv_mask)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    moments = cv2.moments(mask)

    if moments["m00"] > 1000:

        cx = int(moments["m10"] / moments["m00"])
        error = cx - (w // 2)
        normalized_error = error / (w//2)

        steer = STEERING_CENTER - int(normalized_error * 120)
        values["steering"] = max(STEERING_MIN, min(STEERING_MAX, steer))

        upper = mask[:mask.shape[0]//2, :]
        lower = mask[mask.shape[0]//2:, :]

        m1 = cv2.moments(upper)
        m2 = cv2.moments(lower)

        if m1["m00"] > 500 and m2["m00"] > 500:
            cx1 = int(m1["m10"] / m1["m00"])
            cx2 = int(m2["m10"] / m2["m00"])
            curvature = abs(cx1 - cx2)

            if curvature > 40:
                values["throttle"] = THROTTLE_SLOW
            else:
                values["throttle"] = THROTTLE_FORWARD - 5
        else:
            values["throttle"] = THROTTLE_SLOW
    else:
        values["throttle"] = THROTTLE_STOPPED

# =========================
# CAMERA THREAD
# =========================
frame_counter = 0

def camera_worker():
    global frame_counter, CURRENT_LABEL

    cap = cv2.VideoCapture(DEVICE)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1

        if frame_counter % 4 == 0:
            if MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and not E_STOP:

                img = cv2.resize(frame, (224, 224))
                blob = cv2.dnn.blobFromImage(
                    img,
                    scalefactor=1.0/255.0,
                    size=(224, 224),
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

# =========================
# CONTROL LOOP
# =========================
def control_loop():
    pwm = PCA9685(I2C_BUS, PCA9685_ADDR, PCA9685_FREQ)
    while True:
        pwm.set_pwm_12bit(THROTTLE_CHANNEL, values["throttle"])
        pwm.set_pwm_12bit(STEERING_CHANNEL, values["steering"])
        time.sleep(CONTROL_DT)

# =========================
# FLASK ROUTES
# =========================
@app.route("/")
def home():
    return """
    <h1>PiCar Control</h1>
    <p><a href="/status">Status</a></p>
    <p><a href="/mode">Toggle Mode</a></p>
    <p><a href="/autopilot/start">Start Autopilot</a></p>
    <p><a href="/autopilot/pause">Pause Autopilot</a></p>
    """

@app.route("/status")
def status():
    return jsonify(values)

@app.route("/mode")
def toggle_mode():
    global MODE
    MODE = "AUTOPILOT" if MODE == "MANUAL" else "MANUAL"
    return "OK"

@app.route("/autopilot/start")
def auto_start():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = True
    return "STARTED"

@app.route("/autopilot/pause")
def auto_pause():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = False
    return "PAUSED"

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    init_model()

    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=control_loop, daemon=True).start()

    print("✅ ONNX CPU mode active")
    print("Open http://<board-ip>:6088/")
    app.run(host="0.0.0.0", port=PORT)
  
