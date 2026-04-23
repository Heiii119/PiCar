#!/usr/bin/env python3

from flask import Flask, Response, render_template_string, jsonify, request
import time
import threading
import cv2
import numpy as np
from smbus2 import SMBus
import os

# =========================
# CONFIG
# =========================
DEVICE = "/dev/video4"
PORT = 6088

MODEL_PATH = "model.tflite"
TPU_MODEL_PATH = "model_edgetpu.tflite"

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
I2C_BUS = 0

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
# MODEL INITIALIZATION
# =========================
interpreter = None
input_details = None
output_details = None
USE_TPU = False

def init_model():
    global interpreter, input_details, output_details, USE_TPU

    try:
        from tflite_runtime.interpreter import Interpreter
        from tflite_runtime.interpreter import load_delegate

        if os.path.exists(TPU_MODEL_PATH):
            print("✅ Loading Edge TPU model...")
            interpreter = Interpreter(
                model_path=TPU_MODEL_PATH,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
            USE_TPU = True
        else:
            print("✅ Loading CPU TFLite model...")
            interpreter = Interpreter(model_path=MODEL_PATH)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    except Exception as e:
        print("Model load failed:", e)

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

        # Curvature approximation
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
_latest_lock = threading.Lock()
_latest_frame = None
_latest_jpeg = None
_latest_seq = 0
frame_counter = 0

def camera_worker():
    global _latest_frame, _latest_jpeg, _latest_seq
    global frame_counter, CURRENT_LABEL

    cap = cv2.VideoCapture(DEVICE)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _latest_frame = frame.copy()
        frame_counter += 1

        if frame_counter % 4 == 0:
            if MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and not E_STOP:

                input_shape = input_details[0]['shape']
                height = input_shape[1]
                width = input_shape[2]

                img = cv2.resize(frame, (width, height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)

                if input_details[0]['dtype'] == np.float32:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]

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

        _, enc = cv2.imencode(".jpg", frame)
        with _latest_lock:
            _latest_jpeg = enc.tobytes()
            _latest_seq += 1

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
    if _latest_frame is not None:
        calibrate_line(_latest_frame)
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

    print("Open http://<board-ip>:6088/")
    app.run(host="0.0.0.0", port=PORT)
