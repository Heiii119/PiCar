#!/usr/bin/env python3

import time
import threading
import signal
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
from smbus2 import SMBus

# ============================================================
# CONFIG
# ============================================================

WIDTH = 320
HEIGHT = 320
FPS = 30

MODEL_PATH = "best_int8_edgetpu.tflite"
STOP_CLASS_ID = 0
CONF_THRESHOLD = 0.5

AUTO_THROTTLE = 405
THROTTLE_STOP = 370
STEER_CENTER = 380
STEER_MIN = 305
STEER_MAX = 480

PCA9685_ADDR = 0x40
I2C_BUS = 1
PORT = 8180

# ============================================================
# PCA9685 DRIVER
# ============================================================

class PCA9685:
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus=1, addr=0x40, freq=60):
        self.bus = SMBus(bus)
        self.addr = addr
        self.set_pwm_freq(freq)

    def write(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    def read(self, reg):
        return self.bus.read_byte_data(self.addr, reg)

    def set_pwm_freq(self, freq):
        prescale = int(25000000.0 / (4096 * freq) - 1)
        old = self.read(self.MODE1)
        self.write(self.MODE1, old | 0x10)
        self.write(self.PRESCALE, prescale)
        self.write(self.MODE1, old)
        time.sleep(0.005)
        self.write(self.MODE1, old | 0x80)

    def set_pwm(self, ch, val):
        reg = self.LED0_ON_L + 4 * ch
        self.write(reg, 0)
        self.write(reg+1, 0)
        self.write(reg+2, val & 0xFF)
        self.write(reg+3, val >> 8)

# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
)
picam2.start()

def get_frame():
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ============================================================
# CORAL EDGE TPU
# ============================================================

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[
        tflite.load_delegate("libedgetpu.so.1")
    ]
)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# ============================================================
# GLOBAL STATE
# ============================================================

pwm = PCA9685(I2C_BUS, PCA9685_ADDR)

autopilot = False
stop_until = 0

# ============================================================
# LINE FOLLOWING
# ============================================================

LAB_LOWER = np.array([0, 0, 0])
LAB_UPPER = np.array([255, 255, 255])

def line_follow(frame):
    roi = frame[int(HEIGHT*0.5):HEIGHT, :]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, LAB_LOWER, LAB_UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    error = (cx - WIDTH//2) / (WIDTH//2)
    return error

# ============================================================
# CORAL DETECTION
# ============================================================

def detect_stop(frame):
    global stop_until

    resized = cv2.resize(frame, (input_width, input_height))
    input_data = np.expand_dims(resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD and int(classes[i]) == STOP_CLASS_ID:
            stop_until = time.time() + 2.5

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * WIDTH)
            y1 = int(ymin * HEIGHT)
            x2 = int(xmax * WIDTH)
            y2 = int(ymax * HEIGHT)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, "STOP", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame

# ============================================================
# AUTOPILOT LOOP
# ============================================================

def auto_loop():
    global autopilot

    while True:
        if not autopilot:
            time.sleep(0.1)
            continue

        frame = get_frame()
        frame = detect_stop(frame)

        if time.time() < stop_until:
            pwm.set_pwm(0, THROTTLE_STOP)
            continue

        error = line_follow(frame)
        if error is None:
            pwm.set_pwm(0, THROTTLE_STOP)
            continue

        steer = int(STEER_CENTER - error * 80)
        steer = max(STEER_MIN, min(STEER_MAX, steer))

        pwm.set_pwm(0, AUTO_THROTTLE)
        pwm.set_pwm(1, steer)

# ============================================================
# FLASK STREAM
# ============================================================

app = Flask(__name__)

def generate():
    while True:
        frame = get_frame()
        frame = detect_stop(frame)
        _, jpg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() + b"\r\n")

@app.route("/mjpg")
def mjpg():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/auto/start")
def auto_start():
    global autopilot
    autopilot = True
    return jsonify(ok=True)

@app.route("/auto/stop")
def auto_stop():
    global autopilot
    autopilot = False
    pwm.set_pwm(0, THROTTLE_STOP)
    return jsonify(ok=True)

# ============================================================
# START
# ============================================================

threading.Thread(target=auto_loop, daemon=True).start()

print(f"Open browser: http://<pi-ip>:{PORT}/mjpg")

app.run(host="0.0.0.0", port=PORT)
