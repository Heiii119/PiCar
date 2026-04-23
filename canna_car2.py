#!/usr/bin/env python3

import os
import time
import threading
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# Camera
from picamera2 import Picamera2
from libcamera import Transform
import cv2

# PWM
import board
import busio
from adafruit_pca9685 import PCA9685

# =========================================================
# CONFIG
# =========================================================

PWM_CONFIG = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 320,
}

IMAGE_W = 160
IMAGE_H = 120
ROI_FRACTION = (0.6, 0.95)
BIN_THRESH = 0.45
CONTROL_LOOP_HZ = 120

# =========================================================
# PCA9685
# =========================================================

def parse_pin(pin_str):
    left, chan = pin_str.split(":")
    bus_str = left.split(".")[1]
    addr_str = chan.split(".")[0]
    channel_str = chan.split(".")[1]
    return int(bus_str), int(addr_str, 16), int(channel_str)

class MotorController:
    def __init__(self, cfg):
        _, addr, steer_ch = parse_pin(cfg["PWM_STEERING_PIN"])
        _, _, throttle_ch = parse_pin(cfg["PWM_THROTTLE_PIN"])

        self.cfg = cfg
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=addr)
        self.pca.frequency = 60

        self.steer_ch = steer_ch
        self.throttle_ch = throttle_ch

        self.lock = threading.Lock()
        self.stop()

    def set_pwm(self, ch, val):
        val = int(np.clip(val, 0, 4095))
        duty = int((val / 4095) * 65535)
        with self.lock:
            self.pca.channels[ch].duty_cycle = duty
        return val

    def steer(self, val):
        return self.set_pwm(self.steer_ch, val)

    def throttle(self, val):
        return self.set_pwm(self.throttle_ch, val)

    def stop(self):
        self.throttle(self.cfg["THROTTLE_STOPPED_PWM"])

# =========================================================
# CAMERA
# =========================================================

class Camera:
    def __init__(self):
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (320, 240), "format": "XRGB8888"},
            transform=Transform()
        )
        self.cam.configure(config)
        self.frame = None
        self.running = False

    def start(self):
        self.cam.start()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.running:
            arr = self.cam.capture_array()
            self.frame = arr[..., :3]

    def get(self):
        return self.frame.copy() if self.frame is not None else None

# =========================================================
# LINE FOLLOWER
# =========================================================

class Car:
    def __init__(self):
        self.motors = MotorController(PWM_CONFIG)
        self.camera = Camera()
        self.auto_mode = False
        self.detection_mode = "gray"
        self.last_error = 0

    def start(self):
        self.camera.start()
        threading.Thread(target=self.control_loop, daemon=True).start()

    def control_loop(self):
        period = 1.0 / CONTROL_LOOP_HZ
        while True:
            if self.auto_mode:
                self.auto_drive()
            time.sleep(period)

    def auto_drive(self):
        frame = self.camera.get()
        if frame is None:
            return

        small = cv2.resize(frame, (IMAGE_W, IMAGE_H))
        y0 = int(IMAGE_H * ROI_FRACTION[0])
        roi = small[y0:IMAGE_H, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) / 255.0
        mask = (gray < BIN_THRESH).astype(np.uint8)

        xs = np.where(mask.sum(axis=0) > 0)[0]
        if len(xs) == 0:
            self.motors.stop()
            return

        center = xs.mean()
        error = (center / IMAGE_W) * 2 - 1
        self.last_error = error

        if error < -0.1:
            steer = PWM_CONFIG["STEERING_LEFT_PWM"]
        elif error > 0.1:
            steer = PWM_CONFIG["STEERING_RIGHT_PWM"]
        else:
            steer = (PWM_CONFIG["STEERING_LEFT_PWM"] +
                     PWM_CONFIG["STEERING_RIGHT_PWM"]) // 2

        self.motors.steer(steer)
        self.motors.throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])

    # ===== MANUAL =====
    def manual_forward(self):
        self.auto_mode = False
        self.motors.throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])

    def manual_reverse(self):
        self.auto_mode = False
        self.motors.throttle(PWM_CONFIG["THROTTLE_REVERSE_PWM"])

    def manual_left(self):
        self.auto_mode = False
        self.motors.steer(PWM_CONFIG["STEERING_LEFT_PWM"])

    def manual_right(self):
        self.auto_mode = False
        self.motors.steer(PWM_CONFIG["STEERING_RIGHT_PWM"])

    def manual_stop(self):
        self.auto_mode = False
        self.motors.stop()

    def center(self):
        mid = (PWM_CONFIG["STEERING_LEFT_PWM"] +
               PWM_CONFIG["STEERING_RIGHT_PWM"]) // 2
        self.motors.steer(mid)

# =========================================================
# WEB APP
# =========================================================

app = Flask(__name__)
car = Car()
car.start()

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>PiCar</title>
<style>
body{text-align:center;font-family:sans-serif;}
button{padding:10px 20px;margin:5px;font-size:16px;}
</style>
</head>
<body>

<h1>PiCar Control</h1>
<img src="/video" width="480"><br><br>

<button onclick="cmd('auto')">AUTO</button>
<button onclick="cmd('manual')">MANUAL</button>
<button onclick="cmd('stop')">STOP</button>
<br><br>

<button onclick="cmd('forward')">▲</button><br>
<button onclick="cmd('left')">◀</button>
<button onclick="cmd('center')">■</button>
<button onclick="cmd('right')">▶</button><br>
<button onclick="cmd('reverse')">▼</button>

<script>
function cmd(a){
fetch("/cmd",{method:"POST",
headers:{"Content-Type":"application/x-www-form-urlencoded"},
body:"action="+a});
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/cmd", methods=["POST"])
def cmd():
    a = request.form.get("action")
    if a == "auto":
        car.auto_mode = True
    elif a == "manual":
        car.auto_mode = False
    elif a == "forward":
        car.manual_forward()
    elif a == "reverse":
        car.manual_reverse()
    elif a == "left":
        car.manual_left()
    elif a == "right":
        car.manual_right()
    elif a == "stop":
        car.manual_stop()
    elif a == "center":
        car.center()
    return ("", 204)

@app.route("/video")
def video():
    def gen():
        while True:
            frame = car.camera.get()
            if frame is None:
                continue
            _, buf = cv2.imencode(".jpg", frame[:, :, ::-1])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    return Response(gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
