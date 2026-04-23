#!/usr/bin/env python3
"""
PiCar - Web Line Follower + Manual Control + ONNX Traffic Sign Detection
"""

import os
import time
import csv
import threading
from datetime import datetime
import numpy as np
import cv2
import cv2.dnn

from flask import (
    Flask, request, redirect, url_for, Response,
    render_template_string, jsonify
)

from picamera2 import Picamera2
from libcamera import Transform

import board
import busio
from adafruit_pca9685 import PCA9685

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.onnx")

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

IMAGE_W = 160
IMAGE_H = 120
ROI_FRACTION = (0.55, 0.95)

DEAD_BAND_ON = 0.14
DEAD_BAND_OFF = 0.08

CAM_STREAM_W = 320
CAM_STREAM_H = 240

SLOW_THROTTLE_PWM = 395

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 320,
}

# =========================================================
# PCA9685
# =========================================================

def parse_pca9685_pin(pin_str):
    left, chan = pin_str.split(":")
    bus_str = left.split(".")[1]
    addr_str = chan.split(".")[0]
    channel_str = chan.split(".")[1]
    return int(bus_str), int(addr_str, 16), int(channel_str)


class MotorServoController:

    def __init__(self, config):
        s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])

        self.channel_steer = s_ch
        self.channel_throttle = t_ch

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60

        self.cfg = config
        self.lock = threading.Lock()

        self.current_steer_pwm = self.steering_center_pwm()
        self.current_throttle_pwm = config["THROTTLE_STOPPED_PWM"]

        self.stop()

    def set_pwm_raw(self, channel, pwm_value, is_steer=None):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)

        with self.lock:
            self.pca.channels[channel].duty_cycle = duty16
            if is_steer is True:
                self.current_steer_pwm = pwm_value
            elif is_steer is False:
                self.current_throttle_pwm = pwm_value
        return pwm_value

    def steering_center_pwm(self):
        return int((self.cfg["STEERING_LEFT_PWM"] +
                    self.cfg["STEERING_RIGHT_PWM"]) / 2)

    def stop(self):
        self.set_pwm_raw(self.channel_throttle,
                         self.cfg["THROTTLE_STOPPED_PWM"], False)

    def get_pwm_status(self):
        with self.lock:
            return {
                "steering_pwm": self.current_steer_pwm,
                "throttle_pwm": self.current_throttle_pwm,
            }


# =========================================================
# CAMERA
# =========================================================

class CameraWorker:
    def __init__(self):
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (CAM_STREAM_W, CAM_STREAM_H),
                  "format": "XRGB8888"},
            transform=Transform()
        )
        self.cam.configure(config)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.cam.start()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.running:
            arr = self.cam.capture_array()
            rgb = arr[..., :3]
            with self.lock:
                self.frame = rgb.copy()

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# =========================================================
# MAIN SYSTEM
# =========================================================

class WebLineFollower:

    def __init__(self, cfg):

        self.cfg = cfg
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker()

        self.auto_mode = False
        self.running = False
        self.last_center_err = 0
        self.last_decision = "STRAIGHT"
        self.msg = "System Ready"

        # UTURN STATE MACHINE
        self.uturn_active = False
        self.uturn_stage = 0
        self.uturn_start = 0

        # LOAD ONNX
        try:
            self.net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
            self.msg = "ONNX model loaded"
        except Exception as e:
            self.net = None
            self.msg = f"Model load failed: {e}"

    # =========================
    # UTURN STATE MACHINE
    # =========================

    def start_uturn(self):
        self.uturn_active = True
        self.uturn_stage = 0
        self.uturn_start = time.time()

    def update_uturn(self):

        if not self.uturn_active:
            return False

        elapsed = time.time() - self.uturn_start
        cfg = self.cfg

        if self.uturn_stage == 0:
            self.motors.set_pwm_raw(self.motors.channel_throttle,
                                    SLOW_THROTTLE_PWM, False)
            self.motors.set_pwm_raw(self.motors.channel_steer,
                                    cfg["STEERING_RIGHT_PWM"], True)
            if elapsed > 0.5:
                self.uturn_stage = 1
                self.uturn_start = time.time()

        elif self.uturn_stage == 1:
            self.motors.set_pwm_raw(self.motors.channel_throttle,
                                    cfg["THROTTLE_FORWARD_PWM"], False)
            self.motors.set_pwm_raw(self.motors.channel_steer,
                                    cfg["STEERING_RIGHT_PWM"], True)
            if elapsed > 3:
                self.uturn_stage = 2
                self.uturn_start = time.time()

        elif self.uturn_stage == 2:
            self.motors.set_pwm_raw(self.motors.channel_throttle,
                                    cfg["THROTTLE_REVERSE_PWM"] - 5, False)
            self.motors.set_pwm_raw(self.motors.channel_steer,
                                    cfg["STEERING_LEFT_PWM"], True)
            if elapsed > 3:
                self.uturn_stage = 3
                self.uturn_start = time.time()

        elif self.uturn_stage == 3:
            self.motors.set_pwm_raw(self.motors.channel_steer,
                                    self.motors.steering_center_pwm(), True)
            if elapsed > 0.3:
                self.uturn_active = False

        return True

    # =========================
    # LINE DETECTION
    # =========================

    def detect_line(self, frame):

        small = cv2.resize(frame, (IMAGE_W, IMAGE_H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        y0 = int(IMAGE_H * ROI_FRACTION[0])
        roi = gray[y0:IMAGE_H, :]

        _, thresh = cv2.threshold(roi, 120, 255,
                                  cv2.THRESH_BINARY_INV)

        M = cv2.moments(thresh)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        center_norm = (cx / IMAGE_W) * 2 - 1
        return center_norm

    # =========================
    # ONNX SIGN DETECTION
    # =========================

    def check_signs(self, frame):

        if self.net is None:
            return

        img = cv2.resize(frame, (224, 224))
        blob = cv2.dnn.blobFromImage(img, 1/255.0,
                                     (224, 224))
        blob = blob.transpose(0, 2, 3, 1)

        self.net.setInput(blob)
        output = self.net.forward()[0]

        class_id = int(np.argmax(output))
        confidence = float(output[class_id])
        label = CLASS_NAMES[class_id]

        if confidence < CONF_THRESHOLD:
            return

        self.msg = f"{label} ({confidence:.2f})"

        if label in ["stop", "person"]:
            self.motors.set_pwm_raw(
                self.motors.channel_throttle,
                self.cfg["THROTTLE_STOPPED_PWM"], False)

        elif label == "go":
            self.motors.set_pwm_raw(
                self.motors.channel_throttle,
                self.cfg["THROTTLE_FORWARD_PWM"], False)

        elif label == "slow":
            self.motors.set_pwm_raw(
                self.motors.channel_throttle,
                SLOW_THROTTLE_PWM, False)

        elif label == "Uturn":
            if not self.uturn_active:
                print("U-TURN detected")
                self.start_uturn()

    # =========================
    # CONTROL LOOP
    # =========================

    def control_loop(self):

        while self.running:

            frame = self.camera.get_frame()
            if frame is None:
                continue

            if self.auto_mode:

                # UTURN override
                if self.update_uturn():
                    continue

                # LINE FOLLOW
                err = self.detect_line(frame)

                if err is not None:
                    if err < -DEAD_BAND_ON:
                        steer = self.cfg["STEERING_LEFT_PWM"]
                        self.last_decision = "LEFT"
                    elif err > DEAD_BAND_ON:
                        steer = self.cfg["STEERING_RIGHT_PWM"]
                        self.last_decision = "RIGHT"
                    else:
                        steer = self.motors.steering_center_pwm()
                        self.last_decision = "STRAIGHT"

                    self.motors.set_pwm_raw(
                        self.motors.channel_steer, steer, True)

                    self.motors.set_pwm_raw(
                        self.motors.channel_throttle,
                        self.cfg["THROTTLE_FORWARD_PWM"], False)

                # SIGN DETECTION
                self.check_signs(frame)

            time.sleep(0.02)

    # =========================
    # LIFECYCLE
    # =========================

    def start(self):
        self.camera.start()
        self.running = True
        threading.Thread(target=self.control_loop,
                         daemon=True).start()

    def stop(self):
        self.running = False
        self.motors.stop()

    def get_status(self):
        pwm = self.motors.get_pwm_status()
        return {
            "auto_mode": self.auto_mode,
            "decision": self.last_decision,
            "steering_pwm": pwm["steering_pwm"],
            "throttle_pwm": pwm["throttle_pwm"],
            "msg": self.msg
        }


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
lf = WebLineFollower(PWM_STEERING_THROTTLE)

@app.route("/")
def index():
    return "<h1>PiCar ONNX Running</h1><a href='/video'>Video</a>"

@app.route("/status")
def status():
    return jsonify(lf.get_status())

@app.route("/auto")
def auto():
    lf.auto_mode = True
    return "AUTO ON"

@app.route("/manual")
def manual():
    lf.auto_mode = False
    return "MANUAL"

@app.route("/video")
def video():
    def gen():
        while True:
            frame = lf.camera.get_frame()
            if frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg",
                                       frame[:, :, ::-1])
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        lf.start()
        app.run(host="0.0.0.0", port=5000)
    finally:
        lf.stop()
