#!/usr/bin/env python3
"""
PiCar - Web Line Follower + Manual Control + Color Calibration
"""

import os
import time
import csv
import threading
from datetime import datetime

import numpy as np
from flask import Flask, request, redirect, url_for, Response, render_template_string, jsonify

from picamera2 import Picamera2
from libcamera import Transform
import cv2

import board
import busio
from adafruit_pca9685 import PCA9685

from PIL import Image
from tflite_runtime.interpreter import Interpreter


# =========================================================
# Configuration
# =========================================================

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 320,
}

CONTROL_LOOP_HZ = 50
CAMERA_LOOP_HZ = 20
IMAGE_W = 160
IMAGE_H = 120

CAM_STREAM_W = 320
CAM_STREAM_H = 240


# =========================================================
# PCA9685 Helpers
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
        self.set_pwm_raw(
            self.channel_throttle,
            self.cfg["THROTTLE_STOPPED_PWM"],
            is_steer=False
        )

    def get_pwm_status(self):
        with self.lock:
            return {
                "steering_pwm": self.current_steer_pwm,
                "throttle_pwm": self.current_throttle_pwm
            }


# =========================================================
# Camera Worker
# =========================================================

class CameraWorker:

    def __init__(self):
        self.cam = Picamera2()

        config = self.cam.create_video_configuration(
            main={"size": (CAM_STREAM_W, CAM_STREAM_H), "format": "XRGB8888"},
            transform=Transform(hflip=False, vflip=False)
        )

        self.cam.configure(config)

        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.period = 1.0 / CAMERA_LOOP_HZ

    def start(self):
        self.cam.start()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        next_t = time.time()

        while self.running:
            arr = self.cam.capture_array()
            rgb = arr[..., :3]

            with self.lock:
                self.frame = rgb.copy()

            next_t += self.period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()


# =========================================================
# Web Line Follower
# =========================================================

class WebLineFollower:

    def __init__(self, cfg):
        self.cfg = cfg
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker()

        self.running = False
        self.auto_mode = False
        self.last_error = 0.0
        self.msg = "Startup complete"

    def start(self):
        self.camera.start()
        self.running = True
        threading.Thread(target=self.control_loop, daemon=True).start()

    def control_loop(self):
        next_t = time.time()
        period = 1.0 / CONTROL_LOOP_HZ

        while self.running:
            frame = self.camera.get_frame()

            if frame is not None and self.auto_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                small = cv2.resize(gray, (IMAGE_W, IMAGE_H))

                _, thresh = cv2.threshold(
                    small,
                    100,
                    255,
                    cv2.THRESH_BINARY_INV
                )

                M = cv2.moments(thresh)

                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    error = (cx - IMAGE_W/2) / (IMAGE_W/2)
                    self.last_error = error

                    if error < -0.2:
                        steer = self.cfg["STEERING_LEFT_PWM"]
                    elif error > 0.2:
                        steer = self.cfg["STEERING_RIGHT_PWM"]
                    else:
                        steer = self.motors.steering_center_pwm()

                    self.motors.set_pwm_raw(
                        self.motors.channel_steer,
                        steer,
                        is_steer=True
                    )

                    self.motors.set_pwm_raw(
                        self.motors.channel_throttle,
                        self.cfg["THROTTLE_FORWARD_PWM"],
                        is_steer=False
                    )

            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    def set_auto_mode(self, flag):
        self.auto_mode = flag
        self.msg = "AUTO mode" if flag else "MANUAL mode"

    def manual_stop(self):
        self.auto_mode = False
        self.motors.stop()
        self.msg = "Stopped"

    def get_status(self):
        pwm = self.motors.get_pwm_status()
        return {
            "auto_mode": self.auto_mode,
            "error": round(self.last_error, 3),
            "steering_pwm": pwm["steering_pwm"],
            "throttle_pwm": pwm["throttle_pwm"],
            "msg": self.msg
        }


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)
lf = WebLineFollower(PWM_STEERING_THROTTLE)


@app.route("/")
def index():
    return """
    <h1>PiCar Web Control</h1>
    <img src="/video_feed" width="640"><br><br>

    <form method="post" action="/cmd">
        <button name="action" value="auto_on">AUTO</button>
        <button name="action" value="auto_off">MANUAL</button>
        <button name="action" value="stop">STOP</button>
    </form>
    """


@app.route("/status")
def status():
    return jsonify(lf.get_status())


@app.route("/cmd", methods=["POST"])
def cmd():
    action = request.form.get("action")

    if action == "auto_on":
        lf.set_auto_mode(True)
    elif action == "auto_off":
        lf.set_auto_mode(False)
    elif action == "stop":
        lf.manual_stop()

    return redirect(url_for("index"))


@app.route("/video_feed")
def video_feed():

    def gen():
        while True:
            frame = lf.camera.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            ret, buffer = cv2.imencode(".jpg", frame[:, :, ::-1])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    try:
        lf.start()
        app.run(host="0.0.0.0", port=5000)
    finally:
        lf.running = False
