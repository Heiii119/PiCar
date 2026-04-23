#!/usr/bin/env python3
"""
PiCar - Web Line Follower + Manual Control + Color Calibration
NOW USING model.onnx + Confidence Display
"""

import os
import time
import csv
import threading
from datetime import datetime
import numpy as np
import cv2

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

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 320,
}

MODE_LINE      = "line"
MODE_SLOW      = "slow"
MODE_STOP_SIGN = "stop_sign"
MODE_WAIT_RED  = "wait_red"
MODE_UTURN     = "uturn"

SLOW_THROTTLE_PWM  = 395
UTURN_THROTTLE_PWM = 420

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.onnx")

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

IMAGE_W = 160
IMAGE_H = 120
ROI_FRACTION = (0.55, 0.95)

DEAD_BAND_ON  = 0.14
DEAD_BAND_OFF = 0.08
NO_LINE_TIMEOUT      = 0.5
MAX_REVERSE_DURATION = 6.0

CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ  = 25

CAM_STREAM_W = 320
CAM_STREAM_H = 240

DATA_ROOT = "data"

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

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

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

    def stop(self):
        self.running = False
        self.cam.stop()

    def loop(self):
        while self.running:
            arr = self.cam.capture_array()
            rgb = arr[..., :3]
            with self.lock:
                self.frame = rgb.copy()
            time.sleep(1.0 / CAMERA_LOOP_HZ)

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

        self.running = False
        self.auto_mode = False
        self.recording = False

        self.last_line_time = 0.0
        self.last_center_err = 0.0
        self.last_decision = "STRAIGHT"

        self.current_mode = MODE_LINE
        self.mode_until = 0.0

        self.last_sign_label = "none"
        self.last_sign_conf = 0.0

        self.ctrl_period = 1.0 / CONTROL_LOOP_HZ
        self.msg = "Startup: MANUAL"

        self._init_sign_classifier()

    # =====================================================
    # ONNX SIGN CLASSIFIER
    # =====================================================

    def _init_sign_classifier(self):
        try:
            if not os.path.exists(ONNX_MODEL_PATH):
                self.msg = "ONNX model not found"
                self.sign_net = None
                return

            self.sign_net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
            self.msg = "ONNX model loaded"
        except Exception as e:
            self.sign_net = None
            self.msg = f"ONNX load failed: {e}"

    def _check_traffic_signs(self, frame_rgb, tnow):

        if self.sign_net is None:
            return

        img = cv2.resize(frame_rgb, (224, 224))
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224))
        blob = blob.transpose(0, 2, 3, 1)

        self.sign_net.setInput(blob)
        output = self.sign_net.forward()[0]

        class_id = int(np.argmax(output))
        confidence = float(output[class_id])
        label = CLASS_NAMES[class_id]

        if confidence < CONF_THRESHOLD:
            return

        self.last_sign_label = label
        self.last_sign_conf = confidence

        self._on_sign_detected(label, confidence, tnow)

    def _on_sign_detected(self, label, score, tnow):

        l = label.lower()

        if l in ("stop", "person"):
            self.current_mode = MODE_STOP_SIGN
            self.mode_until = tnow + 10.0

        elif l == "slow":
            self.current_mode = MODE_SLOW
            self.mode_until = tnow + 5.0

        elif l == "uturn":
            self.current_mode = MODE_UTURN
            self.mode_until = tnow + 10.0

        elif l == "go":
            self.current_mode = MODE_LINE
            self.mode_until = 0.0

        self.msg = f"{label} ({score:.2f})"

    # =====================================================
    # CONTROL LOOP
    # =====================================================

    def control_loop(self):

        next_t = time.time()

        while self.running:

            tnow = time.time()
            frame = self.camera.get_frame()

            if frame is not None and self.auto_mode:
                self._check_traffic_signs(frame, tnow)

                if self.current_mode == MODE_STOP_SIGN:
                    self.motors.set_pwm_raw(
                        self.motors.channel_throttle,
                        self.cfg["THROTTLE_STOPPED_PWM"], False)

                elif self.current_mode == MODE_SLOW:
                    self.motors.set_pwm_raw(
                        self.motors.channel_throttle,
                        SLOW_THROTTLE_PWM, False)

                elif self.current_mode == MODE_LINE:
                    self.motors.set_pwm_raw(
                        self.motors.channel_throttle,
                        self.cfg["THROTTLE_FORWARD_PWM"], False)

            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def start(self):
        self.camera.start()
        self.running = True
        threading.Thread(target=self.control_loop,
                         daemon=True).start()

    def stop(self):
        self.running = False
        self.camera.stop()
        self.motors.stop()
        self.motors.close()

    def get_status(self):
        pwm = self.motors.get_pwm_status()
        return {
            "auto_mode": self.auto_mode,
            "sign_mode": self.current_mode,
            "sign_label": self.last_sign_label,
            "sign_confidence": round(self.last_sign_conf, 3),
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
    return """
    <h1>PiCar ONNX</h1>
    <a href='/auto'>AUTO</a> |
    <a href='/manual'>MANUAL</a><br><br>
    <img src='/video_feed'><br>
    <pre id='status'></pre>
    <script>
    setInterval(() => {
      fetch('/status')
        .then(r=>r.json())
        .then(d=>{
          document.getElementById('status').innerText =
            JSON.stringify(d,null,2);
        });
    }, 500);
    </script>
    """

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

@app.route("/video_feed")
def video_feed():
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
