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
# CONFIGURATION
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "model.onnx")

CLASS_NAMES = ["background", "stop", "slow", "uturn", "tf_red", "tf_green"]
CONF_THRESHOLD = 0.6

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 410,
    "THROTTLE_SLOW_PWM": 400,
    "THROTTLE_STOPPED_PWM": 393,
    "THROTTLE_REVERSE_PWM": 320,
}

MODE_LINE = "line"
MODE_SLOW = "slow"
MODE_STOP_SIGN = "stop_sign"
MODE_WAIT_RED = "wait_red"
MODE_UTURN = "uturn"

SLOW_THROTTLE_PWM = 400
UTURN_THROTTLE_PWM = 420

CONTROL_LOOP_HZ = 60
CAMERA_LOOP_HZ = 20

IMAGE_W = 160
IMAGE_H = 120
ROI_FRACTION = (0.55, 0.95)

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

        self.current_steer_pwm = self.steering_center_pwm()
        self.current_throttle_pwm = config["THROTTLE_STOPPED_PWM"]

        self.stop()

    def set_pwm_raw(self, channel, pwm_value, is_steer=None):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty16

        if is_steer is True:
            self.current_steer_pwm = pwm_value
        elif is_steer is False:
            self.current_throttle_pwm = pwm_value

        return pwm_value

    def steering_center_pwm(self):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        return int((left + right) / 2)

    def stop(self):
        self.set_pwm_raw(self.channel_throttle,
                         self.cfg["THROTTLE_STOPPED_PWM"],
                         is_steer=False)

    def close(self):
        self.stop()
        self.pca.deinit()

    def get_pwm_status(self):
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
            main={"size": (320, 240), "format": "XRGB8888"},
            transform=Transform(hflip=False, vflip=False)
        )
        self.cam.configure(config)
        self.frame = None
        self.running = False

    def start(self):
        self.cam.start()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.cam.stop()

    def loop(self):
        period = 1.0 / CAMERA_LOOP_HZ
        while self.running:
            arr = self.cam.capture_array()
            rgb = arr[..., :3]
            self.frame = rgb.copy()
            time.sleep(period)

    def get_frame(self):
        return None if self.frame is None else self.frame.copy()

# =========================================================
# LINE FOLLOWER + ONNX
# =========================================================

class WebLineFollower:

    def __init__(self, cfg):
        self.cfg = cfg
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker()

        self.auto_mode = False
        self.running = False

        self.current_mode = MODE_LINE
        self.mode_until = 0.0
        self.last_sign_check = 0.0
        self.sign_check_interval = 0.3

        self.net = None
        self._init_onnx()

        self.msg = "Startup complete."

    def _init_onnx(self):
        if not os.path.exists(ONNX_MODEL_PATH):
            self.msg = "ONNX model not found"
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.msg = "ONNX model loaded"
        except Exception as e:
            self.msg = f"ONNX load error: {e}"
            self.net = None

    def start(self):
        self.camera.start()
        self.running = True
        threading.Thread(target=self.control_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.camera.stop()
        self.motors.close()

    def set_auto_mode(self, flag):
        self.auto_mode = flag
        self.msg = "AUTO" if flag else "MANUAL"

    def _run_onnx(self, frame, tnow):
        if self.net is None:
            return

        img = cv2.resize(frame, (224, 224))
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1/255.0,
            size=(224, 224),
            swapRB=True
        )

        if blob.shape != (1,3,224,224):
            return

        self.net.setInput(blob)
        out = self.net.forward()[0]

        class_id = int(np.argmax(out))
        conf = float(out[class_id])

        if conf < CONF_THRESHOLD:
            return

        label = CLASS_NAMES[class_id]
        self._on_onnx_detected(label, conf, tnow)

    def _on_onnx_detected(self, label, score, tnow):

        if label == "stop":
            self.current_mode = MODE_STOP_SIGN
            self.mode_until = tnow + 5

        elif label == "slow":
            self.current_mode = MODE_SLOW
            self.mode_until = tnow + 5

        elif label == "uturn":
            self.current_mode = MODE_UTURN
            self.mode_until = tnow + 6

        elif label == "tf_red":
            self.current_mode = MODE_WAIT_RED

        elif label == "tf_green":
            if self.current_mode == MODE_WAIT_RED:
                self.current_mode = MODE_LINE

        self.msg = f"{label.upper()} ({score:.2f})"

    def control_loop(self):
        period = 1.0 / CONTROL_LOOP_HZ

        while self.running:
            tnow = time.time()
            frame = self.camera.get_frame()

            if frame is not None and (tnow - self.last_sign_check) > self.sign_check_interval:
                self.last_sign_check = tnow
                try:
                    self._run_onnx(frame, tnow)
                except:
                    pass

            if self.auto_mode:
                steer = self.motors.steering_center_pwm()
                throttle = self.cfg["THROTTLE_FORWARD_PWM"]

                if self.current_mode == MODE_SLOW:
                    throttle = SLOW_THROTTLE_PWM
                elif self.current_mode in (MODE_STOP_SIGN, MODE_WAIT_RED):
                    throttle = self.cfg["THROTTLE_STOPPED_PWM"]
                elif self.current_mode == MODE_UTURN:
                    steer = self.cfg["STEERING_RIGHT_PWM"]
                    throttle = UTURN_THROTTLE_PWM

                self.motors.set_pwm_raw(self.motors.channel_steer, steer, True)
                self.motors.set_pwm_raw(self.motors.channel_throttle, throttle, False)

            time.sleep(period)

# =========================================================
# FLASK WEB UI
# =========================================================

app = Flask(__name__)
lf = WebLineFollower(PWM_STEERING_THROTTLE)

@app.route("/")
def index():
    return """
    <h1>PiCar ONNX Line Follower</h1>
    <img src='/video_feed' width='640'><br>
    <form method='post' action='/cmd'>
        <button name='action' value='auto_on'>AUTO</button>
        <button name='action' value='auto_off'>MANUAL</button>
    </form>
    <p id='status'></p>
    <script>
    setInterval(()=>{
        fetch('/status').then(r=>r.json()).then(d=>{
            document.getElementById('status').innerText =
            'Mode: '+(d.auto_mode?'AUTO':'MANUAL')+
            ' | Steering: '+d.steering_pwm+
            ' | Throttle: '+d.throttle_pwm+
            ' | '+d.msg;
        });
    },500);
    </script>
    """

@app.route("/status")
def status():
    return jsonify({
        "auto_mode": lf.auto_mode,
        "steering_pwm": lf.motors.current_steer_pwm,
        "throttle_pwm": lf.motors.current_throttle_pwm,
        "msg": lf.msg
    })

@app.route("/cmd", methods=["POST"])
def cmd():
    action = request.form.get("action")
    if action == "auto_on":
        lf.set_auto_mode(True)
    elif action == "auto_off":
        lf.set_auto_mode(False)
    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = lf.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buf = cv2.imencode(".jpg", frame[:, :, ::-1])
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    try:
        lf.start()
        app.run(host="0.0.0.0", port=5000)
    finally:
        lf.stop()
