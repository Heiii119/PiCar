#!/usr/bin/env python3
"""
PiCar - Web Line Follower + Manual Control + ONNX (NHWC-safe)
"""

import os
import time
import threading
import numpy as np
import cv2
import cv2.dnn

from flask import Flask, request, redirect, url_for, Response, jsonify

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

CLASS_NAMES = ["background", "stop", "slow", "uturn", "tf_red", "tf_green"]
CONF_THRESHOLD = 0.6

PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 410,
    "THROTTLE_SLOW_PWM": 400,
    "THROTTLE_STOPPED_PWM": 393,
    "THROTTLE_REVERSE_PWM": 320,
}

MODE_LINE = "line"
MODE_SLOW = "slow"
MODE_STOP = "stop"
MODE_WAIT_RED = "wait_red"
MODE_UTURN = "uturn"

SLOW_THROTTLE_PWM = 400
UTURN_THROTTLE_PWM = 420

CONTROL_LOOP_HZ = 60
CAMERA_LOOP_HZ = 20

# =========================================================
# MOTOR CONTROLLER
# =========================================================

def parse_pca9685_pin(pin_str):
    left, chan = pin_str.split(":")
    bus_str = left.split(".")[1]
    addr_str = chan.split(".")[0]
    channel_str = chan.split(".")[1]
    return int(bus_str), int(addr_str, 16), int(channel_str)

class MotorServoController:

    def __init__(self, config):
        _, addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        _, _, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])

        self.channel_steer = s_ch
        self.channel_throttle = t_ch
        self.cfg = config

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=addr)
        self.pca.frequency = 60

        self.current_steer_pwm = self.center_pwm()
        self.current_throttle_pwm = config["THROTTLE_STOPPED_PWM"]

        self.stop()

    def center_pwm(self):
        return int((self.cfg["STEERING_LEFT_PWM"] +
                    self.cfg["STEERING_RIGHT_PWM"]) / 2)

    def set_pwm(self, channel, value, is_steer=None):
        value = int(np.clip(value, 0, 4095))
        duty16 = int((value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty16

        if is_steer:
            self.current_steer_pwm = value
        else:
            self.current_throttle_pwm = value

    def stop(self):
        self.set_pwm(self.channel_throttle,
                     self.cfg["THROTTLE_STOPPED_PWM"],
                     is_steer=False)

    def close(self):
        self.stop()
        self.pca.deinit()

# =========================================================
# CAMERA
# =========================================================

class CameraWorker:

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
        period = 1.0 / CAMERA_LOOP_HZ
        while self.running:
            arr = self.cam.capture_array()
            self.frame = arr[..., :3].copy()
            time.sleep(period)

    def get_frame(self):
        return None if self.frame is None else self.frame.copy()

# =========================================================
# MAIN CONTROLLER
# =========================================================

class PiCar:

    def __init__(self):
        self.motors = MotorServoController(PWM_STEERING_THROTTLE)
        self.camera = CameraWorker()
        self.auto_mode = False
        self.current_mode = MODE_LINE
        self.mode_until = 0
        self.last_infer = 0

        self.net = None
        self.load_onnx()

    def load_onnx(self):
        if not os.path.exists(ONNX_MODEL_PATH):
            print("ONNX model not found")
            return
        self.net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        print("ONNX loaded")

    # ✅ FIXED NHWC INPUT
    def run_onnx(self, frame):

        img = cv2.resize(frame, (224,224))

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1/255.0,
            size=(224,224),
            swapRB=False,
            crop=False
        )

        # NCHW → NHWC
        blob = blob.transpose(0,2,3,1)

        self.net.setInput(blob)
        out = self.net.forward()[0]

        class_id = int(np.argmax(out))
        conf = float(out[class_id])

        if conf < CONF_THRESHOLD:
            return

        label = CLASS_NAMES[class_id]
        print("Detected:", label, conf)

        if label == "stop":
            self.current_mode = MODE_STOP
            self.mode_until = time.time() + 5

        elif label == "slow":
            self.current_mode = MODE_SLOW
            self.mode_until = time.time() + 5

        elif label == "uturn":
            self.current_mode = MODE_UTURN
            self.mode_until = time.time() + 6

        elif label == "tf_red":
            self.current_mode = MODE_WAIT_RED

        elif label == "tf_green":
            if self.current_mode == MODE_WAIT_RED:
                self.current_mode = MODE_LINE

    def control_loop(self):
        period = 1.0 / CONTROL_LOOP_HZ
        while True:
            tnow = time.time()
            frame = self.camera.get_frame()

            if frame is not None and self.net is not None:
                if tnow - self.last_infer > 0.3:
                    self.last_infer = tnow
                    try:
                        self.run_onnx(frame)
                    except Exception as e:
                        print("ONNX error:", e)

            if self.auto_mode:
                steer = self.motors.center_pwm()
                throttle = PWM_STEERING_THROTTLE["THROTTLE_FORWARD_PWM"]

                if self.current_mode == MODE_SLOW:
                    throttle = SLOW_THROTTLE_PWM
                elif self.current_mode in (MODE_STOP, MODE_WAIT_RED):
                    throttle = PWM_STEERING_THROTTLE["THROTTLE_STOPPED_PWM"]
                elif self.current_mode == MODE_UTURN:
                    steer = PWM_STEERING_THROTTLE["STEERING_RIGHT_PWM"]
                    throttle = UTURN_THROTTLE_PWM

                self.motors.set_pwm(self.motors.channel_steer, steer, True)
                self.motors.set_pwm(self.motors.channel_throttle, throttle, False)

            time.sleep(period)

# =========================================================
# WEB SERVER
# =========================================================

app = Flask(__name__)
car = PiCar()
car.camera.start()
threading.Thread(target=car.control_loop, daemon=True).start()

@app.route("/")
def index():
    return """
    <h1>PiCar ONNX</h1>
    <img src='/video'>
    <form method='post' action='/cmd'>
        <button name='a' value='auto'>AUTO</button>
        <button name='a' value='manual'>MANUAL</button>
    </form>
    """

@app.route("/cmd", methods=["POST"])
def cmd():
    if request.form.get("a") == "auto":
        car.auto_mode = True
    else:
        car.auto_mode = False
    return redirect("/")

@app.route("/video")
def video():
    def gen():
        while True:
            frame = car.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buf = cv2.imencode(".jpg", frame[:,:,::-1])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
