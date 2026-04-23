#!/usr/bin/env python3
"""
PiCar - 2 Mode System
MANUAL / AUTOPILOT
Autopilot = Line Following + Sign Detection (Parallel)
"""

import os
import time
import threading
import numpy as np
import cv2
import cv2.dnn

from flask import Flask, request, redirect, Response

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

CURRENT_LABEL = "None"
CURRENT_CONF = 0.0
MODE = "MANUAL"

PWM_CONFIG = {
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 410,
    "THROTTLE_SLOW_PWM": 400,
    "THROTTLE_STOPPED_PWM": 393,
}


# =========================================================
# MOTOR CONTROLLER
# =========================================================

class MotorController:

    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=0x40)
        self.pca.frequency = 60

        self.steer_channel = 1
        self.throttle_channel = 0

        self.stop()

    def center_pwm(self):
        return int((PWM_CONFIG["STEERING_LEFT_PWM"] +
                    PWM_CONFIG["STEERING_RIGHT_PWM"]) / 2)

    def set_pwm(self, channel, value):
        value = int(np.clip(value, 0, 4095))
        duty = int((value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty

    def set_steering(self, value):
        self.set_pwm(self.steer_channel, value)

    def set_throttle(self, value):
        self.set_pwm(self.throttle_channel, value)

    def stop(self):
        self.set_throttle(PWM_CONFIG["THROTTLE_STOPPED_PWM"])


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
            self.frame = arr[..., :3].copy()
            time.sleep(0.03)

    def get_frame(self):
        return None if self.frame is None else self.frame.copy()


# =========================================================
# CAR LOGIC
# =========================================================

class PiCar:

    def __init__(self):
        self.motors = MotorController()
        self.camera = Camera()
        self.auto_mode = False
        self.net = None
        self.load_model()

    def load_model(self):
        if os.path.exists(ONNX_MODEL_PATH):
            self.net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
            print("✅ ONNX Loaded")
        else:
            print("❌ ONNX Not Found")

    # ==============================
    # LINE FOLLOWING
    # ==============================

    def line_follow(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        h, w = thresh.shape
        roi = thresh[int(h*0.6):h, :]

        M = cv2.moments(roi)

        steer = self.motors.center_pwm()

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - w//2
            steer += int(error * 0.6)

        steer = np.clip(
            steer,
            PWM_CONFIG["STEERING_LEFT_PWM"],
            PWM_CONFIG["STEERING_RIGHT_PWM"]
        )

        self.motors.set_steering(steer)
        self.motors.set_throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])

    # ==============================
    # SIGN DETECTION (PARALLEL)
    # ==============================

    def detect_sign(self, frame):

        global CURRENT_LABEL, CURRENT_CONF

        img = cv2.resize(frame, (224,224))

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1/255.0,
            size=(224,224),
            swapRB=False,
            crop=False
        )

        # NHWC fix
        blob = blob.transpose(0,2,3,1)

        self.net.setInput(blob)
        output = self.net.forward()[0]

        class_id = int(np.argmax(output))
        confidence = float(output[class_id])
        label = CLASS_NAMES[class_id]

        CURRENT_LABEL = label
        CURRENT_CONF = float(f"{confidence:.4g}")

        if confidence < CONF_THRESHOLD:
            return

        # Override behavior if sign detected
        if label in ["stop", "person"]:
            self.motors.set_throttle(PWM_CONFIG["THROTTLE_STOPPED_PWM"])

        elif label == "slow":
            self.motors.set_throttle(PWM_CONFIG["THROTTLE_SLOW_PWM"])

        elif label == "go":
            self.motors.set_throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])

        elif label == "Uturn":
            self.motors.set_steering(PWM_CONFIG["STEERING_RIGHT_PWM"])
            self.motors.set_throttle(PWM_CONFIG["THROTTLE_FORWARD_PWM"])


    # ==============================
    # MAIN LOOP
    # ==============================

    def loop(self):

        global MODE

        while True:

            frame = self.camera.get_frame()

            if frame is None:
                continue

            if self.auto_mode:

                MODE = "AUTOPILOT"

                # 1️⃣ Always run line following
                self.line_follow(frame)

                # 2️⃣ Run sign detection in parallel
                if self.net is not None:
                    self.detect_sign(frame)

            else:
                MODE = "MANUAL"

            time.sleep(0.03)


# =========================================================
# WEB SERVER
# =========================================================

app = Flask(__name__)
car = PiCar()
car.camera.start()
threading.Thread(target=car.loop, daemon=True).start()


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
        car.motors.stop()
    return redirect("/")


@app.route("/video")
def video():

    def gen():
        global MODE, CURRENT_LABEL, CURRENT_CONF

        while True:
            frame = car.camera.get_frame()

            if frame is None:
                continue

            cv2.putText(frame, f"Mode: {MODE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame,
                        f"Label: {CURRENT_LABEL} ({CURRENT_CONF})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
