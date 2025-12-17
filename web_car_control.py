#!/usr/bin/env python3
import time
import threading
import subprocess

from flask import Flask, request, redirect, url_for

import board
import busio
from adafruit_pca9685 import PCA9685
import numpy as np

# -------------------------
# PWM configuration (same style as line.py)
# -------------------------
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 393,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 330,
}

# -------------------------
# Helpers to parse pin string
# -------------------------
def parse_pca9685_pin(pin_str):
    try:
        left, chan = pin_str.split(":")
        bus_str = left.split(".")[1]
        addr_str = chan.split(".")[0] if "." in chan else chan
        channel_str = chan.split(".")[1] if "." in chan else "0"
        i2c_bus = int(bus_str)
        i2c_addr = int(addr_str, 16) if addr_str.startswith(("0x", "0X")) else int(addr_str)
        channel = int(channel_str)
        return i2c_bus, i2c_addr, channel
    except Exception as e:
        raise ValueError(f"Invalid PCA9685 pin format: {pin_str}") from e

# -------------------------
# Motor / Servo controller
# -------------------------
class MotorServoController:
    def __init__(self, config):
        s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])
        if s_bus != t_bus or s_addr != t_addr:
            raise ValueError("Steering and Throttle must be on same PCA9685 for this simple driver.")
        self.channel_steer = s_ch
        self.channel_throttle = t_ch
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60
        self.cfg = config
        self.lock = threading.Lock()
        self.stop()

    def set_pwm_raw(self, channel, pwm_value):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty16
        return pwm_value

    def steering_center_pwm(self):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        return int(round((left + right) / 2))

    def stop(self):
        with self.lock:
            self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_STOPPED_PWM"])

    def set_throttle_forward(self):
        with self.lock:
            self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_FORWARD_PWM"])

    def set_throttle_reverse(self):
        with self.lock:
            self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_REVERSE_PWM"])

    def set_throttle_neutral(self):
        with self.lock:
            self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_STOPPED_PWM"])

    def set_steer_left(self):
        with self.lock:
            self.set_pwm_raw(self.channel_steer, self.cfg["STEERING_LEFT_PWM"])

    def set_steer_right(self):
        with self.lock:
            self.set_pwm_raw(self.channel_steer, self.cfg["STEERING_RIGHT_PWM"])

    def set_steer_center(self):
        with self.lock:
            self.set_pwm_raw(self.channel_steer, self.steering_center_pwm())

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

# -------------------------
# Flask app + global car instance
# -------------------------
app = Flask(__name__)
car = MotorServoController(PWM_STEERING_THROTTLE)

# OPTIONAL: holder for line.py subprocess if you decide to integrate it later
line_process = None
line_running = False
line_script_path = "/home/pi/line.py"  # <-- adjust to your actual path

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pi Car Web Control</title>
  <style>
    body { font-family: sans-serif; text-align: center; }
    .btn { padding: 15px 30px; margin: 5px; font-size: 18px; }
    .row { margin: 10px; }
  </style>
</head>
<body>
  <h1>Pi Car Web Control</h1>
  <p>Use these buttons to control the car manually.</p>

  <div class="row">
    <form method="post" action="/cmd">
      <button class="btn" type="submit" name="action" value="forward">▲ Forward</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="left">◀ Left</button>
    </form>
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="stop">■ Stop</button>
    </form>
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="right">▶ Right</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="/cmd">
      <button class="btn" type="submit" name="action" value="reverse">▼ Reverse</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="center">Center Steering</button>
    </form>
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="neutral">Neutral Throttle</button>
    </form>
  </div>

  <hr>
  <!-- OPTIONAL: start/stop line follower (advanced, see notes in code) -->
  <div class="row">
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="start_line">Start Line Follower</button>
    </form>
    <form method="post" action="/cmd" style="display:inline;">
      <button class="btn" type="submit" name="action" value="stop_line">Stop Line Follower</button>
    </form>
  </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return HTML_PAGE

@app.route("/cmd", methods=["POST"])
def cmd():
    global line_process, line_running

    action = request.form.get("action", "")
    print(f"Received action: {action}")

    # Safety: basic rule — don't drive motors manually if line follower is running
    if action in ["forward","reverse","left","right","stop","center","neutral"] and line_running:
        print("Ignoring manual command because line follower is running.")
        return redirect(url_for("index"))

    if action == "forward":
        car.set_steer_center()
        car.set_throttle_forward()

    elif action == "reverse":
        # brief neutral before reverse (some ESCs require this)
        car.set_throttle_neutral()
        time.sleep(0.2)
        car.set_steer_center()
        car.set_throttle_reverse()

    elif action == "left":
        car.set_steer_left()

    elif action == "right":
        car.set_steer_right()

    elif action == "stop":
        car.set_throttle_neutral()

    elif action == "center":
        car.set_steer_center()

    elif action == "neutral":
        car.set_throttle_neutral()

    # ----- OPTIONAL: start/stop line follower script -----
    elif action == "start_line":
        if not line_running:
            # Make sure motors are neutral before giving control to line.py
            car.set_throttle_neutral()
            car.set_steer_center()
            try:
                line_process = subprocess.Popen(
                    ["python3", line_script_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                line_running = True
                print("Started line follower.")
            except Exception as e:
                print(f"Failed to start line follower: {e}")

    elif action == "stop_line":
        if line_running and line_process is not None:
            try:
                line_process.terminate()
                line_process.wait(timeout=2.0)
                print("Stopped line follower.")
            except Exception as e:
                print(f"Error stopping line follower: {e}")
            finally:
                line_running = False
                line_process = None
                # Ensure motors are stopped
                car.set_throttle_neutral()
                car.set_steer_center()

    return redirect(url_for("index"))

if __name__ == "__main__":
    try:
        # Listen on all interfaces so Tailscale traffic can reach it.
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        car.close()
