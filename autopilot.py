import os
import sys
import time
import csv
import tty
import termios
import select
from datetime import datetime
from glob import glob

import numpy as np

# Picamera2
from picamera2 import Picamera2
from libcamera import Transform

# PWM / PCA9685
import board
import busio
from adafruit_pca9685 import PCA9685

# TensorFlow 2.4+
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ------------------------------
# Configuration
# ------------------------------
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_STEERING_SCALE": 1.0,
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "PWM_THROTTLE_SCALE": 1.0,
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 290,
    "STEERING_RIGHT_PWM": 460,
    "THROTTLE_FORWARD_PWM": 500,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 220,
}

DRIVE_LOOP_HZ = 50
MAX_LOOPS = None

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False

DATA_ROOT = "data"

# ------------------------------
# Utility: PCA9685 helper
# ------------------------------
def parse_pca9685_pin(pin_str):
    # Format "PCA9685.<bus>:0x<addr>.<channel>", e.g. "PCA9685.1:0x40.1"
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

class MotorServoController:
    def __init__(self, config):
        s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])
        if s_bus != t_bus or s_addr != t_addr:
            raise ValueError("Steering and Throttle must be on same PCA9685 for this simple driver.")

        self.channel_steer = s_ch
        self.channel_throttle = t_ch

        self.i2c = busio.I2C(board.SCL, board.SDA)
        print(f"[PCA9685] I2C bus {s_bus}, addr 0x{s_addr:02x}, steer ch {s_ch}, throttle ch {t_ch}")
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60  # 60Hz typical

        self.cfg = config
        self.stop()

    def set_pwm_raw(self, channel, pwm_value):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty16
        return pwm_value

    def steering_pwm_from_norm(self, steer_norm):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        if self.cfg["PWM_STEERING_INVERTED"]:
            steer_norm = -steer_norm
        pwm = int(np.interp(steer_norm, [-1, 1], [right, left]))
        return pwm

    def throttle_pwm_from_norm(self, throttle_norm):
        if self.cfg["PWM_THROTTLE_INVERTED"]:
            throttle_norm = -throttle_norm
        rev = self.cfg["THROTTLE_REVERSE_PWM"]
        stop = self.cfg["THROTTLE_STOPPED_PWM"]
        fwd = self.cfg["THROTTLE_FORWARD_PWM"]
        pwm = int(np.interp(throttle_norm, [-1, 0, 1], [rev, stop, fwd]))
        return pwm

    def set_steering(self, steer_norm):
        pwm = self.steering_pwm_from_norm(steer_norm)
        self.set_pwm_raw(self.channel_steer, pwm)
        return pwm

    def set_throttle(self, throttle_norm):
        pwm = self.throttle_pwm_from_norm(throttle_norm)
        self.set_pwm_raw(self.channel_throttle, pwm)
        return pwm

    def stop(self):
        self.set_throttle(0.0)

    def close(self):
        self.stop()
        time.sleep(0.2)
        self.pca.deinit()

# ------------------------------
# Keyboard (no OpenCV)
# ------------------------------
class RawKeyboard:
    # Non-blocking character reader from stdin (terminal)
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return None

class KeyboardDriver:
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.steering_step = 0.25
        self.throttle_step = 0.1
        self.manual_quit = False

    def handle_char(self, ch):
        if ch is None:
            return
        if ch == 'q':
            self.manual_quit = True
            return
        if ch == ' ':
            self.throttle = 0.0
            return
        if ch == 'c':
            self.steering = 0.0
            return
        if ch in ('a',):
            self.steering = float(np.clip(self.steering - self.steering_step, -1, 1))
            return
        if ch in ('d',):
            self.steering = float(np.clip(self.steering + self.steering_step, -1, 1))
            return
        if ch in ('w',):
            self.throttle = float(np.clip(self.throttle + self.throttle_step, -1, 1))
            return
        if ch in ('s',):
            self.throttle = float(np.clip(self.throttle - self.throttle_step, -1, 1))
            return
        if ch == '\x1b':
            seq = ''
            for _ in range(2):
                nxt = kb.get_key(timeout=0.001)
                if nxt:
                    seq += nxt
            if seq == '[D':
                self.steering = float(np.clip(self.steering - self.steering_step, -1, 1))
            elif seq == '[C':
                self.steering = float(np.clip(self.steering + self.steering_step, -1, 1))
            elif seq == '[A':
                self.throttle = float(np.clip(self.throttle + self.throttle_step, -1, 1))
            elif seq == '[B':
                self.throttle = float(np.clip(self.throttle - self.throttle_step, -1, 1))

# ------------------------------
# Data IO
# ------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_session_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(DATA_ROOT, f"session_{stamp}")
    ensure_dir(os.path.join(root, "images"))
    return root

def write_labels_header(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "steering", "throttle"])

def append_label(csv_path, image_name, steering, throttle):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([image_name, f"{steering:.4f}", f"{throttle:.4f}"])

def list_existing_sessions():
    ensure_dir(DATA_ROOT)
    sessions = sorted([p for p in glob(os.path.join(DATA_ROOT, "session_*")) if os.path.isdir(p)])
    return sessions

def select_session_interactive(prompt="Select a session:"):
    sessions = list_existing_sessions()
    if not sessions:
        print("No sessions found in data/.")
        return None
    print(prompt)
    for i, s in enumerate(sessions):
        print(f"[{i}] {s}")
    while True:
        sel = input("Enter index (or blank to cancel): ").strip()
        if sel == "":
            return None
        if sel.isdigit() and 0 <= int(sel) < len(sessions):
            return sessions[int(sel)]
        print("Invalid selection.")

def latest_model_in_session(session_root):
    cands = sorted(glob(os.path.join(session_root, "*.keras"))) or sorted(glob(os.path.join(session_root, "*.h5")))
    return cands[-1] if cands else None

def select_model_from_any_session():
    sessions = list_existing_sessions()
    models = []
    for s in sessions:
        m = latest_model_in_session(s)
        if m:
            models.append(m)
    if not models:
        print("No trained models found in data/ sessions.")
        return None
    print("Select a model:")
    for i, m in enumerate(models):
        print(f"[{i}] {m}")
    while True:
        sel = input("Enter index (or blank to cancel): ").strip()
        if sel == "":
            return None
        if sel.isdigit() and 0 <= int(sel) < len(models):
            return models[int(sel)]
        print("Invalid selection.")

def load_image_for_model(rgb_array):
    if rgb_array.shape[0] != IMAGE_H or rgb_array.shape[1] != IMAGE_W:
        from PIL import Image
        im = Image.fromarray(rgb_array)
        im = im.resize((IMAGE_W, IMAGE_H))
        rgb_array = np.array(im)
    return rgb_array

def load_dataset(session_root):
    import PIL.Image as Image
    csv_path = os.path.join(session_root, "labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No labels.csv in {session_root}")
    X, y = [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            img_path = os.path.join(session_root, "images", row["image"])
            if not os.path.exists(img_path):
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB").resize((IMAGE_W, IMAGE_H))
                X.append(np.array(im))
            steer = float(row["steering"])
            thr = float(row["throttle"])
            y.append([steer, thr])
    if not X:
        raise RuntimeError(f"No images found for labels in {session_root}")
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.float32)
    return X, y

# ------------------------------
# Model
# ------------------------------
def build_model(input_shape=(IMAGE_H, IMAGE_W, IMAGE_DEPTH)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(16, (5,5), strides=2, activation='relu'),
        layers.Conv2D(32, (5,5), strides=2, activation='relu'),
        layers.Conv2D(64, (3,3), strides=2, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='tanh')  # [steering, throttle] in [-1,1]
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# ------------------------------
# Picamera2 manager (HEADLESS, no preview)
# ------------------------------
class PiCam2Manager:
    def __init__(self, width=IMAGE_W, height=IMAGE_H, framerate=CAMERA_FRAMERATE,
                 hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP, retries=1, delay=0.5):
        self.capture_resize = (IMAGE_W, IMAGE_H)
        self.picam2 = None

        last_err = None
        for _ in range(retries + 1):
            try:
                self.picam2 = Picamera2()
                break
            except Exception as e:
                last_err = e
                time.sleep(delay)
        if self.picam2 is None:
            raise RuntimeError(f"Failed to create Picamera2: {last_err}")

        transform = Transform(hflip=hflip, vflip=vflip)

        main_w = max(640, width)
        main_h = max(480, height)
        # Use RGB888 for easy numpy handling
        self.config = self.picam2.create_preview_configuration(
            main={"size": (main_w, main_h), "format": "RGB888"},
            transform=transform
        )
        self.picam2.configure(self.config)
        self.picam2.start()
        time
