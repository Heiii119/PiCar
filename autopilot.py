#!/usr/bin/env python3
# Meta Dot PiCar Control (Headless, ultra-responsive)
# - Decoupled control (250 Hz) and camera (25 Hz) loops
# - Keyboard handled aggressively; PWM updates immediately
# - Smaller camera stream; no blocking on capture during control
# - Keyboard controls: WASD/arrows; space stop; c center; r record; q quit
#
# Suggested run:
#   LIBCAMERA_LOG_LEVELS=*:2 python3 -u autopilot.py

import os
import sys
import time
import csv
import tty
import termios
import select
import threading
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
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",  # ensure 0x40
    "PWM_STEERING_SCALE": 1.0,
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # ensure 0x40
    "PWM_THROTTLE_SCALE": 1.0,
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 290,
    "STEERING_RIGHT_PWM": 460,
    "THROTTLE_FORWARD_PWM": 500,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 220,
}

# Fast control loop, modest camera loop
CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ = 25
MAX_LOOPS = None

# Model input size
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

# Camera settings
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAM_STREAM_W = 320
CAM_STREAM_H = 240

DATA_ROOT = "data"

# ------------------------------
# Utility: PCA9685 helper
# ------------------------------
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

class MotorServoController:
    def __init__(self, config):
        s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])
        if s_bus != t_bus or s_addr != t_addr:
            raise ValueError("Steering and Throttle must be on same PCA9685 for this simple driver.")
        self.channel_steer = s_ch
        self.channel_throttle = t_ch
        self.i2c = busio.I2C(board.SCL, board.SDA)
        print(f"[PCA9685] I2C bus {s_bus}, addr 0x{s_addr:02x}, steer ch {s_ch}, throttle ch {t_ch}", flush=True)
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60
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
        time.sleep(0.1)
        self.pca.deinit()

# ------------------------------
# Keyboard (no OpenCV)
# ------------------------------
class RawKeyboard:
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
        # Tap responsiveness
        self.steering_step = 0.45
        self.throttle_step = 0.50  # updated: throttle increases/decreases by 0.5 per key press
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
                nxt = kb.get_key(timeout=0.0)
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
        layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# ------------------------------
# Camera thread
# ------------------------------
class CameraThread(threading.Thread):
    def __init__(self, hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP):
        super().__init__(daemon=True)
        self.capture_resize = (IMAGE_W, IMAGE_H)
        self.picam2 = Picamera2()
        transform = Transform(hflip=hflip, vflip=vflip)
        self.config = self.picam2.create_preview_configuration(
            main={"size": (CAM_STREAM_W, CAM_STREAM_H), "format": "RGB888"},
            transform=transform
        )
        self.picam2.configure(self.config)
        self.picam2.start()
        time.sleep(0.15)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.period = 1.0 / CAMERA_LOOP_HZ
        print(f"[Camera] Thread started: {CAM_STREAM_W}x{CAM_STREAM_H}, target {CAMERA_LOOP_HZ} Hz", flush=True)

    def run(self):
        from PIL import Image
        next_t = time.time()
        while not self.stop_flag:
            try:
                arr = self.picam2.capture_array()
                if arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                if arr.shape[1] != self.capture_resize[0] or arr.shape[0] != self.capture_resize[1]:
                    arr = np.array(Image.fromarray(arr).resize(self.capture_resize))
                with self.lock:
                    self.latest_frame = arr
            except Exception:
                pass
            next_t += self.period
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.time()

    def get_latest(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.stop_flag = True
        time.sleep(0.02)
        try:
            self.picam2.stop()
        except Exception:
            pass
        print("[Camera] Thread stopped", flush=True)

# ------------------------------
# Status printer
# ------------------------------
class StatusPrinter:
    def __init__(self, rate_hz=1):
        self.period = 1.0 / max(1, rate_hz)
        self._last = 0.0
    def maybe_print(self, camera_ok, steer_norm, thr_norm, steer_pwm, thr_pwm, recording):
        now = time.time()
        if now - self._last >= self.period:
            self._last = now
            rec = "ON" if recording else "OFF"
            cam = "OK" if camera_ok else "N/A"
            print(f"[Status] cam={cam} rec={rec} steer={steer_norm:+.2f} thr={thr_norm:+.2f} "
                  f"PWM(s,t)=({steer_pwm},{thr_pwm})", flush=True)

# ------------------------------
# Main routines (decoupled loops)
# ------------------------------
def preview_and_record_headless():
    print("Drive headless: r=record, q=quit, space=stop, c=center, WASD/arrows.", flush=True)
    cam_th = CameraThread(CAMERA_HFLIP, CAMERA_VFLIP)
    cam_th.start()
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    driver = KeyboardDriver()
    printer = StatusPrinter(rate_hz=1)

    session_root = None
    csv_path = None
    frame_idx = 0

    ctrl.stop()
    time.sleep(0.1)

    control_period = 1.0 / CONTROL_LOOP_HZ
    next_t = time.time()

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            loops = 0
            while True:
                ch = kb.get_key(timeout=0.0)
                for _ in range(3):
                    ch2 = kb.get_key(timeout=0.0)
                    if ch2:
                        ch = ch2

                if ch == 'r':
                    if session_root is None:
                        session_root = create_session_dir()
                        csv_path = os.path.join(session_root, "labels.csv")
                        write_labels_header(csv_path)
                        frame_idx = 0
                        print(f"[Record] Started: {session_root}", flush=True)
                    else:
                        print("[Record] Stopped.", flush=True)
                        session_root = None
                        csv_path = None
                else:
                    driver.handle_char(ch)

                if driver.manual_quit or ch == 'q':
                    break

                steer_pwm = ctrl.set_steering(driver.steering)
                thr_pwm = ctrl.set_throttle(driver.throttle)

                if session_root is not None:
                    frame_rgb = cam_th.get_latest()
                    if frame_rgb is not None:
                        img_name = f"{frame_idx:06d}.jpg"
                        img_path = os.path.join(session_root, "images", img_name)
                        from PIL import Image
                        Image.fromarray(frame_rgb).save(img_path, quality=90)
                        append_label(csv_path, img_name, driver.steering, driver.throttle)
                        frame_idx += 1

                printer.maybe_print(camera_ok=(cam_th.get_latest() is not None),
                                    steer_norm=driver.steering, thr_norm=driver.throttle,
                                    steer_pwm=steer_pwm, thr_pwm=thr_pwm,
                                    recording=(session_root is not None))

                next_t += control_period
                sleep_t = next_t - time.time()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()

                loops += 1
                if MAX_LOOPS is not None and loops >= MAX_LOOPS:
                    break
    finally:
        try:
            cam_th.stop()
        except Exception:
            pass
        ctrl.stop()
        ctrl.close()
    return session_root

def quick_test_headless(duration_sec=5):
    print(f"Quick test: camera thread + neutral PWM for {duration_sec} sec", flush=True)
    cam_th = CameraThread(CAMERA_HFLIP, CAMERA_VFLIP)
    cam_th.start()
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    start = time.time()
    ok = 0
    try:
        while time.time() - start < duration_sec:
            fr = cam_th.get_latest()
            if fr is not None:
                ok += 1
            ctrl.set_steering(0.0)
            ctrl.set_throttle(0.0)
            time.sleep(0.01)
        print(f"Camera provided {ok} frames.", flush=True)
    finally:
        cam_th.stop()
        ctrl.stop()
        ctrl.close()

def train_model_on_session(session_root):
    if session_root is None:
        print("No session selected/recorded.")
        return None
    print(f"Loading dataset from {session_root} ...")
    try:
        X, y = load_dataset(session_root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    if len(X) < 50:
        print("Not enough samples to train (need ~50+).")
        return None
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    model = build_model((IMAGE_H, IMAGE_W, IMAGE_DEPTH))
    epochs = _ask_int("Enter number of training epochs", 30)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]
    _ = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=32, callbacks=callbacks, verbose=1)
    model_path = os.path.join(session_root, "model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model_path

def _robust_load_model(path):
    try:
        if path.endswith(".h5") or path.endswith(".hdf5"):
            print("Loading legacy HDF5 model with compile=False...")
            return tf.keras.models.load_model(path, compile=False)
        return tf.keras.models.load_model(path)
    except Exception:
        return tf.keras.models.load_model(path, compile=False)

def autopilot_loop_headless(model_path):
    if model_path is None or not os.path.exists(model_path):
        print("Model not found; cannot run autopilot.")
        return
    model = _robust_load_model(model_path)
    cam_th = CameraThread(CAMERA_HFLIP, CAMERA_VFLIP)
    cam_th.start()
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    printer = StatusPrinter(rate_hz=1)
    manual_override = False
    driver = KeyboardDriver()
    ctrl.stop()
    time.sleep(0.1)
    control_period = 1.0 / CONTROL_LOOP_HZ
    next_t = time.time()
    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            print("Autopilot: h=manual, a=auto, q=quit. Space stop, c center; WASD in manual.", flush=True)
            while True:
                ch = kb.get_key(timeout=0.0)
                for _ in range(3):
                    ch2 = kb.get_key(timeout=0.0)
                    if ch2:
                        ch = ch2
                if ch == 'q':
                    break
                if ch == 'h':
                    manual_override = True
                elif ch == 'a':
                    manual_override = False
                else:
                    if manual_override:
                        driver.handle_char(ch)
                frame_rgb = cam_th.get_latest()
                if not manual_override and frame_rgb is not None:
                    inp = np.expand_dims(load_image_for_model(frame_rgb), axis=0)
                    pred = model.predict(inp, verbose=0)[0]
                    steer = float(np.clip(pred[0], -1, 1))
                    thr = float(np.clip(pred[1], -1, 1))
                else:
                    steer = driver.steering
                    thr = driver.throttle
                steer_pwm = ctrl.set_steering(steer)
                thr_pwm = ctrl.set_throttle(thr)
                printer.maybe_print(camera_ok=(frame_rgb is not None),
                                    steer_norm=steer, thr_norm=thr,
                                    steer_pwm=steer_pwm, thr_pwm=thr_pwm,
                                    recording=False)
                next_t += control_period
                sleep_t = next_t - time.time()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()
    finally:
        cam_th.stop()
        ctrl.stop()
        ctrl.close()

# ------------------------------
# Small helpers
# ------------------------------
def _ask_int(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s.isdigit():
        return int(s)
    return default

def train_from_existing_session():
    session = select_session_interactive("Select a session to train from existing data:")
    if session is None:
        print("Cancelled.")
        return None
    return train_model_on_session(session)

def run_autopilot_from_existing_model():
    model_path = select_model_from_any_session()
    if model_path is None:
        print("Cancelled.")
        return
    autopilot_loop_headless(model_path)

# ------------------------------
# Main menu
# ------------------------------
def main():
    print("Meta Dot PiCar Control (Headless, ultra-responsive)", flush=True)
    ensure_dir(DATA_ROOT)
    print("1) Drive headless, optionally record a new session")
    print("2) Train model on a recorded session (from data/)")
    print("3) Run autopilot headless using an existing trained model (from data/)")
    print("q) Quit")
    try:
        choice = input("Select an option: ").strip().lower()
    except EOFError:
        print("No TTY detected. Quick test:\n  python3 -u -c \"import autopilot; autopilot.quick_test_headless(5)\"",
              flush=True)
        return
    if choice == "1":
        session_root = preview_and_record_headless()
        print(f"Finished driving. Session: {session_root}")
    elif choice == "2":
        _ = train_from_existing_session()
    elif choice == "3":
        run_autopilot_from_existing_model()
    elif choice == "q":
        print("Bye.")
    else:
        print("Unknown option.")

if __name__ == "__main__":
    main()
