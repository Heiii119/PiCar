import os
import sys
import time
import csv
import tty
import termios
import select
from datetime import datetime

import numpy as np

# Picamera2
from picamera2 import Picamera2, Preview
from libcamera import Transform

# PWM / PCA9685
import board
import busio
from adafruit_pca9685 import PCA9685

# TensorFlow 2.4
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ------------------------------
# Configuration (from your spec)
# ------------------------------
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",  # changed to 0x40
    "PWM_STEERING_SCALE": 1.0,
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # changed to 0x40
    "PWM_THROTTLE_SCALE": 1.0,
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 460,
    "STEERING_RIGHT_PWM": 290,
    "THROTTLE_FORWARD_PWM": 500,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 220,
}

DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False

# ------------------------------
# Utility: PCA9685 helper
# ------------------------------
def parse_pca9685_pin(pin_str):
    # Format "PCA9685.<bus>:<addr>.<channel>", e.g. "PCA9685.1:0x40.1"
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

        # RPi I2C bus is typically 1
        self.i2c = busio.I2C(board.SCL, board.SDA)
        # Debug print to confirm parsed address
        print(f"Using PCA9685 on I2C bus {s_bus}, address 0x{s_addr:02x}, steer ch {s_ch}, throttle ch {t_ch}")
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60  # 60Hz typical

        self.cfg = config
        self.stop()

    def set_pwm_raw(self, channel, pwm_value):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)
        self.pca.channels[channel].duty_cycle = duty16

    def set_steering(self, steer_norm):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        if self.cfg["PWM_STEERING_INVERTED"]:
            steer_norm = -steer_norm
        pwm = int(np.interp(steer_norm, [-1, 1], [right, left]))
        self.set_pwm_raw(self.channel_steer, pwm)

    def set_throttle(self, throttle_norm):
        if self.cfg["PWM_THROTTLE_INVERTED"]:
            throttle_norm = -throttle_norm
        rev = self.cfg["THROTTLE_REVERSE_PWM"]
        stop = self.cfg["THROTTLE_STOPPED_PWM"]
        fwd = self.cfg["THROTTLE_FORWARD_PWM"]
        pwm = int(np.interp(throttle_norm, [-1, 0, 1], [rev, stop, fwd]))
        self.set_pwm_raw(self.channel_throttle, pwm)

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
        # Returns a single character or None
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return None

class KeyboardDriver:
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.steering_step = 0.1
        self.throttle_step = 0.1
        self.manual_quit = False

    def handle_char(self, ch):
        # Arrow keys arrive as escape sequences: '\x1b[A' up, '\x1b[B' down, '\x1b[C' right, '\x1b[D' left
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
        if ch in ('a',):  # left
            self.steering = float(np.clip(self.steering - self.steering_step, -1, 1))
            return
        if ch in ('d',):  # right
            self.steering = float(np.clip(self.steering + self.steering_step, -1, 1))
            return
        if ch in ('w',):  # up
            self.throttle = float(np.clip(self.throttle + self.throttle_step, -1, 1))
            return
        if ch in ('s',):  # down
            self.throttle = float(np.clip(self.throttle - self.throttle_step, -1, 1))
            return
        # Handle escape sequences for arrow keys
        if ch == '\x1b':
            # read next two chars if present
            seq = ''
            for _ in range(2):
                nxt = kb.get_key(timeout=0.001)
                if nxt:
                    seq += nxt
            if seq == '[D':  # left
                self.steering = float(np.clip(self.steering - self.steering_step, -1, 1))
            elif seq == '[C':  # right
                self.steering = float(np.clip(self.steering + self.steering_step, -1, 1))
            elif seq == '[A':  # up
                self.throttle = float(np.clip(self.throttle + self.throttle_step, -1, 1))
            elif seq == '[B':  # down
                self.throttle = float(np.clip(self.throttle - self.throttle_step, -1, 1))

# ------------------------------
# Data IO
# ------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_session_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join("data", f"session_{stamp}")
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

def load_image_for_model(rgb_array):
    # Ensure shape HxWx3 and size to model input
    if rgb_array.shape[0] != IMAGE_H or rgb_array.shape[1] != IMAGE_W:
        # Picamera2 can deliver at desired size; this is a safeguard
        from PIL import Image
        im = Image.fromarray(rgb_array)
        im = im.resize((IMAGE_W, IMAGE_H))
        rgb_array = np.array(im)
    return rgb_array

def load_dataset(session_root):
    import PIL.Image as Image
    csv_path = os.path.join(session_root, "labels.csv")
    X, y = [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            img_path = os.path.join(session_root, "images", row["image"])
            with Image.open(img_path) as im:
                im = im.convert("RGB").resize((IMAGE_W, IMAGE_H))
                X.append(np.array(im))
            steer = float(row["steering"])
            thr = float(row["throttle"])
            y.append([steer, thr])
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
# Picamera2 manager
# ------------------------------
class PiCam2Manager:
    def __init__(self, width=IMAGE_W, height=IMAGE_H, framerate=CAMERA_FRAMERATE,
                 hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP, with_preview=True):
        self.picam2 = Picamera2()
        transform = Transform(hflip=hflip, vflip=vflip)
        self.config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=transform
        )
        self.picam2.configure(self.config)
        self.with_preview = with_preview
        if self.with_preview:
            try:
                self.picam2.start_preview(Preview.DRM)
            except Exception as e:
                print(f"Preview.DRM failed ({e}); continuing without preview.")
                self.with_preview = False
        self.picam2.start()
        # Allow warmup
        time.sleep(0.2)

    def capture_rgb(self):
        # Returns HxWx3 RGB array (uint8)
        frame = self.picam2.capture_array("main")
        return frame

    def annotate(self, text):
        # Preview overlay skipped for portability
        pass

    def stop(self):
        if self.with_preview:
            try:
                self.picam2.stop_preview()
            except Exception:
                pass
        self.picam2.stop()

# ------------------------------
# Main routines
# ------------------------------
def preview_and_record():
    print("Opening Picamera2 preview. Controls: r to start/stop recording, q to quit.")
    cam = PiCam2Manager(IMAGE_W, IMAGE_H, CAMERA_FRAMERATE, CAMERA_HFLIP, CAMERA_VFLIP, with_preview=True)
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    driver = KeyboardDriver()

    session_root = None
    csv_path = None
    frame_idx = 0
    period = 1.0 / DRIVE_LOOP_HZ
    last_loop = time.time()

    # optional: arm ESC by holding stop PWM briefly
    ctrl.stop()
    time.sleep(1.0)

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb  # used in KeyboardDriver arrow handling
            print("Drive with WASD or arrows; space to stop; c to center; r to record; q to quit.")
            while True:
                # Capture frame
                frame_rgb = cam.capture_rgb()  # HxWx3 RGB
                # Poll keyboard
                ch = kb.get_key(timeout=0.0)
                if ch == 'r':
                    if session_root is None:
                        session_root = create_session_dir()
                        csv_path = os.path.join(session_root, "labels.csv")
                        write_labels_header(csv_path)
                        frame_idx = 0
                        print(f"Recording started: {session_root}")
                    else:
                        print("Recording stopped.")
                        session_root = None
                        csv_path = None
                else:
                    driver.handle_char(ch)

                if driver.manual_quit:
                    break

                # Apply controls
                ctrl.set_steering(driver.steering)
                ctrl.set_throttle(driver.throttle)

                # Save data if recording
                if session_root is not None:
                    # Save JPEG
                    img_name = f"{frame_idx:06d}.jpg"
                    img_path = os.path.join(session_root, "images", img_name)
                    # Use PIL to save
                    from PIL import Image
                    Image.fromarray(frame_rgb).save(img_path, quality=90)
                    append_label(csv_path, img_name, driver.steering, driver.throttle)
                    frame_idx += 1

                # Maintain loop rate
                now = time.time()
                dt = now - last_loop
                if dt < period:
                    time.sleep(period - dt)
                last_loop = time.time()
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        ctrl.stop()
        ctrl.close()
    return session_root

def train_model_on_session(session_root):
    if session_root is None:
        print("No session recorded.")
        return None
    print(f"Loading dataset from {session_root} ...")
    X, y = load_dataset(session_root)
    if len(X) < 50:
        print("Not enough samples to train (need ~50+).")
        return None

    # Shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = build_model((IMAGE_H, IMAGE_W, IMAGE_DEPTH))
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    model_path = os.path.join(session_root, "model_tf24.h5")
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model_path

def autopilot_loop(model_path):
    if model_path is None or not os.path.exists(model_path):
        print("Model not found; cannot run autopilot.")
        return

    model = tf.keras.models.load_model(model_path)

    cam = PiCam2Manager(IMAGE_W, IMAGE_H, CAMERA_FRAMERATE, CAMERA_HFLIP, CAMERA_VFLIP, with_preview=True)
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    period = 1.0 / DRIVE_LOOP_HZ
    last_loop = time.time()

    # Allow manual override (hold h to enable manual, a to resume; q to quit)
    manual_override = False
    driver = KeyboardDriver()

    # Arm ESC
    ctrl.stop()
    time.sleep(1.0)

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            print("Autopilot running. h=manual, a=auto, q=quit.")
            while True:
                frame_rgb = cam.capture_rgb()
                ch = kb.get_key(timeout=0.0)
                if ch == 'q':
                    break
                if ch == 'h':
                    manual_override = True
                elif ch == 'a':
                    manual_override = False
                else:
                    if manual_override:
                        driver.handle_char(ch)

                if not manual_override:
                    inp = np.expand_dims(load_image_for_model(frame_rgb), axis=0)
                    pred = model.predict(inp, verbose=0)[0]
                    steer = float(np.clip(pred[0], -1, 1))
                    thr = float(np.clip(pred[1], -1, 1))
                else:
                    steer = driver.steering
                    thr = driver.throttle

                ctrl.set_steering(steer)
                ctrl.set_throttle(thr)

                now = time.time()
                dt = now - last_loop
                if dt < period:
                    time.sleep(period - dt)
                last_loop = time.time()
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        ctrl.stop()
        ctrl.close()

def main():
    print("Picamera2 preview will open. Controls: WASD/Arrows to drive, space stop, c center, r record, q quit.")
    session_root = preview_and_record()
    ans = input("Train model on recorded session? [y/N]: ").strip().lower()
    model_path = None
    if ans == "y" and session_root is not None:
        model_path = train_model_on_session(session_root)

    ans2 = input("Run autopilot now? [y/N]: ").strip().lower()
    if ans2 == "y" and model_path is not None:
        autopilot_loop(model_path)
    else:
        print("Done.")

if __name__ == "__main__":
    main()
