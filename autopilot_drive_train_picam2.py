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
from picamera2 import Picamera2, Preview
from libcamera import Transform

# PWM / PCA9685
import board
import busio
from adafruit_pca9685 import PCA9685

# TensorFlow 2.4+
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# Configuration
# ------------------------------
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",  # ensure 0x40
    "PWM_STEERING_SCALE": 1.0,
    "PWM_STEERING_INVERTED": True,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # ensure 0x40
    "PWM_THROTTLE_SCALE": 1.0,
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 500,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 200,
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

        self.i2c = busio.I2C(board.SCL, board.SDA)
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
# Model - Improved architecture inspired by NVIDIA PilotNet
# ------------------------------
def build_model(input_shape=(IMAGE_H, IMAGE_W, IMAGE_DEPTH)):
    """
    Improved CNN architecture with:
    - Deeper network for better feature extraction
    - Batch normalization for training stability
    - More dropout for regularization
    - Larger fully connected layers
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        
        # Convolutional layers with batch normalization
        layers.Conv2D(24, (5,5), strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(36, (5,5), strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(48, (5,5), strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        # Fully connected layers with dropout
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(2, activation='tanh')  # [steering, throttle] in [-1,1]
    ])
    
    # Use lower learning rate with Adam optimizer
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

def build_lightweight_model(input_shape=(IMAGE_H, IMAGE_W, IMAGE_DEPTH)):
    """
    Lighter model for faster training/inference (if needed)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        
        layers.Conv2D(16, (5,5), strides=2, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5,5), strides=2, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), strides=2, activation='relu'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='tanh')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

# ------------------------------
# Picamera2 manager with Qt windowed preview (robust)
# ------------------------------
class PiCam2Manager:
    def __init__(self, width=IMAGE_W, height=IMAGE_H, framerate=CAMERA_FRAMERATE,
                 hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP, with_preview=True, retries=1, delay=0.5):
        self.with_preview = with_preview
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
        try:
            self.config = self.picam2.create_preview_configuration(
                main={"size": (main_w, main_h), "format": "XBGR8888"},
                transform=transform
            )
        except Exception:
            self.config = self.picam2.create_preview_configuration(
                main={"size": (main_w, main_h), "format": "XRGB8888"},
                transform=transform
            )

        self.picam2.configure(self.config)

        if self.with_preview:
            try:
                self.picam2.start_preview(Preview.QTGL)
            except Exception as e1:
                print(f"Preview.QTGL failed ({e1}); trying Preview.QT ...")
                try:
                    self.picam2.start_preview(Preview.QT)
                except Exception as e2:
                    print(f"Preview.QT also failed ({e2}); preview disabled.")
                    self.with_preview = False

        self.picam2.start()
        time.sleep(0.2)  # warmup

    def capture_rgb(self):
        arr = self.picam2.capture_array()
        if arr.ndim == 3 and arr.shape[2] == 4:
            b, g, r, _ = np.split(arr, 4, axis=2)
            arr = np.concatenate([r, g, b], axis=2)
        if arr.shape[1] != self.capture_resize[0] or arr.shape[0] != self.capture_resize[1]:
            from PIL import Image
            arr = np.array(Image.fromarray(arr).resize(self.capture_resize))
        return arr

    def annotate(self, text):
        pass

    def stop(self):
        try:
            if self.with_preview:
                try:
                    self.picam2.stop_preview()
                except Exception:
                    pass
            self.picam2.stop()
        finally:
            self.picam2 = None
            time.sleep(0.3)

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

    ctrl.stop()
    time.sleep(2.0)

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            print("Drive with WASD/Arrows; space=stop; c=center; r=record toggle; q=quit.")
            while True:
                frame_rgb = cam.capture_rgb()

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

                ctrl.set_steering(driver.steering)
                ctrl.set_throttle(driver.throttle)

                if session_root is not None:
                    img_name = f"{frame_idx:06d}.jpg"
                    img_path = os.path.join(session_root, "images", img_name)
                    from PIL import Image
                    Image.fromarray(frame_rgb).save(img_path, quality=90)
                    append_label(csv_path, img_name, driver.steering, driver.throttle)
                    frame_idx += 1

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
# place this near other small helpers or inside train_model_on_session before model.fit
def _ask_int(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s.isdigit():
        return int(s)
    return default

def _ask_yes_no(prompt, default=False):
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if s == '':
        return default
    return s == 'y'

def augment_data(X, y, augmentation_factor=2):
    """
    Data augmentation: flip images horizontally and adjust steering
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Flip horizontally and invert steering
        for _ in range(augmentation_factor - 1):
            flipped = np.fliplr(X[i])
            X_aug.append(flipped)
            # Invert steering for flipped image
            y_flipped = y[i].copy()
            y_flipped[0] = -y_flipped[0]  # Negate steering
            y_aug.append(y_flipped)
    
    return np.array(X_aug), np.array(y_aug)

def create_data_generator(X, y, batch_size=32, augment=True):
    """
    Create data generator with real-time augmentation
    """
    if not augment:
        while True:
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                yield X[batch_idx], y[batch_idx]
    else:
        while True:
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx].copy()
                y_batch = y[batch_idx].copy()
                
                # Random brightness adjustment
                for j in range(len(X_batch)):
                    if np.random.rand() > 0.5:
                        brightness = np.random.uniform(0.7, 1.3)
                        X_batch[j] = np.clip(X_batch[j] * brightness, 0, 255).astype(np.uint8)
                    
                    # Random horizontal flip
                    if np.random.rand() > 0.5:
                        X_batch[j] = np.fliplr(X_batch[j])
                        y_batch[j][0] = -y_batch[j][0]  # Flip steering
                
                yield X_batch, y_batch

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
    
    print(f"\n=== Training Configuration ===")
    print(f"Total samples: {len(X)}")
    
    # Ask for configuration
    use_augmentation = _ask_yes_no("Use data augmentation (recommended)?", default=True)
    use_lightweight = _ask_yes_no("Use lightweight model (faster but less accurate)?", default=False)
    epochs = _ask_int("Enter number of training epochs", 50)
    batch_size = _ask_int("Enter batch size", 32)
    
    # Augment data if requested
    if use_augmentation:
        print("Augmenting data (horizontal flips)...")
        X, y = augment_data(X, y, augmentation_factor=2)
        print(f"Augmented samples: {len(X)}")
    
    # Shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n = len(X)
    n_train = int(0.85 * n)  # Use 85% for training
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Build model
    if use_lightweight:
        print("Building lightweight model...")
        model = build_lightweight_model((IMAGE_H, IMAGE_W, IMAGE_DEPTH))
    else:
        print("Building improved PilotNet-style model...")
        model = build_model((IMAGE_H, IMAGE_W, IMAGE_DEPTH))
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Setup callbacks
    log_dir = os.path.join(session_root, "logs")
    ensure_dir(log_dir)
    
    training_callbacks = [
        # Early stopping with more patience
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(session_root, "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(session_root, "training_log.csv"),
            append=False
        )
    ]

    print("\n=== Starting Training ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=1
    )

    # Save final model
    model_path = os.path.join(session_root, "model.keras")
    model.save(model_path)
    print(f"\n=== Training Complete ===")
    print(f"Final model saved to: {model_path}")
    print(f"Best model saved to: {os.path.join(session_root, 'best_model.keras')}")
    print(f"Training log saved to: {os.path.join(session_root, 'training_log.csv')}")
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\nFinal Training Loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}, MAE: {final_val_mae:.4f}")
    
    # Plot training history
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Overfitting check
        axes[1, 1].plot(np.array(history.history['loss']) - np.array(history.history['val_loss']))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Loss - Val Loss')
        axes[1, 1].set_title('Overfitting Check (lower is better)')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(session_root, "training_plot.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create training plot: {e}")
    
    return model_path

def _robust_load_model(path):
    try:
        if path.endswith(".h5") or path.endswith(".hdf5"):
            print("Loading legacy HDF5 model with compile=False to avoid metric deserialization issues...")
            return tf.keras.models.load_model(path, compile=False)
        return tf.keras.models.load_model(path)
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("Retrying with compile=False ...")
        return tf.keras.models.load_model(path, compile=False)

def autopilot_loop(model_path):
    if model_path is None or not os.path.exists(model_path):
        print("Model not found; cannot run autopilot.")
        return

    # Check if there's a best_model.keras in the same directory
    best_model_path = os.path.join(os.path.dirname(model_path), "best_model.keras")
    if os.path.exists(best_model_path):
        use_best = _ask_yes_no(f"Found best_model.keras, use it instead of {os.path.basename(model_path)}?", default=True)
        if use_best:
            model_path = best_model_path
            print(f"Using best model: {model_path}")

    model = _robust_load_model(model_path)
    
    # Ask for throttle scaling
    print("\nAutopilot Configuration:")
    throttle_scale = float(input("Enter throttle scaling factor [0.5-1.0, default 0.7]: ").strip() or "0.7")
    throttle_scale = np.clip(throttle_scale, 0.1, 1.0)
    
    steering_smoothing = _ask_yes_no("Enable steering smoothing (recommended)?", default=True)
    
    cam = PiCam2Manager(IMAGE_W, IMAGE_H, CAMERA_FRAMERATE, CAMERA_HFLIP, CAMERA_VFLIP, with_preview=True)
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    period = 1.0 / DRIVE_LOOP_HZ
    last_loop = time.time()

    manual_override = False
    driver = KeyboardDriver()
    
    # Steering smoothing
    prev_steer = 0.0
    steering_alpha = 0.3  # Exponential moving average factor

    ctrl.stop()
    time.sleep(2.0)

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            print("\n=== Autopilot Controls ===")
            print("h = Switch to manual mode")
            print("a = Switch to autopilot mode")
            print("q = Quit")
            print("Space = Emergency stop")
            print("c = Center steering")
            print("WASD/Arrows = Manual control (when in manual mode)")
            print(f"\nThrottle scale: {throttle_scale:.2f}")
            print(f"Steering smoothing: {'Enabled' if steering_smoothing else 'Disabled'}")
            print("\nStarting in AUTOPILOT mode...")
            
            frame_count = 0
            while True:
                frame_rgb = cam.capture_rgb()
                frame_count += 1

                ch = kb.get_key(timeout=0.0)
                if ch == 'q':
                    break
                if ch == 'h':
                    manual_override = True
                    print("\n>>> Switched to MANUAL mode")
                elif ch == 'a':
                    manual_override = False
                    print("\n>>> Switched to AUTOPILOT mode")
                elif ch == ' ':
                    # Emergency stop
                    ctrl.stop()
                    print("\n!!! EMERGENCY STOP !!!")
                    continue
                else:
                    if manual_override:
                        driver.handle_char(ch)

                if not manual_override:
                    inp = np.expand_dims(load_image_for_model(frame_rgb), axis=0)
                    pred = model.predict(inp, verbose=0)[0]
                    steer = float(np.clip(pred[0], -1, 1))
                    thr = float(np.clip(pred[1], -1, 1)) * throttle_scale
                    
                    # Apply steering smoothing
                    if steering_smoothing:
                        steer = steering_alpha * steer + (1 - steering_alpha) * prev_steer
                        prev_steer = steer
                    
                    # Print periodic status
                    if frame_count % 50 == 0:
                        print(f"[Auto] Steer: {steer:+.3f}, Throttle: {thr:+.3f}")
                else:
                    steer = driver.steering
                    thr = driver.throttle
                    prev_steer = steer  # Update for smooth transition back to auto

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
        print("\nAutopilot stopped safely.")

# ------------------------------
# New helpers: train from previous data, run autopilot from existing model
# ------------------------------
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
    autopilot_loop(model_path)

# ------------------------------
# Main menu flow
# ------------------------------
def main():
    ensure_dir(DATA_ROOT)
    print("Meta Dot PiCar Control")
    print("1) Drive, preview, and optionally record a new session")
    print("2) Train model on a previously recorded session (from data/)")
    print("3) Run autopilot using an existing trained model (from data/)")
    print("4) Drive & record, then train immediately, then autopilot")
    print("q) Quit")

    choice = input("Select an option: ").strip().lower()
    if choice == "1":
        session_root = preview_and_record()
        print(f"Finished driving. Session: {session_root}")
    elif choice == "2":
        _ = train_from_existing_session()
    elif choice == "3":
        run_autopilot_from_existing_model()
    elif choice == "4":
        session_root = preview_and_record()
        if session_root:
            ans = input("Train model on recorded session? [y/N]: ").strip().lower()
            model_path = None
            if ans == "y":
                model_path = train_model_on_session(session_root)
            if model_path:
                ans2 = input("Run autopilot now? [y/N]: ").strip().lower()
                if ans2 == "y":
                    autopilot_loop(model_path)
    elif choice == "q":
        print("Bye.")
    else:
        print("Unknown option.")

if __name__ == "__main__":
    main()
