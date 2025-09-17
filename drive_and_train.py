import os 
import time 
import csv 
import glob 
import math 
import signal 
import threading 
from datetime import datetime 
from collections import deque

import numpy as np import cv2

#Camera
from picamera.array import PiRGBArray 
from picamera import PiCamera

PWM / PCA9685
import board import busio from adafruit_pca9685 import PCA9685

TensorFlow 2.4
import tensorflow as tf from tensorflow.keras import layers, models, optimizers

------------------------------
Configuration (from your spec)
------------------------------
PWM_STEERING_THROTTLE = { "PWM_STEERING_PIN": "PCA9685.1:40.1", "PWM_STEERING_SCALE": 1.0, "PWM_STEERING_INVERTED": False, "PWM_THROTTLE_PIN": "PCA9685.1:40.0", "PWM_THROTTLE_SCALE": 1.0, "PWM_THROTTLE_INVERTED": False, "STEERING_LEFT_PWM": 460, "STEERING_RIGHT_PWM": 290, "THROTTLE_FORWARD_PWM": 500, "THROTTLE_STOPPED_PWM": 370, "THROTTLE_REVERSE_PWM": 220, }

DRIVE_LOOP_HZ = 20 MAX_LOOPS = None

CAMERA_TYPE = "PICAM" IMAGE_W = 160 IMAGE_H = 120 IMAGE_DEPTH = 3 CAMERA_FRAMERATE = DRIVE_LOOP_HZ CAMERA_VFLIP = False CAMERA_HFLIP = False

------------------------------
Utility: PWM helper
------------------------------
def parse_pca9685_pin(pin_str): # Example formats: "PCA9685.1:40.1" -> bus 1, address 0x40, channel 1 # or "PCA9685.1:40.0" try: left, chan = pin_str.split(":") bus_str = left.split(".")[1] addr_str = chan.split(".")[0] if "." in chan else chan channel_str = chan.split(".")[1] if "." in chan else "0" i2c_bus = int(bus_str) i2c_addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str) channel = int(channel_str) return i2c_bus, i2c_addr, channel except Exception as e: raise ValueError(f"Invalid PCA9685 pin format: {pin_str}") from e

class MotorServoController: def init(self, config): s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"]) t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"]) if s_bus != t_bus or s_addr != t_addr: raise ValueError("Steering and Throttle must be on same PCA9685 for this simple driver.")

self.channel_steer = s_ch self.channel_throttle = t_ch # Raspberry Pi I2C typically bus 1 self.i2c = busio.I2C(board.SCL, board.SDA) self.pca = PCA9685(self.i2c, address=s_addr) self.pca.frequency = 60 # Hz typical for servos/ESCs self.cfg = config self.stop() def set_pwm_raw(self, channel, pwm_value): # PCA9685 12-bit: value 0..4095. Our config values (220..500) assume "pulse length" ticks at 60Hz. # The Adafruit lib uses duty_cycle 0..65535, but provides set_pwm via older libs. # We'll map our "ticks" to duty_cycle proportionally. # At 60Hz, 1 tick (out of 4096) ~ 4.88us. We'll convert 0..4095 to 16-bit duty. pwm_value = int(np.clip(pwm_value, 0, 4095)) duty16 = int((pwm_value / 4095.0) * 65535) self.pca.channels[channel].duty_cycle = duty16 def set_steering(self, steer_norm): # steer_norm in [-1, 1], map to PWM between RIGHT and LEFT, accounting for inversion left = self.cfg["STEERING_LEFT_PWM"] right = self.cfg["STEERING_RIGHT_PWM"] if self.cfg["PWM_STEERING_INVERTED"]: steer_norm = -steer_norm pwm = int(np.interp(steer_norm, [-1, 1], [right, left])) self.set_pwm_raw(self.channel_steer, pwm) def set_throttle(self, throttle_norm): # throttle_norm in [-1, 1], map to REVERSE..FORWARD via STOP if self.cfg["PWM_THROTTLE_INVERTED"]: throttle_norm = -throttle_norm rev = self.cfg["THROTTLE_REVERSE_PWM"] stop = self.cfg["THROTTLE_STOPPED_PWM"] fwd = self.cfg["THROTTLE_FORWARD_PWM"] pwm = int(np.interp(throttle_norm, [-1, 0, 1], [rev, stop, fwd])) self.set_pwm_raw(self.channel_throttle, pwm) def stop(self): self.set_throttle(0.0) def close(self): self.stop() time.sleep(0.2) self.pca.deinit()
------------------------------
Keyboard control via OpenCV
------------------------------
class KeyboardDriver: def init(self): self.steering = 0.0 # -1..1 self.throttle = 0.0 # -1..1 self.steering_step = 0.1 self.throttle_step = 0.1

def handle_key(self, key): # key is integer from cv2.waitKey if key == ord('q'): return "quit" elif key == 81 or key == ord('a'): # Left arrow or 'a' self.steering = np.clip(self.steering - self.steering_step, -1, 1) elif key == 83 or key == ord('d'): # Right arrow or 'd' self.steering = np.clip(self.steering + self.steering_step, -1, 1) elif key == 82 or key == ord('w'): # Up arrow or 'w' self.throttle = np.clip(self.throttle + self.throttle_step, -1, 1) elif key == 84 or key == ord('s'): # Down arrow or 's' self.throttle = np.clip(self.throttle - self.throttle_step, -1, 1) elif key == 32: # Space bar self.throttle = 0.0 elif key == ord('c'): # center steering self.steering = 0.0 return None
------------------------------
Data collection
------------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def create_session_dir(): stamp = datetime.now().strftime("%Y%m%d_%H%M%S") root = os.path.join("data", f"session_{stamp}") ensure_dir(root) ensure_dir(os.path.join(root, "images")) return root

def save_frame_and_label(root, idx, frame_bgr, steering, throttle): img_name = f"{idx:06d}.jpg" img_path = os.path.join(root, "images", img_name) cv2.imwrite(img_path, frame_bgr) return img_name

def write_labels_header(csv_path): with open(csv_path, "w", newline="") as f: w = csv.writer(f) w.writerow(["image", "steering", "throttle"])

def append_label(csv_path, image_name, steering, throttle): with open(csv_path, "a", newline="") as f: w = csv.writer(f) w.writerow([image_name, f"{steering:.4f}", f"{throttle:.4f}"])

------------------------------
Model
------------------------------
def build_model(input_shape=(IMAGE_H, IMAGE_W, IMAGE_DEPTH)): model = models.Sequential([ layers.Input(shape=input_shape), layers.Rescaling(1./255), layers.Conv2D(16, (5,5), strides=2, activation='relu'), layers.Conv2D(32, (5,5), strides=2, activation='relu'), layers.Conv2D(64, (3,3), strides=2, activation='relu'), layers.Flatten(), layers.Dense(64, activation='relu'), layers.Dropout(0.2), layers.Dense(2, activation='tanh') # outputs: [steering, throttle] in [-1,1] ]) opt = optimizers.Adam(1e-3) model.compile(optimizer=opt, loss='mse', metrics=['mae']) return model

def load_image(path): img = cv2.imread(path) if img is None: raise IOError(f"Failed to read image: {path}") img = cv2.resize(img, (IMAGE_W, IMAGE_H)) img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) return img

def load_dataset(session_root): csv_path = os.path.join(session_root, "labels.csv") X, y = [], [] with open(csv_path, "r") as f: r = csv.DictReader(f) for row in r: img_path = os.path.join(session_root, "images", row["image"]) img = load_image(img_path) steer = float(row["steering"]) thr = float(row["throttle"]) X.append(img) y.append([steer, thr]) X = np.array(X, dtype=np.uint8) y = np.array(y, dtype=np.float32) return X, y

------------------------------
Camera manager
------------------------------
class PiCam: def init(self, width=IMAGE_W, height=IMAGE_H, framerate=CAMERA_FRAMERATE, hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP): self.camera = PiCamera() self.camera.resolution = (width, height) self.camera.framerate = framerate self.camera.hflip = hflip self.camera.vflip = vflip self.rawCapture = PiRGBArray(self.camera, size=(width, height)) # Allow camera to warm up time.sleep(0.2)

def close(self): self.camera.close()
------------------------------
Main routines
------------------------------
def preview_and_record(): cam = PiCam(IMAGE_W, IMAGE_H, CAMERA_FRAMERATE, CAMERA_HFLIP, CAMERA_VFLIP) ctrl = MotorServoController(PWM_STEERING_THROTTLE) driver = KeyboardDriver()

print("Opening camera preview window. Press 'r' to start/stop recording. Press 'q' to quit.") session_root = None csv_path = None frame_idx = 0 last_loop = time.time() period = 1.0 / DRIVE_LOOP_HZ try: # Use continuous capture for speed for frame in cam.camera.capture_continuous(cam.rawCapture, format="bgr", use_video_port=True): img = frame.array # BGR # For human display, we can add HUD text hud = img.copy() cv2.putText(hud, f"Steer: {driver.steering:+.2f} Thr: {driver.throttle:+.2f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA) rec_flag = session_root is not None cv2.putText(hud, "REC" if rec_flag else "IDLE", (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) if rec_flag else (0,255,255), 1, cv2.LINE_AA) cv2.imshow("PiCam", hud) key = cv2.waitKey(1) & 0xFF if key == ord('r'): if session_root is None: session_root = create_session_dir() csv_path = os.path.join(session_root, "labels.csv") write_labels_header(csv_path) frame_idx = 0 print(f"Recording started: {session_root}") else: print("Recording stopped.") session_root = None csv_path = None action = driver.handle_key(key) if action == "quit": break # Apply controls to hardware ctrl.set_steering(driver.steering) ctrl.set_throttle(driver.throttle) # Save data if recording if session_root is not None: img_name = save_frame_and_label(session_root, frame_idx, img, driver.steering, driver.throttle) append_label(csv_path, img_name, driver.steering, driver.throttle) frame_idx += 1 # Throttle loop to DRIVE_LOOP_HZ now = time.time() dt = now - last_loop if dt < period: time.sleep(period - dt) last_loop = time.time() cam.rawCapture.truncate(0) finally: ctrl.stop() cam.close() cv2.destroyAllWindows() ctrl.close() return session_root # may be None if no recording
def train_model_on_session(session_root): if session_root is None: print("No session to train on.") return None

print(f"Loading dataset from {session_root} ...") X, y = load_dataset(session_root) if len(X) < 50: print("Not enough samples to train (need ~50+).") return None # Shuffle idx = np.arange(len(X)) np.random.shuffle(idx) X = X[idx] y = y[idx] # Split n = len(X) n_train = int(0.8 * n) X_train, y_train = X[:n_train], y[:n_train] X_val, y_val = X[n_train:], y[n_train:] print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}") model = build_model((IMAGE_H, IMAGE_W, IMAGE_DEPTH)) callbacks = [ tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss") ] history = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=callbacks, verbose=1 ) model_path = os.path.join(session_root, "model_tf24.h5") model.save(model_path) print(f"Saved model to {model_path}") return model_path
def autopilot_loop(model_path): if model_path is None or not os.path.exists(model_path): print("Model path invalid; cannot run autopilot.") return

model = tf.keras.models.load_model(model_path) cam = PiCam(IMAGE_W, IMAGE_H, CAMERA_FRAMERATE, CAMERA_HFLIP, CAMERA_VFLIP) ctrl = MotorServoController(PWM_STEERING_THROTTLE) print("Autopilot running. Press 'q' to quit, 'h' to handover to manual (hold), 'a' to resume autopilot.") manual_override = False driver = KeyboardDriver() # reuse for manual override last_loop = time.time() period = 1.0 / DRIVE_LOOP_HZ try: for frame in cam.camera.capture_continuous(cam.rawCapture, format="bgr", use_video_port=True): img_bgr = frame.array img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2RGB) inp = np.expand_dims(img_rgb, axis=0) if not manual_override: pred = model.predict(inp, verbose=0)[0] steer, thr = float(np.clip(pred[0], -1, 1)), float(np.clip(pred[1], -1, 1)) else: steer, thr = driver.steering, driver.throttle # Apply to hardware ctrl.set_steering(steer) ctrl.set_throttle(thr) # HUD hud = img_bgr.copy() mode = "AUTO" if not manual_override else "MANUAL" cv2.putText(hud, f"{mode} Steer:{steer:+.2f} Thr:{thr:+.2f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA) cv2.imshow("Autopilot", hud) key = cv2.waitKey(1) & 0xFF if key == ord('q'): break elif key == ord('h'): manual_override = True elif key == ord('a'): manual_override = False if manual_override: driver.handle_key(key) # Throttle loop to DRIVE_LOOP_HZ now = time.time() dt = now - last_loop if dt < period: time.sleep(period - dt) last_loop = time.time() cam.rawCapture.truncate(0) finally: ctrl.stop() cam.close() cv2.destroyAllWindows() ctrl.close()
def main(): print("Pi Cam preview will open. Press 'r' to start/stop recording. Drive with arrows/WASD, Space to stop, c to center. q to quit.") session_root = preview_and_record()

ans = input("Train model on recorded session? [y/N]: ").strip().lower() model_path = None if ans == "y" and session_root is not None: model_path = train_model_on_session(session_root) ans2 = input("Run autopilot now? [y/N]: ").strip().lower() if ans2 == "y" and model_path is not None: autopilot_loop(model_path) else: print("Done.")
if name == "main": main()
