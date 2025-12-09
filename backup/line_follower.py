#!/usr/bin/env python3
# Meta Dot PiCar Line Follower (Headless, ultra-responsive)
# - Decoupled control (250 Hz) and camera (25 Hz) loops
# - No preview window
# - Status prints: camera OK, steer/throttle normals and PWMs
# - Keyboard: WASD/arrows; space stop; c center; r record; h manual; a auto; q quit
#
# Suggested run:
#   LIBCAMERA_LOG_LEVELS=*:2 python3 -u line_follow.py

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
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 470,
    "THROTTLE_FORWARD_PWM": 500,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 220,
}

# Fast control loop, modest camera loop
CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ = 25
MAX_LOOPS = None  # None = run forever

# Image sizes
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

# Camera settings
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAM_STREAM_W = 320
CAM_STREAM_H = 240

DATA_ROOT = "data"

# Line detection parameters
LINE_IS_DARK = True         # True if the line is darker than the floor (e.g., black tape)
ROI_FRACTION = (0.55, 0.95) # Use bottom 40% of the image for line search
BIN_THRESH = 0.45           # Threshold on normalized grayscale [0..1]
SMOOTH_STEER_ALPHA = 0.3    # Low-pass filter for steering
MAX_STEER = 1.0             # clamp [-1,1]
BASE_THROTTLE = 0.35        # forward speed (normalized 0..1)
SLOWDOWN_AT_CURVE = True
CURVE_SLOWDOWN_GAIN = 0.4   # reduce throttle when line curvature is high
DERIV_STEER_GAIN = 0.5      # add derivative term to stabilize

# Control gains (pure pursuit-ish + proportional)
STEER_P_GAIN = 0.9
STEER_D_GAIN = 0.05         # derivative on error
# Optionally, you can add integral if drift persists; keep it small
STEER_I_GAIN = 0.0
STEER_I_CLAMP = 0.3

# Safety
NO_LINE_TIMEOUT = 1.0       # seconds since last line detection to trigger stop

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

# ------------------------------
# Camera helper
# ------------------------------
class CameraWorker:
    def __init__(self, stream_w=CAM_STREAM_W, stream_h=CAM_STREAM_H, hflip=False, vflip=False):
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (stream_w, stream_h), "format": "XRGB8888"},
            transform=Transform(hflip=hflip, vflip=vflip),
            buffer_count=4
        )
        self.cam.configure(config)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.period = 1.0 / CAMERA_LOOP_HZ

    def start(self):
        self.cam.start()
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cam.stop()

    def loop(self):
        next_t = time.time()
        while self.running:
            arr = self.cam.capture_array()
            # XRGB8888 -> RGB
            rgb = arr[..., :3]
            with self.lock:
                self.frame = rgb.copy()
            next_t += self.period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

# ------------------------------
# Line detection without OpenCV
# ------------------------------
def to_gray_norm(rgb):
    # Convert to grayscale float32 [0..1], Rec.601 luma
    g = (0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]).astype(np.float32)
    g /= 255.0
    return g

def binary_threshold(gray, thresh, invert=False):
    if invert:
        # target bright line
        mask = (gray >= thresh).astype(np.uint8)
    else:
        # target dark line
        mask = (gray <= thresh).astype(np.uint8)
    return mask

def roi_slice(h, roi_frac):
    y0 = int(h * roi_frac[0])
    y1 = max(y0 + 1, int(h * roi_frac[1]))
    return y0, y1

def find_line_center(mask):
    # Weighted centroid along x across ROI rows
    # Returns normalized center error in [-1,1], where -1 = far left, +1 = far right
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    # Row-wise sum to check if any line present
    row_sums = mask.sum(axis=1)
    total = row_sums.sum()
    if total < 10:  # not enough pixels
        return None, None

    # Compute per-row centroids; then weight nearer rows more (bottom rows)
    # Weight proportional to row index (closer to bottom gets higher weight)
    weights = (ys + 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = (mask * xs[None, :]).sum(axis=1)
        row_centroids = np.where(row_sums > 0, col_sums / np.maximum(row_sums, 1), np.nan)
    valid = ~np.isnan(row_centroids)
    if not np.any(valid):
        return None, None

    weighted_center = np.nansum(row_centroids[valid] * weights[valid]) / np.nansum(weights[valid])
    # Normalize to [-1,1]
    center_norm = (weighted_center / (w - 1)) * 2.0 - 1.0
    # Rough curvature estimate: slope of centroids across rows
    if np.sum(valid) >= 3:
        y_valid = ys[valid]
        c_valid = row_centroids[valid]
        y_mean = np.mean(y_valid)
        c_mean = np.mean(c_valid)
        denom = np.sum((y_valid - y_mean)**2) + 1e-6
        slope = np.sum((y_valid - y_mean) * (c_valid - c_mean)) / denom
        # Normalize slope by width to get unitless curvature-ish indicator
        curvature = slope / (mask.shape[1] + 1e-6)
    else:
        curvature = 0.0

    return float(center_norm), float(curvature)

# ------------------------------
# Main control
# ------------------------------
class LineFollower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker(stream_w=CAM_STREAM_W, stream_h=CAM_STREAM_H,
                                   hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP)

        # Shared state
        self.running = True
        self.auto_mode = True
        self.recording = False
        self.last_line_time = 0.0

        # Control loop scheduling
        self.ctrl_period = 1.0 / CONTROL_LOOP_HZ
        self.cam_period = 1.0 / CAMERA_LOOP_HZ

        # Control state
        self.prev_error = 0.0
        self.i_error = 0.0
        self.filtered_steer = 0.0

        # Latest perception
        self.last_center_err = 0.0
        self.last_curvature = 0.0

        # Thread
        self.thread = None

        # Recording
        self.data_dir = None
        self.frame_id = 0

    def start(self):
        print("[Camera] Starting...", flush=True)
        self.camera.start()
        t0 = time.time()
        # wait a bit for first frame
        while self.camera.get_frame() is None and time.time() - t0 < 2.0:
            time.sleep(0.05)
        print("[Camera] OK", flush=True)

        self.running = True
        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.keyboard_loop()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.stop()
        self.motors.stop()
        self.motors.close()

    # -------------------------- Keyboard --------------------------
    def keyboard_loop(self):
        print("[Keys] h manual, a auto, r toggle record, WASD/Arrows steer/throttle, space stop, c center, q quit", flush=True)
        manual_steer = 0.0
        manual_throttle = 0.0
        with RawKeyboard() as kb:
            while self.running:
                ch = kb.get_key(timeout=0.05)
                if ch is None:
                    continue
                if ch in ('q', 'Q'):
                    print("[Keys] Quit", flush=True)
                    self.running = False
                    break
                elif ch in ('h', 'H'):
                    self.auto_mode = False
                    print("[Mode] Manual", flush=True)
                elif ch in ('a', 'A'):
                    self.auto_mode = True
                    print("[Mode] Auto (line following)", flush=True)
                elif ch in ('r', 'R'):
                    self.recording = not self.recording
                    if self.recording and self.data_dir is None:
                        self._start_recording()
                    print(f"[Rec] {'ON' if self.recording else 'OFF'} -> {self.data_dir}", flush=True)
                elif ch == ' ':
                    manual_throttle = 0.0
                    self.motors.stop()
                    print("[Keys] Emergency stop", flush=True)
                elif ch in ('c', 'C'):
                    manual_steer = 0.0
                    self.prev_error = 0.0
                    self.i_error = 0.0
                    print("[Keys] Center steer", flush=True)
                # Manual driving: WASD/arrows
                elif ch in ('w', 'W', 'A'):  # 'A' to match arrow-up if using some terminals
                    manual_throttle = min(1.0, manual_throttle + 0.05)
                    print(f"[Manual] throttle {manual_throttle:.2f}", flush=True)
                elif ch in ('s', 'S'):
                    manual_throttle = max(-1.0, manual_throttle - 0.05)
                    print(f"[Manual] throttle {manual_throttle:.2f}", flush=True)
                elif ch in ('a',):  # already used for mode; lower-case 'a' toggles auto
                    pass
                elif ch in ('d', 'D'):
                    manual_throttle = manual_throttle  # no-op here
                elif ch == '\x1b':  # possible arrow
                    # Read two more chars if available to detect arrows
                    seq1 = kb.get_key(timeout=0.01)
                    seq2 = kb.get_key(timeout=0.01)
                    if seq1 == '[' and seq2 in ('A','B','C','D'):
                        if seq2 == 'A':  # up
                            manual_throttle = min(1.0, manual_throttle + 0.05)
                        elif seq2 == 'B':  # down
                            manual_throttle = max(-1.0, manual_throttle - 0.05)
                        elif seq2 == 'C':  # right
                            manual_steer = min(1.0, manual_steer + 0.1)
                        elif seq2 == 'D':  # left
                            manual_steer = max(-1.0, manual_steer - 0.1)
                        print(f"[Manual] steer {manual_steer:.2f}, throttle {manual_throttle:.2f}", flush=True)
                # Apply manual if in manual mode
                if not self.auto_mode:
                    spwm = self.motors.set_steering(manual_steer)
                    tpwm = self.motors.set_throttle(manual_throttle)
                    print(f"[Out][MAN] steer {manual_steer:+.2f} -> {spwm:4d}, throttle {manual_throttle:+.2f} -> {tpwm:4d}", flush=True)

        self.stop()

    # -------------------------- Control loop --------------------------
    def control_loop(self):
        next_t = time.time()
        loops = 0
        while self.running and (MAX_LOOPS is None or loops < MAX_LOOPS):
            tnow = time.time()
            dt = self.ctrl_period  # target loop period for derivative/integral stability

            # Get latest frame
            frame = self.camera.get_frame()

            if frame is not None:
                # Downscale to processing size for speed
                # Simple nearest-neighbor via slicing (no OpenCV): stride down
                scale_y = frame.shape[0] / IMAGE_H
                scale_x = frame.shape[1] / IMAGE_W
                small = frame[::int(max(1, round(scale_y))), ::int(max(1, round(scale_x))), :]
                # Ensure exact target size by cropping/padding if needed
                small = small[:IMAGE_H, :IMAGE_W, :]
                if small.shape[0] != IMAGE_H or small.shape[1] != IMAGE_W:
                    # pad if necessary
                    pad_y = IMAGE_H - small.shape[0]
                    pad_x = IMAGE_W - small.shape[1]
                    small = np.pad(small,
                                   ((0, max(0, pad_y)), (0, max(0, pad_x)), (0,0)),
                                   mode='edge')
                    small = small[:IMAGE_H, :IMAGE_W, :]

                gray = to_gray_norm(small)
                y0, y1 = roi_slice(IMAGE_H, ROI_FRACTION)
                roi = gray[y0:y1, :]

                # Threshold
                mask = binary_threshold(roi, BIN_THRESH, invert=not LINE_IS_DARK)

                # Morphological cleanup (very light, CV-free)
                # - Horizontal 1D smoothing by convolution with [1,1,1]
                kernel = np.array([1,1,1], dtype=np.uint8)
                # pad left/right
                pad = np.pad(mask, ((0,0),(1,1)), mode='edge')
                conv = (pad[:,0:-2] + pad[:,1:-1] + pad[:,2:]) >= 2  # majority
                mask = conv.astype(np.uint8)

                center_norm, curvature = find_line_center(mask)
                if center_norm is not None:
                    self.last_line_time = tnow
                    self.last_center_err = center_norm  # negative=left, positive=right
                    self.last_curvature = curvature
                # else keep previous values for a short time

            # Compute control
            steer_cmd, throttle_cmd, spwm, tpwm = self.compute_and_drive(dt, tnow)

            # Optional logging/recording
            if self.recording and frame is not None:
                self.save_sample(frame, steer_cmd, throttle_cmd, tnow)

            # Pace loop
            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            loops += 1

        self.running = False

    def compute_and_drive(self, dt, tnow):
        # If no line for too long: stop
        if (tnow - self.last_line_time) > NO_LINE_TIMEOUT and self.auto_mode:
            self.i_error = 0.0
            self.prev_error = 0.0
            self.filtered_steer = 0.0
            self.motors.stop()
            return 0.0, 0.0, self.motors.steering_pwm_from_norm(0.0), self.motors.throttle_pwm_from_norm(0.0)

        if not self.auto_mode:
            # When manual mode, do not overwrite outputs here
            return 0.0, 0.0, 0, 0

        err = np.clip(self.last_center_err, -1.0, 1.0)
        d_err = (err - self.prev_error) / max(1e-3, dt)
        self.prev_error = err

        # PI(D)
        self.i_error = np.clip(self.i_error + err * dt, -STEER_I_CLAMP, STEER_I_CLAMP)
        steer_raw = (STEER_P_GAIN * err) + (STEER_I_GAIN * self.i_error) + (STEER_D_GAIN * d_err) + (DERIV_STEER_GAIN * d_err * 0.0)

        # Smooth steering
        steer_raw = np.clip(steer_raw, -MAX_STEER, MAX_STEER)
        self.filtered_steer = (1.0 - SMOOTH_STEER_ALPHA) * self.filtered_steer + SMOOTH_STEER_ALPHA * steer_raw
        steer_cmd = float(np.clip(self.filtered_steer, -1.0, 1.0))

        # Throttle scheduling
        throttle = BASE_THROTTLE
        if SLOWDOWN_AT_CURVE:
            curve_mag = min(1.0, abs(self.last_curvature) * 50.0)  # scale heuristic
            throttle = BASE_THROTTLE * (1.0 - CURVE_SLOWDOWN_GAIN * curve_mag)
        throttle_cmd = float(np.clip(throttle, 0.0, 1.0))

        # Output to motors
        spwm = self.motors.set_steering(steer_cmd)
        tpwm = self.motors.set_throttle(throttle_cmd)

        if int(time.time() * 10) % 10 == 0:  # roughly 10 Hz print
            print(f"[Out][AUTO] err {err:+.3f} curv {self.last_curvature:+.4f} | steer {steer_cmd:+.2f} -> {spwm:4d} | throttle {throttle_cmd:+.2f} -> {tpwm:4d}", flush=True)

        return steer_cmd, throttle_cmd, spwm, tpwm

    # -------------------------- Recording --------------------------
    def _start_recording(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(DATA_ROOT, f"lf_{ts}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta_path = os.path.join(self.data_dir, "meta.csv")
        with open(self.meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","timestamp","steer","throttle"])

    def save_sample(self, frame_rgb, steer_cmd, throttle_cmd, tnow):
        if self.data_dir is None:
            return
        # Save small JPEG-like npy for speed (no PIL/OpenCV here)
        img_name = f"img_{self.frame_id:06d}.npy"
        np.save(os.path.join(self.data_dir, img_name), frame_rgb)
        with open(self.meta_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.frame_id, f"{tnow:.6f}", f"{steer_cmd:.4f}", f"{throttle_cmd:.4f}"])
        self.frame_id += 1

# ------------------------------
# Entry
# ------------------------------
def main():
    lf = LineFollower(PWM_STEERING_THROTTLE)
    try:
        lf.start()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C received; stopping...", flush=True)
    finally:
        lf.stop()
        print("[Main] Stopped cleanly.", flush=True)

if __name__ == "__main__":
    main()
