#!/usr/bin/env python3
# Ultra-Responsive Simple Line Follower (no TF, no OpenCV, no recording)
# Target: Black track with yellow dotted line
# - Decoupled loops: control (250 Hz), camera (25 Hz)
# - Keyboard: h manual, a auto, arrows steer/throttle, space stop, c center, q quit
# - PID-style steering with smoothing; curvature-aware throttle slowdown
# - Robust bright-line detection with small 2D majority filter to handle dotted gaps

import time
import sys
import tty
import termios
import select
import threading
import numpy as np

# Picamera2
from picamera2 import Picamera2
from libcamera import Transform

# PWM / PCA9685
import board
import busio
from adafruit_pca9685 import PCA9685

# ------------------------------
# Configuration (updated PWMs)
# ------------------------------
CFG = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 270,
    "STEERING_RIGHT_PWM": 470,
    "THROTTLE_FORWARD_PWM": 420,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 290,
}

# Camera
CAM_W, CAM_H = 320, 240
HFLIP, VFLIP = False, False

# Processing size (downscale for speed)
PROC_W, PROC_H = 160, 120

# Line detection (yellow on black -> bright line)
LINE_IS_DARK = False
ROI_FRACTION = (0.60, 0.92)
THRESH = 0.60
MIN_PIXELS = 40

# Loops
CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ  = 25

# Control: sharper steering
STEER_P_GAIN = 1.30
STEER_I_GAIN = 0.00
STEER_D_GAIN = 0.07
STEER_I_CLAMP = 0.3
SMOOTH_STEER_ALPHA = 0.50
MAX_STEER = 1.0

# Throttle (raised so commanded PWM is clearly above the deadband)
BASE_THROTTLE = 0.50
SLOWDOWN_AT_CURVE = True
CURVE_SLOWDOWN_GAIN = 0.30
CURVATURE_SCALE = 40.0
MIN_RUN_THROTTLE = 0.20  # never let it fall below this when auto

# Safety
NO_LINE_TIMEOUT = 1.0  # seconds since last detection -> stop

# ------------------------------
# PCA9685 helpers
# ------------------------------
def parse_pca9685_pin(pin_str):
    left, chan = pin_str.split(":")
    bus_str = left.split(".")[1]
    addr_str = chan.split(".")[0] if "." in chan else chan
    channel_str = chan.split(".")[1] if "." in chan else "0"
    i2c_bus = int(bus_str)
    i2c_addr = int(addr_str, 16) if addr_str.startswith(("0x","0X")) else int(addr_str)
    channel = int(channel_str)
    return i2c_bus, i2c_addr, channel

class Driver:
    def __init__(self, cfg):
        _, addr_s, s_ch = parse_pca9685_pin(cfg["PWM_STEERING_PIN"])
        _, addr_t, t_ch = parse_pca9685_pin(cfg["PWM_THROTTLE_PIN"])
        if addr_s != addr_t:
            raise ValueError("Steering and Throttle must be on same PCA9685 for this driver.")
        self.cfg = cfg
        self.s_ch = s_ch
        self.t_ch = t_ch
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=addr_s)
        self.pca.frequency = 60
        self.stop()

    def _to_duty(self, pwm4095):
        pwm4095 = int(np.clip(pwm4095, 0, 4095))
        return int((pwm4095/4095.0)*65535)

    def steering_pwm_from_norm(self, steer):
        if self.cfg["PWM_STEERING_INVERTED"]:
            steer = -steer
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        pwm = int(np.interp(steer, [-1, 1], [right, left]))
        return pwm

    def throttle_pwm_from_norm(self, throttle):
        if self.cfg["PWM_THROTTLE_INVERTED"]:
            throttle = -throttle
        rev = self.cfg["THROTTLE_REVERSE_PWM"]
        stop = self.cfg["THROTTLE_STOPPED_PWM"]
        fwd = self.cfg["THROTTLE_FORWARD_PWM"]
        pwm = int(np.interp(throttle, [-1, 0, 1], [rev, stop, fwd]))
        return pwm

    def set_steering(self, steer):
        pwm = self.steering_pwm_from_norm(steer)
        self.pca.channels[self.s_ch].duty_cycle = self._to_duty(pwm)
        return pwm

    def set_throttle(self, throttle):
        pwm = self.throttle_pwm_from_norm(throttle)
        self.pca.channels[self.t_ch].duty_cycle = self._to_duty(pwm)
        return pwm

    def stop(self):
        self.set_throttle(0.0)

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

# ------------------------------
# Camera worker (decoupled loop)
# ------------------------------
class CameraWorker:
    def __init__(self, w=CAM_W, h=CAM_H, hflip=HFLIP, vflip=VFLIP):
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (w, h), "format": "XRGB8888"},
            transform=Transform(hflip=hflip, vflip=vflip),
            buffer_count=4
        )
        self.cam.configure(config)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.period = 1.0 / CAMERA_LOOP_HZ
        self.thread = None

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
# Keyboard (non-blocking)
# ------------------------------
class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, a, b, c):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def get_key(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

# ------------------------------
# Vision
# ------------------------------
def to_gray_norm(rgb):
    g = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)
    return g/255.0

def get_roi(gray, roi_frac):
    h = gray.shape[0]
    y0 = int(h*roi_frac[0])
    y1 = max(y0+1, int(h*roi_frac[1]))
    return gray[y0:y1, :]

def threshold_mask(gray_roi, thresh, dark=True):
    if dark:
        return (gray_roi <= thresh).astype(np.uint8)
    else:
        return (gray_roi >= thresh).astype(np.uint8)

def majority3x3(mask):
    pad = np.pad(mask, ((1,1),(1,1)), mode='edge')
    s = (
        pad[0:-2,0:-2] + pad[0:-2,1:-1] + pad[0:-2,2:] +
        pad[1:-1,0:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
        pad[2:  ,0:-2] + pad[2:  ,1:-1] + pad[2:  ,2:]
    )
    return (s >= 5).astype(np.uint8)

def find_line_center_and_curvature(mask):
    h, w = mask.shape
    row_sums = mask.sum(axis=1)
    total = int(row_sums.sum())
    if total < MIN_PIXELS:
        return None, None
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = (mask * xs[None, :]).sum(axis=1)
        row_centroids = np.where(row_sums > 0, col_sums / np.maximum(row_sums, 1), np.nan)
    valid = ~np.isnan(row_centroids)
    if not np.any(valid):
        return None, None
    weights = (ys + 1.0)
    weighted_center = np.nansum(row_centroids[valid]*weights[valid]) / np.nansum(weights[valid])
    center_norm = (weighted_center / (w - 1)) * 2.0 - 1.0
    if np.sum(valid) >= 3:
        yv = ys[valid]; cv = row_centroids[valid]
        y_mean = np.mean(yv); c_mean = np.mean(cv)
        denom = np.sum((yv - y_mean)**2) + 1e-6
        slope = np.sum((yv - y_mean)*(cv - c_mean)) / denom
        curvature = slope / (w + 1e-6)
    else:
        curvature = 0.0
    return float(center_norm), float(curvature)

# ------------------------------
# Controller
# ------------------------------
class UltraSimpleLF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.drv = Driver(cfg)
        self.cam = CameraWorker()
        self.ctrl_period = 1.0 / CONTROL_LOOP_HZ

        # State
        self.running = True
        self.auto_mode = True
        self.last_line_time = 0.0
        self.last_err = 0.0
        self.i_err = 0.0
        self.filtered_steer = 0.0
        self.last_center_err = 0.0
        self.last_curvature = 0.0

        # Manual
        self.manual_steer = 0.0
        self.manual_throttle = 0.0

        self.thread = None

    def start(self):
        print("[Camera] Starting...")
        self.cam.start()
        t0 = time.time()
        while self.cam.get_frame() is None and (time.time()-t0) < 2.0:
            time.sleep(0.05)
        print("[Camera] OK")

        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.keyboard_loop()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cam.stop()
        self.drv.stop()
        self.drv.close()

    # Keyboard
    def keyboard_loop(self):
        print("[Keys] h manual, a auto, arrows steer/throttle, space stop, c center, q quit")
        with RawKeyboard() as kb:
            while self.running:
                ch = kb.get_key(timeout=0.05)
                if ch is None:
                    continue
                if ch in ('q', 'Q'):
                    print("[Keys] Quit")
                    self.running = False
                    break
                elif ch in ('h', 'H'):
                    self.auto_mode = False
                    print("[Mode] Manual")
                elif ch in ('a', 'A'):
                    self.auto_mode = True
                    print("[Mode] Auto")
                elif ch == ' ':
                    # Emergency stop: cut throttle immediately and disable auto
                    self.auto_mode = False
                    self.manual_throttle = 0.0
                    self.drv.stop()
                    print("[Keys] Emergency stop (auto disabled)")
                elif ch in ('c', 'C'):
                    # Center steering: zero controller states and command center
                    self.manual_steer = 0.0
                    self.last_err = 0.0
                    self.i_err = 0.0
                    self.filtered_steer = 0.0
                    self.drv.set_steering(0.0)
                    print("[Keys] Center steer")
                elif ch == '\x1b':  # ANSI arrows
                    s1 = kb.get_key(timeout=0.01)
                    s2 = kb.get_key(timeout=0.01)
                    if s1 == '[' and s2 in ('A','B','C','D'):
                        if s2 == 'A':   # up
                            self.manual_throttle = min(1.0, self.manual_throttle + 0.05)
                        elif s2 == 'B': # down
                            self.manual_throttle = max(-1.0, self.manual_throttle - 0.05)
                        elif s2 == 'C': # right
                            self.manual_steer = min(1.0, self.manual_steer + 0.12)
                        elif s2 == 'D': # left
                            self.manual_steer = max(-1.0, self.manual_steer - 0.12)
                        print(f"[Manual] steer {self.manual_steer:+.2f}, throttle {self.manual_throttle:+.2f}")
                # Apply manual output only when auto is off
                if not self.auto_mode:
                    spwm = self.drv.set_steering(self.manual_steer)
                    tpwm = self.drv.set_throttle(self.manual_throttle)
                    print(f"[Out][MAN] steer {self.manual_steer:+.2f}->{spwm:4d} | throttle {self.manual_throttle:+.2f}->{tpwm:4d}")

        self.stop()

    # Control loop
    def control_loop(self):
        next_t = time.time()
        while self.running:
            tnow = time.time()
            dt = self.ctrl_period

            # Perception
            frame = self.cam.get_frame()
            if frame is not None:
                sy = max(1, round(frame.shape[0]/PROC_H))
                sx = max(1, round(frame.shape[1]/PROC_W))
                small = frame[::sy, ::sx, :][:PROC_H, :PROC_W, :]
                gray = to_gray_norm(small)
                roi = get_roi(gray, ROI_FRACTION)
                mask = threshold_mask(roi, THRESH, dark=LINE_IS_DARK)
                mask = majority3x3(mask)
                center_norm, curvature = find_line_center_and_curvature(mask)
                if center_norm is not None:
                    self.last_line_time = tnow
                    self.last_center_err = center_norm
                    self.last_curvature = curvature

            # Control and actuation
            self.compute_and_drive(dt, tnow)

            # Pace
            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

    def compute_and_drive(self, dt, tnow):
        if not self.auto_mode:
            return

        if (tnow - self.last_line_time) > NO_LINE_TIMEOUT:
            self.i_err = 0.0
            self.last_err = 0.0
            self.filtered_steer = 0.0
            self.drv.stop()
            return

        err = float(np.clip(self.last_center_err, -1.0, 1.0))
        d_err = (err - self.last_err) / max(1e-3, dt)
        self.last_err = err
        self.i_err = np.clip(self.i_err + err * dt, -STEER_I_CLAMP, STEER_I_CLAMP)

        steer_raw = (STEER_P_GAIN * err) + (STEER_I_GAIN * self.i_err) + (STEER_D_GAIN * d_err)
        steer_raw = float(np.clip(steer_raw, -MAX_STEER, MAX_STEER))
        self.filtered_steer = (1.0 - SMOOTH_STEER_ALPHA) * self.filtered_steer + SMOOTH_STEER_ALPHA * steer_raw
        steer_cmd = float(np.clip(self.filtered_steer, -1.0, 1.0))

        # Curve-based throttle schedule with minimum run throttle
        throttle = BASE_THROTTLE
        if SLOWDOWN_AT_CURVE:
            curve_mag = min(1.0, abs(self.last_curvature) * CURVATURE_SCALE)
            throttle = BASE_THROTTLE * (1.0 - CURVE_SLOWDOWN_GAIN * curve_mag)
        throttle = max(throttle, MIN_RUN_THROTTLE)
        throttle_cmd = float(np.clip(throttle, 0.0, 1.0))

        spwm = self.drv.set_steering(steer_cmd)
        tpwm = self.drv.set_throttle(throttle_cmd)

        # ~10 Hz status
        if int(time.time() * 10) % 10 == 0:
            print(f"[AUTO] err {err:+.3f} curv {self.last_curvature:+.4f} | steer {steer_cmd:+.2f}->{spwm:4d} | throttle {throttle_cmd:+.2f}->{tpwm:4d}")

# ------------------------------
# Main
# ------------------------------
def main():
    lf = UltraSimpleLF(CFG)
    try:
        lf.start()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C; stopping...")
    finally:
        lf.stop()
        print("[Main] Stopped cleanly.")

if __name__ == "__main__":
    main()
