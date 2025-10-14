#!/usr/bin/env python3
# Ultra-Responsive Simple Line Follower (no TF, no OpenCV, no recording)
# Target: Black track with YELLOW dotted line on BLACK background
# - Decoupled loops: control (250 Hz), camera (25 Hz)
# - Keyboard: h manual, a auto, arrows steer/throttle, space stop (latched), c center, q quit
# - PID-style steering with smoothing; curvature-aware throttle slowdown
# - Robust yellow detection using HSV + small morphological cleanup; grayscale fallback
# - Safety: emergency stop latch with neutral-hold to prevent runaway on glitches

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
    "THROTTLE_FORWARD_PWM": 420,  # adjust to match your ESC forward endpoint
    "THROTTLE_STOPPED_PWM": 370,  # adjust to true neutral; test 368–374 if needed
    "THROTTLE_REVERSE_PWM": 290,
}

# Camera
CAM_W, CAM_H = 320, 240
HFLIP, VFLIP = False, False

# Processing size (downscale for speed)
PROC_W, PROC_H = 160, 120

# Line detection (yellow on black)
# We use HSV to target yellow; fallback to grayscale threshold if needed.
USE_ADAPTIVE_GRAY = False  # set True to try adaptive fallback when lighting is tough
ROI_FRACTION = (0.55, 0.95)    # look lower in the image
MIN_PIXELS = 40                # minimum pixels to trust a detection (after cleanup)

# HSV yellow thresholds (0-1 normalized HSV we compute manually)
# Hue for yellow ~ [0.10, 0.18], saturation high, value moderate-high
HSV_YELLOW = {
    "H_MIN": 0.10, "H_MAX": 0.18,
    "S_MIN": 0.45, "V_MIN": 0.35
}

# Grayscale fallback thresholds
THRESH_BRIGHT = 0.58     # for bright-on-dark
ADAPTIVE_OFFSET = 0.10   # if adaptive used: local mean + offset

# Morphological cleanup
USE_MORPH = True
MORPH_MIN_NEIGHBORS = 4  # majority-like cleanup (center + neighbors)

# Loops
CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ  = 25

# Control: sharper steering
STEER_P_GAIN = 1.30
STEER_I_GAIN = 0.00
STEER_D_GAIN = 0.08
STEER_I_CLAMP = 0.3
SMOOTH_STEER_ALPHA = 0.55
MAX_STEER = 1.0

# Throttle (raised so commanded PWM is clearly above deadband)
BASE_THROTTLE = 0.50
SLOWDOWN_AT_CURVE = True
CURVE_SLOWDOWN_GAIN = 0.30
CURVATURE_SCALE = 40.0
MIN_RUN_THROTTLE = 0.22   # never let it fall below this when auto

# Safety
NO_LINE_TIMEOUT = 1.0  # seconds since last detection -> stop

# Debug
PRINT_DETECT_INFO = True  # print detection area and center sometimes

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
# Vision utilities
# ------------------------------
def to_gray_norm(rgb):
    g = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)
    return g/255.0

def rgb_to_hsv01(rgb):
    # rgb in [0..255], returns hsv in [0..1]
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[...,0], f[...,1], f[...,2]
    mx = np.max(f, axis=-1)
    mn = np.min(f, axis=-1)
    diff = mx - mn + 1e-6

    # Hue
    h = np.zeros_like(mx)
    mask = (mx == r)
    h[mask] = ((g - b)[mask] / diff[mask]) % 6.0
    mask = (mx == g)
    h[mask] = ((b - r)[mask] / diff[mask]) + 2.0
    mask = (mx == b)
    h[mask] = ((r - g)[mask] / diff[mask]) + 4.0
    h = (h / 6.0) % 1.0

    # Saturation
    s = diff / (mx + 1e-6)

    # Value
    v = mx
    return np.stack([h, s, v], axis=-1)

def get_roi(img, roi_frac):
    h = img.shape[0]
    y0 = int(h*roi_frac[0])
    y1 = max(y0+1, int(h*roi_frac[1]))
    return img[y0:y1, :], y0, y1

def morph_cleanup(mask, min_neighbors=4):
    # 3x3 neighborhood cleanup: keep pixels with >= min_neighbors of 1s in neighborhood including self
    pad = np.pad(mask, ((1,1),(1,1)), mode='edge')
    s = (
        pad[0:-2,0:-2] + pad[0:-2,1:-1] + pad[0:-2,2:] +
        pad[1:-1,0:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
        pad[2:  ,0:-2] + pad[2:  ,1:-1] + pad[2:  ,2:]
    )
    return (s >= min_neighbors).astype(np.uint8)

def hsv_yellow_mask(rgb_roi):
    hsv = rgb_to_hsv01(rgb_roi)
    h = hsv[...,0]; s = hsv[...,1]; v = hsv[...,2]
    m = (
        (h >= HSV_YELLOW["H_MIN"]) & (h <= HSV_YELLOW["H_MAX"]) &
        (s >= HSV_YELLOW["S_MIN"]) &
        (v >= HSV_YELLOW["V_MIN"])
    ).astype(np.uint8)
    if USE_MORPH:
        m = morph_cleanup(m, MORPH_MIN_NEIGHBORS)
    return m

def gray_bright_mask(gray_roi, thresh):
    m = (gray_roi >= thresh).astype(np.uint8)
    if USE_MORPH:
        m = morph_cleanup(m, MORPH_MIN_NEIGHBORS)
    return m

def adaptive_gray_mask(gray_roi, offset=ADAPTIVE_OFFSET):
    # Simple local adaptive: compare to row-wise mean + offset
    row_means = gray_roi.mean(axis=1, keepdims=True)
    m = (gray_roi >= (row_means + offset)).astype(np.uint8)
    if USE_MORPH:
        m = morph_cleanup(m, MORPH_MIN_NEIGHBORS)
    return m

def find_line_center_and_curvature(mask):
    h, w = mask.shape
    row_sums = mask.sum(axis=1)
    total = int(row_sums.sum())
    if total < MIN_PIXELS:
        return None, None, total
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = (mask * xs[None, :]).sum(axis=1)
        row_centroids = np.where(row_sums > 0, col_sums / np.maximum(row_sums, 1), np.nan)
    valid = ~np.isnan(row_centroids)
    if not np.any(valid):
        return None, None, total
    weights = (ys + 1.0)
    weighted_center = np.nansum(row_centroids[valid]*weights[valid]) / np.nansum(weights[valid])
    center_norm = (weighted_center / (w - 1)) * 2.0 - 1.0
    # curvature proxy: slope of centroid vs. row index
    if np.sum(valid) >= 3:
        yv = ys[valid]; cv = row_centroids[valid]
        y_mean = np.mean(yv); c_mean = np.mean(cv)
        denom = np.sum((yv - y_mean)**2) + 1e-6
        slope = np.sum((yv - y_mean)*(cv - c_mean)) / denom
        curvature = slope / (w + 1e-6)
    else:
        curvature = 0.0
    return float(center_norm), float(curvature), total

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

        # Emergency stop latch
        self.stop_latched = False
        self.stop_hold_until = 0.0

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
        # Latch stop and hold neutral a few frames for safety
        for _ in range(5):
            self.drv.set_throttle(0.0)
            time.sleep(0.05)
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
                    if not self.stop_latched:
                        self.auto_mode = True
                        print("[Mode] Auto")
                    else:
                        print("[Mode] Auto requested, but stop is latched. Press up/down to manually move or reset latch with 'c' then 'a'.")
                elif ch == ' ':
                    # Emergency stop: disable auto, latch stop, center steer, hold neutral repeatedly
                    self.auto_mode = False
                    self.stop_latched = True
                    self.stop_hold_until = time.time() + 0.8
                    self.manual_throttle = 0.0
                    self.filtered_steer = 0.0
                    self.drv.set_steering(0.0)
                    print("[Keys] EMERGENCY STOP: auto disabled, neutral hold")
                    for _ in range(8):
                        self.drv.set_throttle(0.0)
                        time.sleep(0.1)
                elif ch in ('c', 'C'):
                    # Center steering and clear controller states; also clears stop latch
                    self.manual_steer = 0.0
                    self.last_err = 0.0
                    self.i_err = 0.0
                    self.filtered_steer = 0.0
                    self.stop_latched = False
                    self.drv.set_steering(0.0)
                    print("[Keys] Center steer and cleared stop latch")
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
                    # Respect stop latch: while latched, force neutral unless user deliberately commands reverse/forward
                    if self.stop_latched and time.time() < self.stop_hold_until:
                        tpwm = self.drv.set_throttle(0.0)
                    else:
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
                # Downscale by stride
                sy = max(1, round(frame.shape[0]/PROC_H))
                sx = max(1, round(frame.shape[1]/PROC_W))
                small = frame[::sy, ::sx, :][:PROC_H, :PROC_W, :]

                # ROI
                roi_rgb, _, _ = get_roi(small, ROI_FRACTION)

                # Primary: HSV yellow detection
                mask = hsv_yellow_mask(roi_rgb)
                total_pix = int(mask.sum())

                # Fallbacks if too few pixels
                if total_pix < MIN_PIXELS:
                    gray = to_gray_norm(roi_rgb)
                    # Try bright-on-dark threshold
                    mask = gray_bright_mask(gray, THRESH_BRIGHT)
                    total_pix = int(mask.sum())
                    # Adaptive option if still weak
                    if total_pix < MIN_PIXELS and USE_ADAPTIVE_GRAY:
                        mask = adaptive_gray_mask(gray, ADAPTIVE_OFFSET)
                        total_pix = int(mask.sum())

                center_norm, curvature, total = find_line_center_and_curvature(mask)
                if center_norm is not None:
                    self.last_line_time = tnow
                    self.last_center_err = center_norm
                    self.last_curvature = curvature
                    if PRINT_DETECT_INFO and int(time.time() * 5) % 5 == 0:
                        print(f"[DETECT] pix {total:4d} center {center_norm:+.3f} curv {curvature:+.4f}")
                else:
                    if PRINT_DETECT_INFO and int(time.time() * 5) % 5 == 0:
                        print("[DETECT] no line")

            # Control and actuation
            self.compute_and_drive(dt, tnow)

            # Pace
            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

    def compute_and_drive(self, dt, tnow):
        # Respect emergency stop latch
        if self.stop_latched:
            self.drv.set_throttle(0.0)
            # Keep latch until user clears with 'c' and re-enters auto
            return

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

        # Steering
        steer_raw = (STEER_P_GAIN * err) + (STEER_I_GAIN * self.i_err) + (STEER_D_GAIN * d_err)
        steer_raw = float(np.clip(steer_raw, -MAX_STEER, MAX_STEER))
        self.filtered_steer = (1.0 - SMOOTH_STEER_ALPHA) * self.filtered_steer + SMOOTH_STEER_ALPHA * steer_raw
        steer_cmd = float(np.clip(self.filtered_steer, -1.0, 1.0))

        # Throttle
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
