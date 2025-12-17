#!/usr/bin/env python3
"""
Meta Dot PiCar - Web Line Follower + Manual Control + Color Calibration

- Integrates the discrete line follower logic from line.py
- Supports GRAY or COLOR (HSV band) line detection
- Optional traffic sign / light handling using model.tflite + labels.txt
- Web UI:
  * Live camera preview
  * Live PWM values (steering + throttle)
  * AUTO / MANUAL toggle
  * Manual drive buttons
  * Recording toggle
  * Emergency stop, center steering, throttle up/down
  * Line detection mode (GRAY / COLOR)
  * One-shot color calibration (bottom ROI) like line.py
"""

import os
import time
import csv
import threading
from datetime import datetime

import numpy as np

from flask import (
    Flask, request, redirect, url_for, Response,
    render_template_string, jsonify
)

# Picamera2 / imaging
from picamera2 import Picamera2
from libcamera import Transform
import cv2

# PCA9685 / PWM
import board
import busio
from adafruit_pca9685 import PCA9685

# TFLite sign classifier
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# =========================================================
# Configuration (copied from line.py, slightly adjusted)
# =========================================================
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",  # ensure 0x40
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # ensure 0x40
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 400,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 320,
}

MODE_LINE      = "line"
MODE_SLOW      = "slow"
MODE_STOP_SIGN = "stop_sign"
MODE_WAIT_RED  = "wait_red"
MODE_UTURN     = "uturn"

SLOW_THROTTLE_PWM  = 395
UTURN_THROTTLE_PWM = 420

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGN_MODEL_PATH  = os.path.join(SCRIPT_DIR, "model.tflite")
SIGN_LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")

DEFAULT_SIGN_THRESHOLD = 0.40
SIGN_CLASS_THRESHOLDS = {
    "background": 0.9,
    "stop": 0.30,
    "slow": 0.30,
    "uturn": 0.60,
    "tf_red": 0.50,
    "tf_green": 0.40,
}

# Discrete line follower
DEAD_BAND_ON  = 0.14
DEAD_BAND_OFF = 0.08

NO_LINE_TIMEOUT       = 0.5
MAX_REVERSE_DURATION  = 6.0

USE_CURVE_SLOWDOWN    = True
CURVE_SLOWDOWN_GAIN   = 0.4

CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ  = 25
STATUS_HZ       = 10
MAX_LOOPS       = None

IMAGE_W = 160
IMAGE_H = 120

CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAM_STREAM_W = 320
CAM_STREAM_H = 240

DATA_ROOT = "data"

# Gray-mode line params
LINE_IS_DARK   = False
ROI_FRACTION   = (0.55, 0.95)
BIN_THRESH     = 0.45  # gray threshold [0..1]

# Color-mode HSV thresholds (same defaults as line.py)
H_LO_DEG = 10.0
H_HI_DEG = 50.0
S_MIN    = 70      # 0..100
V_MIN    = 30      # 0..100

# =========================================================
# PCA9685 helpers
# =========================================================
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
    """
    Same motor controller as in line.py, but we also track current PWM
    so we can display it in the UI.
    """
    def __init__(self, config):
        s_bus, s_addr, s_ch = parse_pca9685_pin(config["PWM_STEERING_PIN"])
        t_bus, t_addr, t_ch = parse_pca9685_pin(config["PWM_THROTTLE_PIN"])
        if s_bus != t_bus or s_addr != t_addr:
            raise ValueError("Steering and Throttle must be on same PCA9685.")
        self.channel_steer = s_ch
        self.channel_throttle = t_ch

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60
        self.cfg = config

        self.lock = threading.Lock()
        self.current_steer_pwm = self.steering_center_pwm()
        self.current_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]
        self.stop()

    def set_pwm_raw(self, channel, pwm_value, is_steer=None):
        pwm_value = int(np.clip(pwm_value, 0, 4095))
        duty16 = int((pwm_value / 4095.0) * 65535)
        with self.lock:
            self.pca.channels[channel].duty_cycle = duty16
            if is_steer is True:
                self.current_steer_pwm = pwm_value
            elif is_steer is False:
                self.current_throttle_pwm = pwm_value
        return pwm_value

    def steering_center_pwm(self):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        return int(round((left + right) / 2))

    def stop(self):
        self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_STOPPED_PWM"], is_steer=False)

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

    def get_pwm_status(self):
        with self.lock:
            return {
                "steering_pwm": self.current_steer_pwm,
                "throttle_pwm": self.current_throttle_pwm,
            }

# =========================================================
# TFLite Traffic sign helpers (same as line.py)
# =========================================================
def sign_load_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                line = parts[1]
            labels.append(line)
    return labels

def sign_set_input_tensor(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    height, width = input_details["shape"][1], input_details["shape"][2]
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)
    input_data = np.array(image)
    input_data = np.expand_dims(input_data, axis=0)

    if input_details["dtype"] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32) / 255.0

    interpreter.set_tensor(input_details["index"], input_data)

def sign_classify_top_k(interpreter, top_k=3):
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] == np.uint8:
        scale, zero_point = output_details.get("quantization", (1.0, 0))
        scores = (output_data.astype(np.float32) - zero_point) * scale
    else:
        scores = output_data.astype(np.float32)

    top_k_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_k_indices]

def sign_select_best_label(results, labels):
    if not results:
        return None, None
    best_label = None
    best_score = -1.0
    for class_id, score in results:
        if 0 <= class_id < len(labels):
            label = labels[class_id]
        else:
            continue
        key = label.lower()
        threshold = SIGN_CLASS_THRESHOLDS.get(key, DEFAULT_SIGN_THRESHOLD)
        if score >= threshold and score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return None, None
    return best_label, best_score

# =========================================================
# Camera worker (from line.py)
# =========================================================
class CameraWorker:
    def __init__(self, stream_w=CAM_STREAM_W, stream_h=CAM_STREAM_H,
                 hflip=False, vflip=False):
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

# =========================================================
# Image helpers (gray + HSV color)
# =========================================================
def to_gray_norm(rgb):
    g = (0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]).astype(np.float32)
    g /= 255.0
    return g

def binary_threshold(gray, thresh, invert=False):
    if invert:
        return (gray >= thresh).astype(np.uint8)
    else:
        return (gray <= thresh).astype(np.uint8)

def roi_slice(h, roi_frac):
    y0 = int(h * roi_frac[0])
    y1 = max(y0 + 1, int(h * roi_frac[1]))
    return y0, y1

def find_line_center(mask):
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    row_sums = mask.sum(axis=1)
    total = row_sums.sum()
    if total < 10:
        return None, None
    weights = (ys + 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        col_sums = (mask * xs[None,:]).sum(axis=1)
        row_centroids = np.where(row_sums > 0,
                                 col_sums / np.maximum(row_sums, 1),
                                 np.nan)
    valid = ~np.isnan(row_centroids)
    if not np.any(valid):
        return None, None
    weighted_center = np.nansum(row_centroids[valid] * weights[valid]) / np.nansum(weights[valid])
    center_norm = (weighted_center / (w - 1)) * 2.0 - 1.0

    if np.sum(valid) >= 3:
        y_valid = ys[valid]
        c_valid = row_centroids[valid]
        y_mean = np.mean(y_valid)
        c_mean = np.mean(c_valid)
        denom = np.sum((y_valid - y_mean)**2) + 1e-6
        slope = np.sum((y_valid - y_mean) * (c_valid - c_mean)) / denom
        curvature = slope / (w + 1e-6)
    else:
        curvature = 0.0
    return float(center_norm), float(curvature)

def rgb_to_hsv_np(rgb):
    """
    rgb uint8 -> hsv (H deg 0..360, S% 0..100, V% 0..100), vectorized
    """
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-8

    h = np.zeros_like(cmax)
    mask = delta > 1e-8
    r_eq = (cmax == r) & mask
    g_eq = (cmax == g) & mask
    b_eq = (cmax == b) & mask
    h[r_eq] = (60.0 * ((g[r_eq] - b[r_eq]) / delta[r_eq]) + 360.0) % 360.0
    h[g_eq] = (60.0 * ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 120.0) % 360.0
    h[b_eq] = (60.0 * ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 240.0) % 360.0

    s = np.zeros_like(cmax)
    s[mask] = (delta[mask] / cmax[mask]) * 100.0

    v = cmax * 100.0
    return h, s, v

def estimate_hue_band_and_vmin(
    H, S, V,
    min_s_for_color=60,
    center_crop_frac=0.5,
    hue_margin_deg=8.0,
    v_lo_percentile=30,
    v_min_margin=5
):
    """
    Same logic as in line.py: estimate hue band [h_lo,h_hi] and Vmin
    from a region containing the colored line.
    """
    Hh, Hw = H.shape
    cy0 = int((1.0 - center_crop_frac)/2.0 * Hh)
    cy1 = Hh - cy0
    cx0 = int((1.0 - center_crop_frac)/2.0 * Hw)
    cx1 = Hw - cx0

    Hc = H[cy0:cy1, cx0:cx1]
    Sc = S[cy0:cy1, cx0:cx1]
    Vc = V[cy0:cy1, cx0:cx1]

    mask = Sc >= float(min_s_for_color)
    if np.count_nonzero(mask) < 20:
        mask = np.ones_like(Sc, dtype=bool)

    h_vals = Hc[mask].reshape(-1)
    if h_vals.size == 0:
        return 0.0, 360.0, 30

    h_med = float(np.median(h_vals))
    h_lo = (h_med - hue_margin_deg) % 360.0
    h_hi = (h_med + hue_margin_deg) % 360.0

    v_vals = Vc[mask].reshape(-1)
    v_lo = float(np.percentile(v_vals, v_lo_percentile))
    v_min_est = int(round(max(0.0, min(100.0, v_lo - v_min_margin))))
    return h_lo, h_hi, v_min_est

def hsv_band_mask(rgb, h_lo_deg, h_hi_deg, s_min, v_min):
    """
    Binary mask for hue band (with wrap-around) and S,V minimums.
    """
    H, S, V = rgb_to_hsv_np(rgb)
    s_ok = (S >= float(s_min))
    v_ok = (V >= float(v_min))
    if h_lo_deg <= h_hi_deg:
        h_ok = (H >= h_lo_deg) & (H <= h_hi_deg)
    else:
        h_ok = (H >= h_lo_deg) | (H <= h_hi_deg)
    m = (h_ok & s_ok & v_ok).astype(np.uint8)
    return m

# =========================================================
# Line follower (no curses, driven by background thread)
# =========================================================
class WebLineFollower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker(stream_w=CAM_STREAM_W,
                                   stream_h=CAM_STREAM_H,
                                   hflip=CAMERA_HFLIP,
                                   vflip=CAMERA_VFLIP)

        self.running = False
        self.auto_mode = False
        self.recording = False

        self.last_line_time = 0.0
        self.last_center_err = 0.0
        self.last_curvature = 0.0
        self.last_decision = "STRAIGHT"

        self.data_dir = None
        self.frame_id = 0

        self.ctrl_period = 1.0 / CONTROL_LOOP_HZ

        self.manual_steer_pwm = self.motors.steering_center_pwm()
        self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]

        self.msg = "Startup: MANUAL, car stopped. Press 'AUTO (Line Follower)' to start."

        # Reverse search state
        self.no_line_start_time = None
        self.no_line_phase = "idle"
        self.no_line_phase_start = None

        # Traffic sign / light
        self.current_mode = MODE_LINE
        self.mode_until   = 0.0
        self.last_sign_check = 0.0
        self.sign_check_interval = 0.3
        self.sign_interpreter = None
        self.sign_labels = None
        self._init_sign_classifier()

        # Detection mode: "gray" or "color"
        self.detection_mode = "gray"

        # HSV thresholds (color mode)
        self.h_lo_deg = H_LO_DEG
        self.h_hi_deg = H_HI_DEG
        self.s_min    = S_MIN
        self.v_min    = V_MIN

        # Perf
        self.ctrl_count = 0
        self.ctrl_t0 = time.time()

        # Lock for status
        self.state_lock = threading.Lock()

    # ---------- TFLite init ----------
    def _init_sign_classifier(self):
        try:
            if not (os.path.exists(SIGN_MODEL_PATH) and os.path.exists(SIGN_LABELS_PATH)):
                self.msg = "Sign model/labels not found -> traffic signs disabled"
                return
            self.sign_interpreter = Interpreter(model_path=SIGN_MODEL_PATH)
            self.sign_interpreter.allocate_tensors()
            self.sign_labels = sign_load_labels(SIGN_LABELS_PATH)
            self.msg = f"Loaded sign model with {len(self.sign_labels)} labels"
        except Exception as e:
            self.msg = f"Sign classifier init failed: {e}"
            self.sign_interpreter = None
            self.sign_labels = None

    # ---------- Lifecycle ----------
    def start(self):
        self.camera.start()
        # wait for first frame
        t0 = time.time()
        while self.camera.get_frame() is None and time.time() - t0 < 2.0:
            time.sleep(0.02)
        self.running = True
        threading.Thread(target=self.control_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.camera.stop()
        self.motors.stop()
        self.motors.close()

    # ---------- External controls (from web UI) ----------
    def set_auto_mode(self, flag):
        with self.state_lock:
            self.auto_mode = bool(flag)
            if not self.auto_mode:
                self.msg = "Mode: MANUAL"
            else:
                self.msg = "Mode: AUTO (discrete)"

    def toggle_recording(self):
        with self.state_lock:
            self.recording = not self.recording
            if self.recording and self.data_dir is None:
                self._start_recording()
            self.msg = f"Recording: {'ON' if self.recording else 'OFF'}"

    def emergency_stop(self):
        with self.state_lock:
            self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]
            self.motors.stop()
            self.auto_mode = False
            self.msg = "Emergency stop -> MANUAL, neutral throttle"

    def center_steering(self):
        with self.state_lock:
            self.manual_steer_pwm = self.motors.steering_center_pwm()
            if not self.auto_mode:
                self.motors.set_pwm_raw(self.motors.channel_steer,
                                        self.manual_steer_pwm, is_steer=True)
            self.msg = f"Center steer -> {self.manual_steer_pwm}"

    def manual_forward(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_steer_pwm = self.motors.steering_center_pwm()
            self.manual_throttle_pwm = min(4095, self.manual_throttle_pwm + 5)
            self._apply_manual_pwm()
            self.msg = f"Manual throttle PWM {self.manual_throttle_pwm}"

    def manual_reverse(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_steer_pwm = self.motors.steering_center_pwm()
            self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]
            self._apply_manual_pwm()
            time.sleep(0.25)
            self.manual_throttle_pwm = self.cfg["THROTTLE_REVERSE_PWM"]
            self._apply_manual_pwm()
            self.msg = "Manual: reverse"

    def manual_left(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_steer_pwm = self.cfg["STEERING_LEFT_PWM"]
            self._apply_manual_pwm()
            self.msg = "Manual: left"

    def manual_right(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_steer_pwm = self.cfg["STEERING_RIGHT_PWM"]
            self._apply_manual_pwm()
            self.msg = "Manual: right"

    def manual_stop(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]
            self._apply_manual_pwm()
            self.msg = "Manual: stop"

    def manual_throttle_up(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_throttle_pwm = min(4095, self.manual_throttle_pwm + 5)
            self._apply_manual_pwm()
            self.msg = f"Manual throttle up -> {self.manual_throttle_pwm}"

    def manual_throttle_down(self):
        with self.state_lock:
            self.auto_mode = False
            self.manual_throttle_pwm = max(0, self.manual_throttle_pwm - 5)
            self._apply_manual_pwm()
            self.msg = f"Manual throttle down -> {self.manual_throttle_pwm}"

    def _apply_manual_pwm(self):
        self.motors.set_pwm_raw(self.motors.channel_steer,
                                self.manual_steer_pwm, is_steer=True)
        self.motors.set_pwm_raw(self.motors.channel_throttle,
                                self.manual_throttle_pwm, is_steer=False)

    # Detection mode + color calibration
    def set_detection_mode(self, mode):
        with self.state_lock:
            if mode in ("gray", "color"):
                self.detection_mode = mode
                self.msg = f"Detection mode -> {mode.upper()}"

    def calibrate_color(self):
        """
        One-shot color calibration:
        - Uses bottom ROI (same ROI_FRACTION as line detection)
        - Estimates hue band + Vmin
        - Keeps current S_min
        """
        frame = self.camera.get_frame()
        if frame is None:
            with self.state_lock:
                self.msg = "Calibration: no camera frame"
            return

        Hfull = frame.shape[0]
        y0 = int(Hfull * ROI_FRACTION[0])
        roi = frame[y0:Hfull, :, :]   # bottom region
        Hc, Sc, Vc = rgb_to_hsv_np(roi)

        h_lo_deg, h_hi_deg, v_min_est = estimate_hue_band_and_vmin(
            Hc, Sc, Vc,
            min_s_for_color=max(60, self.s_min - 10),
            center_crop_frac=0.5,
            hue_margin_deg=8.0,
            v_lo_percentile=30,
            v_min_margin=5
        )

        with self.state_lock:
            self.h_lo_deg = h_lo_deg
            self.h_hi_deg = h_hi_deg
            self.v_min = max(self.v_min, v_min_est)
            self.msg = (
                f"Calibrated: h=[{self.h_lo_deg:.1f},{self.h_hi_deg:.1f}]°, "
                f"s_min={self.s_min}, v_min={self.v_min}"
            )

    # ---------- Control loop (auto mode) ----------
    def control_loop(self):
        next_t = time.time()
        loops = 0
        while self.running and (MAX_LOOPS is None or loops < MAX_LOOPS):
            tnow = time.time()
            frame = self.camera.get_frame()

            if frame is not None:
                # downscale for line detection
                scale_y = frame.shape[0] / IMAGE_H
                scale_x = frame.shape[1] / IMAGE_W
                small = frame[::int(max(1, round(scale_y))),
                             ::int(max(1, round(scale_x))), :]
                small = small[:IMAGE_H, :IMAGE_W, :]
                if small.shape[0] != IMAGE_H or small.shape[1] != IMAGE_W:
                    pad_y = IMAGE_H - small.shape[0]
                    pad_x = IMAGE_W - small.shape[1]
                    small = np.pad(
                        small,
                        ((0, max(0, pad_y)), (0, max(0, pad_x)), (0,0)),
                        mode="edge"
                    )
                    small = small[:IMAGE_H, :IMAGE_W, :]

                y0, y1 = roi_slice(IMAGE_H, ROI_FRACTION)

                # read detection mode + HSV thresholds under lock
                with self.state_lock:
                    mode = self.detection_mode
                    h_lo = self.h_lo_deg
                    h_hi = self.h_hi_deg
                    s_min = self.s_min
                    v_min = self.v_min

                if mode == "gray":
                    gray = to_gray_norm(small)
                    roi_gray = gray[y0:y1, :]
                    mask = binary_threshold(roi_gray, BIN_THRESH, invert=not LINE_IS_DARK)
                else:
                    roi_rgb = small[y0:y1, :, :]
                    mask = hsv_band_mask(roi_rgb, h_lo, h_hi, s_min, v_min)

                # Light 1D majority filter horizontally
                pad = np.pad(mask, ((0,0), (1,1)), mode="edge")
                conv = (pad[:,0:-2] + pad[:,1:-1] + pad[:,2:]) >= 2
                mask = conv.astype(np.uint8)

                center_norm, curvature = find_line_center(mask)
                with self.state_lock:
                    if center_norm is not None:
                        self.last_line_time = tnow
                        self.last_center_err = center_norm
                        self.last_curvature = curvature
                        self.no_line_start_time = None
                        self.no_line_phase = "idle"
                        self.no_line_phase_start = None

                # traffic signs (optional)
                if (self.sign_interpreter is not None and
                    self.sign_labels is not None and
                    (tnow - self.last_sign_check) >= self.sign_check_interval):
                    self.last_sign_check = tnow
                    self._check_traffic_signs(frame, tnow)

            # Driving
            with self.state_lock:
                if self.auto_mode:
                    spwm, tpwm = self._compute_and_drive_discrete(tnow)
                else:
                    pwm_status = self.motors.get_pwm_status()
                    spwm = pwm_status["steering_pwm"]
                    tpwm = pwm_status["throttle_pwm"]

                # Recording
                if self.recording and frame is not None:
                    self._save_sample(frame, spwm, tpwm, tnow)

            # pace loop
            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

            self.ctrl_count += 1
            loops += 1

        self.running = False

    # ---------- Discrete auto controller (from line.py) ----------
    def _compute_and_drive_discrete(self, tnow):
        cfg = self.cfg
        left_pwm = cfg["STEERING_LEFT_PWM"]
        right_pwm = cfg["STEERING_RIGHT_PWM"]
        center_pwm = int(round((left_pwm + right_pwm) / 2))
        fwd_pwm = cfg["THROTTLE_FORWARD_PWM"]
        stop_pwm = cfg["THROTTLE_STOPPED_PWM"]
        reverse_pwm = cfg["THROTTLE_REVERSE_PWM"]

        time_since_line = tnow - self.last_line_time
        in_no_line = (time_since_line > NO_LINE_TIMEOUT)

        if in_no_line:
            if self.no_line_start_time is None:
                self.no_line_start_time = tnow
                self.no_line_phase = "neutral"
                self.no_line_phase_start = tnow
                self.msg = "Line lost -> reversing to re-acquire"

            elapsed_rev = tnow - self.no_line_start_time
            if elapsed_rev <= MAX_REVERSE_DURATION:
                if self.last_decision == "LEFT":
                    steer_pwm = left_pwm
                elif self.last_decision == "RIGHT":
                    steer_pwm = right_pwm
                else:
                    steer_pwm = center_pwm
                self.motors.set_pwm_raw(self.motors.channel_steer,
                                        steer_pwm, is_steer=True)

                if self.no_line_phase == "neutral":
                    self.motors.set_pwm_raw(self.motors.channel_throttle,
                                            stop_pwm, is_steer=False)
                    if (tnow - self.no_line_phase_start) >= 0.25:
                        self.no_line_phase = "reverse"
                        self.no_line_phase_start = tnow
                    self.last_decision = "NO_LINE"
                    return steer_pwm, stop_pwm

                self.motors.set_pwm_raw(self.motors.channel_throttle,
                                        reverse_pwm, is_steer=False)
                self.last_decision = "NO_LINE"
                return steer_pwm, reverse_pwm
            else:
                self.no_line_phase = "idle"
                self.no_line_phase_start = None
                spwm = self.motors.set_pwm_raw(self.motors.channel_steer,
                                               center_pwm, is_steer=True)
                tpwm = self.motors.set_pwm_raw(self.motors.channel_throttle,
                                               stop_pwm, is_steer=False)
                self.last_decision = "NO_LINE"
                self.msg = "Reverse timeout -> stopped; waiting for line"
                return spwm, tpwm

        self.no_line_start_time = None

        err = float(np.clip(self.last_center_err, -1.0, 1.0))

        decision = self.last_decision
        if decision in ("STRAIGHT", "NO_LINE"):
            if err < -DEAD_BAND_ON:
                decision = "LEFT"
            elif err > DEAD_BAND_ON:
                decision = "RIGHT"
            else:
                decision = "STRAIGHT"
        elif decision == "LEFT":
            if -DEAD_BAND_OFF <= err <= DEAD_BAND_OFF:
                decision = "STRAIGHT"
            elif err > DEAD_BAND_ON:
                decision = "RIGHT"
        elif decision == "RIGHT":
            if -DEAD_BAND_OFF <= err <= DEAD_BAND_OFF:
                decision = "STRAIGHT"
            elif err < -DEAD_BAND_ON:
                decision = "LEFT"
        self.last_decision = decision

        if decision == "LEFT":
            steer_pwm = left_pwm
        elif decision == "RIGHT":
            steer_pwm = right_pwm
        else:
            steer_pwm = center_pwm

        throttle_pwm = fwd_pwm
        if USE_CURVE_SLOWDOWN:
            curve_mag = min(1.0, abs(self.last_curvature) * 50.0)
            k = 1.0 - CURVE_SLOWDOWN_GAIN * curve_mag
            throttle_pwm = int(np.interp(k, [0.0, 1.0], [stop_pwm, fwd_pwm]))

        steer_pwm_final, throttle_pwm_final = self._apply_sign_mode(
            steer_pwm, throttle_pwm, tnow
        )

        spwm = self.motors.set_pwm_raw(self.motors.channel_steer,
                                       steer_pwm_final, is_steer=True)
        tpwm = self.motors.set_pwm_raw(self.motors.channel_throttle,
                                       throttle_pwm_final, is_steer=False)
        return spwm, tpwm

    # ---------- Traffic signs ----------
    def _check_traffic_signs(self, frame_rgb, tnow):
        if self.sign_interpreter is None or self.sign_labels is None:
            return
        h, w, _ = frame_rgb.shape
        CROP_FRACTION = 0.6
        cw = int(w * CROP_FRACTION)
        ch = int(h * CROP_FRACTION)
        x1 = (w - cw) // 2
        y1 = (h - ch) // 2
        x2 = x1 + cw
        y2 = y1 + ch
        crop = frame_rgb[y1:y2, x1:x2, :]
        image = Image.fromarray(crop)
        sign_set_input_tensor(self.sign_interpreter, image)
        results = sign_classify_top_k(self.sign_interpreter, top_k=3)
        label, score = sign_select_best_label(results, self.sign_labels)
        if label is None:
            return
        self._on_sign_detected(label, score, tnow)

    def _on_sign_detected(self, label, score, tnow):
        l = label.lower()
        if l == "stop":
            self.msg = f"STOP sign ({score:.2f}) -> stop 10 s"
            self.current_mode = MODE_STOP_SIGN
            self.mode_until   = tnow + 10.0
        elif l in ("tf_red", "tl_red"):
            self.msg = f"RED light ({score:.2f}) -> wait for GREEN"
            self.current_mode = MODE_WAIT_RED
        elif l in ("tf_green", "tl_green"):
            if self.current_mode == MODE_WAIT_RED:
                self.msg = f"GREEN light ({score:.2f}) -> resume line"
                self.current_mode = MODE_LINE
                self.mode_until   = 0.0
        elif l == "slow":
            self.msg = f"SLOW sign ({score:.2f}) -> slow mode 5 s"
            self.current_mode = MODE_SLOW
            self.mode_until = tnow + 5.0
        elif l == "uturn":
            if self.current_mode != MODE_UTURN:
                self.msg = f"UTURN sign ({score:.2f}) -> U-turn 10 s"
                self.current_mode = MODE_UTURN
                self.mode_until   = tnow + 10.0

    def _apply_sign_mode(self, steer_pwm_line, throttle_pwm_line, tnow):
        cfg = self.cfg
        stop_pwm    = cfg["THROTTLE_STOPPED_PWM"]
        left_pwm    = cfg["STEERING_LEFT_PWM"]
        right_pwm   = cfg["STEERING_RIGHT_PWM"]

        if self.current_mode in (MODE_STOP_SIGN, MODE_UTURN, MODE_SLOW) and self.mode_until > 0.0:
            if tnow >= self.mode_until:
                self.msg = f"Mode {self.current_mode} finished -> LINE"
                self.current_mode = MODE_LINE
                self.mode_until   = 0.0

        if self.current_mode == MODE_LINE:
            final_steer   = steer_pwm_line
            final_throttle = throttle_pwm_line
        elif self.current_mode == MODE_SLOW:
            final_steer   = steer_pwm_line
            final_throttle = SLOW_THROTTLE_PWM
        elif self.current_mode in (MODE_STOP_SIGN, MODE_WAIT_RED):
            final_steer   = steer_pwm_line
            final_throttle = stop_pwm
        elif self.current_mode == MODE_UTURN:
            final_steer   = right_pwm
            final_throttle = UTURN_THROTTLE_PWM
        else:
            final_steer   = steer_pwm_line
            final_throttle = throttle_pwm_line

        return final_steer, final_throttle

    # ---------- Recording ----------
    def _start_recording(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(DATA_ROOT, f"lfd_{ts}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta_path = os.path.join(self.data_dir, "meta.csv")
        with open(self.meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","timestamp","steer_pwm","throttle_pwm"])

    def _save_sample(self, frame_rgb, steer_pwm, throttle_pwm, tnow):
        if self.data_dir is None:
            return
        img_name = f"img_{self.frame_id:06d}.npy"
        np.save(os.path.join(self.data_dir, img_name), frame_rgb)
        with open(self.meta_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.frame_id, f"{tnow:.6f}", f"{steer_pwm:d}", f"{throttle_pwm:d}"])
        self.frame_id += 1

    # ---------- Status for UI ----------
    def get_status(self):
        with self.state_lock:
            pwm = self.motors.get_pwm_status()
            now = time.time()
            ctrl_fps = self.ctrl_count / max(1e-3, (now - self.ctrl_t0))
            self.ctrl_count = 0
            self.ctrl_t0 = now

            rem = 0.0
            if (self.mode_until > 0.0 and
                self.current_mode in (MODE_STOP_SIGN, MODE_SLOW, MODE_UTURN)):
                rem = max(0.0, self.mode_until - now)

            return {
                "auto_mode": self.auto_mode,
                "recording": self.recording,
                "decision": self.last_decision,
                "center_error": round(self.last_center_err, 3),
                "curvature": round(self.last_curvature, 4),
                "sign_mode": self.current_mode,
                "sign_mode_remaining": round(rem, 1),
                "steering_pwm": pwm["steering_pwm"],
                "throttle_pwm": pwm["throttle_pwm"],
                "msg": self.msg,
                "ctrl_fps": round(ctrl_fps, 1),
                "detection_mode": self.detection_mode,
                "h_lo_deg": round(self.h_lo_deg, 1),
                "h_hi_deg": round(self.h_hi_deg, 1),
                "s_min": self.s_min,
                "v_min": self.v_min,
            }

# =========================================================
# Flask app
# =========================================================
app = Flask(__name__)
lf = WebLineFollower(PWM_STEERING_THROTTLE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Meta Dot PiCar - Web Line Follower</title>
  <style>
    body { font-family: sans-serif; text-align: center; }
    .btn { padding: 10px 20px; margin: 4px; font-size: 16px; }
    .row { margin: 8px; }
    #video { border: 2px solid #333; margin-top: 10px; }
    #status { margin-top: 10px; text-align: left; display:inline-block; }
  </style>
</head>
<body>
  <h1>Meta Dot PiCar - Web Line Follower</h1>

  <h2>Camera Preview</h2>
  <img id="video" src="{{ url_for('video_feed') }}" width="640" height="480" />

  <h2>Status</h2>
  <div id="status">
    <div>Mode: <span id="mode_text"></span></div>
    <div>Recording: <span id="rec_text"></span></div>
    <div>Decision: <span id="decision"></span></div>
    <div>Error (norm): <span id="center_error"></span></div>
    <div>Curvature: <span id="curvature"></span></div>
    <div>Sign mode: <span id="sign_mode"></span>
      (rem: <span id="sign_mode_rem"></span> s)</div>
    <div>Steering PWM: <span id="steer_pwm"></span></div>
    <div>Throttle PWM: <span id="throttle_pwm"></span></div>
    <div>Detection: <span id="det_mode"></span></div>
    <div>HSV: h_lo=<span id="h_lo"></span>°, h_hi=<span id="h_hi"></span>°,
         s_min=<span id="s_min"></span>, v_min=<span id="v_min"></span></div>
    <div>Control loop (est): <span id="ctrl_fps"></span> Hz</div>
    <div>Msg: <span id="msg"></span></div>
  </div>

  <script>
    function refreshStatus() {
      fetch("{{ url_for('status') }}")
        .then(r => r.json())
        .then(d => {
          document.getElementById("mode_text").textContent =
            d.auto_mode ? "AUTO (line follower)" : "MANUAL";
          document.getElementById("rec_text").textContent =
            d.recording ? "ON" : "OFF";
          document.getElementById("decision").textContent = d.decision;
          document.getElementById("center_error").textContent = d.center_error;
          document.getElementById("curvature").textContent = d.curvature;
          document.getElementById("sign_mode").textContent = d.sign_mode;
          document.getElementById("sign_mode_rem").textContent = d.sign_mode_remaining;
          document.getElementById("steer_pwm").textContent = d.steering_pwm;
          document.getElementById("throttle_pwm").textContent = d.throttle_pwm;
          document.getElementById("ctrl_fps").textContent = d.ctrl_fps;
          document.getElementById("msg").textContent = d.msg;

          document.getElementById("det_mode").textContent = d.detection_mode.toUpperCase();
          document.getElementById("h_lo").textContent = d.h_lo_deg;
          document.getElementById("h_hi").textContent = d.h_hi_deg;
          document.getElementById("s_min").textContent = d.s_min;
          document.getElementById("v_min").textContent = d.v_min;
        })
        .catch(err => console.error("Status error:", err));
    }
    setInterval(refreshStatus, 300);
    window.onload = refreshStatus;
  </script>

  <hr>

  <h2>Mode & Safety</h2>
  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="auto_on">AUTO (Line Follower)</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="auto_off">MANUAL</button>
    </form>
  </div>
  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="record_toggle">Toggle Recording</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="estop" style="background:#f66;">EMERGENCY STOP</button>
    </form>
  </div>

  <hr>

  <h2>Line Detection Mode & Color Calibration</h2>
  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="mode_gray">Use GRAY mode</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="mode_color">Use COLOR (HSV) mode</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="calibrate_color">Calibrate Color (ROI)</button>
    </form>
  </div>

  <hr>

  <h2>Manual Drive</h2>
  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}">
      <button class="btn" name="action" value="forward">▲ Forward</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="left">◀ Left</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="stop">■ Stop</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="right">▶ Right</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}">
      <button class="btn" name="action" value="reverse">▼ Reverse</button>
    </form>
  </div>

  <div class="row">
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="center">Center Steering</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="throttle_up">Throttle +</button>
    </form>
    <form method="post" action="{{ url_for('cmd') }}" style="display:inline;">
      <button class="btn" name="action" value="throttle_down">Throttle -</button>
    </form>
  </div>

</body>
</html>
"""

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/status", methods=["GET"])
def status():
    return jsonify(lf.get_status())

@app.route("/cmd", methods=["POST"])
def cmd():
    action = request.form.get("action", "")
    print("CMD:", action)

    if action == "auto_on":
        lf.set_auto_mode(True)
    elif action == "auto_off":
        lf.set_auto_mode(False)
    elif action == "record_toggle":
        lf.toggle_recording()
    elif action == "estop":
        lf.emergency_stop()

    elif action == "mode_gray":
        lf.set_detection_mode("gray")
    elif action == "mode_color":
        lf.set_detection_mode("color")
    elif action == "calibrate_color":
        lf.calibrate_color()

    elif action == "forward":
        lf.manual_forward()
    elif action == "reverse":
        lf.manual_reverse()
    elif action == "left":
        lf.manual_left()
    elif action == "right":
        lf.manual_right()
    elif action == "stop":
        lf.manual_stop()
    elif action == "center":
        lf.center_steering()
    elif action == "throttle_up":
        lf.manual_throttle_up()
    elif action == "throttle_down":
        lf.manual_throttle_down()

    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = lf.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            # frame is RGB; OpenCV expects BGR
            ret, buffer = cv2.imencode(".jpg", frame[:, :, ::-1])
            if not ret:
                continue
            jpg = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- Main ----------
if __name__ == "__main__":
    try:
        lf.start()
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        lf.stop()
