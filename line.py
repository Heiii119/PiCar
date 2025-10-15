#!/usr/bin/env python3
# Meta Dot PiCar Line Follower (Discrete) with curses dashboard
# - Discrete LEFT/STRAIGHT/RIGHT steering (fixed PWMs, hysteresis)
# - Curses dashboard via stdscr.addstr()
# - Keyboard inside curses: h manual, a auto, r record, space stop, c center, p print decision line, q quit
# - Arrows/WASD for manual throttle/steer (when in manual)
#
# Run:
#   LIBCAMERA_LOG_LEVELS=*:2 python3 -u line_follow_discrete_curses.py
#
# Tip: If your terminal becomes garbled after crash, run `reset`.

import os
import sys
import time
import csv
import threading
from datetime import datetime
import curses

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
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # ensure 0x40
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 280,
    "STEERING_RIGHT_PWM": 480,
    "THROTTLE_FORWARD_PWM": 393,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 220,
}

# Discrete decision thresholds (normalized error in [-1,1])
DEAD_BAND_ON = 0.14   # must exceed this to ENTER a turn
DEAD_BAND_OFF = 0.08  # must fall within this to EXIT a turn (hysteresis)

# Safety
NO_LINE_TIMEOUT = 1.0  # seconds without detection -> neutral throttle

# Throttle/curve behavior
USE_CURVE_SLOWDOWN = True
CURVE_SLOWDOWN_GAIN = 0.4  # 0..1 scale

# Loops
CONTROL_LOOP_HZ = 250
CAMERA_LOOP_HZ = 25
STATUS_HZ = 10
MAX_LOOPS = None  # None = run forever

# Image sizes
IMAGE_W = 160
IMAGE_H = 120

# Camera settings
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAM_STREAM_W = 320
CAM_STREAM_H = 240

DATA_ROOT = "data"

# Line detection parameters
LINE_IS_DARK = True         # True if the line is darker than the floor
ROI_FRACTION = (0.55, 0.95) # Use bottom 40% of the image for line search
BIN_THRESH = 0.45           # Threshold on normalized grayscale [0..1]

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
        self.pca = PCA9685(self.i2c, address=s_addr)
        self.pca.frequency = 60
        self.cfg = config
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
        self.set_pwm_raw(self.channel_throttle, self.cfg["THROTTLE_STOPPED_PWM"])

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

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
        col_sums = (mask * xs[None, :]).sum(axis=1)
        row_centroids = np.where(row_sums > 0, col_sums / np.maximum(row_sums, 1), np.nan)
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

# ------------------------------
# Discrete line follower
# ------------------------------
class LineFollowerDiscrete:
    def __init__(self, cfg, stdscr):
        self.cfg = cfg
        self.stdscr = stdscr
        self.motors = MotorServoController(cfg)
        self.camera = CameraWorker(stream_w=CAM_STREAM_W, stream_h=CAM_STREAM_H,
                                   hflip=CAMERA_HFLIP, vflip=CAMERA_VFLIP)
        # State
        self.running = True
        self.auto_mode = True
        self.recording = False
        self.last_line_time = 0.0

        self.ctrl_period = 1.0 / CONTROL_LOOP_HZ
        self.status_period = 1.0 / STATUS_HZ
        self.next_status_t = time.time()

        self.last_center_err = 0.0
        self.last_curvature = 0.0
        self.last_decision = "STRAIGHT"

        # Recording
        self.data_dir = None
        self.frame_id = 0

        # Perf meters
        self.ctrl_count = 0
        self.cam_count = 0
        self.ctrl_t0 = time.time()
        self.cam_t0 = time.time()

        # Manual control
        self.manual_steer_pwm = self.motors.steering_center_pwm()
        self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]

        # Message line
        self.msg = ""

    # ------------- Lifecycle -------------
    def start(self):
        self.camera.start()
        # Wait for first frame (briefly)
        t0 = time.time()
        while self.camera.get_frame() is None and time.time() - t0 < 2.0:
            time.sleep(0.02)

        # Curses setup
        self.stdscr.nodelay(True)  # non-blocking getch
        curses.curs_set(0)         # hide cursor
        self.draw_status(force=True)

        # Start control loop thread
        self.running = True
        t = threading.Thread(target=self.control_loop, daemon=True)
        t.start()

        # Keyboard loop inside curses main thread
        self.keyboard_loop()

        # Cleanup
        self.camera.stop()
        self.motors.stop()
        self.motors.close()

    def stop(self):
        self.running = False

    # ------------- Keyboard -------------
    def keyboard_loop(self):
        while self.running:
            ch = self.stdscr.getch()
            if ch == -1:
                # update status at STATUS_HZ
                now = time.time()
                if now >= self.next_status_t:
                    self.draw_status()
                    self.next_status_t = now + self.status_period
                time.sleep(0.01)
                continue

            # Normalize codes
            if ch in (ord('q'), ord('Q')):
                self.msg = "Quit requested"
                self.stop()
                break
            elif ch in (ord('h'), ord('H')):
                self.auto_mode = False
                self.msg = "Mode: MANUAL"
            elif ch in (ord('a'), ord('A')):
                self.auto_mode = True
                self.msg = "Mode: AUTO (discrete)"
            elif ch in (ord('r'), ord('R')):
                self.recording = not self.recording
                if self.recording and self.data_dir is None:
                    self._start_recording()
                self.msg = f"Recording: {'ON' if self.recording else 'OFF'}"
            elif ch == ord(' '):
                self.manual_throttle_pwm = self.cfg["THROTTLE_STOPPED_PWM"]
                self.motors.stop()
                self.msg = "Emergency stop (neutral throttle)"
            elif ch in (ord('c'), ord('C')):
                self.manual_steer_pwm = self.motors.steering_center_pwm()
                self.msg = f"Center steer -> {self.manual_steer_pwm}"
            elif ch in (ord('p'), ord('P')):
                self.msg = f"Decision={self.last_decision}, err={self.last_center_err:+.3f}"

            # Arrows / WASD
            elif ch in (ord('w'), ord('W')):
                self.manual_throttle_pwm = min(4095, self.manual_throttle_pwm + 10)
                self.msg = f"Manual throttle PWM {self.manual_throttle_pwm}"
            elif ch in (ord('s'), ord('S')):
                self.manual_throttle_pwm = max(0, self.manual_throttle_pwm - 10)
                self.msg = f"Manual throttle PWM {self.manual_throttle_pwm}"
            elif ch == curses.KEY_UP:
                self.manual_throttle_pwm = min(4095, self.manual_throttle_pwm + 10)
                self.msg = f"Manual throttle PWM {self.manual_throttle_pwm}"
            elif ch == curses.KEY_DOWN:
                self.manual_throttle_pwm = max(0, self.manual_throttle_pwm - 10)
                self.msg = f"Manual throttle PWM {self.manual_throttle_pwm}"
            elif ch == curses.KEY_LEFT:
                self.manual_steer_pwm = self.cfg["STEERING_LEFT_PWM"]
                self.msg = f"Manual steer LEFT {self.manual_steer_pwm}"
            elif ch == curses.KEY_RIGHT:
                self.manual_steer_pwm = self.cfg["STEERING_RIGHT_PWM"]
                self.msg = f"Manual steer RIGHT {self.manual_steer_pwm}"

            # Apply manual immediately in manual mode
            if not self.auto_mode:
                spwm = self.motors.set_pwm_raw(self.motors.channel_steer, self.manual_steer_pwm)
                tpwm = self.motors.set_pwm_raw(self.motors.channel_throttle, self.manual_throttle_pwm)

            self.draw_status()

    # ------------- Control loop -------------
    def control_loop(self):
        next_t = time.time()
        loops = 0
        while self.running and (MAX_LOOPS is None or loops < MAX_LOOPS):
            tnow = time.time()

            # Perception
            frame = self.camera.get_frame()
            if frame is not None:
                # Downscale by striding
                scale_y = frame.shape[0] / IMAGE_H
                scale_x = frame.shape[1] / IMAGE_W
                small = frame[::int(max(1, round(scale_y))), ::int(max(1, round(scale_x))), :]
                small = small[:IMAGE_H, :IMAGE_W, :]
                if small.shape[0] != IMAGE_H or small.shape[1] != IMAGE_W:
                    pad_y = IMAGE_H - small.shape[0]
                    pad_x = IMAGE_W - small.shape[1]
                    small = np.pad(small, ((0, max(0, pad_y)), (0, max(0, pad_x)), (0,0)), mode='edge')
                    small = small[:IMAGE_H, :IMAGE_W, :]

                gray = to_gray_norm(small)
                y0, y1 = roi_slice(IMAGE_H, ROI_FRACTION)
                roi = gray[y0:y1, :]
                mask = binary_threshold(roi, BIN_THRESH, invert=not LINE_IS_DARK)

                # Light 1D majority filter horizontally
                pad = np.pad(mask, ((0,0),(1,1)), mode='edge')
                conv = (pad[:,0:-2] + pad[:,1:-1] + pad[:,2:]) >= 2
                mask = conv.astype(np.uint8)

                center_norm, curvature = find_line_center(mask)
                if center_norm is not None:
                    self.last_line_time = tnow
                    self.last_center_err = center_norm
                    self.last_curvature = curvature

            # Drive
            spwm, tpwm = self.compute_and_drive_discrete(tnow)

            # Recording
            if self.recording and frame is not None:
                self.save_sample(frame, spwm, tpwm, tnow)

            # Pace loop
            next_t += self.ctrl_period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

            # Perf counters
            self.ctrl_count += 1
            loops += 1

        self.running = False

    # ------------- Discrete controller -------------
    def compute_and_drive_discrete(self, tnow):
        cfg = self.cfg
        left_pwm = cfg["STEERING_LEFT_PWM"]
        right_pwm = cfg["STEERING_RIGHT_PWM"]
        center_pwm = int(round((left_pwm + right_pwm) / 2))
        fwd_pwm = cfg["THROTTLE_FORWARD_PWM"]
        stop_pwm = cfg["THROTTLE_STOPPED_PWM"]

        # Safety: no line recently
        if (tnow - self.last_line_time) > NO_LINE_TIMEOUT and self.auto_mode:
            spwm = self.motors.set_pwm_raw(self.motors.channel_steer, center_pwm)
            tpwm = self.motors.set_pwm_raw(self.motors.channel_throttle, stop_pwm)
            self.last_decision = "NO_LINE"
            return spwm, tpwm

        if not self.auto_mode:
            return 0, 0

        err = float(np.clip(self.last_center_err, -1.0, 1.0))

        # Hysteresis-based decision
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

        # Steering PWM by decision
        if decision == "LEFT":
            steer_pwm = left_pwm
        elif decision == "RIGHT":
            steer_pwm = right_pwm
        else:
            steer_pwm = center_pwm

        # Throttle PWM with optional curve slowdown
        throttle_pwm = fwd_pwm
        if USE_CURVE_SLOWDOWN:
            curve_mag = min(1.0, abs(self.last_curvature) * 50.0)
            k = 1.0 - CURVE_SLOWDOWN_GAIN * curve_mag
            throttle_pwm = int(np.interp(k, [0.0, 1.0], [stop_pwm, fwd_pwm]))

        spwm = self.motors.set_pwm_raw(self.motors.channel_steer, steer_pwm)
        tpwm = self.motors.set_pwm_raw(self.motors.channel_throttle, throttle_pwm)
        return spwm, tpwm

    # ------------- Recording -------------
    def _start_recording(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(DATA_ROOT, f"lfd_{ts}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta_path = os.path.join(self.data_dir, "meta.csv")
        with open(self.meta_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id","timestamp","steer_pwm","throttle_pwm"])

    def save_sample(self, frame_rgb, steer_pwm, throttle_pwm, tnow):
        if self.data_dir is None:
            return
        img_name = f"img_{self.frame_id:06d}.npy"
        np.save(os.path.join(self.data_dir, img_name), frame_rgb)
        with open(self.meta_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.frame_id, f"{tnow:.6f}", f"{steer_pwm:d}", f"{throttle_pwm:d}"])
        self.frame_id += 1

    # ------------- Dashboard -------------
    def draw_status(self, force=False):
        # Basic layout
        try:
            self.stdscr.erase()
            rows, cols = self.stdscr.getmaxyx()
            # Header
            self.stdscr.addstr(0, 0, "Meta Dot PiCar - Discrete Line Follower [curses dashboard]".ljust(cols-1))

            # Mode/State
            self.stdscr.addstr(2, 0, f"Mode         : {'AUTO' if self.auto_mode else 'MANUAL'}")
            self.stdscr.addstr(3, 0, f"Recording    : {'ON' if self.recording else 'OFF'}")
            self.stdscr.addstr(4, 0, f"Decision     : {self.last_decision}")
            self.stdscr.addstr(5, 0, f"Error (norm) : {self.last_center_err:+.3f}")
            self.stdscr.addstr(6, 0, f"Curvature    : {self.last_curvature:+.4f}")

            # PWM outputs (manual shown in MANUAL, last applied in AUTO)
            if self.auto_mode:
                # We cannot easily fetch last set values back; show targets instead
                self.stdscr.addstr(8, 0, f"AUTO Target PWMs -> steer: {self._target_steer_pwm_str():>4} | throttle: {self._target_throttle_pwm_str():>4}")
            else:
                self.stdscr.addstr(8, 0, f"MANUAL PWMs     -> steer: {self.manual_steer_pwm:>4} | throttle: {self.manual_throttle_pwm:>4}")

            # ROI/Thresholds
            self.stdscr.addstr(10, 0, f"ROI Fraction : {ROI_FRACTION[0]:.2f}..{ROI_FRACTION[1]:.2f}   BIN_THRESH: {BIN_THRESH:.2f}   LINE_IS_DARK: {LINE_IS_DARK}")
            self.stdscr.addstr(11, 0, f"Deadband ON/OFF: {DEAD_BAND_ON:.2f}/{DEAD_BAND_OFF:.2f}   No-line timeout: {NO_LINE_TIMEOUT:.1f}s")

            # Perf (rough estimates)
            now = time.time()
            ctrl_fps = self.ctrl_count / max(1e-3, (now - self.ctrl_t0))
            self.ctrl_count = 0
            self.ctrl_t0 = now

            self.stdscr.addstr(13, 0, f"Control loop target: {CONTROL_LOOP_HZ} Hz | est: {ctrl_fps:5.1f} Hz")
            self.stdscr.addstr(14, 0, f"Camera target: {CAMERA_LOOP_HZ} Hz   | (Picamera2 internal)")

            # Controls help
            self.stdscr.addstr(16, 0, "Keys: h manual | a auto | r record | space stop | c center | p status | arrows/WASD manual | q quit")

            # Message line
            if self.msg:
                self.stdscr.addstr(18, 0, f"Msg: {self.msg}".ljust(cols-1))

            self.stdscr.refresh()
        except curses.error:
            # Ignore drawing errors if terminal too small
            pass

    def _target_steer_pwm_str(self):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        center = int(round((left + right) / 2))
        if self.last_decision == "LEFT":
            return f"{left}"
        elif self.last_decision == "RIGHT":
            return f"{right}"
        elif self.last_decision == "NO_LINE":
            return f"{center}"
        else:
            return f"{center}"

    def _target_throttle_pwm_str(self):
        fwd = self.cfg["THROTTLE_FORWARD_PWM"]
        stop = self.cfg["THROTTLE_STOPPED_PWM"]
        if self.last_decision == "NO_LINE":
            return f"{stop}"
        if not USE_CURVE_SLOWDOWN:
            return f"{fwd}"
        curve_mag = min(1.0, abs(self.last_curvature) * 50.0)
        k = 1.0 - CURVE_SLOWDOWN_GAIN * curve_mag
        tpwm = int(np.interp(k, [0.0, 1.0], [stop, fwd]))
        return f"{tpwm}"

# ------------------------------
# Entry (curses wrapper)
# ------------------------------
def main(stdscr):
    lf = LineFollowerDiscrete(PWM_STEERING_THROTTLE, stdscr)
    try:
        lf.start()
    except KeyboardInterrupt:
        lf.msg = "Ctrl-C received; stopping..."
        lf.draw_status()
    finally:
        lf.stop()
        # Ensure screen restored before exiting
        time.sleep(0.05)

if __name__ == "__main__":
    curses.wrapper(main)
