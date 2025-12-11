# main_line_follower.py

import time
import math
import curses
import argparse

import numpy as np
import cv2

from traffic_light_detector import TrafficLightDetector

# ===================== Camera & Motor placeholders =====================

class SimpleCamera:
    """Simple USB camera using OpenCV. Replace with your libcamera version if needed."""
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """Return an RGB frame, or None if failed."""
        ret, frame_bgr = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def release(self):
        self.cap.release()

class SimpleMotors:
    """
    Stub motor/steering class.
    REPLACE set() with your robot's motor control (PWM / KART API, etc.).
    """
    def __init__(self):
        self._last_print = 0.0

    def set(self, steering, throttle):
        # TODO: replace with real hardware commands.
        # steering, throttle are in [-1, 1].

        # For debug: print at most 10 times per second
        now = time.time()
        if now - self._last_print > 0.1:
            print(f"[MOTOR] steering={steering:+.2f}, throttle={throttle:+.2f}")
            self._last_print = now

# ===================== Global configuration =====================

# Resized image used for line detection
IMAGE_W = 80
IMAGE_H = 60

# Line detection ROI in the resized image (bottom part)
LINE_Y0 = int(0.5 * IMAGE_H)
LINE_Y1 = IMAGE_H

# Simple binary threshold for line (assuming dark line on light floor)
LINE_BINARY_THRESH = 100

# Basic control gains
STEER_KP = 0.8
STEER_KD = 0.1
BASE_THROTTLE = 0.35  # in [-1, 1]

# ===================== LINE-FOLLOWING ALGORITHM (kept separate) =====================
# NOTE: As requested, the line-following logic is in its own function.
#       We do NOT put any traffic-light logic inside this function.

def line_following_step(gray_small, last_center_err):
    """
    Pure line-following step.
    Inputs:
      - gray_small: grayscale image (IMAGE_H x IMAGE_W), uint8.
      - last_center_err: previous center error (float).

    Outputs:
      - center_err: new normalized center error in [-1, 1] (0 means centered), or None if no line.
      - curvature: (here 0.0, placeholder).
      - steer: steering command in [-1, 1].
      - throttle: base throttle in [-1, 1] (no traffic-light override).
      - decision_str: human-readable description of decision.
      - new_last_center_err: center_err if line found; otherwise last_center_err.
    """
    # ---- Line detection ----
    roi = gray_small[LINE_Y0:LINE_Y1, :]
    _, bw = cv2.threshold(roi, LINE_BINARY_THRESH, 255, cv2.THRESH_BINARY_INV)

    ys, xs = np.where(bw == 255)
    if xs.size == 0:
        # No line found
        center_err = None
        curvature = 0.0
        steer = 0.0
        throttle = 0.0
        decision_str = "NO LINE"
        return center_err, curvature, steer, throttle, decision_str, last_center_err

    x_mean = xs.mean()
    # Normalize to [-1, 1]: left=-1, right=+1
    center = (IMAGE_W - 1) / 2.0
    center_err = (x_mean - center) / center
    curvature = 0.0  # could use history of errors; kept simple

    # ---- Control (pure line-following) ----
    d_err = center_err - last_center_err
    steer = -(STEER_KP * center_err + STEER_KD * d_err)  # negative: line to left -> steer left

    # Clip steering to [-1, 1]
    steer = max(-1.0, min(1.0, steer))

    # Base throttle (NO traffic-light logic here)
    throttle = BASE_THROTTLE

    decision_str = f"steer={steer:+.2f}, thr={throttle:+.2f}"

    return center_err, curvature, steer, throttle, decision_str, center_err

# ===================== Main line follower class (integrates modules) =====================

class LineFollowerWithTL:
    def __init__(self, stdscr, args):
        self.stdscr = stdscr
        self.args = args

        self.camera = SimpleCamera()
        self.motors = SimpleMotors()

        # Line-following state
        self.last_center_err = 0.0
        self.last_curvature = 0.0
        self.last_decision = "NONE"

        # Traffic light detector module (separate)
        self.tl_detector = TrafficLightDetector()
        self.traffic_state = "NONE"  # "RED", "GREEN", "NONE"

        # Misc
        self.running = True
        self.msg = ""

        # For drawing FPS
        self.last_loop_time = time.time()
        self.fps = 0.0

    # ---------------- Lifecycle ----------------

    def start(self):
        self.control_loop()

    def stop(self):
        self.running = False
        self.camera.release()
        self.motors.set(0.0, 0.0)

    # ---------------- UI drawing ----------------

    def draw_status(self, tnow):
        # Compute FPS (simple EWMA)
        dt = tnow - self.last_loop_time
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps > 0 else inst_fps
        self.last_loop_time = tnow

        self.stdscr.erase()
        self.stdscr.addstr(0, 0, "Line Follower with Traffic Light (separate modules)")
        self.stdscr.addstr(1, 0, f"FPS           : {self.fps:5.1f}")
        self.stdscr.addstr(2, 0, f"Traffic state : {self.traffic_state}")
        self.stdscr.addstr(3, 0, f"Decision      : {self.last_decision}")
        self.stdscr.addstr(4, 0, f"Error (norm)  : {self.last_center_err:+.3f}")
        self.stdscr.addstr(5, 0, f"Curvature     : {self.last_curvature:+.3f}")
        self.stdscr.addstr(7, 0, "Keys: q=quit")
        self.stdscr.addstr(9, 0, f"Msg           : {self.msg}")

        self.stdscr.refresh()

    # ---------------- Main control loop ----------------

    def control_loop(self):
        self.stdscr.nodelay(True)  # non-blocking getch
        curses.curs_set(0)

        while self.running:
            tnow = time.time()

            # Keyboard
            ch = self.stdscr.getch()
            if ch == ord('q'):
                self.running = False
                break

            # Get camera frame (RGB numpy array)
            frame = self.camera.get_frame()
            if frame is None:
                self.msg = "No camera frame"
                self.motors.set(0.0, 0.0)
                self.draw_status(tnow)
                time.sleep(0.05)
                continue

            # ---------- MODULE 1: Traffic-light detection ----------
            # This uses the separate TrafficLightDetector module with RGB numpy frame.
            self.traffic_state = self.tl_detector.update(frame, tnow)

            # ---------- MODULE 2: Line-following (kept separate) ----------
            # Downscale for line detection
            h, w, _ = frame.shape
            scale_y = h / IMAGE_H
            scale_x = w / IMAGE_W
            small = frame[::int(max(1, round(scale_y))),
                          ::int(max(1, round(scale_x))), :]

            # Ensure exact size
            small = cv2.resize(small, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

            (
                center_err,
                curvature,
                steer,
                base_throttle,
                decision_str,
                self.last_center_err,
            ) = line_following_step(gray_small, self.last_center_err)

            if center_err is None:
                # No line detected
                self.last_decision = "NO LINE"
                self.msg = "No line detected"
                # Optionally: stop motors
                throttle_cmd = 0.0
                steer_cmd = 0.0
            else:
                # Apply traffic-light override OUTSIDE the line-following function
                steer_cmd = steer
                throttle_cmd = base_throttle

                if self.traffic_state == "RED":
                    throttle_cmd = 0.0
                    self.msg = "RED light -> STOP"
                elif self.traffic_state == "GREEN":
                    self.msg = "GREEN light -> GO"
                else:
                    self.msg = "No TL / NONE"

                self.last_curvature = curvature
                self.last_decision = decision_str

            # Send commands to motors
            self.motors.set(steer_cmd, throttle_cmd)

            # Draw UI
            self.draw_status(tnow)
            time.sleep(0.02)  # ~50 Hz

        self.stop()

# ===================== Main entry =====================

def main(stdscr, args):
    lf = LineFollowerWithTL(stdscr, args)
    lf.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # You can add arguments here if needed (e.g. camera index)
    args = parser.parse_args()

    curses.wrapper(main, args)
