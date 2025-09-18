#!/usr/bin/env python3
import os
import sys
import time
import threading
import signal
import termios
import tty
import select
from datetime import datetime
from pathlib import Path

import cv2

# ============== PCA9685 config ==============
PCA9685_I2C_ADDR   = 0x40
PCA9685_I2C_BUSNUM = None  # None = default bus (usually 1 on Raspberry Pi)
PCA9685_FREQUENCY  = 50    # Hz

# Camera settings
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS          = 30

# Helper: microseconds to 12-bit ticks at given frequency
def us_to_12bit(us, freq=PCA9685_FREQUENCY):
    period_us = 1_000_000.0 / freq  # e.g., 20_000 at 50 Hz
    ticks = round((us / period_us) * 4095.0)
    return max(0, min(4095, int(ticks)))

# THROTTLE endpoints (tune if needed)
THROTTLE_CHANNEL       = 0
THROTTLE_REVERSE_PWM   = us_to_12bit(1200)  # ≈ 246
THROTTLE_STOPPED_PWM   = us_to_12bit(1500)  # ≈ 304
THROTTLE_FORWARD_PWM   = us_to_12bit(1800)  # ≈ 369

# STEERING endpoints (tune for your servo; avoid binding)
STEERING_CHANNEL     = 1
STEERING_RIGHT_PWM   = us_to_12bit(1200)    # ≈ 246
STEERING_LEFT_PWM    = us_to_12bit(1800)    # ≈ 369

STEERING_CENTER_PWM  = (STEERING_LEFT_PWM + STEERING_RIGHT_PWM) // 2

SWITCH_PAUSE_S = 0.08  # neutral pause when switching FWD<->REV

# ============== PCA9685 driver ==============
try:
    import Adafruit_PCA9685 as LegacyPCA9685
except ImportError:
    print("Missing Adafruit-PCA9685. Install: pip3 install Adafruit-PCA9685")
    sys.exit(1)

def clamp12(x): return max(0, min(4095, int(x)))

class PWM:
    def __init__(self, address, busnum, freq_hz):
        # If busnum is None, Adafruit lib defaults to bus 1 on Pi
        self.dev = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
        self.dev.set_pwm_freq(freq_hz)

    def set(self, channel, value_12bit):
        v = clamp12(value_12bit)
        self.dev.set_pwm(channel, 0, v)

# ============== Car control ==============
class Car:
    def __init__(self):
        self.pwm = PWM(PCA9685_I2C_ADDR, PCA9685_I2C_BUSNUM, PCA9685_FREQUENCY)
        self.lock = threading.Lock()
        self.last_throttle = THROTTLE_STOPPED_PWM
        self.neutral_all()

    def neutral_all(self):
        with self.lock:
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.pwm.set(STEERING_CHANNEL, STEERING_CENTER_PWM)
            self.last_throttle = THROTTLE_STOPPED_PWM

    def throttle_forward(self):
        with self.lock:
            if self.last_throttle == THROTTLE_REVERSE_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_FORWARD_PWM)
            self.last_throttle = THROTTLE_FORWARD_PWM

    def throttle_reverse(self):
        with self.lock:
            if self.last_throttle == THROTTLE_FORWARD_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_REVERSE_PWM)
            self.last_throttle = THROTTLE_REVERSE_PWM

    def throttle_stop(self):
        with self.lock:
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.last_throttle = THROTTLE_STOPPED_PWM

    def steer_left(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)

    def steer_right(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)

    def steer_center(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, STEERING_CENTER_PWM)

# ============== Keyboard helper ==============
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll_keys(self):
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # collect full escape sequence for arrows
                time.sleep(0.001)
                seq = ch
                while select.select([sys.stdin], [], [], 0)[0]:
                    seq += sys.stdin.read(1)
                keys.append(seq)
            else:
                keys.append(ch)
        return keys

def decode_key(k):
    # Arrow keys: ESC [ A/B/C/D
    if k in ('\x1b[A','w','W'): return 'UP'
    if k in ('\x1b[B','s','S'): return 'DOWN'
    if k in ('\x1b[D','a','A'): return 'LEFT'
    if k in ('\x1b[C','d','D'): return 'RIGHT'
    if k in ('y','Y'): return 'REC_START'
    if k in ('n','N'): return 'REC_STOP'
    if k in ('q','Q','\x03'): return 'QUIT'
    if k == ' ': return 'SPACE'
    if k in ('c','C'): return 'CENTER'
    return None

# ============== Video recorder ==============
class VideoRecorder:
    def __init__(self, index=0, w=FRAME_WIDTH, h=FRAME_HEIGHT, fps=FPS, out_dir="~/PiCar/data"):
        self.index = index
        self.w = w
        self.h = h
        self.fps = fps
        self.out_dir = Path(os.path.expanduser(out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.cap = None
        self.writer = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def _open_cam(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            print("Warning: Could not open camera.")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _new_writer(self):
        ts = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        path = self.out_dir / f"{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.w, self.h))
        if not writer.isOpened():
            print("Error: Could not open VideoWriter.")
            return None
        print(f"Recording to {path}")
        return writer

    def start(self):
        with self.lock:
            if self.running:
                print("Recording already running.")
                return
            self.cap = self._open_cam()
            if self.cap is None:
                return
            self.writer = self._new_writer()
            if self.writer is None:
                self.cap.release()
                self.cap = None
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        prev = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Ensure correct size
            if frame.shape[1] != self.w or frame.shape[0] != self.h:
                frame = cv2.resize(frame, (self.w, self.h))
            self.writer.write(frame)

            # simple pacing if needed
            now = time.time()
            dt = now - prev
            target = 1.0 / max(1, self.fps)
            if dt < target:
                time.sleep(target - dt)
            prev = now

    def stop(self):
        with self.lock:
            if not self.running:
                print("Recording not running.")
                return
            self.running = False
            # give loop a moment to exit
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.writer:
                self.writer.release()
                self.writer = None
            if self.cap:
                self.cap.release()
                self.cap = None
        print("Recording stopped.")

# ============== Main control loop ==============
def print_help():
    print("Controls:")
    print("  Arrow keys or WASD: steer/throttle while held")
    print("  Space: throttle STOP")
    print("  C: center steering")
    print("  Y: start recording   N: stop recording")
    print("  Q or Ctrl-C: quit (safely stop and center)")

def main():
    car = Car()
    recorder = VideoRecorder()
    print_help()

    def cleanup_and_exit():
        try:
            car.neutral_all()
            recorder.stop()
        except Exception:
            pass

    def on_sigint(signum, frame):
        cleanup_and_exit()
        print("\nExiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Momentary control: default to neutral every cycle unless a key says otherwise
    with KB() as kb:
        while True:
            # Defaults each loop
            want_throttle = 'STOP'
            want_steer    = 'CENTER'

            # Collect keys available right now
            for raw in kb.poll_keys():
                k = decode_key(raw)
                if k == 'UP':
                    want_throttle = 'FWD'
                elif k == 'DOWN':
                    want_throttle = 'REV'
                elif k == 'LEFT':
                    want_steer = 'LEFT'
                elif k == 'RIGHT':
                    want_steer = 'RIGHT'
                elif k == 'SPACE':
                    want_throttle = 'STOP'
                elif k == 'CENTER':
                    want_steer = 'CENTER'
                elif k == 'REC_START':
                    recorder.start()
                elif k == 'REC_STOP':
                    recorder.stop()
                elif k == 'QUIT':
                    cleanup_and_exit()
                    return

            # Apply desired actions
            if want_throttle == 'FWD':
                car.throttle_forward()
            elif want_throttle == 'REV':
                car.throttle_reverse()
            else:
                car.throttle_stop()

            if want_steer == 'LEFT':
                car.steer_left()
            elif want_steer == 'RIGHT':
                car.steer_right()
            else:
                car.steer_center()

            time.sleep(0.05)  # ~20 Hz control loop

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Tips:")
        print("- Ensure I2C is enabled on the Pi")
        print("- Install deps: sudo apt-get install -y python3-opencv; pip3 install Adafruit-PCA9685")
        print("- Check wiring and PCA9685 address")
        sys.exit(1)
