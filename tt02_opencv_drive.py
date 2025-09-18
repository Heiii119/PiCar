#!/usr/bin/env python3
# Picamera2 preview + keyboard control for TT02 RC car via Adafruit PCA9685 (CircuitPython).
# Smooth steering/throttle via ramping in a fixed-rate actuator loop, with robust cleanup.
# - Drive with arrow keys or WASD.
# - Space: immediate throttle stop
# - c: center steering
# - q or Ctrl-C: quit safely

import sys
import time
import termios
import tty
import select
import signal
import threading

from picamera2 import Picamera2, Preview

# ==========================
# User-configurable settings
# ==========================
PCA9685_I2C_ADDR  = 0x40
PCA9685_FREQUENCY = 50     # 50 Hz is standard for RC servo/ESC pulses

# Channels
THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# RC pulse ranges (microseconds). Tune to your hardware.
# Servo (steering)
STEER_CENTER_US = 1500     # adjust to your mechanical center
STEER_RANGE_US  = 300      # +/- from center (e.g., 300 => 1200..1800)
STEER_MIN_US    = STEER_CENTER_US - STEER_RANGE_US
STEER_MAX_US    = STEER_CENTER_US + STEER_RANGE_US

# ESC (throttle)
THROTTLE_NEUTRAL_US = 1500
THROTTLE_FWD_MAX_US = 2000
THROTTLE_REV_MAX_US = 1000

# Ramping and update rate (smoothness)
ACTUATOR_HZ   = 200        # actuator update loop frequency
STEER_STEP_US = 10         # max change per cycle (µs)
THR_STEP_US   = 6          # smaller for traction
KEY_POLL_SLEEP = 0.01      # main loop sleep to reduce CPU

# Camera
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ==========================
# PCA9685 (CircuitPython) init
# ==========================
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
except Exception as e:
    sys.exit(
        "Missing CircuitPython PCA9685 deps. In your venv run:\n"
        "  pip install adafruit-circuitpython-pca9685 adafruit-blinka\n"
        f"Import error: {e}"
    )

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def us_to_duty_cycle(us, freq=PCA9685_FREQUENCY):
    period_us = 1_000_000.0 / freq
    return int(clamp(round((us / period_us) * 65535.0), 0, 65535))

class PWM:
    def __init__(self, address, freq_hz):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.dev = PCA9685(i2c, address=address)
        self.dev.frequency = freq_hz

    def set_us(self, channel, pulse_us):
        self.dev.channels[channel].duty_cycle = us_to_duty_cycle(pulse_us, self.dev.frequency)

# ==========================
# Keyboard (non-blocking)
# ==========================
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)   # immediate key reads
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll_all(self):
        chars = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # possible arrow key escape
                time.sleep(0.001)
                seq = ch
                while select.select([sys.stdin], [], [], 0)[0]:
                    seq += sys.stdin.read(1)
                chars.append(seq)
            else:
                chars.append(ch)
        return chars

def decode_key(ch):
    if ch in ('w','W','\x1b[A'):
        return 'UP'
    if ch in ('s','S','\x1b[B'):
        return 'DOWN'
    if ch in ('a','A','\x1b[D'):
        return 'LEFT'
    if ch in ('d','D','\x1b[C'):
        return 'RIGHT'
    if ch == ' ':
        return 'SPACE'
    if ch in ('c','C'):
        return 'CENTER'
    if ch in ('q','Q','\x03'):
        return 'QUIT'
    return None

# ==========================
# Drive helpers and state
# ==========================
def print_controls():
    print("# Keyboard control for TT02 (PCA9685) with Picamera2 preview")
    print("# - Up/W:    throttle forward")
    print("# - Down/S:  throttle reverse")
    print("# - Left/A:  steer left")
    print("# - Right/D: steer right")
    print("# - Space:   immediate throttle stop")
    print("# - c:       center steering")
    print("# - q:       quit (safe stop + center)")
    print()

class DriveState:
    def __init__(self):
        # Targets (desired) in microseconds
        self.target_steer_us    = STEER_CENTER_US
        self.target_throttle_us = THROTTLE_NEUTRAL_US
        # Actual outputs (ramped) in microseconds
        self.out_steer_us       = STEER_CENTER_US
        self.out_throttle_us    = THROTTLE_NEUTRAL_US
        # Threading
        self.lock = threading.Lock()
        self.alive = True
        # Resources for cleanup
        self._cleanup_done = False
        self.picam2 = None
        self.pwm = None
        self.actuator_thread = None

    def set_steer_norm(self, norm):
        us = STEER_CENTER_US + clamp(norm, -1.0, 1.0) * (STEER_MAX_US - STEER_CENTER_US)
        with self.lock:
            self.target_steer_us = clamp(us, STEER_MIN_US, STEER_MAX_US)

    def set_throttle_norm(self, norm):
        norm = clamp(norm, -1.0, 1.0)
        if norm >= 0:
            us = THROTTLE_NEUTRAL_US + norm * (THROTTLE_FWD_MAX_US - THROTTLE_NEUTRAL_US)
        else:
            us = THROTTLE_NEUTRAL_US + norm * (THROTTLE_NEUTRAL_US - THROTTLE_REV_MAX_US)
        with self.lock:
            self.target_throttle_us = clamp(us, THROTTLE_REV_MAX_US, THROTTLE_FWD_MAX_US)

    def stop_all(self):
        with self.lock:
            self.target_throttle_us = THROTTLE_NEUTRAL_US
            self.target_steer_us = STEER_CENTER_US

# ==========================
# Actuator thread (smooth ramping)
# ==========================
def actuator_loop(state: DriveState):
    pwm = state.pwm
    dt = 1.0 / ACTUATOR_HZ
    while state.alive:
        with state.lock:
            # Ramp steering
            if state.target_steer_us > state.out_steer_us:
                state.out_steer_us = min(state.out_steer_us + STEER_STEP_US, state.target_steer_us)
            else:
                state.out_steer_us = max(state.out_steer_us - STEER_STEP_US, state.target_steer_us)
            # Ramp throttle
            if state.target_throttle_us > state.out_throttle_us:
                state.out_throttle_us = min(state.out_throttle_us + THR_STEP_US, state.target_throttle_us)
            else:
                state.out_throttle_us = max(state.out_throttle_us - THR_STEP_US, state.target_throttle_us)
            steer_out = state.out_steer_us
            thr_out   = state.out_throttle_us
        # Write to hardware outside the lock
        try:
            pwm.set_us(STEERING_CHANNEL, steer_out)
            pwm.set_us(THROTTLE_CHANNEL, thr_out)
        except Exception:
            # On transient I2C error, continue trying
            pass
        time.sleep(dt)

# ==========================
# Cleanup helpers (idempotent)
# ==========================
def safe_neutral(state: DriveState):
    # Send neutral commands a few times to ensure ESC/servo receive them
    if not state.pwm:
        return
    for _ in range(3):
        try:
            state.pwm.set_us(THROTTLE_CHANNEL, THROTTLE_NEUTRAL_US)
            state.pwm.set_us(STEERING_CHANNEL, STEER_CENTER_US)
        except Exception:
            pass
        time.sleep(0.05)

def shutdown_actuator(state: DriveState, join_timeout=0.5):
    state.alive = False
    t = state.actuator_thread
    if t and t.is_alive():
        t.join(timeout=join_timeout)

def stop_camera(state: DriveState):
    cam = state.picam2
    if not cam:
        return
    try:
        cam.stop()
    except Exception:
        pass
    try:
        cam.stop_preview()
    except Exception:
        pass

def on_exit(state: DriveState):
    # Guard against double cleanup
    if getattr(state, "_cleanup_done", False):
        return
    state._cleanup_done = True
    try:
        state.stop_all()
        safe_neutral(state)
    finally:
        try:
            shutdown_actuator(state)
        except Exception:
            pass
        try:
            stop_camera(state)
        except Exception:
            pass
        print("Exited.")

# ==========================
# Main
# ==========================
def main():
    state = DriveState()

    # Init PWM and neutral
    pwm = PWM(PCA9685_I2C_ADDR, PCA9685_FREQUENCY)
    state.pwm = pwm
    safe_neutral(state)

    # Init Picamera2 and start preview
    picam2 = Picamera2()
    state.picam2 = picam2
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)

    using_qt = True
    try:
        picam2.start_preview(Preview.QT)
    except Exception:
        picam2.start_preview(Preview.DRM)
        using_qt = False

    picam2.start()

    # Start actuator thread
    state.actuator_thread = threading.Thread(target=actuator_loop, args=(state,), daemon=True)
    state.actuator_thread.start()

    print_controls()
    print(f"Camera preview started ({'QT' if using_qt else 'DRM'} mode).")
    print("Driving active. Press 'q' to quit.")

    def sig_exit(signum, frame):
        print("\nExiting safely (signal).")
        on_exit(state)
        sys.exit(0)

    # Handle Ctrl-C and termination cleanly
    signal.signal(signal.SIGINT, sig_exit)
    signal.signal(signal.SIGTERM, sig_exit)

    # Main key loop: set targets based on keys; actuator thread handles smoothing
    try:
        with KB() as kb:
            while True:
                pressed = kb.poll_all()
                up = down = left = right = stop = center = quit_flag = False
                for raw in pressed:
                    k = decode_key(raw)
                    if k == 'UP': up = True
                    elif k == 'DOWN': down = True
                    elif k == 'LEFT': left = True
                    elif k == 'RIGHT': right = True
                    elif k == 'SPACE': stop = True
                    elif k == 'CENTER': center = True
                    elif k == 'QUIT': quit_flag = True

                if quit_flag:
                    break

                # Targets from keys
                # Throttle
                if stop:
                    state.set_throttle_norm(0.0)
                elif up and not down:
                    state.set_throttle_norm(1.0)
                elif down and not up:
                    state.set_throttle_norm(-1.0)
                else:
                    state.set_throttle_norm(0.0)

                # Steering
                if center:
                    state.set_steer_norm(0.0)
                elif left and not right:
                    state.set_steer_norm(-1.0)
                elif right and not left:
                    state.set_steer_norm(1.0)
                else:
                    state.set_steer_norm(0.0)

                time.sleep(KEY_POLL_SLEEP)
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Ensure python3-picamera2 is available and I2C is enabled.")
    finally:
        on_exit(state)

if __name__ == "__main__":
    main()
