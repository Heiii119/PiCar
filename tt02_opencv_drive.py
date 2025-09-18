#!/usr/bin/env python3
# Picamera2 preview + keyboard control for TT02 RC car via Adafruit PCA9685 (CircuitPython).
# - Uses Picamera2 preview (QT on desktop, DRM on console), no OpenCV windows.
# - Drive with arrow keys or WASD.
# - On key release: throttle returns to STOP, steering to CENTER.
# - Press q or Ctrl-C to exit safely.

import sys
import time
import termios
import tty
import select
import signal

from picamera2 import Picamera2, Preview

# ==========================
# User-configurable settings
# ==========================
PCA9685_I2C_ADDR   = 0x40
# I2C_BUSNUM is not needed with CircuitPython; we use board.SCL/SDA.
PCA9685_FREQUENCY  = 50 # 50 Hz is standard for RC servo/ESC pulses

#Channels
THROTTLE_CHANNEL   = 0
STEERING_CHANNEL   = 1


# RC pulse ranges (microseconds). Tune to your hardware.
# Servo (steering)
STEER_CENTER_US = 1500     # mechanically centered value
STEER_RANGE_US  = 300      # +/- range from center (e.g., 300 => 1200..1800)
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
    # CircuitPython drivers via Blinka
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
    # Convert microseconds to 16-bit duty_cycle for PCA9685 at given freq
    # duty = us / period_us * 65535
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
# Drive helpers
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
        self.target_steer_us   = STEER_CENTER_US
        self.target_throttle_us= THROTTLE_NEUTRAL_US
        # Actual outputs (ramped) in microseconds
        self.out_steer_us      = STEER_CENTER_US
        self.out_throttle_us   = THROTTLE_NEUTRAL_US
        # Threading
        self.lock = threading.Lock()
        self.alive = True

    def set_steer_norm(self, norm):
        # norm in [-1, 1]
        us = STEER_CENTER_US + clamp(norm, -1.0, 1.0) * (STEER_MAX_US - STEER_CENTER_US)
        with self.lock:
            self.target_steer_us = clamp(us, STEER_MIN_US, STEER_MAX_US)

    def set_throttle_norm(self, norm):
        # norm in [-1, 1]; asymmetric range typical for ESC
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
def actuator_loop(pwm, state: DriveState):
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
        pwm.set_us(STEERING_CHANNEL, steer_out)
        pwm.set_us(THROTTLE_CHANNEL, thr_out)

        time.sleep(dt)
        
# ==========================
# Main
# ==========================
def main():
    # Init PWM and neutral
    pwm = PWM(PCA9685_I2C_ADDR, PCA9685_FREQUENCY)
    state = DriveState()
    pwm.set_us(THROTTLE_CHANNEL, THROTTLE_NEUTRAL_US)
    pwm.set_us(STEERING_CHANNEL, STEER_CENTER_US)

    # Start actuator thread
    worker = threading.Thread(target=actuator_loop, args=(pwm, state), daemon=True)
    worker.start()

    # Init Picamera2 and start preview
    picam2 = Picamera2()
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
    print_controls()
    print(f"Camera preview started ({'QT' if using_qt else 'DRM'} mode).")
    print("Driving active. Press 'q' to quit.")

    def on_exit():
        state.stop_all()
        time.sleep(0.2)
        state.alive = False
        try:
            pwm.set_us(THROTTLE_CHANNEL, THROTTLE_NEUTRAL_US)
            pwm.set_us(STEERING_CHANNEL, STEER_CENTER_US)
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.stop_preview()
        except Exception:
            pass
        print("Exited.")

    def on_sigint(signum, frame):
        print("\nExiting safely.")
        on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Main key loop: set targets based on keys; actuator thread handles smoothing
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

            # Targets from keys (hold = command; release = neutral/center)
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

    on_exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure python3-picamera2 is available (system site-packages) and I2C is enabled.")
        sys.exit(1)
