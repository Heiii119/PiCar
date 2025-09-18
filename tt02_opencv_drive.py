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
PCA9685_FREQUENCY  = 60

THROTTLE_CHANNEL   = 0
STEERING_CHANNEL   = 1

# Tune these to your ESC/servo (12-bit scale like original: 0..4095)
THROTTLE_FORWARD_PWM  = 420
THROTTLE_STOPPED_PWM  = 370
THROTTLE_REVERSE_PWM  = 270

STEERING_LEFT_PWM     = 470
STEERING_RIGHT_PWM    = 290

SWITCH_PAUSE_S = 0.06

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

def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

def clamp12(x):
    return max(0, min(4095, int(x)))

def to16_from12(v12):
    # Map 0..4095 to 0..65535 for CircuitPython duty_cycle
    v12 = clamp12(v12)
    return int(round(v12 * 65535 / 4095))

class PWM:
    def __init__(self, address, freq_hz):
        # Initialize I2C using default Pi pins
        # Ensure I2C is enabled: sudo raspi-config -> Interface Options -> I2C -> Enable
        i2c = busio.I2C(board.SCL, board.SDA)
        self.dev = PCA9685(i2c, address=address)
        self.dev.frequency = freq_hz

    def set(self, channel, value_12bit):
        v16 = to16_from12(value_12bit)
        self.dev.channels[channel].duty_cycle = v16

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
def neutral_all(pwm):
    pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
    pwm.set(STEERING_CHANNEL, steering_center_pwm())

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

# ==========================
# Main
# ==========================
def main():
    # Init PWM
    pwm = PWM(PCA9685_I2C_ADDR, PCA9685_FREQUENCY)
    neutral_all(pwm)
    last_throttle = THROTTLE_STOPPED_PWM

    # Init Picamera2 and start preview
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)

    # Prefer QT if a desktop session is running; otherwise fallback to DRM
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

    def on_sigint(signum, frame):
        try:
            neutral_all(pwm)
            time.sleep(0.1)
        finally:
            print("\nExiting safely.")
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.stop_preview()
            except Exception:
                pass
            sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Drive loop: key press = action; release = STOP/CENTER
    with KB() as kb:
        while True:
            pressed = kb.poll_all()
            up = down = left = right = stop = center = quit_flag = False

            for raw in pressed:
                k = decode_key(raw)
                if k == 'UP':
                    up = True
                elif k == 'DOWN':
                    down = True
                elif k == 'LEFT':
                    left = True
                elif k == 'RIGHT':
                    right = True
                elif k == 'SPACE':
                    stop = True
                elif k == 'CENTER':
                    center = True
                elif k == 'QUIT':
                    quit_flag = True

            if quit_flag:
                neutral_all(pwm)
                break

            # Throttle logic
            if stop:
                pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                last_throttle = THROTTLE_STOPPED_PWM
            elif up and not down:
                if last_throttle == THROTTLE_REVERSE_PWM:
                    pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    time.sleep(SWITCH_PAUSE_S)
                pwm.set(THROTTLE_CHANNEL, THROTTLE_FORWARD_PWM)
                last_throttle = THROTTLE_FORWARD_PWM
            elif down and not up:
                if last_throttle == THROTTLE_FORWARD_PWM:
                    pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    time.sleep(SWITCH_PAUSE_S)
                pwm.set(THROTTLE_CHANNEL, THROTTLE_REVERSE_PWM)
                last_throttle = THROTTLE_REVERSE_PWM
            else:
                # No throttle key pressed -> STOP
                pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                last_throttle = THROTTLE_STOPPED_PWM

            # Steering logic
            if center:
                pwm.set(STEERING_CHANNEL, steering_center_pwm())
            elif left and not right:
                pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)
            elif right and not left:
                pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)
            else:
                # No steering key pressed -> CENTER
                pwm.set(STEERING_CHANNEL, steering_center_pwm())

            time.sleep(0.01)

    # Cleanup
    try:
        neutral_all(pwm)
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure python3-picamera2 is available (system site-packages) and I2C is enabled.")
        sys.exit(1)
