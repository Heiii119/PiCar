#!/usr/bin/env python3
# tt02_keyboard_drive.py
# Control TT02 RC car (PCA9685) with keyboard arrow keys in a terminal.
# Keyboard control for ESC (throttle) and servo (steering) via Adafruit PCA9685.
# Works in a regular terminal (SSH friendly). Requires: Adafruit-PCA9685 pip package.
# Keys: arrows or WASD, Space=stop, C=center, Q=quit.
# - Up:     throttle forward
# - Down:   throttle reverse
# - Left:   steer left
# - Right:  steer right
# - Space:  immediate throttle stop
# - c:      center steering
# - q:      quit (safely stops and centers)
#
# Requires:
# - Either "Adafruit-PCA9685" (legacy) or "adafruit-circuitpython-pca9685" (CircuitPython) library
# - I2C enabled on the Raspberry Pi and PCA9685 wired correctly
#
# Wiring reminder:
# - PCA9685 VCC -> Pi 3.3V, GND -> Pi GND, SDA/SCL -> Pi SDA/SCL
# - External servo/ESC power to PCA9685 V+ rail (e.g., 5–6V, common ground with Pi)



import sys
import time
import termios
import tty
import select
import signal

# ==========================
# User-configurable settings
# ==========================
PCA9685_I2C_ADDR   = 0x40
I2C_BUSNUM         = 1      # Force Raspberry Pi primary I2C bus

PCA9685_FREQUENCY  = 60     # 50–60 Hz typical for servo/ESC

# Channels
THROTTLE_CHANNEL   = 0
STEERING_CHANNEL   = 1

# PWM values (0..4095). Tune for your hardware!
THROTTLE_FORWARD_PWM  = 420
THROTTLE_STOPPED_PWM  = 370
THROTTLE_REVERSE_PWM  = 220

STEERING_LEFT_PWM     = 510
STEERING_RIGHT_PWM    = 230

# Safety: require stop before switching FWD<->REV
SWITCH_PAUSE_S = 0.06

# ==========================
# PCA9685 init (legacy lib)
# ==========================
try:
    import Adafruit_PCA9685 as LegacyPCA9685
except ImportError:
    sys.exit("Missing Adafruit-PCA9685. Activate your venv and run: pip install Adafruit-PCA9685")

def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

def clamp12(x):
    return max(0, min(4095, int(x)))

class PWM:
    def __init__(self, address, busnum, freq_hz):
        # Force bus selection to avoid auto-detect issues
        self.dev = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
        self.dev.set_pwm_freq(freq_hz)

    def set(self, channel, value_12bit):
        v = clamp12(value_12bit)
        self.dev.set_pwm(channel, 0, v)

# ==========================
# Non-blocking keyboard input
# ==========================
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self, timeout=0.0):
        # Return a list of raw chars read without blocking
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return []
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # possible escape sequence
            time.sleep(0.001)
            while select.select([sys.stdin], [], [], 0)[0]:
                ch += sys.stdin.read(1)
        return [ch]

def decode_key(ch):
    # Arrow sequences commonly: '\x1b[A' up, '\x1b[B' down, '\x1b[D' left, '\x1b[C' right
    if ch in ('w', 'W', '\x1b[A'):
        return 'UP'
    if ch in ('s', 'S', '\x1b[B'):
        return 'DOWN'
    if ch in ('a', 'A', '\x1b[D'):
        return 'LEFT'
    if ch in ('d', 'D', '\x1b[C'):
        return 'RIGHT'
    if ch == ' ':
        return 'SPACE'
    if ch in ('c', 'C'):
        return 'CENTER'
    if ch in ('q', 'Q', '\x03'):  # Ctrl-C as quit
        return 'QUIT'
    return None

# ==========================
# Main
# ==========================
def main():
    pwm = PWM(PCA9685_I2C_ADDR, I2C_BUSNUM, PCA9685_FREQUENCY)

    def neutral_all():
        pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
        pwm.set(STEERING_CHANNEL, steering_center_pwm())

    def on_sigint(signum, frame):
        try:
            neutral_all()
            time.sleep(0.1)
        finally:
            print("\nExiting safely.")
            sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Startup instructions
    print("# Control TT02 RC car (PCA9685) with keyboard arrow keys in a terminal.")
    print("# - Up:     throttle forward")
    print("# - Down:   throttle reverse")
    print("# - Left:   steer left")
    print("# - Right:  steer right")
    print("# - Space:  immediate throttle stop")
    print("# - c:      center steering")
    print("# - q:      quit (safely stops and centers)")
    print()

    neutral_all()
    time.sleep(1.0)  # Allow ESC to arm

    last_throttle = THROTTLE_STOPPED_PWM

    print("TT02 Keyboard Drive (simple)")
    print(f"Throttle: FWD={THROTTLE_FORWARD_PWM} STOP={THROTTLE_STOPPED_PWM} REV={THROTTLE_REVERSE_PWM}")
    print(f"Steering: L={STEERING_LEFT_PWM} C={steering_center_pwm()} R={STEERING_RIGHT_PWM}")

    with KB() as kb:
        while True:
            keys = kb.poll(timeout=0.02)
            for raw in keys:
                key = decode_key(raw)
                if key == 'QUIT':
                    neutral_all()
                    return
                elif key == 'SPACE':
                    pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    last_throttle = THROTTLE_STOPPED_PWM
                    print("Throttle: STOP")
                elif key == 'CENTER':
                    val = steering_center_pwm()
                    pwm.set(STEERING_CHANNEL, val)
                    print("Steering: CENTER")
                elif key == 'UP':
                    if last_throttle == THROTTLE_REVERSE_PWM:
                        pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                        time.sleep(SWITCH_PAUSE_S)
                    pwm.set(THROTTLE_CHANNEL, THROTTLE_FORWARD_PWM)
                    last_throttle = THROTTLE_FORWARD_PWM
                    print("Throttle: FORWARD")
                elif key == 'DOWN':
                    if last_throttle == THROTTLE_FORWARD_PWM:
                        pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                        time.sleep(SWITCH_PAUSE_S)
                    pwm.set(THROTTLE_CHANNEL, THROTTLE_REVERSE_PWM)
                    last_throttle = THROTTLE_REVERSE_PWM
                    print("Throttle: REVERSE")
                elif key == 'LEFT':
                    pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)
                    print("Steering: LEFT")
                elif key == 'RIGHT':
                    pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)
                    print("Steering: RIGHT")

            time.sleep(0.005)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure I2C is enabled and Adafruit-PCA9685 is installed in your venv.")
        sys.exit(1)
