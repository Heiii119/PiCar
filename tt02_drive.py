#!/usr/bin/env python3
# Camera preview + keyboard control for TT02 RC car via Adafruit PCA9685.
# - Opens a camera preview window, waits 10 seconds
# - Prints controls and asks user to confirm start (y/n)
# - Drive with arrow keys or WASD
# - When you release keys: throttle returns to STOP, steering returns to CENTER
# Requirements:
#   - Adafruit-PCA9685 (pip install Adafruit-PCA9685)
#   - OpenCV for camera preview (pip install opencv-python)
#   - I2C enabled; PCA9685 on I2C bus 1 at address 0x40

import sys
import time
import termios
import tty
import select
import signal
import cv2

# ==========================
# User-configurable settings
# ==========================
PCA9685_I2C_ADDR   = 0x40
I2C_BUSNUM         = 1       # Force Raspberry Pi primary I2C bus

PCA9685_FREQUENCY  = 60      # 50–60 Hz typical for servo/ESC

# PCA9685 channels
THROTTLE_CHANNEL   = 0
STEERING_CHANNEL   = 1

# PWM values (0..4095). Provided by user
THROTTLE_FORWARD_PWM  = 400
THROTTLE_STOPPED_PWM  = 370
THROTTLE_REVERSE_PWM  = 220

STEERING_LEFT_PWM     = 470
STEERING_RIGHT_PWM    = 270

# Safety delay passing through STOP when changing direction
SWITCH_PAUSE_S = 0.06

# Camera settings (adjust index and resolution if needed)
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

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
# Non-blocking keyboard input (raw mode)
# ==========================
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll_all(self):
        # Return list of all pending chars (non-blocking)
        chars = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # possible escape sequence (arrow keys)
                time.sleep(0.001)
                seq = ch
                while select.select([sys.stdin], [], [], 0)[0]:
                    seq += sys.stdin.read(1)
                chars.append(seq)
            else:
                chars.append(ch)
        return chars

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
# Drive control helpers
# ==========================
def neutral_all(pwm):
    pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
    pwm.set(STEERING_CHANNEL, steering_center_pwm())

def print_controls():
    print("# Control TT02 RC car (PCA9685) with keyboard arrow keys in a terminal.")
    print("# - Up:     throttle forward")
    print("# - Down:   throttle reverse")
    print("# - Left:   steer left")
    print("# - Right:  steer right")
    print("# - Space:  immediate throttle stop")
    print("# - c:      center steering")
    print("# - q:      quit (safely stops and centers)")
    print()

# ==========================
# Main
# ==========================
def main():
    # Init PWM driver
    pwm = PWM(PCA9685_I2C_ADDR, I2C_BUSNUM, PCA9685_FREQUENCY)
    neutral_all(pwm)

    # Init camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Warning: Could not open camera. Continuing without preview...")
        cap = None
    else:
        # Try to set resolution (best effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Create preview window
    cv2.namedWindow("TT02 Camera", cv2.WINDOW_NORMAL)

    # Startup instructions
    print_controls()
    print("Starting camera preview and waiting 10 seconds to ensure the window starts...")
    start = time.time()

    # Show frames for ~10 seconds before asking to start
    while time.time() - start < 10.0:
        if cap:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("TT02 Camera", frame)
        # Keep UI responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Early quit from preview
            cleanup(cap, pwm)
            return

    # Ask for readiness
    ready = input("Are you ready to start driving? (y/n): ").strip().lower()
    if ready != 'y':
        print("Not starting. Exiting.")
        cleanup(cap, pwm)
        return

    print_controls()
    print("Driving active. Press 'q' to quit.")

    last_throttle = THROTTLE_STOPPED_PWM

    # SIGINT safe exit
    def on_sigint(signum, frame):
        try:
            neutral_all(pwm)
            time.sleep(0.1)
        finally:
            print("\nExiting safely.")
            cleanup(cap, pwm)
            sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    # Driving loop with key release behavior
    with KB() as kb:
        while True:
            # Update preview
            if cap:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("TT02 Camera", frame)
            # Allow OpenCV window events to process
            key = cv2.waitKey(1)  # don't use for drive, just to keep window responsive

            # Read all pending keystrokes from terminal
            pressed = kb.poll_all()

            # State flags for this frame
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
                cleanup(cap, pwm)
                return

            # Apply throttle logic
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
                # No throttle key pressed this cycle -> return to STOP
                pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                last_throttle = THROTTLE_STOPPED_PWM

            # Apply steering logic
            if center:
                pwm.set(STEERING_CHANNEL, steering_center_pwm())
            elif left and not right:
                pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)
            elif right and not left:
                pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)
            else:
                # No steering key pressed this cycle -> return to CENTER
                pwm.set(STEERING_CHANNEL, steering_center_pwm())

            # Small loop delay
            time.sleep(0.01)

def cleanup(cap, pwm):
    try:
        neutral_all(pwm)
    except Exception:
        pass
    if cap:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure I2C is enabled, Adafruit-PCA9685 and opencv-python are installed, and you have camera access.")
        sys.exit(1)
