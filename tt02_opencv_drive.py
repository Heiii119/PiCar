#!/usr/bin/env python3
# Picamera2 preview + keyboard control for TT02 RC car via Adafruit PCA9685 (CircuitPython).
# Uses user-provided 12-bit PWM values (0..4095), PCA9685 at 50 Hz, and shows a live status board.
# Smooth ramping is applied in 12-bit units; direction changes pass through STOP with a safety pause.

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
PCA9685_FREQUENCY = 50   # per your request

# PCA9685 channels
THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# PWM values (0..4095). Provided by user
THROTTLE_FORWARD_PWM  = 400
THROTTLE_STOPPED_PWM  = 370
THROTTLE_REVERSE_PWM  = 220

STEERING_LEFT_PWM     = 470
STEERING_RIGHT_PWM    = 270

# Safety delay passing through STOP when changing direction
SWITCH_PAUSE_S = 0.06

# Camera settings
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Smoothness and telemetry
ACTUATOR_HZ    = 200      # actuator update frequency
STEER_STEP_12  = 2        # 12-bit ticks per cycle for steering ramp
THR_STEP_12    = 2        # 12-bit ticks per cycle for throttle ramp
KEY_POLL_SLEEP = 0.01     # main loop tick
TELEM_HZ       = 10       # status board refresh rate

# ==========================
# CircuitPython PCA9685 init
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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def to16_from12(v12):
    v12 = clamp(int(v12), 0, 4095)
    return int(round(v12 * 65535 / 4095))

def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

# ==========================
# PCA9685 wrapper
# ==========================
class PWMDev:
    def __init__(self, address, freq_hz):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.dev = PCA9685(i2c, address=address)
        self.dev.frequency = freq_hz

    def set12(self, ch, value_12):
        v12 = clamp(int(value_12), 0, 4095)
        v16 = to16_from12(v12)
        self.dev.channels[ch].duty_cycle = v16
        return v12, v16

# ==========================
# Keyboard non-blocking
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
        chars = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                time.sleep(0.001)
                seq = ch
                while select.select([sys.stdin], [], [], 0)[0]:
                    seq += sys.stdin.read(1)
                chars.append(seq)
            else:
                chars.append(ch)
        return chars

def decode_key(ch):
    if ch in ('w','W','\x1b[A'): return 'UP'
    if ch in ('s','S','\x1b[B'): return 'DOWN'
    if ch in ('a','A','\x1b[D'): return 'LEFT'
    if ch in ('d','D','\x1b[C'): return 'RIGHT'
    if ch == ' ':               return 'SPACE'
    if ch in ('c','C'):         return 'CENTER'
    if ch in ('q','Q','\x03'):  return 'QUIT'
    return None

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
# Drive state
# ==========================
class DriveState:
    def __init__(self):
        # Targets and outputs in 12-bit PWM units
        self.target_thr_12 = THROTTLE_STOPPED_PWM
        self.target_str_12 = steering_center_pwm()
        self.out_thr_12    = THROTTLE_STOPPED_PWM
        self.out_str_12    = steering_center_pwm()

        # Last set raw duty values (16-bit)
        self.last_thr_dc16 = to16_from12(self.out_thr_12)
        self.last_str_dc16 = to16_from12(self.out_str_12)

        # For direction change safety handling
        self.last_command = "STOP"   # "FWD", "REV", "STOP"
        self.switch_block_until = 0.0

        # Threading and resources
        self.lock = threading.Lock()
        self.alive = True
        self._cleanup_done = False
        self.pwm = None
        self.picam2 = None
        self.actuator_thread = None

        # Telemetry
        self._last_telem = 0.0

    def set_throttle_cmd(self, cmd):  # "FWD", "REV", "STOP"
        now = time.monotonic()
        with self.lock:
            # Direction change logic: enforce pass through STOP with pause
            if cmd == "FWD":
                if self.last_command == "REV" and now < self.switch_block_until:
                    # still in block; ignore to protect gearbox/ESC
                    return
                if self.last_command == "REV":
                    # initiate block
                    self.target_thr_12 = THROTTLE_STOPPED_PWM
                    self.switch_block_until = now + SWITCH_PAUSE_S
                    self.last_command = "STOP"
                    return
                self.target_thr_12 = THROTTLE_FORWARD_PWM
                self.last_command = "FWD"
            elif cmd == "REV":
                if self.last_command == "FWD" and now < self.switch_block_until:
                    return
                if self.last_command == "FWD":
                    self.target_thr_12 = THROTTLE_STOPPED_PWM
                    self.switch_block_until = now + SWITCH_PAUSE_S
                    self.last_command = "STOP"
                    return
                self.target_thr_12 = THROTTLE_REVERSE_PWM
                self.last_command = "REV"
            else:
                self.target_thr_12 = THROTTLE_STOPPED_PWM
                self.last_command = "STOP"

    def set_steer_cmd(self, cmd):  # "LEFT", "RIGHT", "CENTER"
        with self.lock:
            if cmd == "LEFT":
                self.target_str_12 = STEERING_LEFT_PWM
            elif cmd == "RIGHT":
                self.target_str_12 = STEERING_RIGHT_PWM
            else:
                self.target_str_12 = steering_center_pwm()

# ==========================
# Actuator thread (ramping + status board)
# ==========================
def actuator_loop(state: DriveState):
    pwm = state.pwm
    dt = 1.0 / ACTUATOR_HZ
    telem_dt = 1.0 / TELEM_HZ

    while state.alive:
        # Ramp toward targets
        with state.lock:
            # throttle
            if state.target_thr_12 > state.out_thr_12:
                state.out_thr_12 = min(state.out_thr_12 + THR_STEP_12, state.target_thr_12)
            else:
                state.out_thr_12 = max(state.out_thr_12 - THR_STEP_12, state.target_thr_12)
            # steering
            if state.target_str_12 > state.out_str_12:
                state.out_str_12 = min(state.out_str_12 + STEER_STEP_12, state.target_str_12)
            else:
                state.out_str_12 = max(state.out_str_12 - STEER_STEP_12, state.target_str_12)

            out_thr = state.out_thr_12
            out_str = state.out_str_12

        # Write to hardware
        try:
            _, thr_dc16 = pwm.set12(THROTTLE_CHANNEL, out_thr)
            _, str_dc16 = pwm.set12(STEERING_CHANNEL, out_str)
            state.last_thr_dc16 = thr_dc16
            state.last_str_dc16 = str_dc16
        except Exception:
            pass

        # Status board (single refreshing line)
        now = time.monotonic()
        if now - state._last_telem >= telem_dt:
            state._last_telem = now
            with state.lock:
                status = (
                    f"[PCA9685 {pwm.dev.frequency}Hz | THR ch{THROTTLE_CHANNEL} | STR ch{STEERING_CHANNEL}] "
                    f"THR tgt/out: {state.target_thr_12}/{out_thr} (dc16={state.last_thr_dc16}) "
                    f"cmd={state.last_command} "
                    f"{'(dir-wait)' if now < state.switch_block_until else ''} | "
                    f"STR tgt/out: {state.target_str_12}/{out_str} (dc16={state.last_str_dc16})"
                )
            # Print as a carriage-returned HUD
            print("\r" + status + " " * max(0, 10 - len(status) % 10), end="", flush=True)

        time.sleep(dt)

# ==========================
# Cleanup helpers
# ==========================
def safe_neutral(state: DriveState):
    if not state.pwm:
        return
    for _ in range(3):
        try:
            state.pwm.set12(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            state.pwm.set12(STEERING_CHANNEL, steering_center_pwm())
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
    try: cam.stop()
    except Exception: pass
    try: cam.stop_preview()
    except Exception: pass

def on_exit(state: DriveState):
    if getattr(state, "_cleanup_done", False):
        return
    state._cleanup_done = True
    try:
        with state.lock:
            state.target_thr_12 = THROTTLE_STOPPED_PWM
            state.target_str_12 = steering_center_pwm()
            state.last_command = "STOP"
        safe_neutral(state)
    finally:
        try: shutdown_actuator(state)
        except Exception: pass
        try: stop_camera(state)
        except Exception: pass
        print("\nExited.")

# ==========================
# Main
# ==========================
def main():
    # Init state and PCA9685
    state = DriveState()
    pwm = PWMDev(PCA9685_I2C_ADDR, PCA9685_FREQUENCY)
    state.pwm = pwm
    safe_neutral(state)

    # Camera setup
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
    signal.signal(signal.SIGINT, sig_exit)
    signal.signal(signal.SIGTERM, sig_exit)

    # Main key loop
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

                # Throttle intent
                if stop:
                    state.set_throttle_cmd("STOP")
                elif up and not down:
                    state.set_throttle_cmd("FWD")
                elif down and not up:
                    state.set_throttle_cmd("REV")
                else:
                    state.set_throttle_cmd("STOP")

                # Steering intent
                if center:
                    state.set_steer_cmd("CENTER")
                elif left and not right:
                    state.set_steer_cmd("LEFT")
                elif right and not left:
                    state.set_steer_cmd("RIGHT")
                else:
                    state.set_steer_cmd("CENTER")

                time.sleep(KEY_POLL_SLEEP)
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Ensure python3-picamera2 is available and I2C is enabled.")
    finally:
        on_exit(state)

if __name__ == "__main__":
    main()
