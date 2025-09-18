#!/usr/bin/env python3
# TT02 RC car control via Adafruit PCA9685 (CircuitPython) + Picamera2 preview.
# - Uses microsecond pulses at 50 Hz (standard RC) for both ESC and servo.
# - Smooth ramping + STOP-pass safety when changing direction.
# - Single-line status HUD refreshes in place (no console spam).
# - Drive with arrow keys / WASD; Space=stop, c=center, q=quit.

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
PCA9685_FREQUENCY = 50      # Standard RC pulse frequency

# Channels
THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# ESC endpoints (microseconds) — adjust if your ESC needs different values
THROTTLE_REV_US   = 1000    # full reverse
THROTTLE_NEUTRAL  = 1500    # neutral/stop
THROTTLE_FWD_US   = 2000    # full forward

# Steering endpoints (microseconds) — tune to avoid binding
STEER_LEFT_US   = 1800
STEER_CENTER_US = 1500
STEER_RIGHT_US  = 1200

# Safety delay passing through STOP when changing direction
SWITCH_PAUSE_S = 0.06

# Smoothness and UI
ACTUATOR_HZ    = 200        # actuator update loop frequency
STEER_STEP_US  = 10         # per-cycle µs change for steering
THR_STEP_US    = 6          # per-cycle µs change for throttle
KEY_POLL_SLEEP = 0.01       # main loop tick
HUD_HZ         = 12         # HUD refresh rate (in-place)

# Camera settings
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

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

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def us_to_dc16(us, freq=PCA9685_FREQUENCY):
    # Convert microseconds to PCA9685 16-bit duty at given freq
    period_us = 1_000_000.0 / freq
    return int(clamp(round((us / period_us) * 65535.0), 0, 65535))

class PWMDev:
    def __init__(self, address, freq_hz):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.dev = PCA9685(i2c, address=address)
        self.dev.frequency = freq_hz
    def set_us(self, ch, pulse_us):
        dc = us_to_dc16(pulse_us, self.dev.frequency)
        self.dev.channels[ch].duty_cycle = dc
        return dc

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
        # Targets and outputs in microseconds
        self.target_thr_us = THROTTLE_NEUTRAL
        self.target_str_us = STEER_CENTER_US
        self.out_thr_us    = THROTTLE_NEUTRAL
        self.out_str_us    = STEER_CENTER_US

        # Last written duty cycles (16-bit)
        self.last_thr_dc16 = us_to_dc16(self.out_thr_us)
        self.last_str_dc16 = us_to_dc16(self.out_str_us)

        # Direction change safety
        self.last_cmd = "STOP"          # "FWD","REV","STOP"
        self.switch_block_until = 0.0

        # Threads/resources
        self.lock = threading.Lock()
        self.alive = True
        self._cleanup_done = False
        self.pwm = None
        self.picam2 = None
        self.actuator_thread = None

        # HUD timing
        self._last_hud = 0.0

    def set_throttle_cmd(self, cmd):  # "FWD","REV","STOP"
        now = time.monotonic()
        with self.lock:
            if cmd == "FWD":
                if self.last_cmd == "REV":
                    if now < self.switch_block_until:
                        return
                    # enforce stop first
                    self.target_thr_us = THROTTLE_NEUTRAL
                    self.switch_block_until = now + SWITCH_PAUSE_S
                    self.last_cmd = "STOP"
                    return
                self.target_thr_us = THROTTLE_FWD_US
                self.last_cmd = "FWD"
            elif cmd == "REV":
                if self.last_cmd == "FWD":
                    if now < self.switch_block_until:
                        return
                    self.target_thr_us = THROTTLE_NEUTRAL
                    self.switch_block_until = now + SWITCH_PAUSE_S
                    self.last_cmd = "STOP"
                    return
                self.target_thr_us = THROTTLE_REV_US
                self.last_cmd = "REV"
            else:
                self.target_thr_us = THROTTLE_NEUTRAL
                self.last_cmd = "STOP"

    def set_steer_cmd(self, cmd):  # "LEFT","RIGHT","CENTER"
        with self.lock:
            if cmd == "LEFT":
                self.target_str_us = STEER_LEFT_US
            elif cmd == "RIGHT":
                self.target_str_us = STEER_RIGHT_US
            else:
                self.target_str_us = STEER_CENTER_US

# ==========================
# Actuator thread (ramping + HUD)
# ==========================
def actuator_loop(state: DriveState):
    pwm = state.pwm
    dt = 1.0 / ACTUATOR_HZ
    hud_dt = 1.0 / HUD_HZ

    # Prepare HUD line (we’ll overwrite it)
    print()  # reserve one line for HUD

    while state.alive:
        # Ramp toward targets
        with state.lock:
            # Throttle
            if state.target_thr_us > state.out_thr_us:
                state.out_thr_us = min(state.out_thr_us + THR_STEP_US, state.target_thr_us)
            else:
                state.out_thr_us = max(state.out_thr_us - THR_STEP_US, state.target_thr_us)
            # Steering
            if state.target_str_us > state.out_str_us:
                state.out_str_us = min(state.out_str_us + STEER_STEP_US, state.target_str_us)
            else:
                state.out_str_us = max(state.out_str_us - STEER_STEP_US, state.target_str_us)

            thr_out = state.out_thr_us
            str_out = state.out_str_us

        # Write to hardware
        try:
            thr_dc = pwm.set_us(THROTTLE_CHANNEL, thr_out)
            str_dc = pwm.set_us(STEERING_CHANNEL, str_out)
            state.last_thr_dc16 = thr_dc
            state.last_str_dc16 = str_dc
        except Exception:
            # keep trying on next loop if transient I2C error
            pass

        # In-place HUD refresh (one line, no scrolling)
        now = time.monotonic()
        if now - state._last_hud >= hud_dt:
            state._last_hud = now
            with state.lock:
                dir_wait = (now < state.switch_block_until)
                hud = (
                    f"[PCA9685 {pwm.dev.frequency}Hz | THR ch{THROTTLE_CHANNEL} | STR ch{STEERING_CHANNEL}] "
                    f"THR tgt/out: {state.target_thr_us:.0f}/{thr_out:.0f}us (dc16={state.last_thr_dc16}) "
                    f"cmd={state.last_cmd}{' (dir-wait)' if dir_wait else ''} | "
                    f"STR tgt/out: {state.target_str_us:.0f}/{str_out:.0f}us (dc16={state.last_str_dc16})"
                )
            # Clear line, print, and return carriage
            sys.stdout.write("\r\033[2K" + hud)
            sys.stdout.flush()

        time.sleep(dt)

# ==========================
# Cleanup helpers
# ==========================
def safe_neutral(state: DriveState):
    if not state.pwm:
        return
    for _ in range(3):
        try:
            state.pwm.set_us(THROTTLE_CHANNEL, THROTTLE_NEUTRAL)
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
            state.target_thr_us = THROTTLE_NEUTRAL
            state.target_str_us = STEER_CENTER_US
            state.last_cmd = "STOP"
        safe_neutral(state)
    finally:
        try: shutdown_actuator(state)
        except Exception: pass
        try: stop_camera(state)
        except Exception: pass
        # Move to next line so shell prompt isn't on HUD line
        print("\nExited.")

# ==========================
# Main
# ==========================
def main():
    # Init state and PCA9685
    state = DriveState()
    pwm = PWMDev(PCA9685_I2C_ADDR, PCA9685_FREQUENCY)
    state.pwm = pwm
    # Put ESC at neutral before it powers up if possible (arm)
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
