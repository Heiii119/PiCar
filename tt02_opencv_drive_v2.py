#!/usr/bin/env python3
# tt02_keyboard_drive_with_camera.py
# Keyboard drive for TT02 RC car using PCA9685 @ 50 Hz with a curses status UI
# and Picamera2 live preview window.
#
# Why this version:
# - Keeps your working Picamera2 preview (QT/DRM fallback).
# - Uses 50 Hz RC timing and 12-bit ticks chosen to match ~1000/1500/2000 µs so ESCs respond.
# - Non-scrolling curses HUD that continuously updates PWM and direction status.
# - Safe STOP-pass pause when switching FWD <-> REV.
#
# Keys:
# - Up/W:    throttle forward
# - Down/S:  throttle reverse
# - Left/A:  steer left
# - Right/D: steer right
# - Space:   immediate throttle stop
# - c:       center steering
# - q:       quit (safe stop + center)

import time
import curses
import signal
import sys

from picamera2 import Picamera2, Preview

# ==========================
# User-configurable constants
# ==========================
PCA9685_I2C_ADDR   = 0x40
PCA9685_I2C_BUSNUM = None  # None = default I2C bus (usually bus 1 on Pi)

# PCA9685 update rate (Hz) for servos/ESCs
PCA9685_FREQUENCY = 50

# Camera settings
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Helper: convert microseconds to 12-bit ticks for the given freq (50 Hz default)
def us_to_12bit(us, freq=PCA9685_FREQUENCY):
    period_us = 1_000_000.0 / freq  # e.g., 20_000 at 50 Hz
    ticks = round((us / period_us) * 4095.0)
    return max(0, min(4095, int(ticks)))

# THROTTLE endpoints mapped to ~1000/1500/2000 µs (works with most ESCs)
THROTTLE_CHANNEL       = 0
THROTTLE_REVERSE_PWM   = us_to_12bit(1000)  # ≈ 205
THROTTLE_STOPPED_PWM   = us_to_12bit(1500)  # ≈ 307
THROTTLE_FORWARD_PWM   = us_to_12bit(2000)  # ≈ 409

# STEERING endpoints (tune for your servo; avoid mechanical binding)
STEERING_CHANNEL     = 1
STEERING_RIGHT_PWM   = us_to_12bit(1200)    # ≈ 246
STEERING_LEFT_PWM    = us_to_12bit(1800)    # ≈ 369

# How long to treat key as "held" after last repeat (seconds)
KEY_HOLD_TIMEOUT = 0.20

# UI refresh rate (Hz)
UI_FPS = 50.0

# Safety pause when switching FWD <-> REV
SWITCH_PAUSE_S = 0.06

# ==========================
# PCA9685 Driver (supports both legacy and CircuitPython libs)
# ==========================
class PCA9685Driver:
    def __init__(self, address=0x40, busnum=None, frequency=50):
        self._mode = None  # 'legacy' or 'cp'
        self._driver = None
        # Try legacy Adafruit_PCA9685 first
        try:
            import Adafruit_PCA9685 as LegacyPCA9685
            if busnum is None:
                self._driver = LegacyPCA9685.PCA9685(address=address)
            else:
                self._driver = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
            self._driver.set_pwm_freq(frequency)
            self._mode = 'legacy'
        except Exception:
            # Fallback to CircuitPython library
            try:
                import board
                import busio
                from adafruit_pca9685 import PCA9685 as CPPCA9685
                i2c = busio.I2C(board.SCL, board.SDA)
                self._driver = CPPCA9685(i2c, address=address)
                self._driver.frequency = frequency
                self._mode = 'cp'
            except Exception as e:
                raise SystemExit(
                    "Could not initialize PCA9685. Please install one of:\n"
                    "  - Legacy:  pip3 install Adafruit-PCA9685\n"
                    "  - CircuitPython: pip3 install adafruit-circuitpython-pca9685\n"
                    f"Original error: {e}"
                )

    def set_pwm_freq(self, freq_hz):
        if self._mode == 'legacy':
            self._driver.set_pwm_freq(freq_hz)
        elif self._mode == 'cp':
            self._driver.frequency = freq_hz

    def set_pwm(self, channel, value_12bit):
        v = max(0, min(4095, int(value_12bit)))
        if self._mode == 'legacy':
            # Legacy uses 12-bit "off" count directly (on=0, off=v)
            self._driver.set_pwm(channel, 0, v)
        elif self._mode == 'cp':
            # CircuitPython uses 16-bit duty_cycle; convert from 12-bit ticks
            dc16 = int(round((v / 4095.0) * 65535.0))
            self._driver.channels[channel].duty_cycle = dc16

# ==========================
# Helper functions
# ==========================
def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

def neutral_all(pwm: PCA9685Driver):
    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
    pwm.set_pwm(STEERING_CHANNEL, steering_center_pwm())

# ==========================
# Main control (curses UI + camera)
# ==========================
def run(stdscr):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    # Init PCA9685
    pwm = PCA9685Driver(address=PCA9685_I2C_ADDR, busnum=PCA9685_I2C_BUSNUM, frequency=PCA9685_FREQUENCY)
    pwm.set_pwm_freq(PCA9685_FREQUENCY)

    # Ensure safe neutral at start (arm ESC)
    neutral_all(pwm)
    time.sleep(1.0)

    # Camera setup and preview (QT preferred, DRM fallback)
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

    # State
    last_press = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
    last_steer = None
    last_throttle = None
    last_help_refresh = 0.0
    last_dir = "STOP"  # "FWD","REV","STOP"
    dir_block_until = 0.0

    # Draw static help
    def draw_help():
        stdscr.clear()
        stdscr.addstr(0, 0, f"TT02 Keyboard Drive (PCA9685 @ {PCA9685_FREQUENCY} Hz) + Picamera2 [{ 'QT' if using_qt else 'DRM' }]")
        stdscr.addstr(1, 0, "Controls:")
        stdscr.addstr(2, 2, "↑ Up: forward    | ↓ Down: reverse")
        stdscr.addstr(3, 2, "← Left: steer L  | → Right: steer R")
        stdscr.addstr(4, 2, "Space: STOP throttle,  c: center steering,  q: quit")
        stdscr.addstr(6, 0, "Status:")
        stdscr.refresh()

    draw_help()

    # Cleanup handler
    def cleanup_and_exit():
        try:
            neutral_all(pwm)
            time.sleep(0.2)
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.stop_preview()
            except Exception:
                pass

    # SIGINT safe shutdown
    def sigint_handler(signum, frame):
        cleanup_and_exit()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Main loop
    dt = 1.0 / UI_FPS
    try:
        while True:
            t = time.time()
            # Drain input buffer this frame
            while True:
                ch = stdscr.getch()
                if ch == -1:
                    break
                if ch in (curses.KEY_UP, ord('w'), ord('W')):
                    last_press['up'] = t
                elif ch in (curses.KEY_DOWN, ord('s'), ord('S')):
                    last_press['down'] = t
                elif ch in (curses.KEY_LEFT, ord('a'), ord('A')):
                    last_press['left'] = t
                elif ch in (curses.KEY_RIGHT, ord('d'), ord('D')):
                    last_press['right'] = t
                elif ch in (ord(' '),):
                    # Immediate throttle stop
                    last_press['up'] = 0.0
                    last_press['down'] = 0.0
                    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    last_throttle = THROTTLE_STOPPED_PWM
                    last_dir = "STOP"
                    dir_block_until = 0.0
                elif ch in (ord('c'), ord('C')):
                    pwm.set_pwm(STEERING_CHANNEL, steering_center_pwm())
                    last_steer = steering_center_pwm()
                elif ch in (ord('q'), ord('Q')):
                    cleanup_and_exit()
                    return

            # Determine "active" keys using a short hold timeout
            def active(key):
                return (t - last_press[key]) < KEY_HOLD_TIMEOUT

            up = active('up')
            down = active('down')
            left = active('left')
            right = active('right')

            # Decide throttle with STOP-pass safety
            desired_dir = "STOP"
            if up and not down:
                desired_dir = "FWD"
            elif down and not up:
                desired_dir = "REV"

            throttle_pwm = THROTTLE_STOPPED_PWM
            if desired_dir == "FWD":
                if last_dir == "REV" and t < dir_block_until:
                    throttle_pwm = THROTTLE_STOPPED_PWM
                elif last_dir == "REV":
                    # initiate STOP pause before allowing FWD
                    throttle_pwm = THROTTLE_STOPPED_PWM
                    dir_block_until = t + SWITCH_PAUSE_S
                    last_dir = "STOP"
                else:
                    throttle_pwm = THROTTLE_FORWARD_PWM
                    last_dir = "FWD"
            elif desired_dir == "REV":
                if last_dir == "FWD" and t < dir_block_until:
                    throttle_pwm = THROTTLE_STOPPED_PWM
                elif last_dir == "FWD":
                    throttle_pwm = THROTTLE_STOPPED_PWM
                    dir_block_until = t + SWITCH_PAUSE_S
                    last_dir = "STOP"
                else:
                    throttle_pwm = THROTTLE_REVERSE_PWM
                    last_dir = "REV"
            else:
                throttle_pwm = THROTTLE_STOPPED_PWM
                last_dir = "STOP"

            # Decide steering
            if left and not right:
                steer_pwm = STEERING_LEFT_PWM
            elif right and not left:
                steer_pwm = STEERING_RIGHT_PWM
            else:
                steer_pwm = steering_center_pwm()

            # Apply only if changed (reduces I2C traffic)
            if throttle_pwm != last_throttle:
                pwm.set_pwm(THROTTLE_CHANNEL, throttle_pwm)
                last_throttle = throttle_pwm

            if steer_pwm != last_steer:
                pwm.set_pwm(STEERING_CHANNEL, steer_pwm)
                last_steer = steer_pwm

            # Status display (non-scrolling, in-place)
            if (t - last_help_refresh) > 0.02:
                last_help_refresh = t
                stdscr.addstr(7, 0, f"Throttle: {last_throttle:4d}  (FWD:{THROTTLE_FORWARD_PWM} STOP:{THROTTLE_STOPPED_PWM} REV:{THROTTLE_REVERSE_PWM})     ")
                stdscr.addstr(8, 0, f"Steering: {last_steer:4d}   (L:{STEERING_LEFT_PWM} C:{steering_center_pwm()} R:{STEERING_RIGHT_PWM})     ")
                wait_flag = " (dir-wait)" if t < dir_block_until and last_dir == "STOP" else "           "
                stdscr.addstr(9, 0, f"Dir: {last_dir}{wait_flag}                                         ")
                stdscr.addstr(11, 0, "Hold arrows/WASD to drive. Release to stop/center. 'q' to quit.            ")
                stdscr.refresh()

            time.sleep(dt)

    finally:
        # Ensure safe neutral and restore terminal even on exceptions
        try:
            cleanup_and_exit()
        except Exception:
            pass
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

def main():
    curses.wrapper(run)

if __name__ == "__main__":
    main()
