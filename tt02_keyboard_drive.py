#!/usr/bin/env python3
# tt02_keyboard_drive.py
# Control TT02 RC car (PCA9685) with keyboard arrow keys in a terminal.
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

import time
import curses
import signal
import sys

# ==========================
# Optional camera preview
# ==========================
try: 
    from picamera2 import Picamera2, Preview 
except Exception: 
    Picamera2 = None 
    Preview = None
# ==========================
# User-provided constants
# ==========================
PCA9685_I2C_ADDR   = 0x40
PCA9685_I2C_BUSNUM = None  # None = default I2C bus (usually 1 on Pi)

# STEERING
STEERING_CHANNEL     = 1
STEERING_LEFT_PWM    = 500
STEERING_RIGHT_PWM   = 240

# THROTTLE
THROTTLE_CHANNEL       = 0
THROTTLE_FORWARD_PWM   = 480
THROTTLE_STOPPED_PWM   = 370
THROTTLE_REVERSE_PWM   = 220

# PCA9685 update rate (Hz) for servos/ESCs
PCA9685_FREQUENCY = 60

# How long to treat key as "held" after last repeat (seconds)
KEY_HOLD_TIMEOUT = 0.20

# UI refresh rate (Hz)
UI_FPS = 50.0

# ==========================
# PCA9685 Driver (supports both legacy and CircuitPython libs)
# ==========================
class PCA9685Driver:
    def __init__(self, address=0x40, busnum=None, frequency=60):
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
        # Clamp to 0..4095
        v = max(0, min(4095, int(value_12bit)))
        if self._mode == 'legacy':
            # on=0, off=v
            self._driver.set_pwm(channel, 0, v)
        elif self._mode == 'cp':
            # CircuitPython uses 16-bit duty_cycle
            dc = int(round((v / 4095.0) * 65535.0))
            self._driver.channels[channel].duty_cycle = dc

# ==========================
# Helper functions
# ==========================
def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

def neutral_all(pwm: PCA9685Driver):
    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
    pwm.set_pwm(STEERING_CHANNEL, steering_center_pwm())
# ==========================
# Camera preview helper 
# ==========================
picam2 = None

def start_preview(): 
    print("Attempt to start Picamera2 preview; return a status string.")
    global picam2 
    if Picamera2 is None or Preview is None: 
        return "unavailable (Picamera2 not installed)" 
        try: 
            picam2 = Picamera2() 
            picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)})) 
            try: 
                picam2.start_preview(Preview.QTGL) 
                backend = "QTGL"
            except Exception: 
                picam2.start_preview(Preview.QT) 
                backend = "QT"
                picam2.start() 
            return f"started ({backend})" 
        except Exception as e: 
            return f"error: {e}"

def stop_preview(): 
    global picam2 
    try: 
        if picam2: 
            picam2.stop() 
            picam2.close() 
    except Exception: 
        pass 
    finally: 
        picam2 = None
# ==========================
# Main control (curses UI)
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

    # Start camera preview 
    preview_status = start_preview() 
    
    # State
    last_press = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
    last_steer = steering_center_pwm() 
    last_throttle = THROTTLE_STOPPED_PWM
    last_help_refresh = 0.0

    # Draw static help
    def draw_help():
        stdscr.clear()
        stdscr.addstr(0, 0, "TT02 Keyboard Drive (PCA9685)")
        stdscr.addstr(1, 0, "Controls:")
        stdscr.addstr(2, 2, "↑ Up: forward    | ↓ Down: reverse")
        stdscr.addstr(3, 2, "← Left: steer L  | → Right: steer R")
        stdscr.addstr(4, 2, "Space: STOP throttle,  c: center steering,  q: quit")
        stdscr.addstr(5, 0, f"Camera preview: {preview_status}")
        stdscr.addstr(6, 0, "Status:")
        stdscr.refresh()

    draw_help()

    # Cleanup handler
    def cleanup_and_exit():
        try:
            neutral_all(pwm)
            time.sleep(0.2)
        finally:
            stop_preview()

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
                    # Emergency stop throttle
                    last_press['up'] = 0.0
                    last_press['down'] = 0.0
                    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    last_throttle = THROTTLE_STOPPED_PWM
                elif ch in (ord('c'), ord('C')):
                    pwm.set_pwm(STEERING_CHANNEL, steering_center_pwm())
                    last_steer = steering_center_pwm()
                elif ch in (ord('q'), ord('Q')):
                    cleanup_and_exit()
                    return

            # Determine "active" keys using a short hold timeout (smooths key repeat gaps)
            def active(key):
                return (t - last_press[key]) < KEY_HOLD_TIMEOUT

            up = active('up')
            down = active('down')
            left = active('left')
            right = active('right')

            # Decide throttle command
            if up and not down:
                throttle_pwm = THROTTLE_FORWARD_PWM
            elif down and not up:
                # Simple safety: if switching from forward to reverse, pass through stop
                if last_throttle == THROTTLE_FORWARD_PWM:
                    pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                    time.sleep(0.05)
                throttle_pwm = THROTTLE_REVERSE_PWM
            else:
                throttle_pwm = THROTTLE_STOPPED_PWM

            # Decide steering command
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

            # Status display
            if (t - last_help_refresh) > 0.1:
                last_help_refresh = t
                stdscr.addstr(7, 0, f"Throttle: {last_throttle:4d}  (FWD:{THROTTLE_FORWARD_PWM} STOP:{THROTTLE_STOPPED_PWM} REV:{THROTTLE_REVERSE_PWM})      ")
                stdscr.addstr(8, 0, f"Steering: {last_steer:4d}   (L:{STEERING_LEFT_PWM} C:{steering_center_pwm()} R:{STEERING_RIGHT_PWM})      ")
                stdscr.addstr(10, 0, "Hold arrows to drive. Release to stop/center. 'q' to quit.                  ")
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
