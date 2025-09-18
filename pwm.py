#!/usr/bin/env python3
# simple_tt02_pwm_tuner.py
# Interactive PWM tuner for TT02 (PCA9685). No camera.
# - W/Up:    increase current channel by step; S/Down: decrease
# - Shift+W: big increase; Shift+S: big decrease
# - A/Left, D/Right: switch between THROTTLE and STEERING
# - i: set exact 12-bit ticks (0..4095)
# - u: set exact microseconds (converted to ticks at current freq)
# - f: change PCA9685 frequency (Hz)
# - r: reset both to neutral/center defaults
# - Space: set throttle to THROTTLE_STOPPED_PWM immediately
# - c: center steering immediately
# - q: quit
#
# Startup requirement per request:
# - On program start, throttle = 1500 µs, steering = 1600 µs

import time
import curses
import signal
import sys

# ============ User settings ============
PCA9685_I2C_ADDR   = 0x40
PCA9685_I2C_BUSNUM = None   # None = default bus (usually 1 on Pi)
PCA9685_FREQUENCY  = 50     # 50 Hz is standard RC

# Channels
THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# Helper conversions
def us_to_12bit(us, freq=PCA9685_FREQUENCY):
    period_us = 1_000_000.0 / float(freq)
    ticks = round((us / period_us) * 4095.0)
    return max(0, min(4095, int(ticks)))

def ticks_to_us(ticks, freq):
    period_us = 1_000_000.0 / float(freq)
    return int(round((ticks / 4095.0) * period_us))

# Defaults per request
THROTTLE_STOPPED_US = 1500
STEERING_START_US   = 1600

# Derive initial ticks at the configured frequency
THROTTLE_STOPPED_PWM = us_to_12bit(THROTTLE_STOPPED_US, PCA9685_FREQUENCY)

# UI step sizes
STEP = 10       # ticks
BIG_STEP = 50   # ticks
UI_FPS = 30.0

# ============ PCA9685 driver (legacy or CircuitPython) ============
class PCA9685Driver:
    def __init__(self, address=0x40, busnum=None, frequency=50):
        self._mode = None  # 'legacy' or 'cp'
        self._driver = None
        self._freq = frequency
        # Try legacy Adafruit first
        try:
            import Adafruit_PCA9685 as LegacyPCA9685
            if busnum is None:
                self._driver = LegacyPCA9685.PCA9685(address=address)
            else:
                self._driver = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
            self._driver.set_pwm_freq(frequency)
            self._mode = 'legacy'
        except Exception:
            # Fallback to CircuitPython
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
                    "Could not initialize PCA9685. Install one of:\n"
                    "  - pip3 install Adafruit-PCA9685\n"
                    "  - pip3 install adafruit-circuitpython-pca9685\n"
                    f"Original error: {e}"
                )

    @property
    def frequency(self):
        return self._driver.frequency if self._mode == 'cp' else self._freq

    def set_pwm_freq(self, freq_hz):
        if self._mode == 'legacy':
            self._driver.set_pwm_freq(freq_hz)
            self._freq = freq_hz
        elif self._mode == 'cp':
            self._driver.frequency = freq_hz

    def set_pwm_12bit(self, channel, value_12bit):
        v = max(0, min(4095, int(value_12bit)))
        if self._mode == 'legacy':
            self._driver.set_pwm(channel, 0, v)  # on=0, off=v
        else:
            dc16 = int(round((v / 4095.0) * 65535.0))
            self._driver.channels[channel].duty_cycle = dc16

# ============ UI helpers ============
def prompt_input(stdscr, row, col, prompt):
    stdscr.addstr(row, col, " " * 80)
    stdscr.addstr(row, col, prompt)
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    try:
        s = stdscr.getstr(row, col + len(prompt), 20)
        try:
            return s.decode().strip()
        except Exception:
            return ""
    finally:
        curses.noecho()
        curses.curs_set(0)

def draw_help(stdscr):
    stdscr.addstr(0, 0, "TT02 PWM Tuner (PCA9685)")
    stdscr.addstr(1, 0, "Controls:")
    stdscr.addstr(2, 2, "A/Left  = select THROTTLE/STEERING")
    stdscr.addstr(3, 2, "D/Right = select THROTTLE/STEERING")
    stdscr.addstr(4, 2, "W/Up    = +step       | S/Down  = -step")
    stdscr.addstr(5, 2, "Shift+W = +big step   | Shift+S = -big step")
    stdscr.addstr(6, 2, "Space   = throttle -> THROTTLE_STOPPED_PWM")
    stdscr.addstr(7, 2, "c       = center steering (1600 µs by default)")
    stdscr.addstr(8, 2, "i       = input ticks (0..4095)")
    stdscr.addstr(9, 2, "u       = input microseconds")
    stdscr.addstr(10, 2, "f       = change frequency (Hz)")
    stdscr.addstr(11, 2, "r       = reset both to defaults (1500/1600 µs)")
    stdscr.addstr(12, 2, "q       = quit")
    stdscr.addstr(14, 0, "Status:")
    stdscr.refresh()

# ============ Main ============
def run(stdscr):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    pwm = PCA9685Driver(address=PCA9685_I2C_ADDR, busnum=PCA9685_I2C_BUSNUM, frequency=PCA9685_FREQUENCY)

    # Initial values per request: throttle=1500 µs, steering=1600 µs
    values = {
        "throttle": us_to_12bit(1500, pwm.frequency),
        "steering": us_to_12bit(1600, pwm.frequency),
    }
    channels = {
        "throttle": THROTTLE_CHANNEL,
        "steering": STEERING_CHANNEL,
    }

    # Apply startup outputs
    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
    pwm.set_pwm_12bit(channels["steering"], values["steering"])

    items = ["throttle", "steering"]
    sel_idx = 0

    draw_help(stdscr)
    last_ui = 0.0

    def apply_outputs():
        pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
        pwm.set_pwm_12bit(channels["steering"], values["steering"])

    def clamp12(v):
        return max(0, min(4095, int(v)))

    def redraw():
        freq = pwm.frequency
        thr = values["throttle"]
        ste = values["steering"]
        stdscr.addstr(15, 0, f"Selected: {items[sel_idx].upper():9s}    PCA9685 Freq: {freq:4d} Hz                            ")
        stdscr.addstr(17, 0, f"Throttle: ch{channels['throttle']}  ticks={thr:4d}  ~us={ticks_to_us(thr, freq):4d}                    ")
        stdscr.addstr(18, 0, f"Steering: ch{channels['steering']}  ticks={ste:4d}  ~us={ticks_to_us(ste, freq):4d}                    ")
        stdscr.addstr(20, 0, f"THROTTLE_STOPPED_PWM (ticks at {freq} Hz) = {us_to_12bit(THROTTLE_STOPPED_US, freq)}                     ")
        stdscr.refresh()

    def cleanup_and_exit():
        # Leave outputs as-is (no auto center/stop)
        pass

    def sigint_handler(signum, frame):
        cleanup_and_exit()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while True:
            ch = stdscr.getch()
            if ch != -1:
                if ch in (ord('q'), ord('Q')):
                    return
                elif ch in (curses.KEY_LEFT, ord('a'), ord('A')):
                    sel_idx = (sel_idx - 1) % len(items)
                elif ch in (curses.KEY_RIGHT, ord('d'), ord('D')):
                    sel_idx = (sel_idx + 1) % len(items)
                elif ch in (curses.KEY_UP, ord('w')):   # +step
                    key = items[sel_idx]
                    values[key] = clamp12(values[key] + STEP)
                    pwm.set_pwm_12bit(channels[key], values[key])
                elif ch in (curses.KEY_DOWN, ord('s')): # -step
                    key = items[sel_idx]
                    values[key] = clamp12(values[key] - STEP)
                    pwm.set_pwm_12bit(channels[key], values[key])
                elif ch in (ord('W'),):  # +big
                    key = items[sel_idx]
                    values[key] = clamp12(values[key] + BIG_STEP)
                    pwm.set_pwm_12bit(channels[key], values[key])
                elif ch in (ord('S'),):  # -big
                    key = items[sel_idx]
                    values[key] = clamp12(values[key] - BIG_STEP)
                    pwm.set_pwm_12bit(channels[key], values[key])
                elif ch in (ord('i'), ord('I')):
                    key = items[sel_idx]
                    s = prompt_input(stdscr, 22, 0, f"Enter 12-bit ticks (0..4095) for {key}: ")
                    try:
                        v = int(s)
                        values[key] = clamp12(v)
                        pwm.set_pwm_12bit(channels[key], values[key])
                    except Exception:
                        pass
                elif ch in (ord('u'), ord('U')):
                    key = items[sel_idx]
                    s = prompt_input(stdscr, 22, 0, f"Enter microseconds (e.g., 1500) for {key}: ")
                    try:
                        us = float(s)
                        values[key] = clamp12(us_to_12bit(us, pwm.frequency))
                        pwm.set_pwm_12bit(channels[key], values[key])
                    except Exception:
                        pass
                elif ch in (ord('f'), ord('F')):
                    s = prompt_input(stdscr, 22, 0, "Enter PCA9685 frequency in Hz (e.g., 50): ")
                    try:
                        hz = int(float(s))
                        hz = max(24, min(1526, hz))  # PCA9685 practical range
                        pwm.set_pwm_freq(hz)
                        # Reapply current ticks at new freq
                        apply_outputs()
                    except Exception:
                        pass
                elif ch in (ord('r'), ord('R')):
                    # Reset to defaults: throttle 1500 µs, steering 1600 µs
                    values["throttle"] = us_to_12bit(1500, pwm.frequency)
                    values["steering"] = us_to_12bit(1600, pwm.frequency)
                    apply_outputs()
                elif ch in (ord(' '),):  # Space: throttle -> THROTTLE_STOPPED_PWM
                    values["throttle"] = us_to_12bit(THROTTLE_STOPPED_US, pwm.frequency)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (ord('c'), ord('C')):  # Center steering (1600 µs)
                    values["steering"] = us_to_12bit(1600, pwm.frequency)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])

            # UI update
            t = time.time()
            if t - last_ui >= (1.0 / UI_FPS):
                last_ui = t
                redraw()

            time.sleep(1.0 / UI_FPS)

    finally:
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
