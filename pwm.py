#!/usr/bin/env python3
# simple_tt02_pwm_tuner.py

import time
import curses
import signal
import sys

PCA9685_I2C_ADDR   = 0x40
PCA9685_I2C_BUSNUM = None
PCA9685_FREQUENCY  = 60  

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

def us_to_12bit(us, freq=PCA9685_FREQUENCY):
    period_us = 1_000_000.0 / float(freq)
    ticks = round((us / period_us) * 4095.0)
    return max(0, min(4095, int(ticks)))

def ticks_to_us(ticks, freq):
    period_us = 1_000_000.0 / float(freq)
    return int(round((ticks / 4095.0) * period_us))

def ticks_to_duty_pct(ticks):
    return (max(0, min(4095, int(ticks))) / 4095.0) * 100.0

THROTTLE_STOPPED_US = 1500

# Step sizes per request
STEP = 5        # ticks per normal step
BIG_STEP = 25   # ticks per big step (Shift)

UI_FPS = 30.0

class PCA9685Driver:
    def __init__(self, address=0x40, busnum=None, frequency=60):
        self._mode = None
        self._driver = None
        self._freq = frequency
        try:
            import Adafruit_PCA9685 as LegacyPCA9685
            if busnum is None:
                self._driver = LegacyPCA9685.PCA9685(address=address)
            else:
                self._driver = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
            self._driver.set_pwm_freq(frequency)
            self._mode = 'legacy'
        except Exception:
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
            self._driver.set_pwm(channel, 0, v)
        else:
            dc16 = int(round((v / 4095.0) * 65535.0))
            self._driver.channels[channel].duty_cycle = dc16

def prompt_input(stdscr, row, col, prompt):
    stdscr.addstr(row, col, " " * 120)
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
    stdscr.addstr(2, 2, "Arrow Up/Down   = throttle +/- step")
    stdscr.addstr(3, 2, "Arrow Left/Right= steering +/- step")
    stdscr.addstr(4, 2, "W/S             = throttle +/- step (alias)")
    stdscr.addstr(5, 2, "Shift+W/Shift+S = throttle +/- BIG step")
    stdscr.addstr(6, 2, "Space           = throttle -> 1500 us (stop)")
    stdscr.addstr(7, 2, "c               = steering -> 1600 us (center)")
    stdscr.addstr(8, 2, "i               = set ticks (0..4095) for selected channel")
    stdscr.addstr(9, 2, "u               = set microseconds for selected channel")
    stdscr.addstr(10,2, "f               = change PCA9685 frequency (Hz)")
    stdscr.addstr(11,2, "TAB             = switch selected channel (Throttle/Steering)")
    stdscr.addstr(12,2, "q               = quit")
    stdscr.addstr(14,0, "Status:")
    stdscr.refresh()

def run(stdscr):
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    curses.curs_set(0)

    pwm = PCA9685Driver(address=PCA9685_I2C_ADDR, busnum=PCA9685_I2C_BUSNUM, frequency=PCA9685_FREQUENCY)

    # Initial values
    values = {
        "throttle": us_to_12bit(1500, pwm.frequency),
        "steering": us_to_12bit(1600, pwm.frequency),
    }
    channels = {"throttle": THROTTLE_CHANNEL, "steering": STEERING_CHANNEL}

    # Selected channel for i/u input; arrows directly map, but TAB can toggle selection
    items = ["throttle", "steering"]
    sel_idx = 0

    # Apply startup outputs
    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
    pwm.set_pwm_12bit(channels["steering"], values["steering"])

    draw_help(stdscr)
    last_ui = 0.0

    def clamp12(v):
        return max(0, min(4095, int(v)))

    def apply_outputs():
        pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
        pwm.set_pwm_12bit(channels["steering"], values["steering"])

    def redraw():
        freq = float(pwm.frequency)
        period_us = 1_000_000.0 / freq
        thr = values["throttle"]
        ste = values["steering"]
        thr_us = ticks_to_us(thr, freq)
        ste_us = ticks_to_us(ste, freq)
        thr_dc = ticks_to_duty_pct(thr)
        ste_dc = ticks_to_duty_pct(ste)

        # Enhanced status lines (kept section titles/positions)
        stdscr.addstr(15, 0, f"Selected: {items[sel_idx].upper():9s}    PCA9685 Freq: {int(freq):4d} Hz (period ~ {period_us:7.1f} us)                     ")
        stdscr.addstr(17, 0, f"Throttle: ch{channels['throttle']}  ticks={thr:4d}  ~us={thr_us:4d}  duty={thr_dc:6.2f}%                                  ")
        stdscr.addstr(18, 0, f"Steering: ch{channels['steering']}  ticks={ste:4d}  ~us={ste_us:4d}  duty={ste_dc:6.2f}%                                  ")
        stdscr.addstr(20, 0, f"Stop target: 1500 us -> ticks={us_to_12bit(1500, freq)}     Center target: 1600 us -> ticks={us_to_12bit(1600, freq)}     ")
        stdscr.refresh()

    def cleanup_and_exit():
        pass  # leave outputs as-is

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

                # Switch selected channel (for i/u input convenience)
                elif ch in (curses.KEY_BTAB, 9):  # Shift+Tab -> KEY_BTAB; 9 -> Tab
                    sel_idx = (sel_idx + 1) % len(items)

                # Throttle adjustments (Up/Down and W/S)
                elif ch in (curses.KEY_UP, ord('w')):
                    values["throttle"] = clamp12(values["throttle"] + STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (curses.KEY_DOWN, ord('s')):
                    values["throttle"] = clamp12(values["throttle"] - STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (ord('W'),):
                    values["throttle"] = clamp12(values["throttle"] + BIG_STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (ord('S'),):
                    values["throttle"] = clamp12(values["throttle"] - BIG_STEP)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])

                # Steering adjustments (Left/Right)
                elif ch in (curses.KEY_RIGHT,):
                    values["steering"] = clamp12(values["steering"] + STEP)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])
                elif ch in (curses.KEY_LEFT,):
                    values["steering"] = clamp12(values["steering"] - STEP)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])

                # Direct set for selected channel
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

                # Frequency change
                elif ch in (ord('f'), ord('F')):
                    s = prompt_input(stdscr, 22, 0, f"Enter PCA9685 frequency in Hz (current {int(pwm.frequency)}): ")
                    try:
                        hz = int(float(s))
                        hz = max(24, min(1526, hz))
                        pwm.set_pwm_freq(hz)
                        # Re-apply current ticks (mapping inside driver handles duty)
                        pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                        pwm.set_pwm_12bit(channels["steering"], values["steering"])
                    except Exception:
                        pass

                # Space -> throttle stop, c -> steering center
                elif ch in (ord(' '),):
                    values["throttle"] = us_to_12bit(1500, pwm.frequency)
                    pwm.set_pwm_12bit(channels["throttle"], values["throttle"])
                elif ch in (ord('c'), ord('C')):
                    values["steering"] = us_to_12bit(1600, pwm.frequency)
                    pwm.set_pwm_12bit(channels["steering"], values["steering"])

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
