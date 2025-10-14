#!/usr/bin/env python3
# Timed driver with explicit start:
# - Idle at rest after start
# - Press 's' to start the loop: Forward 2s, Right 7s, repeat 6 times
# - Space: emergency stop/pause (neutral)
# - r: reset routine back to beginning (remains paused until 's')
# - c: center steering immediately
# - q: quit

import time
import sys
import tty
import termios
import select
import threading
import numpy as np

# PWM / PCA9685
import board
import busio
from adafruit_pca9685 import PCA9685

# ------------------------------
# Configuration (exact PWMs)
# ------------------------------
CFG = {
    "PWM_STEERING_PIN": "PCA9685.1:0x40.1",  # Bus 1, address 0x40, channel 1
    "PWM_STEERING_INVERTED": False,
    "PWM_THROTTLE_PIN": "PCA9685.1:0x40.0",  # Bus 1, address 0x40, channel 0
    "PWM_THROTTLE_INVERTED": False,
    "STEERING_LEFT_PWM": 270,
    "STEERING_RIGHT_PWM": 495,
    "THROTTLE_FORWARD_PWM": 390,
    "THROTTLE_STOPPED_PWM": 370,
    "THROTTLE_REVERSE_PWM": 290,
}

# Routine settings
FORWARD_SEC = 1.5
RIGHT_SEC = 4.5
LOOPS = 6

# ------------------------------
# Helpers
# ------------------------------
def parse_pca9685_pin(pin_str):
    # Format "PCA9685.<bus>:<i2c_addr>.<channel>"
    left, chan = pin_str.split(":")
    bus_str = left.split(".")[1]
    addr_str = chan.split(".")[0] if "." in chan else chan
    channel_str = chan.split(".")[1] if "." in chan else "0"
    i2c_bus = int(bus_str)
    i2c_addr = int(addr_str, 16) if addr_str.lower().startswith("0x") else int(addr_str)
    channel = int(channel_str)
    return i2c_bus, i2c_addr, channel

class Driver:
    def __init__(self, cfg):
        _, addr_s, s_ch = parse_pca9685_pin(cfg["PWM_STEERING_PIN"])
        _, addr_t, t_ch = parse_pca9685_pin(cfg["PWM_THROTTLE_PIN"])
        if addr_s != addr_t:
            raise ValueError("Steering and Throttle must be on the same PCA9685 address.")
        self.cfg = cfg
        self.s_ch = s_ch
        self.t_ch = t_ch
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=addr_s)
        self.pca.frequency = 60
        self.stop()

    def _to_duty(self, pwm4095):
        pwm4095 = int(np.clip(pwm4095, 0, 4095))
        return int((pwm4095 / 4095.0) * 65535)

    def set_steering_pwm(self, pwm4095):
        self.pca.channels[self.s_ch].duty_cycle = self._to_duty(pwm4095)

    def set_throttle_pwm(self, pwm4095):
        self.pca.channels[self.t_ch].duty_cycle = self._to_duty(pwm4095)

    def center_steering(self):
        left = self.cfg["STEERING_LEFT_PWM"]
        right = self.cfg["STEERING_RIGHT_PWM"]
        center = int(round((left + right) / 2))
        self.set_steering_pwm(center)
        return center

    def stop(self):
        self.set_throttle_pwm(self.cfg["THROTTLE_STOPPED_PWM"])

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.pca.deinit()

# ------------------------------
# Keyboard (non-blocking)
# ------------------------------
class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, a, b, c):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def get_key(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

# ------------------------------
# Timed routine controller
# ------------------------------
class TimedDriver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.drv = Driver(cfg)
        self.running = True
        self.paused = True          # start paused/idle
        self.reset_requested = False
        self.start_requested = False
        self.thread = None

    def start(self):
        # Ensure neutral and centered on start
        self.drv.stop()
        ctr = self.drv.center_steering()
        print(f"[Init] Neutral throttle, center steering PWM {ctr}")
        print("Controls: s=START, space=EMERGENCY STOP/PAUSE, r=RESET, c=CENTER, q=QUIT")
        self.thread = threading.Thread(target=self.run_routine, daemon=True)
        self.thread.start()
        self.keyboard_loop()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        # Send neutral a few times for safety
        for _ in range(5):
            self.drv.stop()
            time.sleep(0.05)
        self.drv.close()

    def keyboard_loop(self):
        with RawKeyboard() as kb:
            while self.running:
                ch = kb.get_key(timeout=0.05)
                if ch is None:
                    continue
                if ch in ('q', 'Q'):
                    print("[Keys] Quit requested")
                    self.running = False
                    break
                elif ch in ('s', 'S'):
                    self.start_requested = True
                    self.paused = False
                    print("[Keys] Start routine")
                elif ch == ' ':
                    # Emergency stop/pause: neutral throttle, keep routine paused
                    self.paused = True
                    print("[Keys] EMERGENCY STOP: neutral throttle and paused")
                    for _ in range(8):
                        self.drv.stop()
                        time.sleep(0.1)
                elif ch in ('r', 'R'):
                    # Reset routine from start; remains paused until 's'
                    self.reset_requested = True
                    self.paused = True
                    self.start_requested = False
                    print("[Keys] Reset requested; routine reset and paused (press 's' to start)")
                elif ch in ('c', 'C'):
                    ctr = self.drv.center_steering()
                    print(f"[Keys] Center steering -> PWM {ctr}")

        self.stop()

    def run_routine(self):
        cfg = self.cfg
        forward_pwm = cfg["THROTTLE_FORWARD_PWM"]
        stop_pwm = cfg["THROTTLE_STOPPED_PWM"]
        right_pwm = cfg["STEERING_RIGHT_PWM"]
        center_pwm = int(round((cfg["STEERING_LEFT_PWM"] + cfg["STEERING_RIGHT_PWM"]) / 2))

        loops_done = 0

        while self.running:
            # Wait for explicit start
            while self.running and (self.paused or not self.start_requested):
                self.drv.set_throttle_pwm(stop_pwm)
                time.sleep(0.05)
            if not self.running:
                break

            # If reset came in, clear counters and continue waiting if still paused
            if self.reset_requested:
                loops_done = 0
                self.reset_requested = False
                print("[Routine] Reset: ready to start from loop 1")

            # Execute loops
            while self.running and self.start_requested and not self.paused and loops_done < LOOPS:
                # Check reset during run
                if self.reset_requested:
                    loops_done = 0
                    self.reset_requested = False
                    self.paused = True
                    self.start_requested = False
                    print("[Routine] Reset during run: paused, press 's' to start again")
                    break

                # Step 1: Forward FORWARD_SEC
                print(f"[Routine] Loop {loops_done+1}/{LOOPS}: Forward {FORWARD_SEC:.1f}s")
                self.drv.set_steering_pwm(center_pwm)
                t_end = time.time() + FORWARD_SEC
                while self.running and time.time() < t_end:
                    if self.paused or not self.start_requested or self.reset_requested:
                        break
                    self.drv.set_throttle_pwm(forward_pwm)
                    time.sleep(0.02)
                self.drv.set_throttle_pwm(stop_pwm)

                if self.paused or not self.start_requested or self.reset_requested or not self.running:
                    continue

                # Step 2: Right turn RIGHT_SEC (hold throttle forward while steering right)
                print(f"[Routine] Loop {loops_done+1}/{LOOPS}: Right turn {RIGHT_SEC:.1f}s")
                self.drv.set_steering_pwm(right_pwm)
                t_end = time.time() + RIGHT_SEC
                while self.running and time.time() < t_end:
                    if self.paused or not self.start_requested or self.reset_requested:
                        break
                    self.drv.set_throttle_pwm(forward_pwm)
                    time.sleep(0.02)
                self.drv.set_throttle_pwm(stop_pwm)

                # Increment loop count if not interrupted
                if self.running and not self.paused and self.start_requested and not self.reset_requested:
                    loops_done += 1

            # If finished all loops, stop and require a new start
            if loops_done >= LOOPS:
                print("[Routine] Completed all loops. Pausing at neutral. Press 's' to run again or 'q' to quit.")
                self.drv.set_throttle_pwm(stop_pwm)
                self.drv.set_steering_pwm(center_pwm)
                # Prepare for next start
                loops_done = 0
                self.paused = True
                self.start_requested = False

        # Final cleanup
        print("[Routine] Exiting; neutral and centering")
        self.drv.set_throttle_pwm(stop_pwm)
        self.drv.set_steering_pwm(center_pwm)

# ------------------------------
# Main
# ------------------------------
def main():
    driver = TimedDriver(CFG)
    try:
        driver.start()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C; stopping...")
    finally:
        driver.stop()
        print("[Main] Stopped cleanly.")

if __name__ == "__main__":
    main()
