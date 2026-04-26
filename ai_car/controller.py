# controller.py
import time
import math

# =========================
# PCA9685 / PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 1
DRIVER_PREFER = "smbus2"

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_FORWARD = 410
THROTTLE_SLOW = 400
THROTTLE_STOPPED = 393
THROTTLE_REVERSE = 370

STEERING_MIN = 280
STEERING_MAX = 480
STEERING_CENTER = 380

START_THROTTLE_TICKS = THROTTLE_STOPPED
START_STEERING_TICKS = STEERING_CENTER

STOP_ON_EXIT = True

STEP = 5
STEERING_STEP = 25

# =========================
# AUTOPILOT MODES
# =========================
MODE_LINE   = "LINE"
MODE_SLOW   = "SLOW"
MODE_STOP   = "STOP"
MODE_UTURN  = "UTURN"
MODE_GO     = "GO"

# =========================
# ROBOT CONTROLLER
# =========================
class RobotController:

    def __init__(self):

        # ---- Mode ----
        self.autopilot_enabled = True
        self.current_mode = MODE_LINE

        # ---- PWM State ----
        self.throttle = START_THROTTLE_TICKS
        self.steering = START_STEERING_TICKS

        # ---- Control parameters ----
        self.deadband = 10
        self.max_offset = 160

        # ---- Reverse / Uturn timers ----
        self.mode_timer = 0.0
        self.uturn_stage = 0

        # ---- Setup PWM driver ----
        self.pwm = None
        self._init_pwm()

        self._apply_pwm(self.throttle, self.steering)

    # =========================
    # PWM INIT
    # =========================
    def _init_pwm(self):
        try:
            if DRIVER_PREFER == "smbus2":
                from smbus2 import SMBus
                import adafruit_pca9685
                import board
                import busio

                i2c = busio.I2C(board.SCL, board.SDA)
                self.pwm = adafruit_pca9685.PCA9685(i2c)
                self.pwm.frequency = PCA9685_FREQ
            else:
                raise Exception("Unsupported driver")

        except Exception as e:
            print("PWM init failed:", e)
            self.pwm = None

    def _apply_pwm(self, throttle, steering):
        if self.pwm is None:
            return

        throttle = int(max(0, min(4095, throttle)))
        steering = int(max(0, min(4095, steering)))

        self.pwm.channels[THROTTLE_CHANNEL].duty_cycle = throttle
        self.pwm.channels[STEERING_CHANNEL].duty_cycle = steering

    # =========================
    # PUBLIC UPDATE (Main Entry)
    # =========================
    def update(self, offset, sign_label=None, confidence=0.0):
        """
        Called from main loop.
        offset: line center offset (pixels)
        sign_label: detected sign
        """

        if self.autopilot_enabled:
            self._apply_sign_mode(sign_label)
            self._compute_and_drive_discrete(offset)
        else:
            # Manual mode does not use offset
            pass

    # =========================
    # MANUAL CONTROL
    # =========================
    def manual_key(self, key):
        """
        Call this from web when arrow keys pressed.
        key: 'up', 'down', 'left', 'right', 'stop'
        """
        self.autopilot_enabled = False

        if key == "up":
            self.throttle += STEP
        elif key == "down":
            self.throttle -= STEP
        elif key == "left":
            self.steering -= STEERING_STEP
        elif key == "right":
            self.steering += STEERING_STEP
        elif key == "stop":
            self.throttle = THROTTLE_STOPPED

        self._clamp_manual()
        self._apply_pwm(self.throttle, self.steering)

    def enable_autopilot(self):
        self.autopilot_enabled = True
        self.current_mode = MODE_LINE

    def _clamp_manual(self):
        self.throttle = max(THROTTLE_REVERSE, min(THROTTLE_FORWARD, self.throttle))
        self.steering = max(STEERING_MIN, min(STEERING_MAX, self.steering))

    # =========================
    # APPLY SIGN MODE
    # =========================
    def _apply_sign_mode(self, label):

        if label is None:
            return

        if label == "stop":
            self.current_mode = MODE_STOP
            self.mode_timer = time.time()

        elif label == "slow":
            self.current_mode = MODE_SLOW

        elif label == "uturn":
            self.current_mode = MODE_UTURN
            self.mode_timer = time.time()
            self.uturn_stage = 0

        elif label == "go":
            self.current_mode = MODE_GO

    # =========================
    # DISCRETE CONTROLLER
    # =========================
    def _compute_and_drive_discrete(self, offset):

        now = time.time()

        # ---- STOP MODE ----
        if self.current_mode == MODE_STOP:
            self.throttle = THROTTLE_STOPPED
            if now - self.mode_timer > 2.0:
                self.current_mode = MODE_LINE

        # ---- SLOW MODE ----
        elif self.current_mode == MODE_SLOW:
            self.throttle = THROTTLE_SLOW

        # ---- GO MODE ----
        elif self.current_mode == MODE_GO:
            self.throttle = THROTTLE_FORWARD

        # ---- UTURN MODE ----
        elif self.current_mode == MODE_UTURN:
            if self.uturn_stage == 0:
                self.throttle = THROTTLE_STOPPED
                if now - self.mode_timer > 1.0:
                    self.uturn_stage = 1
                    self.mode_timer = now

            elif self.uturn_stage == 1:
                self.throttle = THROTTLE_REVERSE
                self.steering = STEERING_MAX
                if now - self.mode_timer > 2.0:
                    self.uturn_stage = 2
                    self.mode_timer = now

            elif self.uturn_stage == 2:
                self.current_mode = MODE_LINE

        # ---- NORMAL LINE MODE ----
        else:
            self.throttle = THROTTLE_FORWARD

        # ---- Steering Discrete Logic ----
        if abs(offset) < self.deadband:
            self.steering = STEERING_CENTER
        elif offset > 0:
            self.steering = STEERING_CENTER + min(80, offset)
        else:
            self.steering = STEERING_CENTER + max(-80, offset)

        self.steering = max(STEERING_MIN, min(STEERING_MAX, self.steering))

        self._apply_pwm(self.throttle, self.steering)

    # =========================
    # SAFE STOP
    # =========================
    def stop(self):
        self.throttle = THROTTLE_STOPPED
        self.steering = STEERING_CENTER
        self._apply_pwm(self.throttle, self.steering)

    def shutdown(self):
        if STOP_ON_EXIT:
            self.stop()
        if self.pwm:
            self.pwm.deinit()
