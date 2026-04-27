# controller.py
import time

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

        # ✅ Manual speed from slider
        self.manual_speed_pwm = 410

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
            from smbus2 import SMBus
            from adafruit_pca9685 import PCA9685
        
            bus = SMBus(1)  # I2C bus 1
            self.pwm = PCA9685(bus, address=0x40)
            self.pwm.frequency = PCA9685_FREQ
        
            print("✅ PCA9685 initialized (SMBus mode)")
        
        except Exception as e:
            import traceback
            print("❌ PWM INIT FAILED")
            traceback.print_exc()
            self.pwm = None

        throttle = int(max(0, min(4095, throttle)))
        steering = int(max(0, min(4095, steering)))

        self.pwm.channels[THROTTLE_CHANNEL].duty_cycle = throttle
        self.pwm.channels[STEERING_CHANNEL].duty_cycle = steering

    # =========================
    # SET MANUAL SPEED (Slider)
    # =========================
    def set_manual_speed(self, pwm):
        self.manual_speed_pwm = int(pwm)

    # =========================
    # MAIN UPDATE
    # =========================
    def update(self, offset, sign_label=None, confidence=0.0):

        if self.autopilot_enabled:
            self._apply_sign_mode(sign_label)
            self._compute_and_drive_discrete(offset)
        # Manual mode does nothing here

    # =========================
    # MANUAL CONTROL
    # =========================
    def manual_key(self, key):

        self.autopilot_enabled = False

        if key == "up":
            # ✅ Use slider speed
            self.throttle = self.manual_speed_pwm

        elif key == "down":
            self.throttle = THROTTLE_REVERSE

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
    # SIGN MODE HANDLING
    # =========================
    def _apply_sign_mode(self, label):

        if label is None:
            return

        if label == "stop":
            self.current_mode = MODE_STOP
            self.mode_timer = time.time()

        elif label == "slow":
            self.current_mode = MODE_SLOW

        elif label == "Uturn":
            self.current_mode = MODE_UTURN
            self.mode_timer = time.time()
            self.uturn_stage = 0

        elif label == "go":
            self.current_mode = MODE_GO

    # =========================
    # AUTOPILOT CONTROLLER
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

            elif self.uturn_stage == 2:
                self.current_mode = MODE_LINE

        # ---- NORMAL LINE MODE ----
        else:
            self.throttle = THROTTLE_FORWARD

        # ---- Steering ----
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
