# controller.py
import time

# =========================
# PCA9685 / PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

# ✅ YOUR TUNED VALUES
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

        # Manual slider speed
        self.manual_speed_pwm = THROTTLE_FORWARD

        # Control parameters
        self.deadband = 10

        # Uturn timers
        self.mode_timer = 0.0
        self.uturn_stage = 0

        # Setup PWM driver
        self.pwm = None
        self._init_pwm()

        self._apply_pwm(self.throttle, self.steering)

    # =========================
    # PWM INIT (LEGACY DRIVER)
    # =========================
    def _init_pwm(self):
        try:
            import Adafruit_PCA9685

            self.pwm = Adafruit_PCA9685.PCA9685(address=PCA9685_ADDR)
            self.pwm.set_pwm_freq(PCA9685_FREQ)

            print("✅ PCA9685 initialized (Legacy driver)")

        except Exception as e:
            import traceback
            print("❌ PWM INIT FAILED")
            traceback.print_exc()
            self.pwm = None

    # =========================
    # APPLY PWM (12-bit)
    # =========================
    def _apply_pwm(self, throttle, steering):

        if self.pwm is None:
            return

        throttle = int(max(0, min(4095, throttle)))
        steering = int(max(0, min(4095, steering)))

        # Legacy driver uses raw 12-bit ticks
        self.pwm.set_pwm(THROTTLE_CHANNEL, 0, throttle)
        self.pwm.set_pwm(STEERING_CHANNEL, 0, steering)

    # =========================
    # SET MANUAL SPEED
    # =========================
    def set_manual_speed(self, pwm):
        self.manual_speed_pwm = int(pwm)

    # =========================
    # MAIN UPDATE (Autopilot)
    # =========================
    def update(self, offset, sign_label=None, confidence=0.0):

        if self.autopilot_enabled:
            self._apply_sign_mode(sign_label)
            self._compute_and_drive_discrete(offset)

    # =========================
    # MANUAL CONTROL
    # =========================
    def manual_key(self, key):

        self.autopilot_enabled = False

        if key == "up":
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

        label = label.lower()

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
    # AUTOPILOT CONTROLLER
    # =========================
    def _compute_and_drive_discrete(self, offset):

        now = time.time()

        # STOP MODE
        if self.current_mode == MODE_STOP:
            self.throttle = THROTTLE_STOPPED
            if now - self.mode_timer > 2.0:
                self.current_mode = MODE_LINE

        # SLOW MODE
        elif self.current_mode == MODE_SLOW:
            self.throttle = THROTTLE_SLOW

        # GO MODE
        elif self.current_mode == MODE_GO:
            self.throttle = THROTTLE_FORWARD

        # UTURN MODE
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
                    self.current_mode = MODE_LINE

        # NORMAL LINE MODE
        else:
            self.throttle = THROTTLE_FORWARD

        # Steering logic
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
