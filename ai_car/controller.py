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
THROTTLE_STOPPED = 375
THROTTLE_REVERSE = 300

STEERING_MIN = 280
STEERING_MAX = 480
STEERING_CENTER = 380

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
        self.autopilot_enabled = False
        self.current_mode = MODE_LINE

        # ---- PWM State ----
        self.throttle = THROTTLE_STOPPED
        self.steering = STEERING_CENTER

        # Manual slider speed
        self.manual_speed_pwm = THROTTLE_FORWARD

        # Control parameters
        self.deadband = 10

        # U-turn state machine
        self.mode_timer = 0.0
        self.uturn_stage = 0

        # Setup PWM driver
        self.pwm = None
        self._init_pwm()

        self._apply_pwm(self.throttle, self.steering)

    # =========================
    # PWM INIT
    # =========================
    def _init_pwm(self):
        try:
            import Adafruit_PCA9685

            self.pwm = Adafruit_PCA9685.PCA9685(
                address=PCA9685_ADDR,
                busnum=1
            )

            self.pwm.set_pwm_freq(PCA9685_FREQ)

            print("✅ PCA9685 initialized")

        except Exception as e:
            import traceback
            print("❌ PWM INIT FAILED")
            traceback.print_exc()
            self.pwm = None

    # =========================
    # APPLY PWM
    # =========================
    def _apply_pwm(self, throttle, steering):

        if self.pwm is None:
            return

        throttle = int(max(0, min(4095, throttle)))
        steering = int(max(0, min(4095, steering)))

        self.pwm.set_pwm(THROTTLE_CHANNEL, 0, throttle)
        self.pwm.set_pwm(STEERING_CHANNEL, 0, steering)

    # =========================
    # MANUAL SPEED
    # =========================
    def set_manual_speed(self, pwm):
        self.manual_speed_pwm = int(
            max(THROTTLE_REVERSE, min(THROTTLE_FORWARD, pwm))
        )

    # =========================
    # MAIN UPDATE
    # =========================
    def update(self, offset, sign_label=None, confidence=0.0):

        if not self.autopilot_enabled:
            return

        self._apply_sign_mode(sign_label)
        self._compute_and_drive_discrete(offset)

    # =========================
    # MANUAL CONTROL
    # =========================
    def manual_key(self, key):

        self.autopilot_enabled = False

        if key == "up":
            self.throttle = int(self.manual_speed_pwm)

        elif key == "down":
            self.throttle = THROTTLE_REVERSE

        elif key == "left":
            self.steering -= STEERING_STEP

        elif key == "right":
            self.steering += STEERING_STEP

        elif key == "stop":
            self.throttle = THROTTLE_STOPPED
            self.steering = STEERING_CENTER

        self._clamp_manual()
        self._apply_pwm(self.throttle, self.steering)

    def enable_autopilot(self):
        self.autopilot_enabled = True
        self.current_mode = MODE_LINE

    def _clamp_manual(self):
        self.throttle = max(THROTTLE_REVERSE, min(THROTTLE_FORWARD, self.throttle))
        self.steering = max(STEERING_MIN, min(STEERING_MAX, self.steering))

    # =========================
    # SIGN HANDLING
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
    # AUTOPILOT LOGIC
    # =========================
    def _compute_and_drive_discrete(self, offset):

        now = time.time()

        # =========================
        # STOP MODE
        # =========================
        if self.current_mode == MODE_STOP:
            self.throttle = THROTTLE_STOPPED
            self.steering = STEERING_CENTER

            if now - self.mode_timer > 2.0:
                self.current_mode = MODE_LINE

        # =========================
        # SLOW MODE
        # =========================
        elif self.current_mode == MODE_SLOW:
            self.throttle = THROTTLE_SLOW

        # =========================
        # GO MODE
        # =========================
        elif self.current_mode == MODE_GO:
            self.throttle = THROTTLE_FORWARD

        # =========================
        # ✅ ADVANCED U-TURN MODE
        # =========================
        elif self.current_mode == MODE_UTURN:

            # Stage 0 → Stop 0.2 sec
            if self.uturn_stage == 0:
                self.throttle = THROTTLE_STOPPED
                self.steering = STEERING_MAX
        
                if now - self.mode_timer > 0.2:
                    self.uturn_stage = 1
                    self.mode_timer = now
        
            # Stage 1 → Right + Forward (3 sec)
            elif self.uturn_stage == 1:
                self.throttle = THROTTLE_FORWARD
                self.steering = STEERING_MAX
        
                if now - self.mode_timer > 3.0:
                    self.uturn_stage = 2
                    self.mode_timer = now
        
            # Stage 2 → Stop 0.2 sec
            elif self.uturn_stage == 2:
                self.throttle = THROTTLE_STOPPED
                self.steering = STEERING_MIN
        
                if now - self.mode_timer > 0.2:   # ✅ fixed
                    self.uturn_stage = 3
                    self.mode_timer = now
        
            # Stage 3 → Left + Backward (3 sec)
            elif self.uturn_stage == 3:
                self.throttle = THROTTLE_REVERSE
                self.steering = STEERING_MIN
        
                if now - self.mode_timer > 3.0:   # ✅ fixed
                    self.current_mode = MODE_LINE
                    self.uturn_stage = 0
                    self.uturn_cooldown_timer = now
                    print("✅ U‑Turn Complete")
        
            self._apply_pwm(self.throttle, self.steering)
            return # IMPORTANT → skip normal steering logic

        # =========================
        # NORMAL LINE MODE
        # =========================
        else:
            self.throttle = THROTTLE_SLOW

        # =========================
        # Steering Correction
        # =========================
        if offset is None:
            self.steering = STEERING_CENTER

        elif abs(offset) < self.deadband:
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
