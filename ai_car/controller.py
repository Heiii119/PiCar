# controller.py
import time

# =========================
# PCA9685 / PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_FORWARD = 410
THROTTLE_SLOW = 404
THROTTLE_STOPPED = 375
THROTTLE_REVERSE = 300

STEERING_MIN = 280
STEERING_MAX = 480
STEERING_CENTER = 380

STOP_ON_EXIT = True
STEERING_STEP = 25

# =========================
# MODES
# =========================
MODE_LINE   = "LINE"
MODE_SLOW   = "SLOW"
MODE_STOP   = "STOP"
MODE_UTURN  = "UTURN"
MODE_GO     = "GO"


class RobotController:

    def __init__(self):

        self.autopilot_enabled = False
        self.current_mode = MODE_LINE

        self.throttle = THROTTLE_STOPPED
        self.steering = STEERING_CENTER

        self.manual_speed_pwm = THROTTLE_FORWARD
        self.deadband = 10

        # ✅ UTURN STATE
        self.mode_timer = 0.0
        self.uturn_stage = 0
        self.uturn_cooldown_timer = 0.0
        self.uturn_cooldown_time = 8.0   # seconds cooldown

        # ✅ LINE LOST PROTECTION
        self.line_lost_timer = None
        self.line_lost_timeout = 3.0   # seconds

        # PWM
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
        except Exception:
            import traceback
            print("❌ PWM INIT FAILED")
            traceback.print_exc()
            self.pwm = None

    def _apply_pwm(self, throttle, steering):

        if self.pwm is None:
            return

        throttle = int(max(0, min(4095, throttle)))
        steering = int(max(0, min(4095, steering)))

        self.pwm.set_pwm(THROTTLE_CHANNEL, 0, throttle)
        self.pwm.set_pwm(STEERING_CHANNEL, 0, steering)

    # =========================
    # UPDATE LOOP
    # =========================
    def update(self, offset, sign_label=None, confidence=0.0):

        if not self.autopilot_enabled:
            return

        self._apply_sign_mode(sign_label)
        self._compute_and_drive(offset)

    # =========================
    # SIGN HANDLER (SAFE)
    # =========================
    def _apply_sign_mode(self, label):

        if label is None:
            return

        label = label.lower()

        # ✅ Only accept new signs while in LINE mode
        if self.current_mode != MODE_LINE:
            return

        # STOP
        if label == "stop":
            self.current_mode = MODE_STOP
            self.mode_timer = time.time()

        # SLOW
        elif label == "slow":
            self.current_mode = MODE_SLOW

        # GO
        elif label == "go":
            self.current_mode = MODE_GO

        # ✅ UTURN (Protected)
        elif label == "uturn":

            # cooldown check
            if time.time() - self.uturn_cooldown_timer < self.uturn_cooldown_time:
                return

            self.current_mode = MODE_UTURN
            self.mode_timer = time.time()
            self.uturn_stage = 0

            print("↩ U‑Turn Triggered")

    # =========================
    # AUTOPILOT LOGIC
    # =========================
    def _compute_and_drive(self, offset):

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
        # ✅ UTURN MODE 
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

                if now - self.mode_timer > 0.2:
                    self.uturn_stage = 3
                    self.mode_timer = now

            # Stage 3 → Left + Backward (3 sec)
            elif self.uturn_stage == 3:
                self.throttle = THROTTLE_REVERSE
                self.steering = STEERING_MIN

                if now - self.mode_timer > 3.0:
                    self.current_mode = MODE_LINE
                    self.uturn_stage = 0
                    self.uturn_cooldown_timer = now
                    print("✅ U‑Turn Complete")

            self._apply_pwm(self.throttle, self.steering)
            return  # skip normal steering

        # =========================
        # NORMAL LINE MODE
        # =========================
        else:
            self.throttle = THROTTLE_SLOW

            # =========================
            # ✅ LINE LOST DETECTION
            # =========================
            if offset is None:
        
                # Start timer if first time lost
                if self.line_lost_timer is None:
                    self.line_lost_timer = time.time()
                    print("⚠ Line lost...")
        
                # If lost too long → disable autopilot
                elif time.time() - self.line_lost_timer > self.line_lost_timeout:
                    print("❌ Line lost for 3 seconds → Switching to MANUAL")
        
                    self.stop()
                    self.autopilot_enabled = False
                    self.line_lost_timer = None
                    return
        
            else:
                # Line detected again → reset timer
                self.line_lost_timer = None

        
        # =========================
        # STEERING CONTROL
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

        self.throttle = max(THROTTLE_REVERSE, min(THROTTLE_FORWARD, self.throttle))
        self.steering = max(STEERING_MIN, min(STEERING_MAX, self.steering))

        self._apply_pwm(self.throttle, self.steering)

    # =========================
    # ENABLE AUTOPILOT
    # =========================
    def enable_autopilot(self):
        self.autopilot_enabled = True
        self.current_mode = MODE_LINE

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
