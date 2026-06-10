# main.py

import cv2
import threading
import time
import subprocess

from line_detection import LineDetector
from sign_detection import SignDetector
from controller import RobotController
from web_interface import create_app


# =========================
# ✅ SHARED STATE OBJECT
# =========================
class RobotState:
    def __init__(self):
        self.latest_debug = None
        self.latest_offset = 0
        self.latest_sign = None
        self.latest_conf = 0.0
        self.latest_frame = None


# =========================
# ✅ TAILSCALE HELPER
# =========================
def get_tailscale_ip():
    try:
        result = subprocess.check_output(
            ["tailscale", "ip", "-4"],
            stderr=subprocess.DEVNULL
        )
        ip = result.decode().strip().split("\n")[0]
        return ip
    except Exception:
        return None


# =========================
# INITIALIZE COMPONENTS
# =========================
line_detector = LineDetector()
line_detector.set_mode("hsv_distance")

sign_detector = SignDetector("model.onnx")
# ✅ FORCE manual mode on startup
controller = RobotController()
controller.autopilot_enabled = False

state = RobotState()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

running = True


# =========================
# ✅ MAIN ROBOT LOOP
# =========================
def robot_loop():
    global running

    frame_count = 0
    cached_label = None
    cached_conf = 0.0

    while running:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        state.latest_frame = frame.copy()

        # ✅ Line detection every frame
        offset, debug_frame = line_detector.process(frame)

        # ✅ Sign detection every 5 frames (reduces lag)
        if frame_count % 5 == 0:
            cached_label, cached_conf = sign_detector.detect(frame)

        # ✅ Update controller using cached sign
        controller.update(offset, cached_label, cached_conf)

        # ✅ Update shared state
        state.latest_debug = debug_frame
        state.latest_offset = offset
        state.latest_sign = cached_label
        state.latest_conf = cached_conf

        # Small sleep to prevent CPU maxing out
        time.sleep(0.01)


# Start robot thread
threading.Thread(target=robot_loop, daemon=True).start()


# =========================
# ✅ CREATE FLASK APP
# =========================
app = create_app(state, controller, line_detector)


# =========================
# CLEAN SHUTDOWN
# =========================
def shutdown():
    global running
    running = False
    controller.shutdown()
    camera.release()
    cv2.destroyAllWindows()


# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    try:
        ts_ip = get_tailscale_ip()

        print("\n==============================")
        print("🚗 Autonomous Car Server Started")
        print("Local:      http://127.0.0.1:5000")

        if ts_ip:
            print(f"Tailscale:  http://{ts_ip}:5000")
            print("Running on Tailscale ✅")

        print("==============================\n")

        app.run(host="0.0.0.0", port=5000, threaded=True)

    finally:
        shutdown()
