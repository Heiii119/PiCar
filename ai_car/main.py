# main.py

import cv2
import threading
import time
import subprocess
from flask import Flask, Response, jsonify, request

from line_detection import LineDetector
from sign_detection import SignDetector
from controller import RobotController


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
line_detector.set_mode("hsv_distance")  # or "red_bgr"

sign_detector = SignDetector("model.onnx")
controller = RobotController()

# ✅ Start in MANUAL mode
controller.autopilot_enabled = False

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

running = True

latest_debug = None
latest_offset = 0
latest_sign = None
latest_conf = 0.0
latest_frame = None  # needed for calibration


# =========================
# MAIN LOOP THREAD (10 FPS)
# =========================
def robot_loop():
    global latest_debug, latest_offset, latest_sign, latest_conf, latest_frame

    while running:

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Save frame for calibration
        latest_frame = frame.copy()

        # ---- Line detection ----
        offset, debug_frame = line_detector.process(frame)

        # ---- Sign detection ----
        label, conf = sign_detector.detect(frame)

        # ---- Controller update ----
        controller.update(offset, label, conf)

        # ---- Store for web ----
        latest_debug = debug_frame
        latest_offset = offset
        latest_sign = label
        latest_conf = conf

        time.sleep(0.1)


threading.Thread(target=robot_loop, daemon=True).start()


# =========================
# VIDEO STREAM
# =========================
def generate_stream():
    while True:

        if latest_debug is None:
            time.sleep(0.01)
            continue

        ret, jpeg = cv2.imencode(".jpg", latest_debug)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")


@app.route("/video")
def video():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# =========================
# STATUS API
# =========================
@app.route("/status")
def status():
    return jsonify({
        "offset": latest_offset,
        "sign": latest_sign,
        "confidence": round(latest_conf, 3),
        "mode": controller.current_mode,
        "autopilot": controller.autopilot_enabled
    })


# =========================
# CONTROL ENDPOINTS
# =========================
@app.route("/manual", methods=["POST"])
def manual():
    key = request.json.get("key")
    controller.manual_key(key)
    return jsonify({"ok": True})


@app.route("/manual_speed", methods=["POST"])
def manual_speed():
    pwm = request.json.get("pwm")
    controller.set_manual_speed(pwm)
    return jsonify({"ok": True})


@app.route("/autopilot", methods=["POST"])
def autopilot():
    enabled = request.json.get("enabled", True)
    if enabled:
        controller.enable_autopilot()
    else:
        controller.autopilot_enabled = False
    return jsonify({"ok": True})


# =========================
# LINE CALIBRATION
# =========================
@app.route("/calibrate_line", methods=["POST"])
def calibrate_line():
    global latest_frame

    if latest_frame is None:
        return jsonify({"ok": False})

    if line_detector.mode == "hsv_distance":
        line_detector.calibrate_color(latest_frame)

    line_detector.calibrate_center()

    print("✅ Line fully calibrated")
    return jsonify({"ok": True})


# =========================
# WEB INTERFACE
# =========================
def get_web_interface():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Car Control</title>
<style>
body { font-family: Arial; text-align: center; background: #f2f2f2; margin: 0; padding: 10px; }
img { width: 100%; max-width: 500px; border: 3px solid #333; border-radius: 10px; }
button { padding: 12px 20px; margin: 10px; font-size: 16px; border-radius: 8px; border: none; cursor: pointer; }
.calibrate-btn { background: orange; color: white; }
</style>
</head>
<body>

<h1>🚗 Autonomous Car Control</h1>
<img src="/video">

<div>
Mode: <span id="mode"></span> |
Autopilot: <span id="auto"></span> |
Sign: <span id="sign"></span>
</div>

<br>

<button onclick="enableAutopilot()">Autopilot</button>
<button onclick="enableManual()">Manual</button>
<button class="calibrate-btn" onclick="calibrateLine()">Calibrate Line</button>

<script>
function enableAutopilot(){
    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:true})
    });
}

function enableManual(){
    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:false})
    });
}

function calibrateLine(){
    if(confirm("Place car centered on the line before calibrating. Continue?")){
        fetch('/calibrate_line', { method:'POST' });
    }
}

async function updateStatus(){
    const res = await fetch('/status');
    const data = await res.json();
    document.getElementById('mode').textContent = data.mode;
    document.getElementById('auto').textContent = data.autopilot;
    document.getElementById('sign').textContent = data.sign;
}

setInterval(updateStatus, 500);
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return get_web_interface()


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
            print(f"Running on http://{ts_ip}:5000 for Tailscale ✅")
        else:
            print("⚠️ Tailscale IP not found")

        print("==============================\n")

        app.run(host="0.0.0.0", port=5000, threaded=True)

    finally:
        shutdown()
