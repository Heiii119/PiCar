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
line_detector.set_mode("hsv_distance")

sign_detector = SignDetector("model.onnx")
controller = RobotController()
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
latest_frame = None


# =========================
# MAIN LOOP THREAD
# =========================
def robot_loop():
    global latest_debug, latest_offset, latest_sign, latest_conf, latest_frame

    while running:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        latest_frame = frame.copy()

        offset, debug_frame = line_detector.process(frame)
        label, conf = sign_detector.detect(frame)

        controller.update(offset, label, conf)

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
# ✅ FULL WEB INTERFACE (RESTORED)
# =========================
@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Autonomous Car Control</title>
<style>
html, body { touch-action: manipulation; }
body {
    font-family: Arial, sans-serif;
    text-align: center;
    background-color: #f2f2f2;
    margin: 0;
    padding: 10px;
    user-select: none;
}
img {
    width: 100%;
    max-width: 500px;
    border: 3px solid #333;
    border-radius: 10px;
}
.mode-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}
.mode-btn {
    padding: 15px 25px;
    font-size: 18px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    width: 160px;
    transition: 0.2s;
    background-color: #ccc;
}
.mode-btn.active-auto { background-color: #28a745; color: white; }
.mode-btn.active-manual { background-color: #007bff; color: white; }

.arrow-grid {
    margin-top: 30px;
    display: inline-grid;
    grid-template-columns: 120px 120px 120px;
    grid-template-rows: 120px 120px 120px;
    gap: 15px;
}
.arrow-btn {
    font-size: 40px;
    border-radius: 15px;
    border: none;
    background-color: #444;
    color: white;
    cursor: pointer;
}
.stop-btn {
    background-color: red;
    color: white;
    font-size: 26px;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    width: 260px;
    border: none;
}
input[type=range] {
    width: 80%;
    max-width: 400px;
}
.calibrate-btn {
    background-color: orange;
    color: white;
    font-size: 18px;
    padding: 12px 20px;
    border-radius: 10px;
    border: none;
    margin-top: 20px;
    cursor: pointer;
}
</style>
</head>
<body>

<h1>🚗 Autonomous Car Control</h1>
<img src="/video">

<div style="margin-top:20px;">
Mode: <span id="mode"></span> |
Autopilot: <span id="auto"></span> |
Sign: <span id="sign"></span>
</div>

<div class="mode-container">
    <button id="autoBtn" class="mode-btn active-auto" onclick="enableAutopilot()">Autopilot</button>
    <button id="manualBtn" class="mode-btn" onclick="enableManual()">Manual</button>
</div>

<div class="arrow-grid">
    <div></div>
    <button class="arrow-btn" onclick="sendKey('up')">↑</button>
    <div></div>

    <button class="arrow-btn" onclick="sendKey('left')">←</button>
    <button class="arrow-btn" onclick="sendKey('stop')">■</button>
    <button class="arrow-btn" onclick="sendKey('right')">→</button>

    <div></div>
    <button class="arrow-btn" onclick="sendKey('down')">↓</button>
    <div></div>
</div>

<button class="stop-btn" onclick="sendKey('stop')">EMERGENCY STOP</button>

<h3>Manual Speed (PWM)</h3>
<input type="range" min="370" max="420" value="410" oninput="updateSpeed(this.value)">
<div>PWM: <span id="speedValue">410</span></div>

<button class="calibrate-btn" onclick="calibrateLine()">Calibrate Line</button>

<script>
let autoBtn = document.getElementById("autoBtn");
let manualBtn = document.getElementById("manualBtn");

function sendKey(key){
    fetch('/manual', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({key:key})
    });
}

function updateSpeed(value){
    document.getElementById("speedValue").innerText = value;
    fetch('/manual_speed', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({pwm:value})
    });
}

function enableAutopilot(){
    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:true})
    });

    autoBtn.classList.add("active-auto");
    manualBtn.classList.remove("active-manual");
}

function enableManual(){
    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:false})
    });

    manualBtn.classList.add("active-manual");
    autoBtn.classList.remove("active-auto");
}

function calibrateLine(){
    fetch('/calibrate_line', { method:'POST' });
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

        print("==============================\n")

        app.run(host="0.0.0.0", port=5000, threaded=True)

    finally:
        shutdown()
