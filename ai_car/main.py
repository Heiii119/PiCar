# main.py

import cv2
import threading
import time
from flask import Flask, Response, jsonify, request

from line_detection import LineDetector
from sign_detection import SignDetector
from controller import RobotController


# =========================
# INITIALIZE COMPONENTS
# =========================
line_detector = LineDetector()
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


# =========================
# MAIN LOOP THREAD (10 FPS)
# =========================
def robot_loop():
    global latest_debug, latest_offset, latest_sign, latest_conf

    while running:

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

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


# ✅ LINE CALIBRATION
@app.route("/calibrate_line", methods=["POST"])
def calibrate_line():
    line_detector.calibrate_center()
    return jsonify({"ok": True})


# =========================
# WEB INTERFACE
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

html, body {
    touch-action: manipulation;
}

body {
    font-family: Arial, sans-serif;
    text-align: center;
    background-color: #f2f2f2;
    margin: 0;
    padding: 10px;

    user-select: none;
    -webkit-user-select: none;
    -ms-user-select: none;
}

img {
    width: 100%;
    max-width: 500px;
    border: 3px solid #333;
    border-radius: 10px;
}

.mode-btn {
    padding: 15px 25px;
    font-size: 18px;
    margin: 10px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    width: 180px;
}

.active-auto { background-color: #28a745; color: white; }
.active-manual { background-color: #007bff; color: white; }
.inactive { background-color: #ccc; }

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

<br>

<button id="autoBtn" class="mode-btn inactive" onclick="enableAutopilot()">Autopilot</button>
<button id="manualBtn" class="mode-btn active-manual" onclick="enableManual()">Manual</button>

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
<input type="range" min="370" max="420" value="410"
       oninput="updateSpeed(this.value)">
<div>PWM: <span id="speedValue">410</span></div>

<button class="calibrate-btn" onclick="calibrateLine()">Calibrate Line</button>

<div style="margin-top:20px;">
Mode: <span id="mode"></span> |
Autopilot: <span id="auto"></span> |
Sign: <span id="sign"></span>
</div>

<script>
// ✅ Disable pinch zoom
document.addEventListener('gesturestart', function (e) {
    e.preventDefault();
});

// ✅ Disable double tap zoom
let lastTouchEnd = 0;
document.addEventListener('touchend', function (event) {
    const now = (new Date()).getTime();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, false);

// ✅ Prevent ctrl + wheel zoom (desktop)
document.addEventListener('wheel', function(e){
    if(e.ctrlKey){
        e.preventDefault();
    }
}, { passive: false });

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

    autoBtn.classList.remove("inactive", "active-manual");
    autoBtn.classList.add("active-auto");

    manualBtn.classList.remove("active-manual", "active-auto");
    manualBtn.classList.add("inactive");
}


function enableManual(){

    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:false})
    });

    manualBtn.classList.remove("inactive", "active-auto");
    manualBtn.classList.add("active-manual");

    autoBtn.classList.remove("active-auto", "active-manual");
    autoBtn.classList.add("inactive");
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


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        shutdown()
