import time
import cv2
from flask import Flask, Response, jsonify, request


def create_app(state, controller, line_detector):
    app = Flask(__name__)

    # =========================
    # VIDEO STREAM
    # =========================
    def generate_stream():
        while True:
            if state.latest_debug is None:
                time.sleep(0.01)
                continue

            ret, jpeg = cv2.imencode(".jpg", state.latest_debug)
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
            "offset": state.latest_offset,
            "sign": state.latest_sign,
            "confidence": round(state.latest_conf, 3),
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
        if state.latest_frame is None:
            return jsonify({"ok": False})

        if line_detector.mode == "hsv_distance":
            line_detector.calibrate_color(state.latest_frame)

        line_detector.calibrate_center()
        print("✅ Line fully calibrated")
        return jsonify({"ok": True})

    # =========================
    # WEB UI
    # =========================
    @app.route("/")
    def index():
        return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Car Control</title>
<style>
body {
    font-family: Arial, sans-serif;
    text-align: center;
    background-color: #f2f2f2;
    margin: 0;
    padding: 10px;
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
}

function enableManual(){
    fetch('/autopilot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({enabled:false})
    });
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

    return app
