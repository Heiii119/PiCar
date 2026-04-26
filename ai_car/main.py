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
sign_detector = SignDetector("model.onnx")  # inference interval handled inside
controller = RobotController()

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

running = True

latest_frame = None
latest_debug = None
latest_offset = 0
latest_sign = None
latest_conf = 0.0


# =========================
# MAIN LOOP THREAD (10 FPS)
# =========================
def robot_loop():
    global latest_frame, latest_debug
    global latest_offset, latest_sign, latest_conf

    while running:

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ---- Line detection ----
        offset, debug_frame = line_detector.process(frame)

        # ---- Sign detection (already runs every 4 frames internally) ----
        label, conf = sign_detector.detect(frame)

        # ---- Controller update ----
        controller.update(offset, label, conf)

        # ---- Store for web ----
        latest_frame = frame
        latest_debug = debug_frame
        latest_offset = offset
        latest_sign = label
        latest_conf = conf

        # ✅ Limit to ~10 FPS
        time.sleep(0.1)


# Start robot thread
threading.Thread(target=robot_loop, daemon=True).start()


# =========================
# VIDEO STREAM (Every 4th Frame)
# =========================
def generate_stream():
    global latest_debug

    stream_counter = 0

    while True:

        if latest_debug is None:
            time.sleep(0.01)
            continue

        stream_counter += 1

        # ✅ Stream only every 4th frame
        if stream_counter % 4 != 0:
            time.sleep(0.01)
            continue

        ret, jpeg = cv2.imencode(".jpg", latest_debug)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")


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


@app.route("/autopilot", methods=["POST"])
def autopilot():
    enabled = request.json.get("enabled", True)
    if enabled:
        controller.enable_autopilot()
    else:
        controller.autopilot_enabled = False
    return jsonify({"ok": True})


# =========================
# WEB INTERFACE
# =========================
@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Robot Control</title>
    </head>
    <body>
        <h1>Autonomous Car Control</h1>

        <img src="/video" width="480"/>

        <h3>Status</h3>
        <div>Offset: <span id="offset"></span></div>
        <div>Sign: <span id="sign"></span></div>
        <div>Confidence: <span id="conf"></span></div>
        <div>Mode: <span id="mode"></span></div>
        <div>Autopilot: <span id="auto"></span></div>

        <br>
        <button onclick="setAutopilot(true)">Enable Autopilot</button>
        <button onclick="setAutopilot(false)">Disable Autopilot</button>

        <script>
        async function updateStatus(){
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById('offset').textContent = data.offset;
            document.getElementById('sign').textContent = data.sign;
            document.getElementById('conf').textContent = data.confidence;
            document.getElementById('mode').textContent = data.mode;
            document.getElementById('auto').textContent = data.autopilot;
        }

        setInterval(updateStatus, 500);

        function sendKey(key){
            fetch('/manual', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({key:key})
            });
        }

        function setAutopilot(enabled){
            fetch('/autopilot', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({enabled:enabled})
            });
        }

        document.addEventListener('keydown', function(e){
            if(e.key === "ArrowUp") sendKey("up");
            if(e.key === "ArrowDown") sendKey("down");
            if(e.key === "ArrowLeft") sendKey("left");
            if(e.key === "ArrowRight") sendKey("right");
            if(e.key === " ") sendKey("stop");
        });
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
