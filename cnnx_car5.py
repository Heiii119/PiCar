#!/usr/bin/env python3

import time
import threading
import numpy as np
import cv2
import cv2.dnn

from flask import Flask, Response, jsonify, request, redirect
from picamera2 import Picamera2
from libcamera import Transform


# =========================
# CONFIG
# =========================

CLASS_NAMES = ["background", "stop", "person", "slow", "Uturn", "go"]
CONF_THRESHOLD = 0.75

MODE = "MANUAL"
AUTOPILOT_RUNNING = False
E_STOP = False

CURRENT_LABEL = "MANUAL | none"

THROTTLE_FORWARD = 410
THROTTLE_SLOW = 400
THROTTLE_STOPPED = 393
THROTTLE_REVERSE = 370

STEERING_MIN = 280
STEERING_MAX = 480
STEERING_CENTER = 380

values = {
    "throttle": THROTTLE_STOPPED,
    "steering": STEERING_CENTER
}

# =========================
# CAMERA
# =========================

picam = Picamera2()
picam.configure(
    picam.create_video_configuration(
        main={"size": (320,240), "format":"XRGB8888"},
        transform=Transform()
    )
)
picam.start()

_latest_frame = None
_latest_jpeg = None
_latest_lock = threading.Lock()


# =========================
# LOAD MODEL
# =========================

net = cv2.dnn.readNetFromONNX("model.onnx")


# =========================
# U-TURN STATE MACHINE
# =========================

uturn_active = False
uturn_stage = 0
uturn_start = 0

def start_uturn():
    global uturn_active, uturn_stage, uturn_start
    uturn_active = True
    uturn_stage = 0
    uturn_start = time.time()

def update_uturn():
    global uturn_active, uturn_stage, uturn_start

    if not uturn_active:
        return False

    elapsed = time.time() - uturn_start

    if uturn_stage == 0:
        values["throttle"] = THROTTLE_SLOW
        values["steering"] = STEERING_MAX
        if elapsed > 0.5:
            uturn_stage = 1
            uturn_start = time.time()

    elif uturn_stage == 1:
        values["throttle"] = THROTTLE_FORWARD
        values["steering"] = STEERING_MAX
        if elapsed > 3:
            uturn_stage = 2
            uturn_start = time.time()

    elif uturn_stage == 2:
        values["throttle"] = THROTTLE_REVERSE - 5
        values["steering"] = STEERING_MIN
        if elapsed > 3:
            uturn_stage = 3
            uturn_start = time.time()

    elif uturn_stage == 3:
        values["steering"] = STEERING_CENTER
        if elapsed > 0.3:
            uturn_active = False

    return True


# =========================
# LINE FOLLOW
# =========================

def line_follow(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    roi = thresh[int(h*0.6):h, :]
    M = cv2.moments(roi)

    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        error = cx - w//2
        steer = STEERING_CENTER + int(error * 0.6)
    else:
        steer = STEERING_CENTER

    steer = np.clip(steer, STEERING_MIN, STEERING_MAX)

    values["steering"] = steer
    values["throttle"] = THROTTLE_FORWARD


# =========================
# AUTOPILOT LOOP
# =========================

def autopilot_loop():
    global CURRENT_LABEL, _latest_frame, _latest_jpeg

    while True:

        frame = picam.capture_array()[..., :3]
        _latest_frame = frame

        if MODE == "AUTOPILOT" and AUTOPILOT_RUNNING and not E_STOP:

            if update_uturn():
                CURRENT_LABEL = "AUTOPILOT | Uturn"
            else:

                # Always line follow
                line_follow(frame)

                # Sign detection
                img = cv2.resize(frame, (224,224))
                blob = cv2.dnn.blobFromImage(img, 1/255.0, (224,224))
                blob = blob.transpose(0,2,3,1)

                net.setInput(blob)
                output = net.forward()[0]

                class_id = int(np.argmax(output))
                confidence = float(output[class_id])
                label = CLASS_NAMES[class_id]

                if confidence > CONF_THRESHOLD:

                    if label in ["stop","person"]:
                        values["throttle"] = THROTTLE_STOPPED

                    elif label == "go":
                        values["throttle"] = THROTTLE_FORWARD

                    elif label == "slow":
                        values["throttle"] = THROTTLE_SLOW

                    elif label == "Uturn":
                        start_uturn()

                    CURRENT_LABEL = f"AUTOPILOT | {label}"

                else:
                    CURRENT_LABEL = "AUTOPILOT | none"

        else:
            CURRENT_LABEL = "MANUAL | none"

        # Encode frame
        cv2.putText(frame, f"Label: {CURRENT_LABEL}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        ret, enc = cv2.imencode(".jpg", frame)
        if ret:
            with _latest_lock:
                _latest_jpeg = enc.tobytes()


threading.Thread(target=autopilot_loop, daemon=True).start()


# =========================
# FLASK
# =========================

app = Flask(__name__)

@app.route("/")
def index():
    return """
    <h1>PiCar ONNX</h1>
    <img src='/video'>
    <form method='post' action='/cmd'>
        <button name='a' value='auto'>AUTO</button>
        <button name='a' value='manual'>MANUAL</button>
    </form>
    """

@app.route("/cmd", methods=["GET", "POST"])
def cmd():
    global MODE
    if request.method == "POST":
        if request.form.get("a") == "auto":
            MODE = "AUTOPILOT"
        else:
            MODE = "MANUAL"
            values["throttle"] = THROTTLE_STOPPED
    return redirect("/")

@app.route("/status")
def status():
    return jsonify({
        "throttle": values["throttle"],
        "steering": values["steering"],
        "mode": MODE,
        "label": CURRENT_LABEL
    })

@app.route("/arrow", methods=["POST"])
def arrow():
    direction = request.json["dir"]

    if direction == "up":
        values["throttle"] = THROTTLE_FORWARD
    elif direction == "down":
        values["throttle"] = THROTTLE_REVERSE
    elif direction == "left":
        values["steering"] = STEERING_MIN
    elif direction == "right":
        values["steering"] = STEERING_MAX
    elif direction == "stop":
        values["throttle"] = THROTTLE_STOPPED
        values["steering"] = STEERING_CENTER

    return "OK"

@app.route("/mode")
def toggle_mode():
    global MODE
    MODE = "AUTOPILOT" if MODE=="MANUAL" else "MANUAL"
    return "OK"

@app.route("/autopilot/start")
def auto_start():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = True
    return "STARTED"

@app.route("/autopilot/pause")
def auto_pause():
    global AUTOPILOT_RUNNING
    AUTOPILOT_RUNNING = False
    return "PAUSED"

@app.route("/estop")
def estop():
    global E_STOP
    E_STOP = True
    values["throttle"] = THROTTLE_STOPPED
    return "STOPPED"

@app.route("/video")
def video():
    def gen():
        while True:
            with _latest_lock:
                if _latest_jpeg is None:
                    continue
                frame = _latest_jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame + b'\r\n')

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
