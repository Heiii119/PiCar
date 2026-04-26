import cv2
import numpy as np
import time
import threading

from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# ===== Coral =====
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ===== PCA9685 =====
import board
import busio
from adafruit_pca9685 import PCA9685

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "model.tflite"
CAMERA_INDEX = 0
PORT = 9090

STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1

STEERING_CENTER = 375  # Adjust for your servo
STEERING_LEFT = 300
STEERING_RIGHT = 450

THROTTLE_FORWARD = 400
THROTTLE_STOP = 350

# ==========================
# INIT HARDWARE
# ==========================

# I2C for PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Coral Interpreter
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# ==========================
# FLASK APP
# ==========================

app = Flask(__name__)
socketio = SocketIO(app)

camera = cv2.VideoCapture(CAMERA_INDEX)

# ==========================
# LINE FOLLOWING
# ==========================

def line_follow(frame):
    height, width, _ = frame.shape
    
    roi = frame[int(height*0.6):height, 0:width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            center_offset = cx - width/2

            if center_offset > 30:
                steer(STEERING_RIGHT)
            elif center_offset < -30:
                steer(STEERING_LEFT)
            else:
                steer(STEERING_CENTER)

            throttle(THROTTLE_FORWARD)

            cv2.drawContours(roi, [c], -1, (0,255,0), 2)

    return frame


# ==========================
# ROAD SIGN DETECTION
# ==========================

def detect_sign(frame):
    input_h, input_w = input_shape[1], input_shape[2]

    resized = cv2.resize(frame, (input_w, input_h))
    resized = np.expand_dims(resized, axis=0)

    common.set_input(interpreter, resized)
    interpreter.invoke()

    output = interpreter.get_output_details()[0]
    results = interpreter.get_tensor(output['index'])

    class_id = np.argmax(results)
    confidence = results[0][class_id]

    if confidence > 0.8:
        handle_sign(class_id)

        cv2.putText(frame, f"Sign: {class_id}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame


def handle_sign(class_id):
    # EDIT according to your label mapping
    if class_id == 0:  # STOP
        throttle(THROTTLE_STOP)
        time.sleep(2)

    elif class_id == 1:  # LEFT
        steer(STEERING_LEFT)
        time.sleep(0.5)

    elif class_id == 2:  # RIGHT
        steer(STEERING_RIGHT)
        time.sleep(0.5)


# ==========================
# MOTOR CONTROL
# ==========================

def steer(value):
    pca.channels[STEERING_CHANNEL].duty_cycle = value

def throttle(value):
    pca.channels[THROTTLE_CHANNEL].duty_cycle = value


# ==========================
# VIDEO STREAM
# ==========================

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = line_follow(frame)
        frame = detect_sign(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print("Starting server on port 9090...")
    throttle(THROTTLE_STOP)
    steer(STEERING_CENTER)
    socketio.run(app, host="0.0.0.0", port=PORT)
