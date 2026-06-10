from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

# Use video0 (change to 1 if needed)
camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Force MJPEG (important for Logitech webcams)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        success, img = camera.read()
        if not success:
            continue

        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            with lock:
                frame = buffer.tobytes()

thread = threading.Thread(target=capture_frames)
thread.daemon = True
thread.start()

def generate():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            output = frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')

@app.route('/')
def index():
    return '''
        <html>
        <head>
            <title>C922 Live Stream</title>
        </head>
        <body>
            <h1>Logitech C922 Live Stream</h1>
            <img src="/video_feed">
        </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
