from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import threading

app = Flask(__name__)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480)}
))
picam2.start()

frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        img = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            with lock:
                frame = buffer.tobytes()

# Background capture thread
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
            <title>Raspberry Pi Live Stream</title>
        </head>
        <body>
            <h1>Live Camera Stream</h1>
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
