#!/usr/bin/env python3
# Remote-drive TT02 RC car over TCP with a simple GUI.
# - The program runs a TCP server on 192.168.50.149:6666 (change HOST if needed)
# - Connect from a phone/other device via a browser-like page (served by this app) or any TCP client sending commands
# - GUI buttons for local control + a "Start Preview" button to open/close the camera window
# - Keyboard control also supported when GUI window is focused (WASD/Arrows, Space stop, C center, Q quit)
#
# Network protocol (text, newline-terminated commands):
#   THROTTLE FWD
#   THROTTLE REV
#   THROTTLE STOP
#   STEER LEFT
#   STEER RIGHT
#   STEER CENTER
#   QUIT
#
# A very simple HTTP page is also served on the same port that opens a WebSocket-like TCP (raw) connection
# using JavaScript fetch fallback (long-poll + POST). If you prefer, use a TCP client app to send the commands above.
#
# Requirements:
#   - pip install Adafruit-PCA9685 opencv-python
#   - I2C enabled; PCA9685 on I2C bus 1 at address 0x40
#!/usr/bin/env python3
# TT02 Remote-drive server with embedded HTML UI
# - Serves UI at / and /index.html
# - Command POSTs at /cmd?c=...
# - Preview toggle at /preview/toggle
# - Health at /health
# - Listens on 0.0.0.0:6666 by default

import sys
import time
import threading
import socket
import queue
import signal
import termios
import tty
import select
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

# ==========================
# PCA9685 section
# ==========================
PCA9685_I2C_ADDR   = 0x40
I2C_BUSNUM         = 1
PCA9685_FREQUENCY  = 60

THROTTLE_CHANNEL   = 0
STEERING_CHANNEL   = 1

# PWM values
THROTTLE_FORWARD_PWM  = 400
THROTTLE_STOPPED_PWM  = 370
THROTTLE_REVERSE_PWM  = 220

STEERING_LEFT_PWM     = 470
STEERING_RIGHT_PWM    = 270

SWITCH_PAUSE_S = 0.06

try:
    import Adafruit_PCA9685 as LegacyPCA9685
except ImportError:
    sys.exit("Missing Adafruit-PCA9685. Run: pip install Adafruit-PCA9685")

def steering_center_pwm():
    return int(round((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2.0))

def clamp12(x):
    return max(0, min(4095, int(x)))

class PWM:
    def __init__(self, address, busnum, freq_hz):
        self.dev = LegacyPCA9685.PCA9685(address=address, busnum=busnum)
        self.dev.set_pwm_freq(freq_hz)

    def set(self, channel, value_12bit):
        v = clamp12(value_12bit)
        self.dev.set_pwm(channel, 0, v)

# ==========================
# Car control core
# ==========================
class CarController:
    def __init__(self):
        self.pwm = PWM(PCA9685_I2C_ADDR, I2C_BUSNUM, PCA9685_FREQUENCY)
        self.last_throttle = THROTTLE_STOPPED_PWM
        self.lock = threading.Lock()
        self.neutral_all()

    def neutral_all(self):
        with self.lock:
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.pwm.set(STEERING_CHANNEL, steering_center_pwm())
            self.last_throttle = THROTTLE_STOPPED_PWM

    def throttle_forward(self):
        with self.lock:
            if self.last_throttle == THROTTLE_REVERSE_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_FORWARD_PWM)
            self.last_throttle = THROTTLE_FORWARD_PWM

    def throttle_reverse(self):
        with self.lock:
            if self.last_throttle == THROTTLE_FORWARD_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_REVERSE_PWM)
            self.last_throttle = THROTTLE_REVERSE_PWM

    def throttle_stop(self):
        with self.lock:
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.last_throttle = THROTTLE_STOPPED_PWM

    def steer_left(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)

    def steer_right(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)

    def steer_center(self):
        with self.lock:
            self.pwm.set(STEERING_CHANNEL, steering_center_pwm())

# ==========================
# Camera preview manager
# ==========================
class PreviewManager:
    def __init__(self, title="TT02 Camera", index=0, w=640, h=480):
        self.title = title
        self.index = index
        self.w = w
        self.h = h
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.running:
                return
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                print("Warning: Could not open camera.")
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while self.running:
            if self.cap:
                ok, frame = self.cap.read()
                if ok:
                    cv2.imshow(self.title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None

# ==========================
# Networking: simple HTTP + command POST
# ==========================
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 6666

car = CarController()
preview = PreviewManager()

HTML_PAGE = """<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TT02 Remote Drive</title>
<style>
body { font-family: sans-serif; margin: 16px; }
.grid { display: grid; grid-template-columns: repeat(3, 100px); grid-gap: 12px; justify-content: center; }
button { font-size: 18px; padding: 12px; }
.row { display:flex; gap:10px; justify-content:center; margin-top:12px; flex-wrap:wrap; }
.big { width: 120px; }
</style>
</head>
<body>
<h2>TT02 Remote Drive</h2>
<p>Tap and hold arrows for momentary control. Release returns to STOP/CENTER.</p>

<div class="row">
  <button id="previewBtn" class="big">Start Preview</button>
  <button id="stopBtn" class="big">STOP</button>
  <button id="centerBtn" class="big">CENTER</button>
</div>

<div class="grid" style="margin-top:20px">
  <div></div>
  <button id="up">Up</button>
  <div></div>

  <button id="left">Left</button>
  <div></div>
  <button id="right">Right</button>

  <div></div>
  <button id="down">Down</button>
  <div></div>
</div>

<script>
const base = '';
const send = (cmd) => fetch(base + '/cmd?c=' + encodeURIComponent(cmd), {method:'POST'});

const holdBehavior = (id, pressCmd, releaseCmd) => {
  const el = document.getElementById(id);
  let touching = false;

  const press = (e) => { e.preventDefault(); if (!touching) { touching = true; send(pressCmd); } };
  const release = (e) => { e.preventDefault(); if (touching) { touching = false; send(releaseCmd); } };

  el.addEventListener('mousedown', press);
  el.addEventListener('mouseup', release);
  el.addEventListener('mouseleave', release);

  el.addEventListener('touchstart', press, {passive:false});
  el.addEventListener('touchend', release);
  el.addEventListener('touchcancel', release);
};

document.getElementById('previewBtn').addEventListener('click', async () => {
  const r = await fetch(base + '/preview/toggle', {method:'POST'});
  const t = await r.text();
  document.getElementById('previewBtn').innerText = t.includes('running') ? 'Stop Preview' : 'Start Preview';
});

document.getElementById('stopBtn').addEventListener('click', () => send('THROTTLE STOP'));
document.getElementById('centerBtn').addEventListener('click', () => send('STEER CENTER'));

holdBehavior('up', 'THROTTLE FWD', 'THROTTLE STOP');
holdBehavior('down', 'THROTTLE REV', 'THROTTLE STOP');
holdBehavior('left', 'STEER LEFT', 'STEER CENTER');
holdBehavior('right', 'STEER RIGHT', 'STEER CENTER');
</script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send_html(HTML_PAGE)
        elif path == '/health':
            self._send_text('ok')
        else:
            self._send_text("Not found", code=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/cmd':
            qs = parse_qs(parsed.query)
            cmd = (qs.get('c', [''])[0] or '').strip().upper()
            ok, msg = self._apply_command(cmd)
            self._send_text("OK" if ok else f"ERR: {msg}", code=200 if ok else 400)
        elif parsed.path == '/preview/toggle':
            running = preview.running
            if running:
                preview.stop()
                self._send_text("preview stopped")
            else:
                preview.start()
                time.sleep(0.2)
                self._send_text("preview running")
        else:
            self._send_text("Not found", code=404)

    def _apply_command(self, cmd: str):
        try:
            if cmd == 'THROTTLE FWD':
                car.throttle_forward()
            elif cmd == 'THROTTLE REV':
                car.throttle_reverse()
            elif cmd == 'THROTTLE STOP':
                car.throttle_stop()
            elif cmd == 'STEER LEFT':
                car.steer_left()
            elif cmd == 'STEER RIGHT':
                car.steer_right()
            elif cmd == 'STEER CENTER':
                car.steer_center()
            elif cmd == 'QUIT':
                car.neutral_all()
            elif cmd == '':
                return False, "empty"
            else:
                return False, f"unknown: {cmd}"
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def _send_html(self, html):
        data = html.encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, code=200):
        data = text.encode('utf-8')
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return  # quieter

# ==========================
# Local keyboard UI (optional)
# ==========================
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll_all(self):
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
               
