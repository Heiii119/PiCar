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
from socketserver import ThreadingMixIn
from datetime import datetime, timedelta

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

STEERING_LEFT_PWM     = 510
STEERING_RIGHT_PWM    = 230

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
        self.command_queue = queue.Queue(maxsize=20)  # Command buffer for unstable connections
        self.last_command_time = datetime.now()
        self.command_timeout = timedelta(seconds=2)  # Auto-stop if no commands for 2 seconds
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.running = True
        self.neutral_all()
        self.watchdog_thread.start()

    def neutral_all(self):
        with self.lock:
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.pwm.set(STEERING_CHANNEL, steering_center_pwm())
            self.last_throttle = THROTTLE_STOPPED_PWM
            self.last_command_time = datetime.now()

    def _watchdog_loop(self):
        """Safety watchdog: auto-stop if no commands received for timeout period"""
        while self.running:
            time.sleep(0.5)
            if datetime.now() - self.last_command_time > self.command_timeout:
                # No commands received recently - auto-stop for safety
                with self.lock:
                    if self.last_throttle != THROTTLE_STOPPED_PWM:
                        self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                        self.last_throttle = THROTTLE_STOPPED_PWM
                        print("[Watchdog] Auto-stopped due to connection timeout")

    def shutdown(self):
        self.running = False
        self.neutral_all()

    def throttle_forward(self):
        with self.lock:
            self.last_command_time = datetime.now()
            if self.last_throttle == THROTTLE_REVERSE_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_FORWARD_PWM)
            self.last_throttle = THROTTLE_FORWARD_PWM

    def throttle_reverse(self):
        with self.lock:
            self.last_command_time = datetime.now()
            if self.last_throttle == THROTTLE_FORWARD_PWM:
                self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
                time.sleep(SWITCH_PAUSE_S)
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_REVERSE_PWM)
            self.last_throttle = THROTTLE_REVERSE_PWM

    def throttle_stop(self):
        with self.lock:
            self.last_command_time = datetime.now()
            self.pwm.set(THROTTLE_CHANNEL, THROTTLE_STOPPED_PWM)
            self.last_throttle = THROTTLE_STOPPED_PWM

    def steer_left(self):
        with self.lock:
            self.last_command_time = datetime.now()
            self.pwm.set(STEERING_CHANNEL, STEERING_LEFT_PWM)

    def steer_right(self):
        with self.lock:
            self.last_command_time = datetime.now()
            self.pwm.set(STEERING_CHANNEL, STEERING_RIGHT_PWM)

    def steer_center(self):
        with self.lock:
            self.last_command_time = datetime.now()
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
# Networking: HTTP server with threading for better connection handling
# ==========================
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 6666

car = CarController()
preview = PreviewManager()

# Connection tracking
connection_stats = {
    'total_commands': 0,
    'failed_commands': 0,
    'last_client_ip': None,
    'last_connection_time': None
}
stats_lock = threading.Lock()

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
<p style="display:flex;align-items:center;justify-content:center;gap:8px;">
  Connection: <span id="health" style="width:16px;height:16px;border-radius:50%;background:gray;display:inline-block;"></span>
</p>

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
let failCount = 0;
let lastSuccess = Date.now();

// Send command with retry logic for unstable connections
const send = async (cmd, retries = 2) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(base + '/cmd?c=' + encodeURIComponent(cmd), {
        method: 'POST',
        timeout: 3000,
        signal: AbortSignal.timeout(3000)
      });
      if (response.ok) {
        failCount = 0;
        lastSuccess = Date.now();
        return true;
      }
    } catch (err) {
      console.warn(`Command ${cmd} failed (attempt ${i+1}):`, err.message);
      if (i < retries - 1) await new Promise(r => setTimeout(r, 100)); // Wait 100ms before retry
    }
  }
  failCount++;
  console.error(`Command ${cmd} failed after ${retries} attempts`);
  return false;
};

// Connection health indicator
const updateHealth = () => {
  const timeSinceSuccess = Date.now() - lastSuccess;
  const healthDot = document.getElementById('health');
  if (!healthDot) return;
  
  if (timeSinceSuccess < 2000 && failCount < 3) {
    healthDot.style.background = 'green';
    healthDot.title = 'Connection: Good';
  } else if (timeSinceSuccess < 5000 && failCount < 5) {
    healthDot.style.background = 'orange';
    healthDot.title = 'Connection: Unstable';
  } else {
    healthDot.style.background = 'red';
    healthDot.title = 'Connection: Poor';
  }
};
setInterval(updateHealth, 1000);

const holdBehavior = (id, pressCmd, releaseCmd) => {
  const el = document.getElementById(id);
  let touching = false;
  let pressInterval = null;

  const press = (e) => { 
    e.preventDefault(); 
    if (!touching) { 
      touching = true; 
      send(pressCmd);
      // Send command repeatedly every 500ms while held for unstable connections
      pressInterval = setInterval(() => send(pressCmd), 500);
    } 
  };
  const release = (e) => { 
    e.preventDefault(); 
    if (touching) { 
      touching = false; 
      clearInterval(pressInterval);
      send(releaseCmd); 
    } 
  };

  el.addEventListener('mousedown', press);
  el.addEventListener('mouseup', release);
  el.addEventListener('mouseleave', release);

  el.addEventListener('touchstart', press, {passive:false});
  el.addEventListener('touchend', release);
  el.addEventListener('touchcancel', release);
};

document.getElementById('previewBtn').addEventListener('click', async () => {
  try {
    const r = await fetch(base + '/preview/toggle', {method:'POST', timeout: 5000});
    const t = await r.text();
    document.getElementById('previewBtn').innerText = t.includes('running') ? 'Stop Preview' : 'Start Preview';
  } catch (err) {
    console.error('Preview toggle failed:', err);
  }
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
    timeout = 5  # Socket timeout to handle slow/unstable connections

    def setup(self):
        """Set socket timeout for better handling of unstable connections"""
        BaseHTTPRequestHandler.setup(self)
        self.request.settimeout(5)

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                self._send_html(HTML_PAGE)
            elif path == '/health':
                with stats_lock:
                    stats = f"ok|cmds:{connection_stats['total_commands']}|fails:{connection_stats['failed_commands']}"
                self._send_text(stats)
            elif path == '/stats':
                with stats_lock:
                    stats_str = (f"Total Commands: {connection_stats['total_commands']}\n"
                                f"Failed Commands: {connection_stats['failed_commands']}\n"
                                f"Last Client: {connection_stats['last_client_ip']}\n"
                                f"Last Connection: {connection_stats['last_connection_time']}")
                self._send_text(stats_str)
            else:
                self._send_text("Not found", code=404)
        except socket.timeout:
            print(f"[Timeout] GET request timed out from {self.client_address[0]}")
        except Exception as e:
            print(f"[Error] GET handler: {e}")

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == '/cmd':
                qs = parse_qs(parsed.query)
                cmd = (qs.get('c', [''])[0] or '').strip().upper()
                
                # Track connection stats
                with stats_lock:
                    connection_stats['total_commands'] += 1
                    connection_stats['last_client_ip'] = self.client_address[0]
                    connection_stats['last_connection_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                ok, msg = self._apply_command(cmd)
                
                if not ok:
                    with stats_lock:
                        connection_stats['failed_commands'] += 1
                
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
        except socket.timeout:
            print(f"[Timeout] POST request timed out from {self.client_address[0]}")
            with stats_lock:
                connection_stats['failed_commands'] += 1
        except Exception as e:
            print(f"[Error] POST handler: {e}")
            with stats_lock:
                connection_stats['failed_commands'] += 1

    def _apply_command(self, cmd: str):
        """Apply command with error handling and retry logic"""
        max_retries = 2
        retry_delay = 0.05
        
        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    print(f"[Retry] Command '{cmd}' failed (attempt {attempt + 1}), retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"[Error] Command '{cmd}' failed after {max_retries} attempts: {e}")
                    return False, str(e)
        return False, "max retries exceeded"

    def _send_html(self, html):
        try:
            data = html.encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "keep-alive")  # Allow connection reuse
            self.send_header("Keep-Alive", "timeout=5, max=100")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[Warning] Connection broken while sending HTML: {e}")

    def _send_text(self, text, code=200):
        try:
            data = text.encode('utf-8')
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "keep-alive")  # Allow connection reuse
            self.send_header("Keep-Alive", "timeout=5, max=100")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[Warning] Connection broken while sending response: {e}")

    def log_message(self, fmt, *args):
        # Log connection issues but keep quiet for successful requests
        if "code 40" in str(args) or "code 50" in str(args):
            print(f"[HTTP] {fmt % args}")
        return

# ==========================
# Local keyboard UI (optional)
# ==========================
class KB:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
               # Save current terminal settings
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
                time.sleep(0.001)
                seq = ch
                while select.select([sys.stdin], [], [], 0)[0]:
                    seq += sys.stdin.read(1)
                keys.append(seq)
            else:
                keys.append(ch)
        return keys

def decode_key(ch):
    if ch in ('w','W','\x1b[A'):
        return 'UP'
    if ch in ('s','S','\x1b[B'):
        return 'DOWN'
    if ch in ('a','A','\x1b[D'):
        return 'LEFT'
    if ch in ('d','D','\x1b[C'):
        return 'RIGHT'
    if ch == ' ':
        return 'SPACE'
    if ch in ('c','C'):
        return 'CENTER'
    if ch in ('q','Q','\x03'):
        return 'QUIT'
    return None

def print_controls():
    print("# Control TT02 RC car over network or locally.")
    print("# Buttons on / page control throttle/steering.")
    print("# - Up:     throttle forward")
    print("# - Down:   throttle reverse")
    print("# - Left:   steer left")
    print("# - Right:  steer right")
    print("# - Space:  throttle stop")
    print("# - c:      center steering")
    print("# - q:      quit (safely stops and centers)")
    print()

# ==========================
# Main server runner with threading for better connection handling
# ==========================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for better concurrent connection handling"""
    daemon_threads = True  # Don't wait for threads to finish on shutdown
    request_queue_size = 10  # Buffer for incoming connections

def run_server():
    server = ThreadedHTTPServer((HOST, PORT), RequestHandler)
    server.timeout = 1.0  # Server socket timeout
    print(f"Server running at http://0.0.0.0:{PORT}/")
    print("Open http://<this-device-ip>:{PORT}/ from your phone on the same Wi‑Fi.")
    print("\n[Connection] Using threaded server for better stability with unstable connections")
    print("[Watchdog] Auto-stop enabled - car will stop if no commands for 2 seconds")
    return server

def main():
    print_controls()

    def on_sigint(signum, frame):
        try:
            car.shutdown()
            preview.stop()
            time.sleep(0.1)
        finally:
            print("\nExiting safely.")
            print(f"[Stats] Total commands: {connection_stats['total_commands']}, Failed: {connection_stats['failed_commands']}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)

    server = run_server()
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Status monitoring thread
    def status_monitor():
        last_cmd_count = 0
        while True:
            time.sleep(10)  # Print stats every 10 seconds
            with stats_lock:
                if connection_stats['total_commands'] != last_cmd_count:
                    print(f"[Status] Commands: {connection_stats['total_commands']}, "
                          f"Failed: {connection_stats['failed_commands']}, "
                          f"Last client: {connection_stats['last_client_ip']}")
                    last_cmd_count = connection_stats['total_commands']
    
    status_thread = threading.Thread(target=status_monitor, daemon=True)
    status_thread.start()

    print("Local keys active in this terminal. Press 'q' or Ctrl-C to quit.")
    print("Visit http://<your-ip>:6666/stats for connection statistics")
    with KB() as kb:
        while True:
            for raw in kb.poll_all():
                k = decode_key(raw)
                if k == 'UP':
                    car.throttle_forward()
                elif k == 'DOWN':
                    car.throttle_reverse()
                elif k == 'LEFT':
                    car.steer_left()
                elif k == 'RIGHT':
                    car.steer_right()
                elif k == 'SPACE':
                    car.throttle_stop()
                elif k == 'CENTER':
                    car.steer_center()
                elif k == 'QUIT':
                    car.shutdown()
                    preview.stop()
                    server.shutdown()
                    return
            # Light decay to neutral for local control loop
            car.throttle_stop()
            car.steer_center()
            time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(f"Socket error: {e}")
        print("Tip: If binding fails, ensure no other process uses the port and that you have network up.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure I2C is enabled and required packages are installed.")
        sys.exit(1)
