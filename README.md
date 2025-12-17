# PiCar
Playing around with Raspberry Pi controlled RC car!

## Set up on Pi
https://docs.donkeycar.com/guide/robot_sbc/setup_raspberry_pi/

### setup virtual env
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-full (optional)
python3 -m venv tflite-env
source tflite-env/bin/activate

```
Later, to use again: ```source tflite-env/bin/activate```

### Configure I2C PCA9685 servo board (activate the venv first)
```bash
pip3 install Adafruit-PCA9685
pip3 install adafruit-circuitpython-pca9685
sudo apt-get install -y i2c-tools
sudo i2cdetect -y 1
```
### Install opencv
```bash
pip install opencv-python
```

### install tensorflowLite
```bash
python3 -m venv tflite-env
source tflite-env/bin/activate
pip install --upgrade pip
pip install tflite-runtime
pip3 install --upgrade tflite-runtime
```

### get open cv
```bash
sudo apt update
sudo apt install -y python3-opencv
```

##image regconition
```bash
pip install pillow
# backup (if any)
[ -s labels.txt ] && cp labels.txt labels_backup.txt

# download new labels
wget \
  https://github.com/leferrad/tensorflow-mobilenet/raw/refs/heads/master/imagenet/labels.txt \
  -O labels.txt
```

## Install and setup Tailscale
### 1. on the Raspberry Pi: 
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up #login
```
- A URL will appear in the terminal.
- Open that URL in a browser (on any device), log in, and approve the Pi.
- After that, run:
```bash
tailscale ip
# You should see an IP like 100.x.y.z. That is the Pi’s Tailscale IP.
```
if tailscale is not activate
```bash
sudo systemctl enable tailscaled
sudo systemctl start tailscaled
```
### 2. on your computer or phone or other device:
- Install the Tailscale app (iOS App Store / Google Play).
- Log in with the same account.
- Connect to the Tailscale network.

### 3. run the program on Pi
```bash
python3 web_car_control.py
```

### 4. connect to the pi on your device:
- Open the Tailscale app → ensure it’s connected.
- Open the browser app (Safari, Chrome, etc.).
```bash
http://<Pi_Tailscale_IP>:5000
```

Enter the same URL:

# PiCam RGB/HSV Tools (No OpenCV)

Live previews and color-space demos for Raspberry Pi cameras using Picamera2 + PyQt5 + Pillow.

## Prerequisites
- Raspberry Pi with Raspberry Pi OS (Bookworm/Bullseye) and a CSI camera
- Desktop session (PyQt5 opens a window). If headless, ask us for a CLI-only variant.

## Install
```bash
git clone git@github.com:Heiii119/PiCar.git
cd PiCar
bash install.sh
```

## Quick Camera Check
for Bookworm ```rpicam-hello -t 0```
for Bullseye (older version): ```libcamera-hello -t 0```
if there is any detection error:
1.	```sudo apt update```
2.	```sudo apt install -y rpicam-apps```
3.	```sudo apt full-upgrade -y```
4.	```rpicam-hello --list-cameras```



# Camera Programs and usage
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
bash install.sh
python3 test_res_fps_qt.py   # or the other tests
```

## Test 1: Resolution/FPS live preview
### Purpose: 
Explore resolution (解析度) and frame rate (幀率) with a live view and measured FPS.

### Run:
```bash
python3 test_res_fps_qt.py
```
### Controls: 
r = next resolution, f = next FPS, s = save frame(PNG in captures/), Ctrl+C = stop, q/Esc = quit

### Console:
Prints available sensor modes at startup.

### Saved files:
captures/x_fps_YYYYMMDD_HHMMSS.png

## Test 2: RGB + HSV channels live viewer
### Purpose: 
Visualize RGB and HSV channels live in a 2x2 grid.

### Run:
```bash
python3 test_hsv_channels_qt.py
```
What you see (tile order):
Top-left: RGB (original)
Top-right: Hue colorized (H with S=255, V=255)
Bottom-left: Saturation (grayscale; dark=desaturated, bright=vivid)
Bottom-right: Value (grayscale brightness)

### Controls:
s = save composite, Ctrl+C = stop, q/Esc = quit

### Saved files:
captures/rgb_hsv_grid_YYYYMMDD_HHMMSS.png

## Test 3: Hue rotation (色彩空間轉換) live demo
### Purpose: 
Rotate hue in HSV, keep S and V, convert back to RGB for display.

### Run:
```bash
python3 test_hue_rotation_qt.py
```
### Controls: 
←/→ = ±10 hue, ↑/↓ = ±1 hue, 0 = reset, s = save, Ctrl+C = stop, q/Esc = quit

### Saved files:
captures/hue_rot__YYYYMMDD_HHMMSS.png

## Task 4: Scene color and edge/line metrics (console)
### Purpose: 
Print what the camera “sees” as numbers: mean color, dominant hue, brightness, and simple edge orientation in two ROIs.
- Center ROI (middle third) for scene color
- Bottom ROI (lower band) for line-like features

### Run:
```bash
python3 camera_inspector.py --preview
```
Default SIZE=1280x720, FPS=30, PRINT_RATE=1.0.
You may change the setting of resolution(, fps and print rate:
```bash
camera_inspector.py [--size SIZE] [--fps FPS] [--print-rate PRINT_RATE] --preview
```

### Console/Output (prints every interval):
- Mean RGB: (R,G,B) for each ROI
- Mean HSV: H in degrees, S and V in 0–255 scale
- Dominant hue: mode of hue histogram on sufficiently saturated pixels
- V_p50: median brightness
- Edge strength: average gradient magnitude
- Orientation tendency: “vertical-dominant”, “horizontal-dominant”, or “mixed” with dx/dy ratio

## Task 5: Line detection and steering decision (console)
### Purpose: 
For a line-following task. Looks at the bottom ROI, detects a line (black or colored), and prints position/steering decisions like “line LEFT of center → turn LEFT”.

### Run:
```bash
python3 line_follower_console.py --preview
```

For Black Line on Bright Floor:
```bash
python3 line_follower_console.py --mode black --v-max 80 --s-max 100
```

For Colored line (e.g., yellow ≈ 50–65°):
```bash
python3 line_follower_console.py --mode color --h-lo 50 --h-hi 65 --s-min 80 --v-min 60
```

Adjust ROI height (use more/less of the bottom):
```bash
python3 line_follower_console.py --roi-height 0.35
```

#### Key parameters:
--mode black | color: choose dark-line detection or a hue range
Black mode thresholds: --v-max, --s-max
Color mode thresholds: --h-lo, --h-hi (degrees; wrap-around supported), --s-min, --v-min
Detection region: --roi-height (fraction of image height at bottom)
Decision tuning: --deadband (center tolerance), --min-coverage (min mask fraction)
Mirroring fix: --invert-steer (if camera/steering mapping is flipped)

### Outputs (printed about 10× per second by default):
coverage: fraction of ROI pixels flagged as “line”
e: normalized lateral error of the line centroid
angle: approximate line angle via PCA (optional; may be None on sparse masks)
status and decision:
- “line LEFT/RIGHT of center → turn LEFT/RIGHT”
- or “on line (centered) → go straight”
- or “line lost → search”

#### Lateral error definition: 
e= (cx−W/2)/(W/2)
W = ROI width, cx = centroid x of the detected line; e < 0 means line is left of center, e > 0 right of center.

#### Example output:
ROI: 1280x252 | coverage=4.12% | e=-0.183 | angle=+86.4° | line LEFT of center -> turn LEFT ROI: 1280x252 | coverage=0.05% | line lost | decision=search ROI: 1280x252 | coverage=6.90% | e=+0.021 | on line (centered) -> go straight

#### Tuning tips:
If the line is faint/thin, increase --roi-height or relax thresholds.
For red lines near wrap-around, e.g., --h-lo 350 --h-hi 10 works.
If decisions feel reversed, add --invert-steer.

## Notes:
These examples avoid OpenCV. Dependencies are installed via APT: Picamera2, PyQt5, Pillow, NumPy.
If the preview window doesn’t show under Wayland, try:
```bash
export QT_QPA_PLATFORM=xcb
```
For smoother previews, try 1280x720 at 30 fps.
All saved images land in captures/.

# Driving Programs and usage
### Requires:
- I2C enabled on the Raspberry Pi
- PCA9685 wired correctly
  - Throttle ESC → PCA9685 channel 0 (pin 0)
  - Steering servo → PCA9685 channel 1 (pin 1)

### Run:
```bash
python3 tt02_keyboard_drive.py
```
### Controls: 
↑/↓ = throttle forward/reverse
←/→ = steering turn left/right
Space = immediate throttle stop
c = center steering
Crtl+C = stop

## autopilot
- Opens a Picamera2 preview window
- Lets you record training data while you drive using keyboard controls
- Trains a small TF 2.4 model on frames + steering/throttle labels
- Runs a basic autopilot loop using the trained model

Notes:
- Uses Picamera2 preview with DRM/KMS via its built-in preview window; we also add a simple HUD by overlaying text with the Picamera2 annotation helper.
- Keyboard input is read from stdin in raw mode (no OpenCV). Terminal must be focused.
- Controls: a/d or Left/Right to steer, w/s or Up/Down for throttle, space to stop, c to center steering, r to start/stop recording, q to quit.
- Data stored under ./data/session_<timestamp>/images plus labels.csv.
- Uses Adafruit PCA9685 for steering/motor as per your PWM config.
- If running headless (no display), set with_preview=False in PiCam2Manager, and it will still capture frames without opening a window.
- Some environments may require Preview.DRM rather than Preview.QTGL; switch start_preview(Preview.DRM) if QTGL isn’t available.
- First tests: lift wheels off the ground. Many ESCs require “stop” PWM for 1–2 seconds before throttle.
- If steering or throttle directions are reversed, set PWM_STEERING_INVERTED or PWM_THROTTLE_INVERTED to True.

Before running:
- sudo apt-get install -y python3-libcamera python3-picamera2
- pip3 install adafruit-circuitpython-pca9685 adafruit-blinka RPi.GPIO
- Enable I2C (sudo raspi-config) and ensure PCA9685 at address 0x40.
- Ensure you can open a Picamera2 preview (KMS/DRM). If running headless/SSH, set preview="null" and skip preview.

### Run:
```bash
python3 drive_train_autopilot_picam2.py
```
