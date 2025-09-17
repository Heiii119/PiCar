# PiCar
Playing around with Raspberry Pi controlled RC car!

## Set up on Pi
https://docs.donkeycar.com/guide/robot_sbc/setup_raspberry_pi/

### setup virtual env
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-full
python3 -m venv ~/tt02-venv
source ~/tt02-venv/bin/activate
pip3 install Adafruit-PCA9685
```
Later, to use again: ```source ~/tt02-venv/bin/activate```

### Configure I2C PCA9685 servo board
```bash
sudo apt-get install -y i2c-tools
sudo i2cdetect -y 1
```


### install tensorflow
for Python 3.7.3 on Raspbian Buster. 
- Use a fresh virtualenv Avoid the piwheels hash issue
- Pins: numpy==1.19.5, h5py==2.10.0 (TF 2.4-friendly)
- Uses piwheels and disables cache to avoid mirror/hash mismatches
- Preinstalls scikit-build to satisfy builds when using --no-build-isolation
- Install Donkeycar 4.5.1 and TensorFlow 2.4.0 (cp37/armv7l)

Step 0: System prep
```bash
sudo apt update
sudo apt install -y python3-venv libatlas-base-dev libhdf5-103 libhdf5-dev cmake ninja-build git
```

Step 1: Create and activate a clean virtualenv 
```bash
python3 -m venv ~/dkc451
source ~/dkc451/bin/activate
```

Step 2: Upgrade pip (keep a 3.7-compatible version) and clear cache
```bash
python -m pip install --upgrade "pip<24" setuptools wheel
python -m pip cache purge
unset PIP_REQUIRE_HASHES 2>/dev/null || true
```

Step 3: Get Donkeycar 4.5.1 (done)
```bash
git clone https://github.com/autorope/donkeycar
cd donkeycar
git fetch --all --tags -f
git checkout 4.5.1
```

Step 4: Create version constraints for TF 2.4 
```bash
printf "numpy==1.19.5\nh5py<3\n" > constraints.txt
```

Step 5: Preinstall pinned numpy and h5py from piwheels (wheels only, no cache) 

```bash
python -m pip install --no-cache-dir --only-binary=:all: --index-url https://www.piwheels.org/simple --extra-index-url https://pypi.org/simple -c constraints.txt numpy==1.19.5 h5py==2.10.0
```
Preinstall build helper (prevents “No module named skbuild” with --no-build-isolation)
```bash
python -m pip install --no-cache-dir "scikit-build<0.18"
```
If a package insists on building and fails, try wheels-only to identify the culprit:
This is to check if everything has wheels. 
If it succeeds, you’re done and can skip step 6.

```bash
python -m pip install --only-binary=:all: -e .[pi] --index-url https://www.piwheels.org/simple --extra-index-url https://pypi.org/simple
```


Step 6: Install Donkeycar (editable) with Pi extras, no build isolation, respecting constraints 
```bash
python -m pip install --no-cache-dir --no-build-isolation -e .[pi] -c constraints.txt --index-url https://www.piwheels.org/simple --extra-index-url https://pypi.org/simple
```

Step 7:Install TensorFlow 2.4.0 (cp37, armv7l) 
```bash
python -m pip install --no-cache-dir https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
```

Quick verification
verify the tensorflow install:
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```
```bash
python - <<'PY' import sys, numpy as np import tensorflow as tf print(sys.version) print("numpy:", np.version) print("tensorflow:", tf.version) print("tf test:", tf.reduce_sum(tf.constant([[1.0,2.0],[3.0,4.0]])).numpy()) PY
````

Troubleshooting (only if needed)

Hash mismatch from piwheels:
Ensure you used --no-cache-dir and the explicit --index-url https://www.piwheels.org/simple
```bash
python -m pip cache purge
unset PIP_REQUIRE_HASHES
grep -nR 'numpy==1.19' setup.* setup.cfg pyproject.toml requirements || true
sed -i 's/numpy==1.19\b/numpy==1.19.5/g' setup.py 2>/dev/null || true
sed -i 's/numpy==1.19\b/numpy==1.19.5/g' setup.cfg 2>/dev/null || true
```
If piwheels keeps failing for h5py, build from PyPI (slower):
```bash
python -m pip install --no-cache-dir --no-binary=:all: --index-url https://pypi.org/simple "h5py==2.10.0"
```
Then re-run step 6.

Notes and tips:
Always use python -m pip inside the venv to avoid calling a system pip by mistake.
If the sed commands don’t find anything to change, tell me what grep printed and I’ll adjust the one-liner.
If you ever see the hash mismatch again, keep using --no-cache-dir and the explicit --index-url https://www.piwheels.org/simple flags.

Why edit the pin? Donkeycar 4.5.1 pins numpy==1.19.0, but TF 2.4.x requires numpy >= 1.19.2. Moving to 1.19.5 keeps both happy.
Hash mismatch fix: using the main piwheels index plus --no-cache-dir avoids stale/mirrored files (the “archive1” hash you saw).
If you still see “hashes do not match”:
Ensure no global hash enforcement is set: echo $PIP_REQUIRE_HASHES; if set, run: unset PIP_REQUIRE_HASHES
Re-run installs with --no-cache-dir and the explicit --index-url shown above
If TF import complains about missing BLAS/HDF5 at runtime, install:
sudo apt install -y libhdf5-103 libhdf5-dev
Alternative (no file edits): install Donkeycar without deps, then manage deps yourself with numpy==1.19.5

python -m pip install -e . --no-deps
Then add its extras piece by piece (excluding numpy) or from its requirements after changing the numpy pin there. This is more manual; the edit approach above is simpler.

## set up on computer

https://docs.donkeycar.com/guide/create_application/

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
