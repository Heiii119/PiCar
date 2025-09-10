# PiCar
Playing around with Raspberry Pi controlled RC car!

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



# Programs and usage
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
python3 camera_inspector.py --size 1280x720 --fps 30 --interval 1.0
```

### Console/Output (prints every interval):
Mean RGB: (R,G,B) for each ROI
Mean HSV: H in degrees, S and V in 0–255 scale
Dominant hue: mode of hue histogram on sufficiently saturated pixels
V_p50: median brightness
Edge strength: average gradient magnitude
Orientation tendency: “vertical-dominant”, “horizontal-dominant”, or “mixed” with dx/dy ratio

## Task 5: Line detection and steering decision (console)
### Purpose: 
For a line-following task. Looks at the bottom ROI, detects a line (black or colored), and prints position/steering decisions like “line LEFT of center → turn LEFT”.

### Run:
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
