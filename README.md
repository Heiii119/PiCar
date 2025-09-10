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



# Run the tests
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
bash install.sh
python3 scripts/test_res_fps_qt.py   # or the other tests
```

## Test 1: Resolution/FPS live preview
```bash
python3 scripts/test_res_fps_qt.py
```
Keys: r = next resolution, f = next FPS, s = save frame, q/Esc = quit

## Test 2: RGB + HSV channels live viewer
```bash
python3 scripts/test_hsv_channels_qt.py
```
Keys: s = save composite, q/Esc = quit

## Test 3: Hue rotation (色彩空間轉換) live demo
```bash
python3 scripts/test_hue_rotation_qt.py
```
Keys: ←/→ = ±10 hue, ↑/↓ = ±1 hue, 0 = reset, s = save, q/Esc = quit



