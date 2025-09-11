#!/usr/bin/env bash
set -e

# For Raspberry Pi OS Bookworm/Bullseye (with APT)
if ! command -v apt >/dev/null 2>&1; then
  echo "This installer expects apt (Debian/Raspberry Pi OS)."
  echo "Manual install: Picamera2 + PyQt5 + Pillow + NumPy"
  exit 1
fi

sudo apt update
# Picamera2 (camera access), PyQt5 (GUI), Pillow + NumPy (processing)
sudo apt install -y python3-picamera2 python3-pyqt5 python3-pil python3-numpy

echo
echo "Install done. Test the camera with: rpicam-hello -t 0"
echo "Then run scripts, e.g.: python3 test_res_fps_qt.py"
