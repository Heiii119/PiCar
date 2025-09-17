#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Update apt package lists..."
sudo apt-get update

echo "[2/6] Enable Legacy Camera via raspi-config (non-interactive fallback)"
# Try non-interactive enabling; if it fails, prompt user to run raspi-config manually
if command -v raspi-config >/dev/null 2>&1; then
  # raspi-config nonint is available on most images; if not, we echo instructions
  if sudo raspi-config nonint get_camera >/dev/null 2>&1; then
    # 0 = enabled, 1 = disabled
    sudo raspi-config nonint do_camera 0 || true
  else
    echo "raspi-config nonint not available. Please run: sudo raspi-config -> Interface Options -> Legacy Camera -> Enable"
  fi
else
  echo "raspi-config not found. Please install or enable camera manually."
fi

echo "[3/6] Install legacy camera tools and dependencies..."
sudo apt-get install -y \
  libraspberrypi-bin \
  python3-picamera \
  python3-pip \
  python3-numpy \
  python3-pil \
  git

echo "[4/6] Basic camera test (will fail if camera not connected/enabled)"
set +e
raspistill -o /tmp/legacy_cam_test.jpg -t 1000
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "Warning: raspistill test failed. Ensure ribbon connection is correct and Legacy Camera is enabled, then reboot."
else
  echo "raspistill test image saved to /tmp/legacy_cam_test.jpg"
fi

echo "[5/6] Optional: install v4l-utils for debugging"
sudo apt-get install -y v4l-utils

echo "[6/6] Done. Reboot is recommended."
echo "Run: sudo reboot"
