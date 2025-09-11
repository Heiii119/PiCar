#!/usr/bin/env bash
# install.sh — Setup for TT02 keyboard control + Picamera2 preview on Raspberry Pi OS (Bookworm)
# - Installs system packages (Picamera2, PyQt5, I2C tools, NumPy, Pillow)
# - Enables I2C and adds current user to i2c group
# - Installs PCA9685 Python libs (both legacy and CircuitPython options)
# - Hardens Qt (Wayland/X11) by installing qtwayland5
# Usage:
#   chmod +x install.sh
#   ./install.sh
set -euo pipefail

REBOOT_NEEDED=0

have_cmd() { command -v "$1" >/dev/null 2>&1; }

require_cmd() {
  if ! have_cmd "$1"; then
    echo "ERROR: Required command '$1' not found."
    exit 1
  fi
}

echo "=== TT02 + Picamera2 setup ==="
echo "This script uses sudo where needed. You may be prompted for your password."

require_cmd sudo
require_cmd python3
require_cmd pip3 || true

# -------------------------------------------------------------------
# 1) System packages
# -------------------------------------------------------------------
echo "==> Updating APT package lists..."
sudo apt-get update -y

echo "==> Installing system dependencies via APT..."
sudo apt-get install -y \
  git \
  i2c-tools \
  python3 \
  python3-pip \
  python3-numpy \
  python3-pil \
  python3-smbus \
  python3-smbus2 \
  python3-pyqt5 \
  qtwayland5 \
  python3-picamera2

# -------------------------------------------------------------------
# 2) Enable I2C and add user to i2c group
# -------------------------------------------------------------------
echo "==> Enabling I2C interface (raspi-config)..."
if have_cmd raspi-config; then
  # 0 = enable
  sudo raspi-config nonint do_i2c 0 || true
else
  echo "WARNING: raspi-config not found. Please enable I2C manually if needed."
fi

# Add current user to the i2c group (takes effect next login or reboot)
if id -nG "$USER" | grep -qw "i2c"; then
  echo "User '$USER' is already in the 'i2c' group."
else
  echo "==> Adding '$USER' to 'i2c' group..."
  sudo usermod -aG i2c "$USER"
  REBOOT_NEEDED=1
fi

# Check if /dev/i2c-1 exists; if not, a reboot is likely required
if [[ ! -e /dev/i2c-1 ]]; then
  echo "NOTICE: /dev/i2c-1 not present yet; enabling I2C may require a reboot."
  REBOOT_NEEDED=1
fi

# -------------------------------------------------------------------
# 3) Python packages (PCA9685 drivers)
#    We install to user site-packages to avoid system conflicts.
# -------------------------------------------------------------------
echo "==> Installing Python PCA9685 libraries (user site)..."
# CircuitPython stack
python3 -m pip install --user --upgrade adafruit-circuitpython-pca9685 adafruit-blinka
# Legacy Adafruit (optional; our code supports both)
python3 -m pip install --user --upgrade Adafruit-PCA9685

# -------------------------------------------------------------------
# 4) Helpful notes and quick checks
# -------------------------------------------------------------------
echo
echo "==> Quick checks (optional):"
echo "   - Verify PCA9685 at address 0x40 is visible on I2C bus 1:"
echo "       i2cdetect -y 1 | grep 40 || echo 'PCA9685 not detected yet'"
echo "   - If using Wayland and preview windows don’t show, try:"
echo "       export QT_QPA_PLATFORM=wayland"
echo "     Or force X11:"
echo "       export QT_QPA_PLATFORM=xcb"

echo
echo "==> Done installing dependencies."

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  echo
  echo "A reboot is recommended for I2C/group changes to take effect."
  echo "Please reboot now or log out/in before using the PCA9685."
fi

echo "Install done. Test the camera with: rpicam-hello -t 0"
echo "Then run scripts"
echo "  - Run camera testing scripts, e.g.: python3 test_res_fps_qt.py"
echo "  - Run camera inspector with preview:"
echo "      python3 camera_inspector.py --preview"
echo "  - Run line follower with preview (example black line):"
echo "      python3 line_follower_console.py --mode black --preview"
echo "  - Run keyboard control (ensure terminal focus):"
echo "      python3 tt02_keyboard_drive.py"

