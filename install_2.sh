#!/usr/bin/env bash
# install.sh — TT02 + Picamera2 setup (PiCam only; disables Coral/EdgeTPU APT repos)
# Keeps original packages:
#   python3-picamera2 python3-pyqt5 i2c-tools python3-numpy python3-pil
# Adds minimal extras:
#   python3-pip python3-smbus python3-smbus2 qtwayland5 git
#   rpicam-apps (or libcamera-apps as fallback) for rpicam-hello
# Creates venv at ~/tt02-venv and installs OpenCV + Adafruit PCA9685 only inside it

set -u
set -o pipefail

FAILED_PKGS=()
DISABLED_CORAL_FILES=()
REBOOT_NEEDED=0

have_cmd() { command -v "$1" >/dev/null 2>&1; }

apt_install_one() {
  local pkg="$1"
  echo "==> Installing $pkg ..."
  if ! sudo apt-get install -y "$pkg"; then
    echo "WARNING: Failed to install $pkg"
    FAILED_PKGS+=("$pkg")
    return 0
  fi
}

disable_coral_repos() {
  echo "==> Checking for Coral/EdgeTPU APT sources to disable..."
  local changed=0
  for f in /etc/apt/sources.list.d/*coral*.list /etc/apt/sources.list.d/*edgetpu*.list /etc/apt/sources.list.d/*cloud*.list; do
    [[ -f "$f" ]] || continue
    if grep -Ei 'coral|edgetpu|packages\.cloud\.google\.com' "$f" >/dev/null 2>&1; then
      echo "   - Disabling $f (commenting out deb lines)"
      if [[ ! -f "${f}.bak" ]]; then
        sudo cp "$f" "${f}.bak" || true
      fi
      if sudo sed -i -E 's/^[[:space:]]*deb /# deb /' "$f"; then
        DISABLED_CORAL_FILES+=("$f")
        changed=1
      fi
    fi
  done
  if (( changed == 0 )); then
    echo "   No Coral/EdgeTPU sources found (nothing to disable)."
  fi
}

echo "=== TT02 setup (PiCam only) ==="
if ! have_cmd sudo; then echo "ERROR: sudo not found."; exit 1; fi
if ! have_cmd python3; then echo "ERROR: python3 not found."; exit 1; fi

echo "System info:"
uname -a || true
cat /etc/os-release || true
echo

# Disable Coral/EdgeTPU sources if present
disable_coral_repos

echo "==> Updating APT indexes..."
if ! sudo apt-get update; then
  echo "APT update failed. Retrying while allowing Release info changes..."
  if ! sudo apt-get update --allow-releaseinfo-change; then
    echo "ERROR: apt-get update failed. Check your network and APT sources."
    exit 1
  fi
fi

# Core and extras (system-wide)
apt_install_one python3-picamera2
apt_install_one python3-pyqt5
apt_install_one i2c-tools
apt_install_one python3-numpy
apt_install_one python3-pil

# Minimal extras
apt_install_one python3-pip
apt_install_one python3-smbus
apt_install_one python3-smbus2
apt_install_one qtwayland5
apt_install_one git

# Venv support and full Python (headers etc.)
apt_install_one python3-venv
apt_install_one python3-full

# Camera apps for rpicam-hello (try new then old package name)
apt_install_one rpicam-apps
apt_install_one libcamera-apps

# Enable I2C if possible
echo "==> Enabling I2C (raspi-config if available)..."
if have_cmd raspi-config; then
  sudo raspi-config nonint do_i2c 0 || echo "NOTE: raspi-config I2C enable returned non-zero (continuing)."
else
  echo "NOTE: raspi-config not found; skipping automatic I2C enable."
fi

# Add user to i2c group
if id -nG "$USER" | grep -qw "i2c"; then
  echo "User '$USER' already in 'i2c' group."
else
  echo "==> Adding '$USER' to 'i2c' group..."
  if sudo usermod -aG i2c "$USER"; then
    REBOOT_NEEDED=1
  else
    echo "WARNING: Could not add '$USER' to 'i2c' group."
  fi
fi

# Device presence hint
if [[ ! -e /dev/i2c-1 ]]; then
  echo "NOTICE: /dev/i2c-1 not present; a reboot may be required after enabling I2C."
  REBOOT_NEEDED=1
fi

# Create and populate a dedicated venv at ~/tt02-venv
echo "==> Creating Python venv at ~/tt02-venv ..."
python3 -m venv "$HOME/tt02-venv" || { echo "ERROR: Failed to create venv."; exit 1; }

echo "==> Activating venv and installing packages inside it ..."
# shellcheck disable=SC1090
source "$HOME/tt02-venv/bin/activate"

# Upgrade tooling in venv
python -m pip install --upgrade pip wheel setuptools || echo "WARNING: venv pip upgrade failed (continuing)."

# Install OpenCV and PCA9685 only in the venv
# Use headless OpenCV unless you specifically need GUI windows from cv2.imshow
pip install --no-cache-dir opencv-python-headless || { echo "ERROR: Failed to install OpenCV in venv."; deactivate; exit 1; }

# PCA9685 libs
pip install --no-cache-dir adafruit-circuitpython-pca9685 adafruit-blinka Adafruit-PCA9685 || echo "WARNING: PCA9685 install in venv encountered an issue."

deactivate || true

# PATH hint for user-site scripts (not strictly required now, but helpful if user-site tools get used)
USER_BASE=$(python3 -m site --user-base 2>/dev/null || echo "$HOME/.local")
BIN_DIR="$USER_BASE/bin"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo
  echo "NOTE: Your PATH may not include $BIN_DIR."
  echo "Add this to your ~/.bashrc to use user-site tools:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

if (( ${#FAILED_PKGS[@]} )); then
  echo
  echo "The following APT packages could not be installed:"
  printf '  - %s\n' "${FAILED_PKGS[@]}"
  echo "This may happen if you're not on Raspberry Pi OS or APT sources are missing."
fi

if (( ${#DISABLED_CORAL_FILES[@]} )); then
  echo
  echo "Temporarily disabled Coral/EdgeTPU APT entries in:"
  printf '  - %s\n' "${DISABLED_CORAL_FILES[@]}"
  echo "Backups (if created) have .bak suffix. You can re-enable by restoring the file or uncommenting lines."
fi

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  echo
  echo "Reboot recommended so I2C and group changes take effect."
fi

echo "Install done. Test the camera with: rpicam-hello -t 0"
echo "Then run scripts"
echo "  - Activate venv: source ~/tt02-venv/bin/activate"
echo "  - Run camera testing scripts, e.g.: python3 test_res_fps_qt.py"
echo "  - Run camera inspector with preview:"
echo "      python3 camera_inspector.py --preview"
echo "  - Run line follower with preview (example black line):"
echo "      python3 line_follower_console.py --mode black --preview"
echo "  - Run keyboard control (ensure terminal focus):"
echo "      python3 tt02_keyboard_drive.py"
