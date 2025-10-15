#!/usr/bin/env bash
# install.sh — TT02 + Picamera2 setup (PiCam only; disables Coral/EdgeTPU APT repos)
# Ensures Debian ABI-aligned camera stack:
#   python3-picamera2 python3-numpy python3-simplejpeg (and friends)
# Keeps original packages:
#   python3-picamera2 python3-pyqt5 i2c-tools python3-numpy python3-pil
# Adds minimal extras:
#   python3-pip python3-smbus python3-smbus2 qtwayland5 git
#   rpicam-apps (or libcamera-apps as fallback) for rpicam-hello
# Creates venv at ~/tt02-venv 
# - Creates venv with --system-site-packages so apt site-packages (picamera2) are visible
# - Installs OpenCV + PCA9685 libs only inside the venv
# - Ensures libcamera/rpicam-hello present, enables I2C, and suggests KMS camera stack

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
  fi
}

disable_coral_repos() {
  echo "==> Checking for Coral/EdgeTPU APT sources to disable..."
  local changed=0
  shopt -s nullglob
  for f in /etc/apt/sources.list.d/*coral*.list /etc/apt/sources.list.d/*edgetpu*.list /etc/apt/sources.list.d/*cloud*.list; do
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
  shopt -u nullglob
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

# Basic sanity: require Bullseye or newer
if ! grep -Eq 'bullseye|bookworm' /etc/os-release; then
  echo "WARNING: This script expects Raspberry Pi OS Bullseye/Bookworm. Picamera2 apt packages may be unavailable."
fi

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

# Upgrade core to ensure ABI alignment
echo "==> Upgrading base system (recommended for libcamera alignment)..."
sudo apt-get dist-upgrade -y || echo "NOTICE: dist-upgrade failed or was skipped."

# Core camera stack
apt_install_one python3-numpy
apt_install_one python3-simplejpeg
apt_install_one python3-picamera2
apt_install_one libcamera-apps || true   # older name on some images
apt_install_one rpicam-apps || true      # newer name replaces libcamera-apps

# GUI support for Picamera2 preview (QT)
apt_install_one python3-pyqt5
apt_install_one qtwayland5
apt_install_one python3-kms++ || true

# I2C + utils
apt_install_one i2c-tools
apt_install_one python3-smbus
apt_install_one python3-smbus2

# Imaging utils
apt_install_one python3-pil

# General tools
apt_install_one python3-pip
apt_install_one git

# Venv support and build tools
apt_install_one python3-venv
apt_install_one python3-full
apt_install_one build-essential
apt_install_one libjpeg-dev

# Enable I2C and KMS camera stack if possible
echo "==> Enabling I2C and KMS (raspi-config if available)..."
if have_cmd raspi-config; then
  sudo raspi-config nonint do_i2c 0 || echo "NOTE: raspi-config I2C enable returned non-zero (continuing)."
  # Ensure GL driver uses KMS (not FKMS/Legacy)
  sudo raspi-config nonint do_gldriver KMS || echo "NOTE: Could not set KMS via raspi-config (continuing)."
  REBOOT_NEEDED=1
else
  echo "NOTE: raspi-config not found; skipping automatic I2C/KMS enable."
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

# Create and populate a venv that can see system site-packages (for picamera2 from apt)
VENV_DIR="$HOME/tt02-venv"
echo "==> Creating Python venv at $VENV_DIR (with system site packages) ..."
python3 -m venv --system-site-packages "$VENV_DIR" || { echo "ERROR: Failed to create venv."; exit 1; }

echo "==> Activating venv and installing packages inside it ..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Upgrade tooling in venv
python -m pip install --upgrade pip wheel setuptools || echo "WARNING: venv pip upgrade failed (continuing)."

# Install OpenCV and PCA9685 only in the venv
pip install --no-cache-dir opencv-python-headless || { echo "ERROR: Failed to install OpenCV in venv."; deactivate; exit 1; }
pip install --no-cache-dir adafruit-circuitpython-pca9685 adafruit-blinka Adafruit-PCA9685 || echo "WARNING: PCA9685 install in venv encountered an issue."

# Optional: small helpers for console UIs
pip install --no-cache-dir rich || true

deactivate || true

# Post-install verification (system Python)
echo "==> Verifying Picamera2 stack (system Python) ..."
SYS_PY="$(command -v python3 || echo /usr/bin/python3)"
echo "Using: $SYS_PY"
"$SYS_PY" - <<'PY'
import sys
print("Python:", sys.version)
def try_import(name):
    try:
        __import__(name)
        print(f"OK: import {name}")
        return True
    except Exception as e:
        print(f"FAIL: import {name} -> {e}")
        return False
ok_np = try_import("numpy")
ok_sj = try_import("simplejpeg")
ok_pc2 = try_import("picamera2")
PY

# Post-install verification (venv Python — should also see picamera2 thanks to --system-site-packages)
echo "==> Verifying imports in venv ..."
source "$VENV_DIR/bin/activate"
python - <<'PY'
import sys
print("VENV Python:", sys.version)
def try_import(name):
    try:
        __import__(name)
        print(f"OK (venv): import {name}")
        return True
    except Exception as e:
        print(f"FAIL (venv): import {name} -> {e}")
        return False
try_import("numpy")
try_import("simplejpeg")
try_import("picamera2")  # should succeed because venv sees system site-packages
PY
deactivate

# PATH hint for user-site scripts
USER_BASE=$(python3 -m site --user-base 2>/dev/null || echo "$HOME/.local")
BIN_DIR="$USER_BASE/bin"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo
  echo "NOTE: Your PATH may not include $BIN_DIR."
  echo "Add this to your ~/.bashrc:"
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
  echo "Backups have .bak suffix. Re-enable by restoring or uncommenting."
fi

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  echo
  echo "Reboot recommended so I2C/KMS changes take effect."
fi

echo
echo "Install done."
echo "Test the camera: rpicam-hello -t 0  (on HDMI/local)"
echo "Headless scripts should use Preview.DRM or Preview.NULL."
echo
echo "Run scripts:"
echo "  - Activate venv: source ~/tt02-venv/bin/activate"
echo "  - Example: python3 test_qtgl_preview.py"
echo "If you still get ModuleNotFoundError for picamera2, run with system Python (no venv):"
echo "  - deactivate"
echo "  - python3 test_qtgl_preview.py"
