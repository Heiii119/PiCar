#!/usr/bin/env bash
# install.sh — PiCar setup: Picamera2 + OpenCV (apt) + TFLite (venv) + PCA9685 + I2C tools

set -u
set -o pipefail

FAILED_PKGS=()
REBOOT_NEEDED=0

have_cmd() { command -v "$1" >/dev/null 2>&1; }

apt_install_one() {
  local pkg="$1"
  echo "==> Installing APT package: $pkg"
  if ! sudo apt-get install -y "$pkg"; then
    echo "WARNING: Failed to install $pkg"
    FAILED_PKGS+=("$pkg")
  fi
}

echo "=== PiCar setup ==="
if ! have_cmd sudo; then echo "ERROR: sudo not found."; exit 1; fi
if ! have_cmd python3; then echo "ERROR: python3 not found."; exit 1; fi

echo "==> Updating APT indexes..."
if ! sudo apt-get update; then
  echo "APT update failed. Retrying with --allow-releaseinfo-change..."
  sudo apt-get update --allow-releaseinfo-change || { echo "ERROR: apt-get update failed."; exit 1; }
fi

echo "==> (Optional) Upgrading base system..."
sudo apt-get dist-upgrade -y || echo "NOTICE: dist-upgrade failed or was skipped."

# ---------- Core packages ----------
# Venv support
apt_install_one python3-venv
# Optional but useful on newer Debian/RPiOS for packaging completeness
apt_install_one python3-full || true

# Camera stack (Picamera2 from apt is recommended)
apt_install_one python3-numpy
apt_install_one python3-simplejpeg
apt_install_one python3-picamera2
apt_install_one python3-pil

# OpenCV via APT (matches your request; venv will see it if we use --system-site-packages)
apt_install_one python3-opencv

# I2C tools + SMBus helpers
apt_install_one i2c-tools
apt_install_one python3-smbus
apt_install_one python3-smbus2

# Common tooling
apt_install_one python3-pip
apt_install_one git

# Optional: rpicam/libcamera apps for quick camera test on HDMI/local
apt_install_one rpicam-apps || true
apt_install_one libcamera-apps || true

# ---------- Enable I2C ----------
echo "==> Enabling I2C (raspi-config if available)..."
if have_cmd raspi-config; then
  sudo raspi-config nonint do_i2c 0 || echo "NOTE: raspi-config I2C enable returned non-zero (continuing)."
  REBOOT_NEEDED=1
else
  echo "NOTE: raspi-config not found; skipping automatic I2C enable."
fi

# Add user to i2c group (helps running without sudo later)
if id -nG "$USER" | grep -qw "i2c"; then
  echo "User '$USER' already in 'i2c' group."
else
  echo "==> Adding '$USER' to i2c group..."
  sudo usermod -aG i2c "$USER" || echo "WARNING: Could not add user to i2c group."
  REBOOT_NEEDED=1
fi

if [[ ! -e /dev/i2c-1 ]]; then
  echo "NOTICE: /dev/i2c-1 not present yet (reboot may be required)."
  REBOOT_NEEDED=1
fi

# ---------- Create venv: tflite-env ----------
VENV_DIR="$HOME/tflite-env"
echo "==> Creating venv at: $VENV_DIR"
# Use system-site-packages so apt packages (picamera2, cv2) are visible in venv
python3 -m venv --system-site-packages "$VENV_DIR" || { echo "ERROR: Failed to create venv."; exit 1; }

echo "==> Activating venv..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools || echo "WARNING: pip upgrade failed (continuing)."

# ---------- pip installs inside venv ----------
echo "==> Installing TFLite runtime in venv..."
python -m pip install --upgrade tflite-runtime || { echo "ERROR: Failed to install tflite-runtime."; deactivate; exit 1; }

echo "==> Installing PCA9685 libraries in venv..."
python -m pip install --upgrade adafruit-blinka adafruit-circuitpython-pca9685 Adafruit-PCA9685 \
  || echo "WARNING: PCA9685 pip installs had an issue (continuing)."

deactivate || true

# ---------- Quick checks ----------
echo "==> I2C scan (may show nothing if nothing is connected):"
sudo i2cdetect -y 1 || true

echo "==> Verifying imports inside venv..."
source "$VENV_DIR/bin/activate"
python - <<'PY'
def check(mod):
    try:
        __import__(mod)
        print("OK:", mod)
    except Exception as e:
        print("FAIL:", mod, "->", e)

check("picamera2")
check("cv2")
check("numpy")
check("tflite_runtime.interpreter")
check("adafruit_pca9685")
PY
deactivate || true

# ---------- Summary ----------
if (( ${#FAILED_PKGS[@]} )); then
  echo
  echo "The following APT packages failed to install:"
  printf '  - %s\n' "${FAILED_PKGS[@]}"
fi

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  echo
  echo "Reboot recommended so I2C/group changes take effect:"
  echo "  sudo reboot"
fi

echo
echo "Done."
echo "Activate your environment with:"
echo "  source ~/tflite-env/bin/activate"
echo "Run your program with:"
echo "  python3 your_program.py"
