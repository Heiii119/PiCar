#!/usr/bin/env bash
set -euo pipefail

# Colors for logs
GREEN="$(printf '\033[1;32m')"
YELLOW="$(printf '\033[1;33m')"
RED="$(printf '\033[1;31m')"
NC="$(printf '\033[0m')"

log() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERR ]${NC} $*" >&2; }

# Ensure we are in home directory (adjust if desired)
cd "${HOME}"

log "Updating APT and installing system libraries (BLAS/LAPACK/HDF5, etc.) ..."
sudo apt update
sudo apt install -y libatlas-base-dev libhdf5-dev libblas-dev liblapack-dev gfortran

# Create venv if not exists
if [ ! -d "tf-venv" ]; then
  log "Creating Python venv at ${HOME}/tf-venv ..."
  python3 -m venv --system-site-packages tf-venv
else
  warn "Virtual environment tf-venv already exists. Reusing it."
fi

# Activate venv
# shellcheck disable=SC1091
source tf-venv/bin/activate

log "Upgrading pip, setuptools, wheel, packaging ..."
python -m pip install --upgrade pip setuptools wheel packaging

log "Purging pip cache ..."
pip cache purge || true

log "Installing helper package Send2Trash ..."
pip install "Send2Trash>=1.8.2"

log "Installing TensorFlow 2.16.1 (aarch64 wheel) ..."
pip install --no-cache-dir tensorflow-aarch64==2.16.1

log "Verifying TensorFlow installation ..."
python - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
print("CPU devices:", tf.config.list_physical_devices('CPU'))
PY

log "Installing Adafruit libraries (Blinka, PCA9685, etc.) ..."
pip install Adafruit-Blinka adafruit-circuitpython-busdevice adafruit-circuitpython-pca9685 adafruit-circuitpython-motor adafruit-circuitpython-servokit

log "Installing Picamera2 and libcamera tools ..."
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps libgpiod2

log "Installing i2c-tools ..."
sudo apt update
sudo apt install -y i2c-tools

log "Ensuring I2C kernel modules are loaded (i2c-dev) ..."
if ! lsmod | grep -q '^i2c_dev'; then
  sudo modprobe i2c-dev || true
fi

echo
log "Running verification checks inside venv ..."

# Re-activate in case user runs this section standalone
# shellcheck disable=SC1091
source tf-venv/bin/activate

python - <<'PY'
import board, busio
print("board OK:", hasattr(board, "SCL"), "SCL:", getattr(board, "SCL", None), "SDA:", getattr(board, "SDA", None))
PY

python - <<'PY'
import adafruit_pca9685
print("adafruit_pca9685 OK, module:", adafruit_pca9685.__file__)
PY

python - <<'PY'
from picamera2 import Picamera2
print("picamera2 OK")
PY

echo
log "Scanning I2C bus 1 (expect PCA9685 at 0x40 if connected) ..."
if command -v i2cdetect >/dev/null 2>&1; then
  i2cdetect -y 1 || true
else
  warn "i2cdetect not found. Skipping I2C scan."
fi

echo
log "All done. To use this environment, run:"
echo "  source ${HOME}/tf-venv/bin/activate"
