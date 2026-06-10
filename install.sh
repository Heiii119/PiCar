#!/bin/bash

set -e

echo "======================================"
echo "Updating system..."
echo "======================================"
sudo apt update
sudo apt upgrade -y

echo "======================================"
echo "Installing system dependencies..."
echo "======================================"
# Note: removed python3-opencv (we use pip opencv-python for version-matching).
# These libs are runtime deps that pip opencv/tflite-runtime link against.
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    libjpeg-dev \
    libopenjp2-7 \
    libtiff6 \
    libopenblas-dev \
    libatlas3-base \
    i2c-tools

echo "======================================"
echo "Enabling I2C..."
echo "======================================"
sudo raspi-config nonint do_i2c 0

echo "======================================"
echo "Creating virtual environment: car-venv"
echo "======================================"
# Isolated venv (no --system-site-packages) so OpenCV/NumPy stay version-matched.
python3 -m venv car-venv
source car-venv/bin/activate

echo "======================================"
echo "Upgrading pip..."
echo "======================================"
pip install --upgrade pip setuptools wheel

echo "======================================"
echo "Installing Python packages..."
echo "======================================"
# AI inference stack (Teachable Machine TFLite model — sign_detection.py)
pip install \
    numpy==1.26.4 \
    opencv-python \
    tflite-runtime

# Hardware: Adafruit Blinka + PCA9685 motor/servo HAT
pip install \
    Adafruit-Blinka \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    adafruit-circuitpython-motorkit \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-busdevice \
    adafruit-circuitpython-register \
    Adafruit-PlatformDetect \
    Adafruit-PureIO

# Web interface (matches web_interface.py)
pip install \
    flask \
    flask-socketio \
    eventlet

echo "======================================"
echo "Verifying installation..."
echo "======================================"
python -c "import cv2, numpy, board, busio, flask; \
from tflite_runtime.interpreter import Interpreter; \
print('OpenCV  :', cv2.__version__); \
print('TFLite  : OK (Interpreter imported)'); \
print('NumPy   :', numpy.__version__); \
print('Blinka  : OK (board, busio imported)'); \
print('Flask   :', flask.__version__)"

echo "======================================"
echo "✅ Installation Complete"
echo "======================================"
echo ""
echo "Activate environment with:"
echo "  source car-venv/bin/activate"
echo ""
echo "Check the HAT is on the I2C bus with:"
echo "  i2cdetect -y 1   (expect 0x40 for PCA9685)"
echo ""
echo "Reboot recommended (I2C enable)."
echo "======================================"
