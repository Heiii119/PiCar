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
echo "Adding Coral repository..."
echo "======================================"

# Add Coral GPG key
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
| sudo gpg --dearmor -o /usr/share/keyrings/coral-archive-keyring.gpg

# Add Coral repo
echo "deb [signed-by=/usr/share/keyrings/coral-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt update

echo "======================================"
echo "Installing Coral Edge TPU runtime..."
echo "======================================"
sudo apt install -y libedgetpu1-max

echo "======================================"
echo "Creating virtual environment: car-venv"
echo "======================================"

python3 -m venv car-venv
source car-venv/bin/activate

echo "======================================"
echo "Upgrading pip..."
echo "======================================"
pip install --upgrade pip setuptools wheel

echo "======================================"
echo "Installing Python packages..."
echo "======================================"

pip install \
    numpy \
    opencv-python \
    tflite-runtime \
    Adafruit-Blinka \
    adafruit-circuitpython-busdevice \
    adafruit-circuitpython-connectionmanager \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-register \
    adafruit-circuitpython-requests \
    adafruit-circuitpython-typing \
    Adafruit_GPIO \
    Adafruit_PCA9685 \
    Adafruit-PlatformDetect \
    Adafruit-PureIO \
    ultralytics \
    tensorflow

echo "======================================"
echo "✅ Installation Complete"
echo "======================================"
echo ""
echo "Activate environment with:"
echo "source car-venv/bin/activate"
echo ""
echo "Reboot recommended."
echo "======================================"
