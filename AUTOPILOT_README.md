# Autopilot (PilotNet) module
This workspace includes `autopilot_pilotnet.py` — a self-contained module that reuses
camera and motor helpers from `drive_train_autopilot_picam2.py` and provides:

- PilotNet-like model builder
- tf.data augmentation pipeline (flip + brightness/contrast)
- Training harness that saves a Keras `.h5` model
- Optional TFLite export for on-device inference
- `run_autopilot` loop with keyboard override

## tensorflow installation
```bash
uname -m 
```
If uname -m prints aarch64, proceed with the TensorFlow install steps I shared:
```bash
sudo apt update && sudo apt install -y libatlas-base-dev libhdf5-dev libblas-dev liblapack-dev gfortran
python3 -m venv --system-site-packages tf-venv && source tf-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging
pip cache purge
pip install "Send2Trash>=1.8.2"
pip install --no-cache-dir tensorflow-aarch64
pip install tensorflow-aarch64==2.16.1
python -c "import tensorflow as tf; print(tf.version); print(tf.config.list_physical_devices('CPU'))"
```
```bash
pip install Adafruit-Blinka adafruit-circuitpython-busdevice adafruit-circuitpython-pca9685 adafruit-circuitpython-motor adafruit-circuitpython-servokit
```
```bash
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps libgpiod2
```
```bash
sudo raspi-config # enable I2C and SPI
```
```bash
sudo usermod -aG gpio,i2c,spi,video $USER
```
```bash
sudo reboot
```
verify
```bash
source tf-venv/bin/activate
python -c "import board, busio; print('board OK:', hasattr(board, 'SCL'))"
python -c "from picamera2 import Picamera2; print('picamera2 OK')"
```


Quick examples (PowerShell)

Record a session using the existing tool in the repo (from the main script):
```bash
python .\drive_train_autopilot_picam2.py
```
Train a PilotNet model on a recorded session (replace session folder):
```bash
python .\autopilot_pilotnet.py train data\session_YYYYMMDD_HHMMSS --epochs 25 --batch 32
```
Train and export TFLite:
```bash
python .\autopilot_pilotnet.py train data\session_YYYYMMDD_HHMMSS --tflite
```
Run autopilot using the Keras model:
```bash
python .\autopilot_pilotnet.py run data\session_YYYYMMDD_HHMMSS\model_pilotnet.h5
```
Run autopilot with TFLite:
```bash
python .\autopilot_pilotnet.py run data\session_YYYYMMDD_HHMMSS\model_pilotnet.tflite --tflite
```
Notes

- This module expects TensorFlow to be installed on the device (TensorFlow 2.x).
- The code reuses `PiCam2Manager` and `MotorServoController` which require the Raspberry Pi environment and the PCA9685 driver to be present.
- If you want a smaller model for Raspberry Pi, use the `--tflite` export and consider quantization.
