# Autopilot (PilotNet) module

This workspace includes `autopilot_pilotnet.py` — a self-contained module that reuses
camera and motor helpers from `drive_train_autopilot_picam2.py` and provides:

- PilotNet-like model builder
- tf.data augmentation pipeline (flip + brightness/contrast)
- Training harness that saves a Keras `.h5` model
- Optional TFLite export for on-device inference
- `run_autopilot` loop with keyboard override

Quick examples (PowerShell)

Record a session using the existing tool in the repo (from the main script):

python .\drive_train_autopilot_picam2.py

Train a PilotNet model on a recorded session (replace session folder):

python .\autopilot_pilotnet.py train data\session_YYYYMMDD_HHMMSS --epochs 25 --batch 32

Train and export TFLite:

python .\autopilot_pilotnet.py train data\session_YYYYMMDD_HHMMSS --tflite

Run autopilot using the Keras model:

python .\autopilot_pilotnet.py run data\session_YYYYMMDD_HHMMSS\model_pilotnet.h5

Run autopilot with TFLite:

python .\autopilot_pilotnet.py run data\session_YYYYMMDD_HHMMSS\model_pilotnet.tflite --tflite

Notes

- This module expects TensorFlow to be installed on the device (TensorFlow 2.x).
- The code reuses `PiCam2Manager` and `MotorServoController` which require the Raspberry Pi environment and the PCA9685 driver to be present.
- If you want a smaller model for Raspberry Pi, use the `--tflite` export and consider quantization.
