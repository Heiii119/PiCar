### Install the PyCoral API and other pip packages
```bash
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 Pillow opencv-python opencv-contrib-python
```

## Copy these files into the same folder
```bash
scp model.tflite pi@raspberrypi.local:/home/pi/PiCar/
scp labels.txt   pi@raspberrypi.local:/home/pi/PiCar/
```
