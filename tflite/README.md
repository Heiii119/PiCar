### Install the PyCoral API and other pip packages
```bash
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 Pillow opencv-python opencv-contrib-python
```

### Save the files as
C:\Users\YourName\PiCarModels\traffic_signs\model.tflite

C:\Users\YourName\PiCarModels\traffic_signs\labels.txt

### On PC, Copy these files into the same folder
```bash
cd ~/PiCarModels/traffic_signs
scp model.tflite pi@raspberrypi.local:/home/pi/PiCar/
scp labels.txt   pi@raspberrypi.local:/home/pi/PiCar/
```
