from picamera2 import Picamera2, Preview
import time
import os

# Create a Picamera2 object
picam2 = Picamera2()

# Configure the camera for still images
# A lower resolution 'lores' stream is used for the preview to save memory
camera_config = picam2.create_still_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(camera_config)

# Start the preview window and the camera
picam2.start_preview(Preview.QTGL) # Use "qtgl" or "drm" depending on your system
picam2.start()

# Pause to allow the camera sensors to adjust light levels
time.sleep(2)

# Capture the image and save it to a file
picam2.capture_file("test.jpg")

# Stop the camera and preview
picam2.stop_preview()
picam2.stop()

print("Image saved as test.jpg")
