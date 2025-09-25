# save as test_qtgl_preview.py
import time
from picamera2 import Picamera2, Preview
from libcamera import Transform

def main():
    picam2 = Picamera2()

    # Older API: only pass main; do not pass "preview" keyword
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XBGR8888"},  # if this fails, try "XRGB8888" or omit format
        transform=Transform(hflip=False, vflip=False)
    )
    picam2.configure(config)

    # Start Qt windowed preview
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    print("QTGL preview running. Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            picam2.stop_preview()
        except Exception:
            pass
        picam2.stop()

if __name__ == "__main__":
    main()
