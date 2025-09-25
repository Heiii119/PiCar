#!/usr/bin/env python3
# test_hue_rotation_qt.py
import sys, time, os
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from picamera2 import Picamera2

def fps_to_us(fps):
    fps = max(1, int(fps))
    return max(3333, int(1_000_000 / fps))

def configure(picam2, size=(1280,720), fps=30):
    frame_us = fps_to_us(fps)
    cfg = picam2.create_video_configuration(main={"size": size, "format": "RGB888"},
                                            controls={"FrameDurationLimits": (frame_us, frame_us)})
    picam2.configure(cfg)

class Viewer(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color:black;")
        self._pix = None

    def show_rgb(self, rgb):
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        self._pix = QtGui.QPixmap.fromImage(qimg.copy())
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = Viewer()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.viewer)

        self.picam2 = Picamera2()
        configure(self.picam2)
        self.picam2.start()

        self.h_shift = 0  # 0..255
        self._last_rgb = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        os.makedirs("captures", exist_ok=True)
        self.frames = 0
        self.t0 = time.time()
        self._update_title(0.0)

    def _update_title(self, fps):
        degrees = (self.h_shift / 255.0) * 360.0
        self.setWindowTitle(f"Hue rotation: {self.h_shift}/255 ({degrees:.1f}°)  meas {fps:.1f} fps")

    def keyPressEvent(self, e):
        k = e.key()
        if k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtWidgets.QApplication.quit()
        elif k == QtCore.Qt.Key_Left:
            self.h_shift = (self.h_shift - 10) % 256
        elif k == QtCore.Qt.Key_Right:
            self.h_shift = (self.h_shift + 10) % 256
        elif k == QtCore.Qt.Key_Up:
            self.h_shift = (self.h_shift + 1) % 256
        elif k == QtCore.Qt.Key_Down:
            self.h_shift = (self.h_shift - 1) % 256
        elif k == QtCore.Qt.Key_0:
            self.h_shift = 0
        elif k == QtCore.Qt.Key_S:
            self._save_frame()

    def _save_frame(self):
        if self._last_rgb is None: return
        h, w, _ = self._last_rgb.shape
        qimg = QtGui.QImage(self._last_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"captures/hue_rot_{self.h_shift}_{ts}.png"
        qimg.copy().save(path)
        print(f"Saved {path}")

    def _apply_hue_rotation(self, rgb_np):
        img = Image.fromarray(rgb_np, "RGB").convert("HSV")
        h, s, v = img.split()
        h_np = np.array(h, dtype=np.uint8)
        h_rot = ((h_np.astype(np.uint16) + int(self.h_shift)) % 256).astype(np.uint8)
        img_rot = Image.merge("HSV", (Image.fromarray(h_rot, "L"), s, v)).convert("RGB")
        return np.array(img_rot)

    def update_frame(self):
        rgb = self.picam2.capture_array()
        rgb_rot = self._apply_hue_rotation(rgb)
        self._last_rgb = rgb_rot
        self.viewer.show_rgb(rgb_rot)
        self.frames += 1
        if self.frames % 30 == 0:
            elapsed = time.time() - self.t0
            fps = self.frames / max(1e-6, elapsed)
            self._update_title(fps)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.resize(960, 540)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
