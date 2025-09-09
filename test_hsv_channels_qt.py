#!/usr/bin/env python3
# test_hsv_channels_qt.py
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
        self.setWindowTitle("RGB + HSV Channels Viewer")

        self.picam2 = Picamera2()
        configure(self.picam2)
        self.picam2.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        self._last_grid = None
        os.makedirs("captures", exist_ok=True)

        self.frames = 0
        self.t0 = time.time()

    def keyPressEvent(self, e):
        if e.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtWidgets.QApplication.quit()
        elif e.key() == QtCore.Qt.Key_S:
            self._save_grid()

    def _save_grid(self):
        if self._last_grid is None: return
        h, w, _ = self._last_grid.shape
        qimg = QtGui.QImage(self._last_grid.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"captures/rgb_hsv_grid_{ts}.png"
        qimg.copy().save(path)
        print(f"Saved {path}")

    def _make_grid(self, rgb_np):
        img = Image.fromarray(rgb_np, "RGB")
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        # Colorized Hue (S=255, V=255), and grayscale S/V
        h_color = Image.merge("HSV", (h, Image.new("L", h.size, 255), Image.new("L", h.size, 255))).convert("RGB")
        s_rgb = Image.merge("RGB", (s, s, s))
        v_rgb = Image.merge("RGB", (v, v, v))

        rgb = np.array(img)
        h_col = np.array(h_color)
        s_img = np.array(s_rgb)
        v_img = np.array(v_rgb)

        top = np.hstack([rgb, h_col])
        bot = np.hstack([s_img, v_img])
        return np.vstack([top, bot])

    def update_frame(self):
        rgb = self.picam2.capture_array()
        grid = self._make_grid(rgb)
        self._last_grid = grid
        self.viewer.show_rgb(grid)
        self.frames += 1
        if self.frames % 30 == 0:
            elapsed = time.time() - self.t0
            fps = self.frames / max(1e-6, elapsed)
            self.setWindowTitle(f"RGB + HSV Channels Viewer  (meas {fps:.1f} fps)")

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
