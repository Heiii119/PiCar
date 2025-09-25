#!/usr/bin/env python3
# test_res_fps_qt.py
import sys, time, os
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from picamera2 import Picamera2

def fps_to_us(fps):
    fps = max(1, int(fps))
    return max(3333, int(1_000_000 / fps))

def unique_sizes_from_modes(picam2):
    sizes = []
    modes = getattr(picam2, "sensor_modes", None) or []
    for m in modes:
        sz = tuple(m.get("size", ()))
        if len(sz) == 2 and sz not in sizes:
            sizes.append(sz)
    if not sizes:
        sizes = [(1280, 720), (1920, 1080), (2304, 1296)]
    sizes.sort(key=lambda s: s[0]*s[1])
    return sizes, modes

def configure(picam2, size, fps):
    frame_us = fps_to_us(fps)
    config = picam2.create_video_configuration(
        main={"size": size, "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_us, frame_us)},
    )
    picam2.configure(config)

class Viewer(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color:black;")
        self._last_pix = None

    def show_rgb(self, rgb):
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())
        self._last_pix = pix
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._last_pix is not None:
            scaled = self._last_pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resolution/FPS Preview")
        self.viewer = Viewer()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.viewer)

        self.picam2 = Picamera2()
        self.sizes, modes = unique_sizes_from_modes(self.picam2)
        print("Available sensor modes:")
        for i, m in enumerate(modes or []):
            print(f" [{i}] {m}")
        self.size_idx = min(1, len(self.sizes)-1)
        self.fps_options = [15, 30, 60]
        self.fps_idx = 1

        self.current_size = self.sizes[self.size_idx]
        self.current_fps = self.fps_options[self.fps_idx]
        configure(self.picam2, self.current_size, self.current_fps)
        self.picam2.start()

        self.t_prev = time.time()
        self.frames = 0
        self.meas_fps = 0.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        os.makedirs("captures", exist_ok=True)
        self._last_rgb = None
        self._update_title()

    def _update_title(self):
        self.setWindowTitle(f"{self.current_size[0]}x{self.current_size[1]}  req {self.current_fps} fps  meas {self.meas_fps:.1f} fps")

    def keyPressEvent(self, e):
        k = e.key()
        if k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            QtWidgets.QApplication.quit()
        elif k == QtCore.Qt.Key_R:
            self._change_resolution()
        elif k == QtCore.Qt.Key_F:
            self._change_fps()
        elif k == QtCore.Qt.Key_S:
            self._save_frame()

    def _change_resolution(self):
        self.size_idx = (self.size_idx + 1) % len(self.sizes)
        self.current_size = self.sizes[self.size_idx]
        self._reconfigure()

    def _change_fps(self):
        self.fps_idx = (self.fps_idx + 1) % len(self.fps_options)
        self.current_fps = self.fps_options[self.fps_idx]
        self._reconfigure()

    def _reconfigure(self):
        self.timer.stop()
        self.picam2.stop()
        configure(self.picam2, self.current_size, self.current_fps)
        self.picam2.start()
        self.t_prev = time.time()
        self.frames = 0
        self.meas_fps = 0.0
        self.timer.start(0)
        self._update_title()

    def _save_frame(self):
        if self._last_rgb is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"captures/{self.current_size[0]}x{self.current_size[1]}_{self.current_fps}fps_{ts}.png"
        QtGui.QImage(self._last_rgb.data, self._last_rgb.shape[1], self._last_rgb.shape[0],
                     3*self._last_rgb.shape[1], QtGui.QImage.Format_RGB888).copy().save(path)
        print(f"Saved {path}")

    def update_frame(self):
        rgb = self.picam2.capture_array()  # RGB888
        self._last_rgb = rgb
        self.viewer.show_rgb(rgb)
        self.frames += 1
        now = time.time()
        dt = now - self.t_prev
        if dt >= 0.5:
            self.meas_fps = self.frames / dt
            self.frames = 0
            self.t_prev = now
            self._update_title()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.resize(960, 540)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
