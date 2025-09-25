#!/usr/bin/env python3
# camera_inspector.py
# Opens preview first, keeps updating while asking:
# "Do you want to start inspector? (y/n)"
# Starts console output only after 'y'. 'n' exits.

import os

# Harden Qt startup before importing PyQt5.
# Prefer X11 (xcb) if DISPLAY is present; else Wayland; else offscreen.
if "QT_QPA_PLATFORM" not in os.environ:
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    elif os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    else:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse, time, math, sys, select
import numpy as np
from PIL import Image
from picamera2 import Picamera2

# Optional PyQt5 preview (only used if --preview is set)
try:
    from PyQt5.QtWidgets import QApplication, QLabel
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
    from PyQt5.QtCore import Qt
except Exception:
    QApplication = None

def fps_to_us(fps):
    fps = max(1, int(fps))
    return max(3333, int(1_000_000 / fps))

def configure(picam2, size, fps):
    cfg = picam2.create_video_configuration(
        main={"size": size, "format": "RGB888"},
        controls={"FrameDurationLimits": (fps_to_us(fps), fps_to_us(fps))}
    )
    picam2.configure(cfg)

def rgb_to_hsv_np(rgb):
    img = Image.fromarray(rgb, "RGB").convert("HSV")
    H, S, V = img.split()
    return np.array(H, dtype=np.uint8), np.array(S, dtype=np.uint8), np.array(V, dtype=np.uint8)

# Lightweight PyQt5 preview helper
class PreviewWindow:
    def __init__(self, title="Camera inspector preview"):
        # Block preview if no GUI session
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            raise RuntimeError("No GUI session detected (QT_QPA_PLATFORM=offscreen). Run on the Pi desktop or set DISPLAY.")
        if QApplication is None:
            raise RuntimeError("PyQt5 is not available. Install python3-pyqt5 or run without --preview.")
        self.app = QApplication.instance() or QApplication([])
        self.label = QLabel()
        self.label.setWindowTitle(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 360)
        self.label.show()
        # Bring to front and ensure it paints
        self.label.raise_()
        self.label.activateWindow()
        self.app.processEvents()

    def draw_and_show(self, frame_rgb, overlays=None):
        h, w, _ = frame_rgb.shape
        # Wrap numpy data in QImage, then copy so we can paint safely
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        if overlays:
            painter = QPainter(qimg)

            # center vertical line
            pen = QPen(QColor("cyan"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(w // 2, 0, w // 2, h)

            # Optional text message
            msg = overlays.get("text")
            if msg:
                # Shadow for readability
                painter.setFont(QFont("Sans", 12))
                pen_black = QPen(QColor(0, 0, 0))
                pen_black.setWidth(3)
                painter.setPen(pen_black)
                painter.drawText(12, 24, msg)
                painter.setPen(QPen(QColor("white")))
                painter.drawText(10, 22, msg)

            painter.end()

        self.label.setPixmap(QPixmap.fromImage(qimg))
        # Keep UI responsive without blocking the script
        self.app.processEvents()

def main():
    ap = argparse.ArgumentParser(description="Camera inspector (Picamera2, optional PyQt5 preview)")
    ap.add_argument("--size", default="1280x720", help="Resolution WxH (e.g. 1280x720)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--print-rate", type=float, default=2.0, help="Status lines per second")
    ap.add_argument("--preview", action="store_true", help="Show live camera preview (PyQt5)")
    args = ap.parse_args()

    # 1) Create the preview window FIRST (if requested)
    preview = None
    if args.preview:
        preview = PreviewWindow(title="Camera Inspector Preview")
        # Ensure it shows before other processing
        preview.app.processEvents()
        time.sleep(0.05)

    # 2) Configure and start the camera
    try:
        W, H = map(int, args.size.lower().split("x"))
    except Exception:
        raise SystemExit("Invalid --size. Use WxH, e.g. 1280x720")

    picam2 = Picamera2()
    configure(picam2, (W, H), args.fps)
    picam2.start()

    # 3) Preview-only phase with non-blocking console prompt
    #    The preview keeps updating while we wait for 'y'/'n'.
    want_start = False
    if args.preview:
        print("Do you want to start line inspector? (y/n): ", end="", flush=True)
        try:
            while True:
                frame = picam2.capture_array()
                # Show preview with a hint message; no console prints yet
                if preview is not None:
                    overlays = {
                        "text": "Waiting for input in terminal: y to start, n to exit"
                    }
                    preview.draw_and_show(frame, overlays=overlays)
                    # If user closed the preview window, exit
                    if not preview.label.isVisible():
                        print("\nPreview window closed. Exiting.")
                        return

                # Non-blocking read of user's response
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ans = sys.stdin.readline().strip().lower()
                    if ans.startswith("y"):
                        want_start = True
                        print("\nStarting camera inspector...")
                        break
                    elif ans.startswith("n"):
                        print("\nExiting without starting.")
                        return
                    else:
                        print("Please type y or n: ", end="", flush=True)
        except KeyboardInterrupt:
            return
    else:
        # If no preview requested, just ask once (blocking)
        ans = input("Do you want to start inspector? (y/n): ").strip().lower()
        if ans.startswith("y"):
            want_start = True
        else:
            print("Exiting without starting.")
            return

    # 4) Now announce and start normal console output loop
    print("Running. Ctrl+C to stop.")
    last_print = 0.0
    print_period = 1.0 / max(1e-3, args.print_rate)
    ema_fps = None
    t_prev = time.time()

    try:
        while want_start:
            t_now = time.time()
            dt = max(1e-6, t_now - t_prev)
            t_prev = t_now
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            frame = picam2.capture_array()  # HxWx3 RGB888

            # Basic image stats
            Hc, Sc, Vc = rgb_to_hsv_np(frame)
            mean_v = float(np.mean(Vc))
            mean_r = float(np.mean(frame[:, :, 0]))
            mean_g = float(np.mean(frame[:, :, 1]))
            mean_b = float(np.mean(frame[:, :, 2]))

            # Metadata (if available)
            md = {}
            try:
                md = picam2.capture_metadata() or {}
            except Exception:
                md = {}

            pieces = [f"res={frame.shape[1]}x{frame.shape[0]}", f"fps={ema_fps:.1f}"]
            pieces.append(f"meanV={mean_v:.1f} meanRGB=({mean_r:.1f},{mean_g:.1f},{mean_b:.1f})")

            exp_us = md.get("ExposureTime")
            if isinstance(exp_us, (int, float)):
                pieces.append(f"exp={int(exp_us)}us")
            gain = md.get("AnalogueGain")
            if isinstance(gain, (int, float)):
                pieces.append(f"gain={gain:.2f}x")
            lux = md.get("Lux")
            if isinstance(lux, (int, float)):
                pieces.append(f"lux={lux:.1f}")
            if "ColourGains" in md and isinstance(md["ColourGains"], (tuple, list)) and len(md["ColourGains"]) >= 2:
                r_gain, b_gain = md["ColourGains"][0], md["ColourGains"][1]
                try:
                    pieces.append(f"awb=({float(r_gain):.2f},{float(b_gain):.2f})")
                except Exception:
                    pass

            # Update preview (if enabled)
            if preview is not None:
                overlays = {"text": f"{pieces[0]} | {pieces[1]}"}
                preview.draw_and_show(frame, overlays=overlays)
                if not preview.label.isVisible():
                    print("Preview window closed. Exiting.")
                    return

            # Console output at chosen rate
            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                print(" | ".join(pieces))

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
