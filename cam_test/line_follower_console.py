#!/usr/bin/env python3
# line_follower_console.py
# Preview window opens first, then asks: "Do you want to start line follower? (y/n)"
# Starts console output only after you type 'y'. Type 'n' to exit.

import os

# Choose a Qt platform before importing PyQt5 for more reliable preview startup.
# Prefer X11 (xcb) if DISPLAY is present; otherwise try Wayland; else offscreen.
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
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
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

def make_mask_black(H, S, V, s_max=80, v_max=80):
    # Dark line on light floor
    return (V <= v_max) & (S <= s_max)

def in_hue_range(H, h_lo_deg, h_hi_deg):
    # H is 0..255 ~ 0..360 deg
    lo = int(round(h_lo_deg * 255.0 / 360.0)) % 256
    hi = int(round(h_hi_deg * 255.0 / 360.0)) % 256
    if lo <= hi:
        return (H >= lo) & (H <= hi)
    else:
        # wrap-around
        return (H >= lo) | (H <= hi)

def make_mask_color(H, S, V, h_lo=20, h_hi=40, s_min=60, v_min=60):
    return in_hue_range(H, h_lo, h_hi) & (S >= s_min) & (V >= v_min)

def centroid_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None, 0
    return xs.mean(), ys.mean(), len(xs)

def pca_angle_deg(mask):
    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n < 50:
        return None
    x = xs.astype(np.float32)
    y = ys.astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    M = np.stack([x, y], axis=0)
    C = (M @ M.T) / n
    w, V = np.linalg.eigh(C)
    v = V[:, 1]
    angle = math.degrees(math.atan2(v[1], v[0]))  # 0°=horizontal, 90°=vertical
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

# Lightweight PyQt5 preview helper
class PreviewWindow:
    def __init__(self, title="Line follower preview"):
        # Block preview if no GUI session
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            raise RuntimeError("No GUI session detected (QT_QPA_PLATFORM=offscreen). Run on the Pi desktop.")
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

            # ROI rectangle
            y0 = overlays.get("roi_y0", 0)
            y1 = overlays.get("roi_y1", h)
            pen.setColor(QColor("lime"))
            painter.setPen(pen)
            painter.drawRect(0, y0, w - 1, y1 - y0 - 1)

            # centroid dot (cx, cy are ROI-local; add y0)
            cx = overlays.get("cx")
            cy = overlays.get("cy")
            if cx is not None and cy is not None:
                pen.setColor(QColor("red"))
                pen.setWidth(2)
                painter.setPen(pen)
                r = 6
                painter.drawEllipse(int(cx) - r, int(y0 + cy) - r, 2 * r, 2 * r)

            # angle segment
            ang = overlays.get("angle")
            if ang is not None and cx is not None and cy is not None:
                length = min(w, h) // 8
                rad = math.radians(ang)
                x0 = int(cx - length * math.cos(rad) / 2)
                y0_line = int(y0 + cy - length * math.sin(rad) / 2)
                x1 = int(cx + length * math.cos(rad) / 2)
                y1_line = int(y0 + cy + length * math.sin(rad) / 2)
                pen.setColor(QColor("yellow"))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawLine(x0, y0_line, x1, y1_line)

            painter.end()

        self.label.setPixmap(QPixmap.fromImage(qimg))
        # Keep UI responsive without blocking the script
        self.app.processEvents()

def main():
    ap = argparse.ArgumentParser(description="Line-following console analyzer (no OpenCV)")
    ap.add_argument("--size", default="1280x720", help="Resolution WxH (e.g. 1280x720)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--roi-height", type=float, default=0.35, help="Bottom ROI height fraction (0..1)")
    ap.add_argument("--mode", choices=["black", "color"], default="black", help="Detect dark line or a colored line")
    # thresholds for black mode
    ap.add_argument("--v-max", type=int, default=80, help="Max V for black line")
    ap.add_argument("--s-max", type=int, default=100, help="Max S for black line")
    # thresholds for color mode
    ap.add_argument("--h-lo", type=float, default=20.0, help="Hue low in degrees (0..360)")
    ap.add_argument("--h-hi", type=float, default=40.0, help="Hue high in degrees (0..360)")
    ap.add_argument("--s-min", type=int, default=80, help="Min S for colored line")
    ap.add_argument("--v-min", type=int, default=60, help="Min V for colored line")
    # decision parameters
    ap.add_argument("--deadband", type=float, default=0.05, help="Deadband on normalized error e")
    ap.add_argument("--min-coverage", type=float, default=0.002, help="Min mask fraction to accept line (~0.2%)")
    ap.add_argument("--invert-steer", action="store_true", help="Invert left/right decision if camera mount is mirrored")
    ap.add_argument("--print-rate", type=float, default=10.0, help="Status lines per second")
    ap.add_argument("--preview", action="store_true", help="Show live camera preview with overlays (PyQt5)")
    args = ap.parse_args()

    # 1) Create the preview window FIRST (if requested)
    preview = None
    if args.preview:
        preview = PreviewWindow(title="Line follower preview")
        # Ensure it shows before other processing
        preview.app.processEvents()
        time.sleep(0.05)

    # 2) Configure and start the camera
    W, H = map(int, args.size.lower().split("x"))
    picam2 = Picamera2()
    configure(picam2, (W, H), args.fps)
    picam2.start()

    # 3) Preview-only phase with non-blocking console prompt
    #    The preview keeps updating while we wait for 'y'/'n'.
    want_start = False
    if args.preview:
        print("Do you want to start line follower? (y/n): ", end="", flush=True)
        try:
            while True:
                frame = picam2.capture_array()
                # Draw overlays (ROI + centroid + angle), same as main loop but no console prints
                y0 = int(H * (1.0 - args.roi_height))
                roi = frame[y0:H, :, :]
                hR, wR = roi.shape[0], roi.shape[1]

                Hc, Sc, Vc = rgb_to_hsv_np(roi)
                if args.mode == "black":
                    mask = make_mask_black(Hc, Sc, Vc, s_max=args.s_max, v_max=args.v_max)
                else:
                    mask = make_mask_color(Hc, Sc, Vc, h_lo=args.h_lo, h_hi=args.h_hi, s_min=args.s_min, v_min=args.v_min)

                coverage = mask.mean()
                cx = cy = None
                ang = None
                if coverage >= args.min_coverage:
                    cx, cy, _ = centroid_from_mask(mask)
                    if cx is not None:
                        ang = pca_angle_deg(mask)

                if preview is not None:
                    overlays = {
                        "roi_y0": y0,
                        "roi_y1": H,
                        "cx": cx if coverage >= args.min_coverage else None,
                        "cy": cy if coverage >= args.min_coverage else None,
                        "angle": ang,
                    }
                    preview.draw_and_show(frame, overlays=overlays)

                # Non-blocking read of user's response
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ans = sys.stdin.readline().strip().lower()
                    if ans.startswith("y"):
                        want_start = True
                        print("\nStarting line follower...")
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
        ans = input("Do you want to start line follower? (y/n): ").strip().lower()
        if ans.startswith("y"):
            want_start = True
        else:
            print("Exiting without starting.")
            return

    # 4) Now announce and start normal console output loop
    print("Running. Ctrl+C to stop.")
    last_print = 0.0
    print_period = 1.0 / max(1e-3, args.print_rate)

    try:
        while want_start:
            frame = picam2.capture_array()  # HxWx3 RGB888
            # Bottom ROI
            y0 = int(H * (1.0 - args.roi_height))
            roi = frame[y0:H, :, :]
            hR, wR = roi.shape[0], roi.shape[1]

            Hc, Sc, Vc = rgb_to_hsv_np(roi)

            if args.mode == "black":
                mask = make_mask_black(Hc, Sc, Vc, s_max=args.s_max, v_max=args.v_max)
            else:
                mask = make_mask_color(Hc, Sc, Vc, h_lo=args.h_lo, h_hi=args.h_hi, s_min=args.s_min, v_min=args.v_min)

            coverage = mask.mean()
            status = ""
            decision = ""
            e_val = None
            cx = cy = None
            ang = None

            if coverage < args.min_coverage:
                status = "line lost"
                decision = "search"
            else:
                cx, cy, _ = centroid_from_mask(mask)
                if cx is None:
                    status = "line lost"
                    decision = "search"
                else:
                    # Normalized lateral error: e = (cx - wR/2) / (wR/2)
                    e_val = float((cx - (wR / 2)) / (wR / 2))
                    if abs(e_val) <= args.deadband:
                        decision = "go straight"
                        status = "on line (centered)"
                    elif e_val < -args.deadband:
                        status = "line LEFT of center"
                        decision = "turn LEFT"
                    else:
                        status = "line RIGHT of center"
                        decision = "turn RIGHT"
                    ang = pca_angle_deg(mask)

            # Optional steering inversion
            invert = args.invert_steer
            if invert and ("turn LEFT" in decision or "turn RIGHT" in decision):
                decision = "turn RIGHT" if "LEFT" in decision else "turn LEFT"
                status = status.replace("LEFT", "TEMP").replace("RIGHT", "LEFT").replace("TEMP", "RIGHT")

            # Update preview (if enabled)
            if preview is not None:
                overlays = {
                    "roi_y0": y0,
                    "roi_y1": H,
                    "cx": cx if coverage >= args.min_coverage else None,
                    "cy": cy if coverage >= args.min_coverage else None,
                    "angle": ang,
                }
                preview.draw_and_show(frame, overlays=overlays)

            # Console output at chosen rate
            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                cov_pct = 100.0 * coverage
                if e_val is None:
                    print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | {status} | decision={decision}")
                else:
                    if ang is None:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | {status} -> {decision}")
                    else:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | angle={ang:+.1f}° | {status} -> {decision}")

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
