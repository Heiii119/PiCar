#!/usr/bin/env python3
# line_follower_console.py — presets + color-order toggle + invert steering
# - Choose preset in CONFIG (no long CLI)
# - Toggle COLOR_FLIP_BGR2RGB if your preview shows wrong colors (yellow looks blue)
# - Toggle INVERT_STEER if turn directions are reversed
# - Optional one-shot calibration still available with --calibrate (short)

import os

# Choose a Qt platform before importing PyQt5 for more reliable preview startup.
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

# Optional PyQt5 preview
try:
    from PyQt5.QtWidgets import QApplication, QLabel
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
    from PyQt5.QtCore import Qt
except Exception:
    QApplication = None

# ========== CONFIG: EDIT THESE LINES BEFORE RUNNING ==========
# 1) Choose line type: "black", "yellow", "red", "blue"
mode_choice = "yellow"

# 2) If preview shows wrong colors (yellow appears blue), set this True
#    This flips channel order from BGR->RGB consistently for both preview and HSV.
COLOR_FLIP_BGR2RGB = True

# 3) If the robot turns the wrong way (LEFT/RIGHT reversed), set this True
INVERT_STEER = False

# 4) Global tuning
roi_height_default = 0.45    # bottom 35% of the frame
deadband_default = 0.09      # normalized center error deadband
min_coverage_default = 0.005 # ~0.2% of ROI pixels must match to accept

# 5) Per-color HSV presets (degrees for hue; 0..255 for S, V thresholds)
LINE_COLOR_PRESETS = {
    "black": {  # dark on light
        "type": "black",
        "s_max": 100,
        "v_max": 80,
    },
    "yellow": {  # yellow on dark (black track)
        "type": "color",
        "h_lo": 10.0,
        "h_hi": 50.0,
        "s_min": 80,
        "v_min": 70,
    },
    "red": {  # red on bright floor (wrap-around)
        "type": "color",
        "h_lo": 350.0,  # wrap across 0°
        "h_hi": 10.0,
        "s_min": 90,
        "v_min": 80,
    },
    "blue": {  # blue on bright floor
        "type": "color",
        "h_lo": 200.0,
        "h_hi": 240.0,
        "s_min": 90,
        "v_min": 80,
    },
}
# =============================================================

def fps_to_us(fps):
    fps = max(1, int(fps))
    return max(3333, int(1_000_000 / fps))

def configure(picam2, size, fps):
    cfg = picam2.create_video_configuration(
        main={"size": size, "format": "RGB888"},
        controls={"FrameDurationLimits": (fps_to_us(fps), fps_to_us(fps))}
    )
    picam2.configure(cfg)

def ensure_rgb(img):
    """
    Ensure the frame is in RGB order.
    If COLOR_FLIP_BGR2RGB is True, flip channels from BGR->RGB.
    Returns a contiguous copy suitable for QImage and PIL.
    """
    img = img[:, :, :3]
    if COLOR_FLIP_BGR2RGB:
        return img[:, :, ::-1].copy()
    else:
        return img.copy()

def rgb_to_hsv_np(rgb):
    # rgb must be true RGB order already; call ensure_rgb before passing here if needed.
    img = Image.fromarray(rgb, "RGB").convert("HSV")
    H, S, V = img.split()
    return np.array(H, dtype=np.uint8), np.array(S, dtype=np.uint8), np.array(V, dtype=np.uint8)

def make_mask_black(H, S, V, s_max=80, v_max=80):
    return (V <= v_max) & (S <= s_max)

def in_hue_range(H, h_lo_deg, h_hi_deg):
    # H: 0..255 ~ 0..360 deg
    lo = int(round(h_lo_deg * 255.0 / 360.0)) % 256
    hi = int(round(h_hi_deg * 255.0 / 360.0)) % 256
    if lo <= hi:
        return (H >= lo) & (H <= hi)
    else:
        # wrap-around across 0 deg
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
    angle = math.degrees(math.atan2(v[1], v[0]))
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

# Optional: self-calibration helpers (usable with --calibrate)
def robust_percentiles(a, lo=5, hi=95):
    a = a.reshape(-1)
    return np.percentile(a, lo), np.percentile(a, hi)

def hue_degrees_from_H(H_u8):
    return (H_u8.astype(np.float32) * (360.0 / 255.0))

def estimate_hue_band_and_vmin(H_u8, S_u8, V_u8,
                               min_s_for_color=60,
                               center_crop_frac=0.5,
                               hue_margin_deg=8.0,
                               v_lo_percentile=30,
                               v_min_margin=5):
    h, w = H_u8.shape
    ch = int(h * center_crop_frac)
    cw = int(w * center_crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    Hc = H_u8[y0:y0+ch, x0:x0+cw]
    Sc = S_u8[y0:y0+ch, x0:x0+cw]
    Vc = V_u8[y0:y0+ch, x0:x0+cw]

    s_lo, _ = robust_percentiles(Sc, 50, 95)
    s_thresh = max(min_s_for_color, int(s_lo))
    candidates = Sc >= s_thresh
    if candidates.sum() < 100:
        s_thresh = max(40, int(s_lo * 0.8))
        candidates = Sc >= s_thresh
    if candidates.sum() == 0:
        candidates = np.ones_like(Sc, dtype=bool)

    H_deg = hue_degrees_from_H(Hc[candidates])
    V_sel = Vc[candidates]

    h_lo_deg, h_hi_deg = robust_percentiles(H_deg, 10, 90)
    h_lo_deg = max(0.0, h_lo_deg - hue_margin_deg)
    h_hi_deg = min(360.0, h_hi_deg + hue_margin_deg)

    v_lo, _ = robust_percentiles(V_sel, v_lo_percentile, 90)
    v_min = int(max(0, min(255, v_lo - v_min_margin)))
    return float(h_lo_deg), float(h_hi_deg), int(v_min)

# Preview helper
class PreviewWindow:
    def __init__(self, title="Line follower preview"):
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
        self.label.raise_()
        self.label.activateWindow()
        self.app.processEvents()

    def draw_and_show(self, frame_rgb, overlays=None):
        h, w, _ = frame_rgb.shape
        rgb = ensure_rgb(frame_rgb)  # enforce correct channel order for preview
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

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
        self.app.processEvents()

def main():
    # Minimal CLI: size, fps, preview, calibrate. No long threshold args needed.
    ap = argparse.ArgumentParser(description="Line-following console analyzer with presets and color fix toggles")
    ap.add_argument("--size", default="1280x720", help="Resolution WxH (e.g. 1280x720)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--preview", action="store_true", help="Show live camera preview with overlays (PyQt5)")
    ap.add_argument("--calibrate", action="store_true", help="Auto-estimate hue range and V-min once (color modes)")
    args = ap.parse_args()

    # Apply CONFIG choices
    if mode_choice not in LINE_COLOR_PRESETS:
        raise ValueError(f"Unknown mode_choice '{mode_choice}'. Choose one of: {list(LINE_COLOR_PRESETS.keys())}")
    preset = LINE_COLOR_PRESETS[mode_choice]

    mode = "color" if preset["type"] == "color" else "black"
    roi_height = roi_height_default
    deadband = deadband_default
    min_coverage = min_coverage_default

    # Unpack thresholds
    h_lo = preset.get("h_lo", 20.0)
    h_hi = preset.get("h_hi", 40.0)
    s_min = preset.get("s_min", 80)
    v_min = preset.get("v_min", 70)
    s_max = preset.get("s_max", 100)
    v_max = preset.get("v_max", 80)

    # Preview window (optional)
    preview = None
    if args.preview:
        fix_note = " (BGR->RGB fix ON)" if COLOR_FLIP_BGR2RGB else ""
        preview = PreviewWindow(title=f"Line follower preview [{mode_choice}]{fix_note}")
        preview.app.processEvents()
        time.sleep(0.05)

    # Camera
    W, H = map(int, args.size.lower().split("x"))
    picam2 = Picamera2()
    configure(picam2, (W, H), args.fps)
    picam2.start()

    # Optional one-shot calibration (only for color modes)
    if args.calibrate and mode == "color":
        print("Calibration: place the colored line centered inside the ROI box, then press Enter...")
        while True:
            frame = picam2.capture_array()
            y0 = int(H * (1.0 - roi_height))
            if preview is not None:
                preview.draw_and_show(frame, {"roi_y0": y0, "roi_y1": H})
            rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
            if rlist:
                _ = sys.stdin.readline()
                break
        roi = frame[y0:H, :, :]
        roi_rgb = ensure_rgb(roi)  # ensure correct channel order for HSV
        Hc, Sc, Vc = rgb_to_hsv_np(roi_rgb)
        h_lo_deg, h_hi_deg, v_min_est = estimate_hue_band_and_vmin(
            Hc, Sc, Vc,
            min_s_for_color=max(60, s_min - 10),
            center_crop_frac=0.5,
            hue_margin_deg=8.0,
            v_lo_percentile=30,
            v_min_margin=5
        )
        h_lo, h_hi = h_lo_deg, h_hi_deg
        v_min = max(v_min, v_min_est)  # don’t get darker than preset
        print(f"Calibrated: h=[{h_lo:.1f},{h_hi:.1f}]°, s_min={s_min}, v_min={v_min}")

    # Prompt to start (supports preview updating)
    want_start = False
    if preview is not None:
        print("Do you want to start line follower? (y/n): ", end="", flush=True)
        try:
            while True:
                frame = picam2.capture_array()
                y0 = int(H * (1.0 - roi_height))
                roi = frame[y0:H, :, :]
                roi_rgb = ensure_rgb(roi)  # consistent order for HSV and preview coloring
                Hc, Sc, Vc = rgb_to_hsv_np(roi_rgb)
                if mode == "black":
                    mask = make_mask_black(Hc, Sc, Vc, s_max=s_max, v_max=v_max)
                else:
                    mask = make_mask_color(Hc, Sc, Vc, h_lo=h_lo, h_hi=h_hi, s_min=s_min, v_min=v_min)
                cx = cy = ang = None
                if mask.mean() >= min_coverage:
                    cx, cy, _ = centroid_from_mask(mask)
                    if cx is not None:
                        ang = pca_angle_deg(mask)
                preview.draw_and_show(frame, {"roi_y0": y0, "roi_y1": H, "cx": cx, "cy": cy, "angle": ang})

                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    ans = sys.stdin.readline().strip().lower()
                    if ans.startswith("y"):
                        want_start = True
                        print("\nStarting line follower...")
                        break
                    elif ans.startswith("n"):
                        print("\nExiting without starting.")
                        picam2.stop()
                        return
                    else:
                        print("Please type y or n: ", end="", flush=True)
        except KeyboardInterrupt:
            picam2.stop()
            return
    else:
        ans = input("Do you want to start line follower? (y/n): ").strip().lower()
        if ans.startswith("y"):
            want_start = True
        else:
            print("Exiting without starting.")
            picam2.stop()
            return

    print(f"Running preset '{mode_choice}' (mode={mode}). "
          f"Color fix={'ON' if COLOR_FLIP_BGR2RGB else 'OFF'}, Invert steer={'ON' if INVERT_STEER else 'OFF'}. Ctrl+C to stop.")
    last_print = 0.0
    print_period = 1.0 / 10.0

    try:
        while want_start:
            frame = picam2.capture_array()
            y0 = int(H * (1.0 - roi_height))
            roi = frame[y0:H, :, :]
            hR, wR = roi.shape[0], roi.shape[1]

            roi_rgb = ensure_rgb(roi)  # ensure consistent RGB for HSV
            Hc, Sc, Vc = rgb_to_hsv_np(roi_rgb)

            if mode == "black":
                mask = make_mask_black(Hc, Sc, Vc, s_max=s_max, v_max=v_max)
            else:
                mask = make_mask_color(Hc, Sc, Vc, h_lo=h_lo, h_hi=h_hi, s_min=s_min, v_min=v_min)

            coverage = mask.mean()
            status = decision = ""
            e_val = None
            cx = cy = None
            ang = None

            if coverage < min_coverage:
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
                    # Optional steering inversion
                    if INVERT_STEER:
                        e_val = -e_val
                    if abs(e_val) <= deadband:
                        decision = "go straight"; status = "on line (centered)"
                    elif e_val < -deadband:
                        status = "line LEFT of center"; decision = "turn LEFT"
                    else:
                        status = "line RIGHT of center"; decision = "turn RIGHT"
                    ang = pca_angle_deg(mask)

            if preview is not None:
                preview.draw_and_show(frame, {"roi_y0": y0, "roi_y1": H,
                                              "cx": cx if coverage >= min_coverage else None,
                                              "cy": cy if coverage >= min_coverage else None,
                                              "angle": ang})

            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                cov_pct = 100.0 * coverage
                th_desc = f"mode={mode_choice}"
                if mode == "color":
                    th_desc += f" h=[{h_lo:.1f},{h_hi:.1f}] s>={s_min} v>={v_min}"
                else:
                    th_desc += f" v<={v_max} s<={s_max}"
                if e_val is None:
                    print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | {status} | decision={decision} | {th_desc}")
                else:
                    if ang is None:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | {status} -> {decision} | {th_desc}")
                    else:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | angle={ang:+.1f}° | {status} -> {decision} | {th_desc}")

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
