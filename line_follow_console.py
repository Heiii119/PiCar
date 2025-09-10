#!/usr/bin/env python3
# line_follower_console.py
import argparse, time, math
import numpy as np
from PIL import Image
from picamera2 import Picamera2

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
    # Use Pillow for correctness and simplicity
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
    cx = xs.mean()
    cy = ys.mean()
    area = len(xs)
    return cx, cy, area

def pca_angle_deg(mask):
    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n < 50:
        return None
    x = xs.astype(np.float32)
    y = ys.astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    M = np.stack([x, y], axis=0)   # 2 x N
    C = (M @ M.T) / n              # 2 x 2 covariance
    w, V = np.linalg.eigh(C)       # ascending eigenvalues
    v = V[:, 1]                    # principal direction
    angle = math.degrees(math.atan2(v[1], v[0]))  # 0°=horizontal, 90°=vertical
    # normalize to [-90, 90]
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

def main():
    ap = argparse.ArgumentParser(description="Line-following console analyzer (no OpenCV)")
    ap.add_argument("--size", default="1280x720", help="Resolution WxH (e.g. 1280x720)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--roi-height", type=float, default=0.35, help="Bottom ROI height fraction (0..1)")
    ap.add_argument("--mode", choices=["black","color"], default="black", help="Detect dark line or a colored line")
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
    args = ap.parse_args()

    W, H = map(int, args.size.lower().split("x"))
    picam2 = Picamera2()
    configure(picam2, (W, H), args.fps)
    picam2.start()
    print("Running. Ctrl+C to stop.")

    last_print = 0.0
    print_period = 1.0 / max(1e-3, args.print_rate)

    try:
        while True:
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

            coverage = mask.mean()  # fraction of ROI pixels
            status = ""
            decision = ""
            e_val = None
            angle = None

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
                    e_val = float((cx - (wR/2)) / (wR/2))
                    # Decision logic: steer towards the line to reduce |e|
                    if abs(e_val) <= args.deadband:
                        decision = "go straight"
                        status = "on line (centered)"
                    elif e_val < -args.deadband:
                        # line appears LEFT of center
                        status = "line LEFT of center"
                        decision = "turn LEFT"
                    else:
                        status = "line RIGHT of center"
                        decision = "turn RIGHT"

                    if args.invert-steer:  # noqa: E999 (hyphen in attribute)
                        pass
            # Workaround because argparse doesn't allow hyphen in attribute name
            invert = getattr(args, "invert_steer", False)
            if invert and ("turn LEFT" in decision or "turn RIGHT" in decision):
                decision = "turn RIGHT" if "LEFT" in decision else "turn LEFT"
                status = status.replace("LEFT","TEMP").replace("RIGHT","LEFT").replace("TEMP","RIGHT")

            # Angle estimate (optional)
            angle = pca_angle_deg(mask)
            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                cov_pct = 100.0 * coverage
                if e_val is None:
                    print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | {status} | decision={decision}")
                else:
                    # angle can be None
                    if angle is None:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | {status} -> {decision}")
                    else:
                        print(f"ROI: {wR}x{hR} | coverage={cov_pct:.2f}% | e={e_val:+.3f} | angle={angle:+.1f}° | {status} -> {decision}")
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
