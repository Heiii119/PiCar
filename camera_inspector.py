#!/usr/bin/env python3
# camera_inspector.py
import argparse, time
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

def rgb_to_gray(rgb):
    # rgb uint8 -> gray float32
    r, g, b = rgb[...,0].astype(np.float32), rgb[...,1].astype(np.float32), rgb[...,2].astype(np.float32)
    return 0.299*r + 0.587*g + 0.114*b

def edge_metrics(gray):
    # simple gradient-based edge/line-ness metrics
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    mean_edge = (dx.mean() + dy.mean()) / 2.0
    # Orientation tendency: if dx sum > dy sum, there are more vertical edges (lines)
    sx, sy = dx.sum(), dy.sum()
    if sx > sy * 1.2:
        orientation = "vertical-dominant"
    elif sy > sx * 1.2:
        orientation = "horizontal-dominant"
    else:
        orientation = "mixed"
    ratio = (sx + 1e-6) / (sy + 1e-6)
    return float(mean_edge), orientation, float(ratio)

def hsv_stats(rgb):
    img = Image.fromarray(rgb, "RGB").convert("HSV")
    H, S, V = img.split()
    h = np.array(H, dtype=np.uint8)
    s = np.array(S, dtype=np.uint8)
    v = np.array(V, dtype=np.uint8)
    # Mean HSV
    mean_h = float(h.mean()) * 360.0 / 255.0
    mean_s = float(s.mean())
    mean_v = float(v.mean())
    # Dominant hue: histogram only on sufficiently saturated pixels
    mask_sat = s >= 40
    if mask_sat.any():
        hist = np.bincount(h[mask_sat].ravel(), minlength=256)
        dom_h_bin = int(np.argmax(hist))
        dom_hue_deg = dom_h_bin * 360.0 / 255.0
    else:
        dom_hue_deg = float('nan')
    v_p50 = float(np.percentile(v, 50))
    return (mean_h, mean_s, mean_v, dom_hue_deg, v_p50)

def roi_slices(h, w, kind):
    if kind == "center":
        y0, y1 = int(h*0.33), int(h*0.66)
        x0, x1 = int(w*0.33), int(w*0.66)
    elif kind == "bottom":
        y0, y1 = int(h*0.65), h
        x0, x1 = 0, w
    else:
        y0, y1, x0, x1 = 0, h, 0, w
    return slice(y0, y1), slice(x0, x1)

def main():
    ap = argparse.ArgumentParser(description="Camera inspector: color and line/edge metrics (no OpenCV)")
    ap.add_argument("--size", default="1280x720", help="Resolution WxH (e.g. 1920x1080)")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS")
    ap.add_argument("--interval", type=float, default=1.0, help="Print interval seconds")
    args = ap.parse_args()
    w, h = map(int, args.size.lower().split("x"))

    picam2 = Picamera2()
    configure(picam2, (w, h), args.fps)
    picam2.start()
    print("Running. Press Ctrl+C to stop.")
    last_print = time.time()

    try:
        while True:
            frame = picam2.capture_array()  # RGB888, HxWx3
            now = time.time()
            if now - last_print >= args.interval:
                last_print = now

                # Center ROI
                syc, sxc = roi_slices(frame.shape[0], frame.shape[1], "center")
                roi_c = frame[syc, sxc]
                mean_rgb_c = tuple(map(lambda x: int(round(x)), roi_c.mean(axis=(0,1))))

                mh_c, ms_c, mv_c, domh_c, vp50_c = hsv_stats(roi_c)
                gray_c = rgb_to_gray(roi_c)
                edg_c, orient_c, ratio_c = edge_metrics(gray_c)

                # Bottom ROI
                syb, sxb = roi_slices(frame.shape[0], frame.shape[1], "bottom")
                roi_b = frame[syb, sxb]
                mean_rgb_b = tuple(map(lambda x: int(round(x)), roi_b.mean(axis=(0,1))))
                mh_b, ms_b, mv_b, domh_b, vp50_b = hsv_stats(roi_b)
                gray_b = rgb_to_gray(roi_b)
                edg_b, orient_b, ratio_b = edge_metrics(gray_b)

                print(
                    f"[center] RGB_mean={mean_rgb_c}  HSV_mean=(H={mh_c:.1f}°,S={ms_c:.1f},V={mv_c:.1f})  "
                    f"H_dominant={domh_c:.1f}°  V_p50={vp50_c:.1f}  edge={edg_c:.2f}  orient={orient_c} (dx/dy={ratio_c:.2f})"
                )
                print(
                    f"[bottom] RGB_mean={mean_rgb_b}  HSV_mean=(H={mh_b:.1f}°,S={ms_b:.1f},V={mv_b:.1f})  "
                    f"H_dominant={domh_b:.1f}°  V_p50={vp50_b:.1f}  edge={edg_b:.2f}  orient={orient_b} (dx/dy={ratio_b:.2f})"
                )
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
