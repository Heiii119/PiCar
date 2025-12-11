# traffic_light_detector.py
#
# Separate module to detect traffic-light colour: "RED" / "GREEN" / "NONE"
# from an RGB numpy array (H x W x 3, uint8).
#
# It uses BOTH:
#   - HSV-based detection, and
#   - Simple RGB-based rules (R >> G,B for RED; G >> R,B for GREEN)

import time
import numpy as np

# ===================== HSV conversion (same style as line.py) =====================

def rgb_to_hsv_np(rgb):
    """
    rgb uint8 -> hsv (H deg 0..360, S% 0..100, V% 0..100), vectorized
    """
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-8

    # Hue
    h = np.zeros_like(cmax)
    mask = delta > 1e-8
    r_eq = (cmax == r) & mask
    g_eq = (cmax == g) & mask
    b_eq = (cmax == b) & mask
    h[r_eq] = (60.0 * ((g[r_eq] - b[r_eq]) / delta[r_eq]) + 360.0) % 360.0
    h[g_eq] = (60.0 * ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 120.0) % 360.0
    h[b_eq] = (60.0 * ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 240.0) % 360.0

    # Saturation
    s = np.zeros_like(cmax)
    s[mask] = (delta[mask] / cmax[mask]) * 100.0

    # Value
    v = cmax * 100.0
    return h, s, v

# ===================== Configuration =====================

DEFAULT_TL_HOLD_TIME = 1.5  # seconds to hold last seen RED/GREEN

# Kept for compatibility (not used directly in the final logic)
DEFAULT_TL_MIN_AREA_FRACTION = 0.20

# HSV hue ranges in degrees [0, 360)
RED_H_LO_1 = 0.0
RED_H_HI_1 = 45.0       # wider towards orange
RED_H_LO_2 = 330.0
RED_H_HI_2 = 360.0      # wrap-around red

GREEN_H_LO = 60.0       # allow more yellowish greens
GREEN_H_HI = 170.0      # allow bluish greens

# Minimum saturation & value (brightness) to be considered "coloured"
TL_S_MIN = 15.0         # in [0, 100]
TL_V_MIN = 10.0         # in [0, 100]

# Separate area thresholds inside the ROI (fraction of ROI pixels)
RED_MIN_AREA_FRACTION = 0.35    # fraction of ROI for RED
GREEN_MIN_AREA_FRACTION = 0.10  # fraction of ROI for GREEN

# Extra RGB-based rules (channel differences) for robustness
RED_RGB_MIN = 50        # minimum R value
RED_RGB_DELTA = 40      # R must be at least DELTA above G and B

GREEN_RGB_MIN = 60      # minimum G value
GREEN_RGB_DELTA = 40    # G must be at least DELTA above R and B

# ===================== Stateless single-frame detector =====================

def detect_traffic_light_state_once(
    frame_rgb: np.ndarray,
    area_fraction_threshold: float = DEFAULT_TL_MIN_AREA_FRACTION,  # kept for API, not used
) -> str:
    """
    Stateless, single-frame traffic-light detector.

    Parameters
    ----------
    frame_rgb : np.ndarray
        RGB image (H x W x 3), dtype uint8.
    area_fraction_threshold : float
        Kept for backward compatibility, but not used directly.

    Returns
    -------
    str
        "RED", "GREEN", or "NONE".
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return "NONE"

    h, w, _ = frame_rgb.shape

    # Traffic light ROI: only look in the TOP HALF of the image
    y0 = 0
    y1 = int(0.40 * h)    # top 30% of the frame
    x0 = int(0.20 * w)    # left 30% in from edge
    x1 = int(0.80 * w)    # right 30% in from edge
    roi = frame_rgb[y0:y1, x0:x1, :]

    if roi.size == 0:
        return "NONE"

    roi_h, roi_w, _ = roi.shape
    roi_pixels = roi_h * roi_w

    # Split channels (0..255)
    R = roi[..., 0].astype(np.float32)
    G = roi[..., 1].astype(np.float32)
    B = roi[..., 2].astype(np.float32)

    # HSV for hue-based detection
    H, S, V = rgb_to_hsv_np(roi)

    # Valid pixels: bright enough
    valid = V >= TL_V_MIN
    if np.count_nonzero(valid) < 50:  # very small -> probably just noise
        return "NONE"

    # ----- HSV-based RED & GREEN masks -----
    hsv_valid = valid & (S >= TL_S_MIN)

    red_hsv_mask = hsv_valid & (
        ((H >= RED_H_LO_1) & (H <= RED_H_HI_1)) |
        ((H >= RED_H_LO_2) & (H <= RED_H_HI_2))
    )

    green_hsv_mask = hsv_valid & (H >= GREEN_H_LO) & (H <= GREEN_H_HI)

    # ----- RGB-based fallback masks -----
    # RED: R is high and clearly dominates G & B
    red_rgb_mask = valid & (
        (R >= RED_RGB_MIN) &
        (R >= G + RED_RGB_DELTA) &
        (R >= B + RED_RGB_DELTA)
    )

    # GREEN: G is high and clearly dominates R & B
    green_rgb_mask = valid & (
        (G >= GREEN_RGB_MIN) &
        (G >= R + GREEN_RGB_DELTA) &
        (G >= B + GREEN_RGB_DELTA)
    )

    # Combine HSV and RGB criteria
    red_mask = red_hsv_mask | red_rgb_mask
    green_mask = green_hsv_mask | green_rgb_mask

    red_count = int(np.count_nonzero(red_mask))
    green_count = int(np.count_nonzero(green_mask))

    # FRACTION RELATIVE TO ROI (not whole frame)
    red_fraction = red_count / roi_pixels
    green_fraction = green_count / roi_pixels

    # --- Symmetric decision logic: RED and GREEN same priority ---

    red_strong = red_fraction >= RED_MIN_AREA_FRACTION
    green_strong = green_fraction >= GREEN_MIN_AREA_FRACTION

    # 1) Neither colour is strong enough
    if not red_strong and not green_strong:
        return "NONE"

    # 2) Only one colour strong
    if red_strong and not green_strong:
        return "RED"
    if green_strong and not red_strong:
        return "GREEN"

    # 3) Both strong -> choose the one with larger area
    if red_fraction > green_fraction:
        return "RED"
    elif green_fraction > red_fraction:
        return "GREEN"
    else:
        # Exact tie: ambiguous; let the stateful wrapper hold the last state
        return "NONE"

# ===================== Stateful wrapper with hold time =====================

class TrafficLightDetector:
    """
    Stateful detector that smooths the result over time:
      - Uses detect_traffic_light_state_once(...) on each frame.
      - Holds last RED/GREEN state for 'hold_time' seconds.

    Public API:
      - update(frame_rgb, tnow=None) -> "RED" | "GREEN" | "NONE"
      - current_state property
    """

    def __init__(
        self,
        hold_time: float = DEFAULT_TL_HOLD_TIME,
        area_fraction_threshold: float = DEFAULT_TL_MIN_AREA_FRACTION,
    ):
        self.hold_time = hold_time
        # area_fraction_threshold kept for backwards compatibility
        self.area_fraction_threshold = area_fraction_threshold

        self._current_state = "NONE"
        self._last_state = "NONE"
        self._last_time = 0.0

    @property
    def current_state(self) -> str:
        return self._current_state

    def update(self, frame_rgb: np.ndarray, tnow: float | None = None) -> str:
        """
        Update internal state using current frame and time.

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB numpy array (H x W x 3), dtype uint8.
        tnow : float or None
            Current time in seconds (e.g. time.time()).
            If None, time.time() will be used.

        Returns
        -------
        str
            Smoothed state: "RED", "GREEN", or "NONE".
        """
        if tnow is None:
            tnow = time.time()

        raw_state = detect_traffic_light_state_once(
            frame_rgb,
            area_fraction_threshold=self.area_fraction_threshold,
        )

        if raw_state != "NONE":
            self._last_state = raw_state
            self._last_time = tnow

        # Hold the last non-NONE state for 'hold_time' seconds
        if (tnow - self._last_time) > self.hold_time:
            self._current_state = "NONE"
        else:
            self._current_state = self._last_state

        return self._current_state
