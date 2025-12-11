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

n", red_y_mean)
bash: syntax error near unexpected token `"red_frac",'


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
