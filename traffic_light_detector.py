# traffic_light_detector.py

import time
import numpy as np
import cv2

# ===================== HSV conversion helper =====================

def rgb_to_hsv_np(img_rgb):
    """
    Convert uint8 RGB image to HSV:
    - H in degrees [0, 360)
    - S, V in [0, 255]
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0        # OpenCV: 0..179 -> 0..358
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    return h, s, v

# ===================== Traffic-light detector (stateless core) =====================

# Default configuration values (you can override via the class if needed)
DEFAULT_TL_HOLD_TIME = 1.5          # seconds to hold last seen RED/GREEN
DEFAULT_TL_MIN_AREA_FRACTION = 0.20 # 20% of ROI area

# Hue ranges in degrees [0, 360)
RED_H_LO_1 = 0.0
RED_H_HI_1 = 30.0       # wider red towards orange
RED_H_LO_2 = 330.0
RED_H_HI_2 = 360.0      # wrap-around red

GREEN_H_LO = 60.0       # allow more yellowish greens
GREEN_H_HI = 170.0      # allow bluish greens

# Minimum saturation & value (brightness) to be considered "coloured"
TL_S_MIN = 30.0         # lower = more sensitive, but more noise
TL_V_MIN = 30.0

def detect_traffic_light_state_once(
    frame_rgb: np.ndarray,
    area_fraction_threshold: float = DEFAULT_TL_MIN_AREA_FRACTION,
) -> str:
    """
    Stateless, single-frame traffic-light detector.

    Parameters
    ----------
    frame_rgb : np.ndarray
        RGB image (H x W x 3), dtype uint8.
    area_fraction_threshold : float
        Minimum fraction of the ROI area that must be red/green to trigger.

    Returns
    -------
    str
        "RED", "GREEN", or "NONE".
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return "NONE"

    h, w, _ = frame_rgb.shape

    # Large central ROI to include your light in front of the camera
    y0 = int(0.05 * h)
    y1 = int(0.95 * h)
    x0 = int(0.10 * w)
    x1 = int(0.90 * w)
    roi = frame_rgb[y0:y1, x0:x1, :]

    if roi.size == 0:
        return "NONE"

    roi_h, roi_w, _ = roi.shape
    roi_pixels = roi_h * roi_w

    H, S, V = rgb_to_hsv_np(roi)

    # Valid: reasonably saturated and bright so background doesn't dominate
    valid = (S >= TL_S_MIN) & (V >= TL_V_MIN)
    if np.count_nonzero(valid) < 50:  # very small -> probably just noise
        return "NONE"

    # ----- RED detection with wider hue range -----
    # Two intervals: [RED_H_LO_1, RED_H_HI_1] and [RED_H_LO_2, RED_H_HI_2]
    red_mask = valid & (
        ((H >= RED_H_LO_1) & (H <= RED_H_HI_1)) |
        ((H >= RED_H_LO_2) & (H <= RED_H_HI_2))
    )

    # ----- GREEN detection with wider hue range -----
    green_mask = valid & (H >= GREEN_H_LO) & (H <= GREEN_H_HI)

    red_count = int(np.count_nonzero(red_mask))
    green_count = int(np.count_nonzero(green_mask))

    # FRACTION RELATIVE TO ROI (not whole frame)
    red_fraction = red_count / roi_pixels
    green_fraction = green_count / roi_pixels

    # Require at least this fraction of the ROI to be red/green
    if (
        red_fraction < area_fraction_threshold
        and green_fraction < area_fraction_threshold
    ):
        return "NONE"

    # Decide which colour dominates
    if red_fraction > green_fraction:
        return "RED"
    else:
        return "GREEN"

# ===================== Traffic-light detector (stateful wrapper) =====================

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
            frame_rgb, area_fraction_threshold=self.area_fraction_threshold
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
