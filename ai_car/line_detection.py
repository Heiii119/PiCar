# line_detection.py
import cv2
import numpy as np


class LineDetector:

    def __init__(self):

        # ----- Detection mode -----
        # "hsv_distance" or "red_bgr"
        self.mode = "hsv_distance"

        # ----- ROI -----
        self.roi_ratio = 0.5

        # ----- Morphology -----
        self.kernel = np.ones((5, 5), np.uint8)

        # ----- HSV Distance Settings -----
        self.target_hsv = None
        self.hsv_distance_threshold = 40

        # ----- BGR Red Dominance Settings -----
        self.red_margin = 40

        # ----- Calibration -----
        self.offset_bias = 0
        self.last_offset = 0

        # ----- Smoothing -----
        self.smoothed_offset = 0
        self.alpha = 0.6  # smoothing factor

        # ----- Debug -----
        self.last_mask = None

    # =========================================
    # PUBLIC PROCESS FUNCTION
    # =========================================
    def process(self, frame):

        h, w, _ = frame.shape

        # ---- ROI (bottom part) ----
        roi_top = int(h * (1 - self.roi_ratio))
        roi = frame[roi_top:h, :]
        roi_center_x = w // 2

        # ---- Select pipeline ----
        if self.mode == "hsv_distance":
            mask = self._hsv_distance_pipeline(roi)
        else:
            mask = self._red_bgr_pipeline(roi)

        self.last_mask = mask

        center = self._find_line_center(mask)

        # ---- Offset calculation ----
        if center is None:
            offset = None
        else:
            raw_offset = center - roi_center_x
            corrected = raw_offset - self.offset_bias

            # ✅ smoothing
            self.smoothed_offset = (
                self.alpha * corrected +
                (1 - self.alpha) * self.smoothed_offset
            )

            offset = self.smoothed_offset
            self.last_offset = offset

        debug = self._make_debug_frame(frame, mask, center, roi_top)

        return offset, debug

    # =========================================
    # ✅ HSV DISTANCE PIPELINE
    # =========================================
    def _hsv_distance_pipeline(self, roi):

        if self.target_hsv is None:
            return np.zeros(roi.shape[:2], dtype=np.uint8)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Compute distance in HSV space
        diff = hsv.astype(np.float32) - self.target_hsv
        distance = np.linalg.norm(diff, axis=2)

        mask = np.zeros_like(distance, dtype=np.uint8)
        mask[distance < self.hsv_distance_threshold] = 255

        # Morphology
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    # =========================================
    # ✅ BGR RED DOMINANCE PIPELINE
    # =========================================
    def _red_bgr_pipeline(self, roi):

        b, g, r = cv2.split(roi)

        mask = np.zeros_like(r, dtype=np.uint8)

        red_pixels = (r > g + self.red_margin) & (r > b + self.red_margin)
        mask[red_pixels] = 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    # =========================================
    # FIND LINE CENTER
    # =========================================
    def _find_line_center(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 300:
            return None

        M = cv2.moments(largest)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        return cx

    # =========================================
    # DEBUG FRAME
    # =========================================
    def _make_debug_frame(self, frame, mask, center, roi_top):

        debug = frame.copy()
        h, w, _ = frame.shape

        # Draw ROI
        cv2.rectangle(debug, (0, roi_top), (w, h), (0, 255, 0), 2)

        # Draw image center
        cv2.line(debug, (w // 2, roi_top), (w // 2, h), (255, 0, 0), 2)

        if center is not None:
            cv2.circle(
                debug,
                (center, roi_top + (h - roi_top) // 2),
                8,
                (0, 0, 255),
                -1
            )

        calibrated_center = (w // 2) + self.offset_bias
        cv2.line(debug,
                 (calibrated_center, roi_top),
                 (calibrated_center, h),
                 (0, 255, 255),
                 2)

        return debug

    # =========================================
    # ✅ COLOR CALIBRATION (HSV)
    # =========================================
    def calibrate_color(self, frame):

        h, w, _ = frame.shape

        roi_top = int(h * (1 - self.roi_ratio))
        roi = frame[roi_top:h, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        box = hsv[
            roi.shape[0]//2 - 20 : roi.shape[0]//2 + 20,
            w//2 - 20 : w//2 + 20
        ]

        self.target_hsv = box.mean(axis=(0, 1))

        print("✅ HSV color calibrated:", self.target_hsv)

    # =========================================
    # CENTER CALIBRATION
    # =========================================
    def calibrate_center(self):
        self.offset_bias += self.last_offset
        print("✅ Line center calibrated")

    def reset_calibration(self):
        self.offset_bias = 0
        print("✅ Line calibration reset")

    # =========================================
    # SETTINGS
    # =========================================
    def set_mode(self, mode):
        if mode in ["hsv_distance", "red_bgr"]:
            self.mode = mode
