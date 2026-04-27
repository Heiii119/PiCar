# line_detection.py
import cv2
import numpy as np


class LineDetector:

    def __init__(self):

        # ----- Detection mode -----
        # "gray" or "hsv"
        self.mode = "gray"

        # ----- Gray settings -----
        self.gray_threshold = 120

        # ----- HSV settings -----
        self.hue_low = 15
        self.hue_high = 40
        self.sat_min = 50
        self.val_min = 50

        # ----- ROI -----
        self.roi_ratio = 0.5

        # ----- Morphology -----
        self.kernel = np.ones((5, 5), np.uint8)

        # ----- Calibration -----
        self.offset_bias = 0          # ✅ calibration correction
        self.last_offset = 0          # ✅ store last valid offset

        # ----- Debug -----
        self.last_mask = None

    # =========================================
    # PUBLIC PROCESS FUNCTION
    # =========================================
    def process(self, frame):
        """
        Returns:
            offset (int)
            debug_frame (BGR image)
        """

        h, w, _ = frame.shape

        # ---- ROI (bottom part of image) ----
        roi_top = int(h * (1 - self.roi_ratio))
        roi = frame[roi_top:h, :]
        roi_center_x = w // 2

        # ---- Select pipeline ----
        if self.mode == "gray":
            mask = self._gray_pipeline(roi)
        else:
            mask = self._hsv_pipeline(roi)

        self.last_mask = mask

        center = self._find_line_center(mask)

        # ---- Offset calculation ----
        if center is None:
            offset = self.last_offset  # ✅ keep last value if lost
        else:
            raw_offset = center - roi_center_x
            offset = raw_offset - self.offset_bias
            self.last_offset = offset

        debug = self._make_debug_frame(frame, mask, center, roi_top)

        return int(offset), debug

    # =========================================
    # GRAY PIPELINE
    # =========================================
    def _gray_pipeline(self, roi):

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            self.gray_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

        return binary

    # =========================================
    # HSV PIPELINE
    # =========================================
    def _hsv_pipeline(self, roi):

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array([self.hue_low, self.sat_min, self.val_min])
        upper = np.array([self.hue_high, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    # =========================================
    # FIND LINE CENTER
    # =========================================
    def _find_line_center(self, mask):

        moments = cv2.moments(mask)

        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"])
        return cx

    # =========================================
    # DEBUG FRAME
    # =========================================
    def _make_debug_frame(self, frame, mask, center, roi_top):

        debug = frame.copy()
        h, w, _ = frame.shape

        # Draw ROI
        cv2.rectangle(
            debug,
            (0, roi_top),
            (w, h),
            (0, 255, 0),
            2
        )

        # Draw frame center
        cv2.line(
            debug,
            (w // 2, roi_top),
            (w // 2, h),
            (255, 0, 0),
            2
        )

        # Draw detected line center
        if center is not None:
            cv2.circle(
                debug,
                (center, roi_top + (h - roi_top) // 2),
                8,
                (0, 0, 255),
                -1
            )

        # Draw calibrated center line
        calibrated_center = (w // 2) + self.offset_bias
        cv2.line(
            debug,
            (calibrated_center, roi_top),
            (calibrated_center, h),
            (0, 255, 255),
            2
        )

        return debug

    # =========================================
    # ✅ CALIBRATION FUNCTION
    # =========================================
    def calibrate_center(self):
        """
        Sets current offset as new zero.
        Car must be centered on line when pressed.
        """
        self.offset_bias += self.last_offset
        print("✅ Line center calibrated")

    def reset_calibration(self):
        self.offset_bias = 0
        print("✅ Line calibration reset")

    # =========================================
    # SETTINGS UPDATE
    # =========================================
    def set_gray_threshold(self, value):
        self.gray_threshold = int(value)

    def set_hsv_range(self, h_low, h_high, s_min, v_min):
        self.hue_low = int(h_low)
        self.hue_high = int(h_high)
        self.sat_min = int(s_min)
        self.val_min = int(v_min)

    def set_mode(self, mode):
        if mode in ["gray", "hsv"]:
            self.mode = mode
