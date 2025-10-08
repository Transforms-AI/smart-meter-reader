"""Two‑stage YOLO meter reader for detecting meters and digit dials.

This script extends the YOLO‑based meter reader to support two
separate detection models: one that finds the gas meter in a full
image and another that locates the digit dial within the cropped
meter.  After isolating the digit dial the script applies the same
preprocessing steps used previously (sharpening, thresholding,
contouring) and then classifies each of the eight digits using a
YOLOv8n classifier.

The expected models are:

* **Meter detector** – a YOLOv8n detect model trained on full images
  with class id 1 representing the meter.  The script selects the
  highest‑confidence box with class=1 to crop the meter region.
* **Dial detector** – a YOLOv8n model trained to find the digit dial
  within the cropped meter.  Class id 0 is used for the dial.  If
  your model uses oriented bounding boxes they will be approximated
  with axis‑aligned cropping here; further perspective correction can
  be added if needed.
* **Digit classifier** – a YOLOv8n classifier trained on single digits
  (0–9), used to recognise each of the eight cropped digit tiles.

Usage example:

```
python3 yolo_meter_reader_2stage.py \
    --meter-detector models/meter_detector.pt \
    --dial-detector models/dial_detector.pt \
    --classifier models/digit_classifier.pt \
    --camera-id 0 \
    --results-dir results
```
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as e:
    raise ImportError(
        "Ultralytics package not found.  Install it with 'pip install ultralytics'."
    ) from e

try:
    import paho.mqtt.client as mqtt_client  # type: ignore
except ImportError:
    mqtt_client = None

# -----------------------------------------------------------------------------
#  Helper functions copied from the original meter_reader implementation
#
# The following functions perform the same preprocessing steps as in the
# original project.  They have been copied verbatim with minor stylistic
# adjustments so that this script is completely self‑contained and does
# not depend on any external modules.

def balancing_tilted_image(image: np.ndarray, greyscale_image: np.ndarray, cycles: int) -> np.ndarray:
    """Balance a potentially tilted image using the Hough line transform.

    The gas‑meter dial may not be perfectly horizontal in the captured frame.
    This function attempts to find the dominant line angle and rotates the
    image so the dial is level.  It runs for a fixed number of cycles to
    refine the alignment.

    Args:
        image: Colour BGR image as a NumPy array.
        greyscale_image: Greyscale version of the same image.
        cycles: Number of balancing iterations.

    Returns:
        The rotated BGR image.
    """
    for _ in range(cycles):
        dst = cv2.Canny(greyscale_image, 50, 200, None, 3)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 110, None, 0, 0)
        sizemax = float(np.hypot(cdst.shape[0], cdst.shape[1]))
        if lines is not None:
            average_angle = 0.0
            line_num = 0
            for i in range(len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                cur_angle = theta * 180.0 / np.pi
                if abs(90.0 - cur_angle) < 15:
                    average_angle += cur_angle
                    line_num += 1
                # draw lines for debugging (not needed here)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
                pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
            if line_num > 0:
                rows, cols = greyscale_image.shape[:2]
                angle = (average_angle / line_num) - 90.0
                M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1.0)
                greyscale_image = cv2.warpAffine(greyscale_image, M, (cols, rows))
                image = cv2.warpAffine(image, M, (cols, rows))
    return image


def resize_and_sharpen_image(
    image: np.ndarray,
    dim: Tuple[int, int],
    tb_w: int,
    tb_th: int,
    tb_blur_size: int,
    tb_blur_sigma: int,
) -> np.ndarray:
    """Resize the cropped dial image (sharpening disabled for original quality).

    MODIFIED: Sharpening operations have been commented out to preserve original
    image quality and resolution. Only necessary resizing is performed.

    Args:
        image: BGR image.
        dim: Desired (width, height) after resizing.
        tb_w: Sharpening strength parameter scaled by 10 (unused).
        tb_th: Threshold below which differences are ignored (unused).
        tb_blur_size: Base blur kernel size (unused).
        tb_blur_sigma: Gaussian sigma scaled by 10 (unused).

    Returns:
        Resized BGR image with the given dimensions (no sharpening applied).
    """
    # Only resize, skip all sharpening operations to preserve original quality
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    return resized
    
    # COMMENTED OUT: Sharpening operations disabled to preserve original quality
    # # Convert to Lab
    # im_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
    # l_channel, a_channel, b_channel = cv2.split(im_lab)
    # w = tb_w / 10.0
    # th = tb_th
    # blur_size = tb_blur_size * 2 + 3
    # blur_sigma = tb_blur_sigma / 10.0
    # im_blur = cv2.GaussianBlur(l_channel, (blur_size, blur_size), blur_sigma)
    # im_diff = cv2.subtract(l_channel, im_blur, dtype=cv2.CV_16S)
    # im_abs_diff = cv2.absdiff(l_channel, im_blur)
    # im_diff_masked = im_diff.copy()
    # im_diff_masked[im_abs_diff < th] = 0
    # sharpened_l = cv2.add(l_channel, w * im_diff_masked, dtype=cv2.CV_8UC1)
    # res_lab = cv2.merge([sharpened_l, a_channel, b_channel])
    # return cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR)


def adaptive_threshold_and_median_blur(image: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """Apply the NiBlack adaptive threshold (median blurs disabled for original quality).

    MODIFIED: Median blur operations have been commented out to preserve original
    image quality and detail.

    Args:
        image: BGR image to threshold.
        block_size: Window size for the NiBlack algorithm (odd integer).
        k: NiBlack ``k`` parameter controlling the bias between local mean and
           standard deviation.

    Returns:
        A greyscale image after thresholding (no median blurring applied).
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # NiBlack threshold from the ximgproc module – this requires opencv-contrib-python
    thresh = cv2.ximgproc.niBlackThreshold(
        grey,
        255,
        cv2.THRESH_BINARY,
        block_size,
        k,
        binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK,
        r=106,
    )
    # COMMENTED OUT: Multiple median blurs disabled to preserve original quality
    # for kernel in (3, 5, 7, 9):
    #     thresh = cv2.medianBlur(thresh, kernel)
    return thresh


def find_contours(
    threshold_im: np.ndarray,
    original_im: np.ndarray,
    img_width: int,
    img_height: int,
    h_min: int = 50,
    w_min: int = 25,
    w_max: int = 120,
    x_min: int = 25,
    x_max: int = 400,
    h_w_ratio_max: float = 3.0,
    h_w_ratio_min: float = 1.0,
    y_min: int = 25,
    y_max: int = 125,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Locate digit regions in the threshold image and slice them into 8 tiles.

    Args:
        threshold_im: Greyscale thresholded image.
        original_im: Sharpened BGR image corresponding to ``threshold_im``.
        img_width: Desired width of each digit tile.
        img_height: Desired height of each digit tile.
        h_min, w_min, w_max, x_min, x_max, h_w_ratio_max, h_w_ratio_min,
        y_min, y_max: Filtering parameters controlling which contours are
        accepted.  See the original helper functions for details.

    Returns:
        A list of 8 BGR images sized (img_width, img_height) containing the
        digit tiles, and a colour version of the threshold image with drawn
        rectangles for debugging.
    """
    contours, hierarchy = cv2.findContours(
        threshold_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1
    )
    threshold_col = cv2.cvtColor(threshold_im, cv2.COLOR_GRAY2BGR)
    filtered = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        if (
            h < h_min
            or w < w_min
            or w > w_max
            or x < x_min
            or x > x_max
            or h / w > h_w_ratio_max
            or h / w < h_w_ratio_min
            or y < y_min
            or (y + h) > y_max
        ):
            continue
        start = (x, y)
        end = (x + w, y + h)
        cv2.rectangle(threshold_col, start, end, (0, 0, 255), 1, cv2.LINE_AA)
        filtered.append({"x": x, "y": y, "w": w, "h": h})
    filtered = sorted(filtered, key=lambda d: d["x"])
    if not filtered:
        raise RuntimeError("No contours found for digit regions")
    # Determine the clipping positions between digits.  When there are only
    # two candidate contours (rare but possible) use their midpoints; otherwise
    # compute offsets from the first three.
    if len(filtered) <= 2:
        clipping_1 = round(
            (filtered[1]["x"] - (filtered[0]["x"] + filtered[0]["w"])) / 2
            + (filtered[0]["x"] + filtered[0]["w"])
        )
        clipping_offset = filtered[1]["x"] - filtered[0]["x"]
    else:
        clipping_1 = round(
            (filtered[1]["x"] - (filtered[0]["x"] + filtered[0]["w"])) / 2
            + (filtered[0]["x"] + filtered[0]["w"])
        )
        clipping_2 = round(
            (filtered[2]["x"] - (filtered[1]["x"] + filtered[1]["w"])) / 2
            + (filtered[1]["x"] + filtered[1]["w"])
        )
        clipping_offset = clipping_2 - clipping_1
    image_list: List[np.ndarray] = []
    for i in range(8):
        left = (clipping_1 + i * clipping_offset) - clipping_offset
        right = clipping_1 + i * clipping_offset
        tile = original_im[:, left:right, :]
        tile_resized = cv2.resize(tile, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
        image_list.append(tile_resized)
    return image_list, threshold_col

# -----------------------------------------------------------------------------
#  Oriented bounding box cropping helper

def crop_from_poly(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crop a rotated rectangle from an image using a 4‑point polygon.

    Args:
        image: Source BGR image.
        poly: Flattened array of 8 floats defining the 4 vertices of the
            oriented bounding box in the order provided by Ultralytics
            (x1,y1,x2,y2,x3,y3,x4,y4).  If the order is unknown, the
            function will reorder points to [top‑left, top‑right,
            bottom‑right, bottom‑left].

    Returns:
        The cropped, perspective‑corrected image containing the region
        inside the polygon.
    """
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    # Order points: use sum and diff to find corners
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    # Destination coordinates
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype=np.float32,
    )
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


class YOLOTwoStageMeterReader:
    """Meter reader that uses two YOLO detection stages and a classifier."""

    def __init__(
        self,
        meter_detector_path: str,
        dial_detector_path: str,
        classifier_path: Optional[str] = None,
        digit_detector_path: Optional[str] = None,
    ) -> None:
        """Initialize the two‑stage reader with optional classifier and digit detector models.

        Args:
            meter_detector_path: Path to the YOLO model that detects the meter region (class id 1).
            dial_detector_path: Path to the YOLO model that detects the digit dial (class id 0).
            classifier_path: Path to a YOLO classifier model for digits (optional).
            digit_detector_path: Path to a YOLO detection model trained to detect digits 0–9 (optional).
        """
        if not os.path.exists(meter_detector_path):
            raise FileNotFoundError(f"Meter detector model not found at {meter_detector_path}")
        if not os.path.exists(dial_detector_path):
            raise FileNotFoundError(f"Dial detector model not found at {dial_detector_path}")
        self.meter_detector = YOLO(meter_detector_path)
        self.dial_detector = YOLO(dial_detector_path)
        # Optional classifier (not used in detection‑based recognition)
        self.classifier = YOLO(classifier_path) if classifier_path else None
        # Optional digit detection model
        self.digit_detector = YOLO(digit_detector_path) if digit_detector_path else None
        # Preprocessing parameters (inherited from previous implementation)
        self.resize_dim = (1200, 140)
        self.balancing_cycles = 3
        self.tb_w = 70
        self.tb_th = 0
        self.tb_blur_size = 10
        self.tb_blur_sigma = 50
        self.block_size = 65
        self.k = 0.5
        self.h_min = 60
        self.w_min = 25
        self.w_max = 120
        self.x_min = 30
        self.x_max = 400
        self.y_min = 15
        self.y_max = 135
        self.h_w_ratio_max = 3.99
        self.h_w_ratio_min = 1.0

        # Predefined x-axis ratios for cropping 8 digits from the dial.  The
        # values are normalised to [0, 1] relative to the dial width and
        # correspond to x_start and x_end positions for each digit.  These
        # ranges were converted from percentages (0–100) by dividing by 100.
        # They will be adjusted at runtime based on digit detections.
        self.digit_ranges = [
            (0.013, 0.125),
            (0.125, 0.25),
            (0.25, 0.36),
            (0.36, 0.48),
            (0.48, 0.61),
            (0.61, 0.72),
            (0.72, 0.84),
            (0.86, 0.95),
        ]

    def slice_digits_by_ratio(self, image: np.ndarray, ranges: List[Tuple[float, float]],
                              img_width: int = 28, img_height: int = 28) -> List[np.ndarray]:
        """Slice the dial image into digit tiles based on predefined x‑axis ratios.

        Args:
            image: The dial image after sharpening (BGR).
            ranges: List of (start, end) tuples where values are percentages of the
                image width (0–100).
            img_width: Width to resize each digit tile to.
            img_height: Height to resize each digit tile to.

        Returns:
            A list of resized digit tiles.
        """
        h, w, _ = image.shape
        tiles: List[np.ndarray] = []
        for start_pct, end_pct in ranges:
            # start_pct and end_pct are normalised (0–1) values
            left = int(start_pct * w)
            right = int(end_pct * w)
            # Clip bounds
            left = max(0, min(left, w - 1))
            right = max(left + 1, min(right, w))
            tile = image[:, left:right, :]
            tile_resized = cv2.resize(tile, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            tiles.append(tile_resized)
        return tiles

    def rotate_image(self, image: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate an image by a given angle in radians around its centre.

        Args:
            image: Input BGR image.
            angle_rad: Angle in radians.  Positive values mean counter‑clockwise.

        Returns:
            Rotated image with the same dimensions as the input.
        """
        angle_deg = angle_rad * 180.0 / np.pi
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def assign_digits_from_detections(self, boxes, classes, confs, image_width: int) -> List[str]:
        """Assign detected digits to their positions based on x‑axis ratios.

        Args:
            boxes: Tensor or array of shape (n, 4) with xyxy coordinates.
            classes: Array of class indices (0–9).
            confs: Array of confidence scores.
            image_width: Width of the image on which detections were made.

        Returns:
            A list of 8 strings representing digits in each position.  Empty strings
            indicate no detection for that position.
        """
        positions = [''] * len(self.digit_ranges)
        # Flatten to numpy for easier handling
        boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes
        classes_np = classes.astype(int)
        confs_np = confs
        # Iterate over detections
        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i]
            x_center = (x1 + x2) / 2.0
            # Normalise x coordinate to [0, 1] relative to image width
            x_ratio = x_center / float(image_width)
            # Find which digit range this belongs to
            for idx, (start_pct, end_pct) in enumerate(self.digit_ranges):
                if start_pct <= x_ratio <= end_pct:
                    digit = str(classes_np[i])
                    # Assign if empty or higher confidence
                    if positions[idx] == '' or confs_np[i] > float(positions[idx][1]):
                        positions[idx] = (digit, confs_np[i])
                    break
        # Extract only digits
        result = []
        for pos in positions:
            if isinstance(pos, tuple):
                result.append(pos[0])
            else:
                result.append('')
        return result

    def assign_digits_from_ratios(
        self,
        ratios: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        ranges: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[str], List[float]]:
        """Assign detected digits to positions using precomputed x‑axis ratios.

        This helper takes a list of x‑axis normalised values (0–1) and maps
        each detection to one of the digit positions defined either by
        ``ranges`` if provided or by ``self.digit_ranges``.  When multiple
        detections fall into the same range the one with the highest
        confidence is selected.

        Args:
            ratios: 1‑D array of x‑axis positions in the range [0, 1].
            classes: 1‑D array of class indices (0–9).
            confs: 1‑D array of confidence scores (same length as ratios).
            ranges: Optional list of (start, end) tuples defining the digit
                ranges.  If None, ``self.digit_ranges`` is used.

        Returns:
            A tuple (digits, confs_per_position) where digits is a list of
            strings (one per position) and confs_per_position is a list of
            floats representing the confidence of the selected detection for
            each position.  Positions without a detection will contain
            an empty string and confidence 0.0.
        """
        use_ranges = ranges if ranges is not None else self.digit_ranges
        positions: List[Tuple[str, float] | str] = [''] * len(use_ranges)
        # Iterate over detections
        for ratio, cls, conf in zip(ratios, classes, confs):
            # Skip any detections outside the normalised range [0, 1]
            if ratio < 0.0 or ratio > 1.0:
                continue
            for idx, (start_pct, end_pct) in enumerate(use_ranges):
                # use_ranges are normalised, so compare directly
                if start_pct <= ratio <= end_pct:
                    digit = str(int(cls))
                    current = positions[idx]
                    # Replace if empty or lower confidence
                    if current == '' or (isinstance(current, tuple) and conf > current[1]):
                        positions[idx] = (digit, float(conf))
                    break
        # Extract digits and confidences
        digits_out: List[str] = []
        confs_out: List[float] = []
        for pos in positions:
            if isinstance(pos, tuple):
                digits_out.append(pos[0])
                confs_out.append(pos[1])
            else:
                digits_out.append('')
                confs_out.append(0.0)
        return digits_out, confs_out

    def detect_meter(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect the meter in the full image and return an axis‑aligned crop.

        Many meter detectors produce axis‑aligned bounding boxes (``boxes``)
        instead of oriented bounding boxes.  This method filters detections
        for class id 1 and uses the highest‑confidence bounding box to crop
        the meter region.  The crop is extracted from the original BGR
        image using integer coordinates.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.meter_detector(rgb, verbose=False)
        # Ensure we have axis‑aligned boxes
        if not results or results[0].boxes is None:
            raise RuntimeError("Meter detector returned no detections")
        boxes = results[0].boxes
        # Convert to numpy arrays
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        # Filter for class id 1
        mask = classes == 1
        if not mask.any():
            raise RuntimeError("No meter (class id 1) found in image")
        # Select the highest confidence detection for the meter
        idx = int(np.argmax(confs * mask))
        x1, y1, x2, y2 = xyxy[idx]
        h, w, _ = image_bgr.shape
        # Clamp coordinates to image bounds
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w))
        y2 = max(0, min(int(round(y2)), h))
        return image_bgr[y1:y2, x1:x2].copy()

    def detect_dial(
        self, meter_image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Detect the digit dial within the cropped meter and return its polygon.

        This function uses the dial detector to find the digit dial (class id 0)
        within the provided meter image.  It returns the dial crop, the
        orientation angle of the dial in radians, and the polygon points
        (flattened into a one‑dimensional array of 8 floats) defining the
        oriented bounding box of the dial.  The polygon vertices are
        expected to be in the order provided by the detector.

        Args:
            meter_image_bgr: The BGR image containing the meter region.

        Returns:
            A tuple (dial_crop, angle, poly) where:
            - dial_crop is the cropped dial region extracted by perspective
              warping.
            - angle is the orientation of the dial in radians (from the
              ``xywhr`` representation).
            - poly is an array of shape (8,) containing the four vertices of
              the dial OBB in the format [x1,y1,x2,y2,x3,y3,x4,y4].
        """
        rgb = cv2.cvtColor(meter_image_bgr, cv2.COLOR_BGR2RGB)
        results = self.dial_detector(rgb, verbose=False)
        # Ensure we have oriented bounding boxes
        if not results or not results[0].obb:
            raise RuntimeError("Dial detector returned no detections")
        obb = results[0].obb
        classes = obb.cls.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        # Filter for class id 0 (dial)
        mask = classes == 0
        if not mask.any():
            raise RuntimeError("No dial (class id 0) found in meter image")
        # Choose the highest confidence dial
        idx = int(np.argmax(confs * mask))
        poly = obb.xyxyxyxy.cpu().numpy()[idx]
        xywhr = obb.xywhr.cpu().numpy()[idx]  # [xc, yc, w, h, angle]
        angle = float(xywhr[4])  # radians
        dial_crop = crop_from_poly(meter_image_bgr, poly)
        return dial_crop, angle, poly

    def classify_digits(self, tiles: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """Classify each digit tile using the YOLO classifier and return digits with confidences.

        This method takes a list of digit tiles and runs the classifier on each.
        It returns two lists of equal length: one with the predicted digit strings
        and one with the corresponding classification confidence for the predicted class.

        Args:
            tiles: List of digit images (BGR) to classify.

        Returns:
            A tuple (digits, confs) where digits is a list of strings and confs
            is a list of floats representing the confidence for each prediction.
        """
        digits: List[str] = []
        confs: List[float] = []
        for tile in tiles:
            rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            res = self.classifier(rgb_tile, verbose=False)
            
            try:
                cls_idx = int(res[0].probs.top1)
                #print("success")
            except AttributeError:
                scores = res[0].probs.data
                cls_idx = int(np.argmax(scores))
            # digits.append(str(cls_idx))
            
            # Extract class probabilities from the result; use new API if available
            try:
                # Ultralytics returns a 'probs' attribute with the probabilities
                scores = res[0].probs.data
                #print("success2")
            except AttributeError:
                # Fallback: assume res[0].probs.data exists
                scores = res[0].probs.data
            # Find the class with the highest probability
            cls_idx = int(np.argmax(scores.cpu()))
            prob = float(scores[cls_idx])
            digits.append(str(cls_idx))
            confs.append(prob)
        return digits, confs

    def analyze(
        self, image: np.ndarray
    ) -> Tuple[str, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
        """Run the full pipeline and return the predicted meter reading.

        This method performs the following steps:
        1. Detect and crop the meter region from the full image.
        2. Optionally balance the meter crop using the Hough transform to
           compensate for tilt.
        3. Detect the dial within the balanced meter and obtain its oriented
           bounding box (OBB) and orientation angle.
        4. Rotate the balanced meter so that the dial is horizontally aligned.
        5. Detect digits within the rotated meter using the YOLO digit detector
           (if provided) and assign them to positions based on their x‑axis
           relative to the dial OBB.  Optionally fall back to classification
           for missing digits.
        6. Return the digit string and intermediate images.

        Args:
            image: BGR image containing the full gas meter.

        Returns:
            A tuple (pred_str, sharpened_dial, tiles, meter_crop, dial_crop) where
            - pred_str is the final meter reading string.
            - sharpened_dial is the resized and sharpened dial crop used for
              classification fallback.
            - tiles is a list of digit tiles for classification fallback.
            - meter_crop is the axis‑aligned crop of the meter region.
            - dial_crop is the crop of the dial region extracted via the OBB.
        """
        # Stage 1: detect the meter region in the full image
        meter_crop = self.detect_meter(image)
        # Stage 2: (optional) balance the meter crop to reduce tilt
        meter_grey = cv2.cvtColor(meter_crop, cv2.COLOR_BGR2GRAY)
        balanced_meter = balancing_tilted_image(meter_crop.copy(), meter_grey.copy(), self.balancing_cycles)
        # Stage 3: detect the dial OBB and obtain angle and polygon
        dial_crop, angle, poly = self.detect_dial(balanced_meter)
        # Resize and sharpen the dial for classification fallback and visualisation
        sharpened_dial = resize_and_sharpen_image(
            dial_crop,
            self.resize_dim,
            self.tb_w,
            self.tb_th,
            self.tb_blur_size,
            self.tb_blur_sigma,
        )
        # We will dynamically adjust digit ranges based on detections and then
        #print("here2")# classify each tile.  Start with a copy of the original ranges.
        
        adjusted_ranges = self.digit_ranges.copy()
        detection_digits: List[str] = [''] * len(self.digit_ranges)
        detection_confs: List[float] = [0.0] * len(self.digit_ranges)
        #print("here2")
        if self.digit_detector is not None:
            # Rotate the balanced meter around its centre by the dial's orientation angle
            #print("here2")
            rotated_meter = self.rotate_image(balanced_meter, angle)
            #print("here2")
            h_m, w_m = balanced_meter.shape[:2]
            # Compute rotation matrix used in rotate_image
            #print("here2")
            angle_deg = angle * 180.0 / np.pi
            M = cv2.getRotationMatrix2D((w_m / 2.0, h_m / 2.0), -angle_deg, 1.0)
            # Transform dial OBB points to rotated coordinates to get x bounds
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            transformed = (M @ pts_h.T).T
            #print("here2")
            dial_x_min = float(np.min(transformed[:, 0]))
            dial_x_max = float(np.max(transformed[:, 0]))
            dial_width = max(dial_x_max - dial_x_min, 1.0)
            # Detect digits on the rotated meter using the digit detector
            rgb_rot = cv2.cvtColor(rotated_meter, cv2.COLOR_BGR2RGB)
            results_digit = self.digit_detector(rgb_rot, verbose=False)
            ratios_list: List[float] = []
            classes_list: List[int] = []
            confs_list: List[float] = []
            
            if results_digit and results_digit[0].boxes is not None:
                boxes = results_digit[0].boxes.xyxy
                classes = results_digit[0].boxes.cls
                confs = results_digit[0].boxes.conf
                print("classes:", classes)
                print("confs:", confs)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    x_center = float((x1 + x2) / 2.0)
                    # Compute normalised x ratio relative to dial bounds
                    ratio = (x_center - dial_x_min) / dial_width
                    if 0.0 <= ratio <= 1.0:
                        ratios_list.append(ratio)
                        classes_list.append(int(classes[i]))
                        confs_list.append(float(confs[i]))
            # If there are detections, adjust the digit ranges based on one detection
            if ratios_list:
                ratios_arr = np.array(ratios_list, dtype=np.float32)
                confs_arr = np.array(confs_list, dtype=np.float32)
                classes_arr = np.array(classes_list, dtype=np.int32)

                # For each detection, assign it to a tile and compute offset
                per_tile_offsets = {}  # tile_idx → list of offsets (no weighting by confidence)
                for ratio in ratios_arr:
                    for t_idx, (s, e) in enumerate(self.digit_ranges):
                        if s <= ratio <= e:
                            tile_center = (s + e) / 2.0
                            offset = ratio - tile_center
                            per_tile_offsets.setdefault(t_idx, []).append(offset)
                            break

                # Compute average offset across all detections (simple mean)
                all_offsets = [off for offsets in per_tile_offsets.values() for off in offsets]
                if all_offsets:
                    avg_offset = sum(all_offsets) / len(all_offsets)
                else:
                    avg_offset = 0.0

                print("Computed average offset:", avg_offset)

                # Build adjusted ranges tile by tile
                adjusted_ranges = []
                for t_idx, (s, e) in enumerate(self.digit_ranges):
                    if t_idx in per_tile_offsets:
                        # compute local offset (mean of offsets in that tile)
                        offsets_in_tile = per_tile_offsets[t_idx]
                        local_offset = sum(offsets_in_tile) / len(offsets_in_tile)
                        used_offset = local_offset
                    else:
                        used_offset = avg_offset

                    # Shift the tile by used_offset
                    print("used_offset:",t_idx,used_offset)
                    ns = s #+ used_offset
                    ne = e #+ used_offset

                    # Clamp to [0, 1]
                    ns = max(0.0, min(1.0, ns))
                    ne = max(0.0, min(1.0, ne))

                    # Ensure ordering
                    if ne < ns:
                        ns, ne = ne, ns

                    adjusted_ranges.append((ns, ne))

                # Now perform assignment using adjusted_ranges
                detection_digits, detection_confs = self.assign_digits_from_ratios(
                    ratios_arr, classes_arr, confs_arr, ranges=adjusted_ranges
                )
            # if ratios_list:
            #     # Sort detections by their x‑axis ratio so that the leftmost detection is used
            #     first_idx = int(np.argmax(confs_list))
            #     first_ratio = ratios_list[first_idx]

            #     # sorted_indices = np.argsort(ratios_list)
            #     # first_idx = int(sorted_indices[1])
            #     # first_ratio = ratios_list[first_idx]
            #     # Determine which original range the first detection falls into
            #     tile_idx = None
            #     for idx, (start, end) in enumerate(self.digit_ranges):
            #         if start <= first_ratio <= end:
            #             tile_idx = idx
            #             break
            #     if tile_idx is not None:
            #         start_range, end_range = self.digit_ranges[tile_idx]
            #         tile_center = (start_range + end_range) / 2.0
            #         # Compute offset: difference between detected ratio and expected tile centre
            #         offset = first_ratio - tile_center
            #         print("offset:",offset)
            #         new_ranges: List[Tuple[float, float]] = []
            #         for s, e in self.digit_ranges:
            #             ns = s + offset#*2
            #             ne = e + offset#*2
            #             # Clamp to [0,1]
            #             ns = max(0.0, min(1.0, ns))
            #             ne = max(0.0, min(1.0, ne))
            #             # Ensure order
            #             if ne < ns:
            #                 ns, ne = ne, ns
            #             new_ranges.append((ns, ne))
            #         adjusted_ranges = new_ranges
            #     # Assign detection digits based on adjusted ranges
            #     ratios_arr = np.array(ratios_list, dtype=np.float32)
            #     classes_arr = np.array(classes_list, dtype=np.int32)
            #     confs_arr = np.array(confs_list, dtype=np.float32)
            #     detection_digits, detection_confs = self.assign_digits_from_ratios(
            #         ratios_arr, classes_arr, confs_arr, ranges=adjusted_ranges
            #     )
                #print(detection_digits)
        # Slice tiles according to adjusted ranges
        #print("here3")
        
        print("adjusted_ranges: ",adjusted_ranges)
        tiles = self.slice_digits_by_ratio(sharpened_dial, adjusted_ranges, img_width=28, img_height=28)
        #print("here45")
        
        # print("tiles: ",tiles)
        # Save each tile into a subfolder `tiles`
        save_root =  "results"
        tiles_dir = os.path.join(save_root, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)
        # bbox_info might carry bounding box or positional data for each tile
        for i, tile in enumerate(tiles):
            # tile is a small 28×28 patch (or similar) as a numpy array (grayscale or BGR)
            tile_fname = os.path.join(tiles_dir, f"tile_{i:03d}.png")
            # If tile is grayscale or single channel, cv2.imwrite works too
            cv2.imwrite(tile_fname, tile)
            # Draw tiles and detected digit boxes on the dial crop
            dial_annotated = sharpened_dial.copy()
            h, w = dial_annotated.shape[:2]
            # Draw tile boxes and centers
            for i, (start_pct, end_pct) in enumerate(adjusted_ranges):
                left = int(start_pct * w)
                right = int(end_pct * w)
                top = 0
                bottom = h - 1
                # Rectangle for tile
                cv2.rectangle(dial_annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                # Center point for tile
                cx = int((left + right) / 2)
                cy = int(h / 2)
                cv2.circle(dial_annotated, (cx, cy), 4, (0, 255, 0), -1)
            # Draw detected digit boxes and centers if available
            if self.digit_detector is not None and 'boxes' in locals():
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    # Rectangle for detected digit
                    cv2.rectangle(dial_annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    # Center point for detected digit
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(dial_annotated, (cx, cy), 4, (255, 0, 0), -1)
            # Save the annotated dial image
            annotated_fname = os.path.join(save_root, "dial_annotated.png")
            cv2.imwrite(annotated_fname, dial_annotated)
        # Classify each tile to obtain digit predictions and confidences
        classification_digits: List[str] = [''] * len(tiles)
        classification_confs: List[float] = [0.0] * len(tiles)
        if self.classifier is not None:
            #print("here45")
            classification_digits, classification_confs = self.classify_digits(tiles)
            #print("here45")
        # Combine detection and classification results by confidence
        final_positions: List[str] = []
        #print("here4")
        for idx in range(len(self.digit_ranges)):
            det_digit = detection_digits[idx] if idx < len(detection_digits) else ''
            det_conf = detection_confs[idx] if idx < len(detection_confs) else 0.0
            cls_digit = classification_digits[idx] if idx < len(classification_digits) else ''
            cls_conf = classification_confs[idx] if idx < len(classification_confs) else 0.0
            if det_digit and det_conf >= cls_conf:
                final_positions.append(det_digit)
            else:
                final_positions.append(cls_digit)
        pred_str = ''.join(final_positions)
        #print("here3")
        return pred_str, sharpened_dial, tiles, meter_crop, dial_crop

    def capture_image(self, camera_id: int = 0, width: int = 1920, height: int = 1080) -> np.ndarray:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read from camera during warm up")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        return frame


def publish_mqtt(
    broker: str,
    port: int,
    topic: str,
    payload: str,
    client_id: Optional[str] = None,
) -> None:
    if mqtt_client is None:
        raise ImportError("paho-mqtt is required for MQTT functionality.  Install it via pip.")
    cid = client_id or f"meter-reader-{datetime.datetime.now().timestamp():.0f}"
    client = mqtt_client.Client(client_id=cid)
    client.connect(broker, port)
    client.publish(topic, payload)
    client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two‑stage YOLO gas meter reader")
    parser.add_argument("--meter-detector", type=str, required=True, default=None, help="Path to the meter detector .pt file")
    parser.add_argument("--dial-detector", type=str, required=True, help="Path to the digit dial detector .pt file")
    parser.add_argument(
        "--classifier", type=str, required=False, default=None,
        help="Path to a YOLO classifier model for digits (optional). If omitted,"
             " the script will rely solely on the digit detector or leave digits blank."
    )
    parser.add_argument(
        "--digit-detector", type=str, required=False, default=None,
        help="Path to the YOLO detection model trained to detect digits 0–9."
             " When provided, this model is used to recognise each digit in the dial"
             " and rotated meter."
    )
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save processed images and CSV logs")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera ID for live capture (mutually exclusive with --image-path)")
    parser.add_argument("--image-path", type=str, default=None, help="Process a single image instead of capturing from a camera")
    parser.add_argument("--mqtt-host", type=str, default=None, help="MQTT broker host (optional)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt-topic", type=str, default=None, help="Topic for MQTT publishing")
    args = parser.parse_args()

    if (args.camera_id is None) == (args.image_path is None):
        parser.error("Specify exactly one of --camera-id or --image-path")

    os.makedirs(args.results_dir, exist_ok=True)
    reader = YOLOTwoStageMeterReader(
        args.meter_detector,
        args.dial_detector,
        classifier_path=args.classifier,
        digit_detector_path=args.digit_detector,
    )

    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Image file {args.image_path} does not exist", file=sys.stderr)
            sys.exit(1)
        img_bgr = cv2.imread(args.image_path)
        if img_bgr is None:
            print(f"Failed to load image {args.image_path}", file=sys.stderr)
            sys.exit(1)
    else:
        img_bgr = reader.capture_image(camera_id=args.camera_id)

    try:
        pred_str, processed_img, tiles, meter_crop, dial_crop = reader.analyze(img_bgr)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    # Save images: processed dial, meter crop and dial crop
    timestamp = int(datetime.datetime.now().timestamp())
    seq = len([name for name in os.listdir(args.results_dir) if name.lower().endswith(".jpg")])
    # Save the sharpened dial region
    # Use the predicted string directly in the filename for clarity
    dial_filename = f"{seq}_{pred_str}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(args.results_dir, dial_filename), processed_img)
    # Save meter and dial crops
    meter_filename = f"{seq}_meter_{timestamp}.jpg"
    dial_crop_filename = f"{seq}_dial_{timestamp}.jpg"
    cv2.imwrite(os.path.join(args.results_dir, meter_filename), meter_crop)
    cv2.imwrite(os.path.join(args.results_dir, dial_crop_filename), dial_crop)
    # Append to CSV log
    csv_path = os.path.join(args.results_dir, "result_images.csv")
    with open(csv_path, "a") as f:
        f.write(f"{timestamp}, {pred_str}\n")
    print(f"Meter reading: {pred_str}")
    if args.mqtt_host and args.mqtt_topic:
        try:
            publish_mqtt(args.mqtt_host, args.mqtt_port, args.mqtt_topic, pred_str)
        except Exception as mqtt_exc:
            print(f"Failed to publish MQTT message: {mqtt_exc}", file=sys.stderr)


if __name__ == "__main__":
    main()