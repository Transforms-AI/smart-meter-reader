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
    """Resize and sharpen the cropped dial image.

    The sharpening routine closely follows the upstream implementation.  It
    converts the image to the Lab colour space and accentuates the L
    channel using a difference of Gaussians approach.  Parameters control
    the strength (``tb_w``), threshold (``tb_th``) and blur (``tb_blur_size``
    and ``tb_blur_sigma``).

    Args:
        image: BGR image.
        dim: Desired (width, height) after resizing.
        tb_w: Sharpening strength parameter scaled by 10.
        tb_th: Threshold below which differences are ignored.
        tb_blur_size: Base blur kernel size.
        tb_blur_sigma: Gaussian sigma scaled by 10.

    Returns:
        Sharpened BGR image with the given dimensions.
    """
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    # Convert to Lab
    im_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(im_lab)
    w = tb_w / 10.0
    th = tb_th
    blur_size = tb_blur_size * 2 + 3
    blur_sigma = tb_blur_sigma / 10.0
    im_blur = cv2.GaussianBlur(l_channel, (blur_size, blur_size), blur_sigma)
    im_diff = cv2.subtract(l_channel, im_blur, dtype=cv2.CV_16S)
    im_abs_diff = cv2.absdiff(l_channel, im_blur)
    im_diff_masked = im_diff.copy()
    im_diff_masked[im_abs_diff < th] = 0
    sharpened_l = cv2.add(l_channel, w * im_diff_masked, dtype=cv2.CV_8UC1)
    res_lab = cv2.merge([sharpened_l, a_channel, b_channel])
    return cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR)


def adaptive_threshold_and_median_blur(image: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """Apply the NiBlack adaptive threshold and multiple median blurs.

    Args:
        image: BGR image to threshold.
        block_size: Window size for the NiBlack algorithm (odd integer).
        k: NiBlack ``k`` parameter controlling the bias between local mean and
           standard deviation.

    Returns:
        A greyscale image after thresholding and median blurring.
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
    # Multiple median blurs as in the original script
    for kernel in (3, 5, 7, 9):
        thresh = cv2.medianBlur(thresh, kernel)
    return thresh


def afind_contours(
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


class YOLOTwoStageMeterReader:
    """Meter reader that uses two YOLO detection stages and a classifier."""

    def __init__(
        self,
        meter_detector_path: str,
        dial_detector_path: str,
        classifier_path: str,
    ) -> None:
        if not os.path.exists(meter_detector_path):
            raise FileNotFoundError(f"Meter detector model not found at {meter_detector_path}")
        if not os.path.exists(dial_detector_path):
            raise FileNotFoundError(f"Dial detector model not found at {dial_detector_path}")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Digit classifier model not found at {classifier_path}")
        self.meter_detector = YOLO(meter_detector_path)
        self.dial_detector = YOLO(dial_detector_path)
        self.classifier = YOLO(classifier_path)
        # Preprocessing parameters (inherited from previous implementation)
        self.resize_dim = (1000, 140)
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
        
        
        self.digit_ranges = [
            (0.01463414634, 0.09268292683),
            (0.1219512195, 0.2146341463),
            (0.2390243902, 0.3268292683),
            (0.356097561, 0.4487804878),
            (0.4731707317, 0.5707317073),
            (0.5902439024, 0.6829268293),
            (0.7073170732, 0.8048780488),
            (0.8634146341, 0.9512195122),
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
            left = int(start_pct * w)
            right = int(end_pct  * w)
            # Clip bounds
            left = max(0, min(left, w - 1))
            right = max(left + 1, min(right, w))
            tile = image[:, left:right, :]
            tile_resized = cv2.resize(tile, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            tiles.append(tile_resized)
        return tiles
    

    def detect_meter(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect the meter in the full image and return a cropped region.

        Filters detections for class id 1 and selects the highest confidence.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.meter_detector(rgb, verbose=False)
        if not results or not results[0].boxes:
            raise RuntimeError("Meter detector returned no detections")
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        # Filter for class id 1
        mask = classes == 1
        if not mask.any():
            raise RuntimeError("No meter (class id 1) found in image")
        idx = np.argmax(confs * mask)
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[idx]
        print(x1,y1,x2,y2)
        h, w, _ = image_bgr.shape
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        cropped = image_bgr[y1:y2, x1:x2].copy()
        results_dir = "/home/islam/repon/smart-gas-meter-reader/results"
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Construct full path
        save_path = os.path.join(results_dir, "cropped_meter.jpg")

        # Save using OpenCV
        success = cv2.imwrite(save_path, cropped)
        if not success:
            # Optionally, raise or warn if saving fails
            raise RuntimeError(f"Failed to save cropped image to {save_path}")

        return cropped



    def order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """
        Given an array of shape (4,2) of (x,y) corner points in arbitrary order,
        return them ordered as: top-left, top-right, bottom-right, bottom-left.
        """
        # Compute sums and differences
        s = pts.sum(axis=1)     # x+y
        diff = np.diff(pts, axis=1)  # x - y

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]        # top-left has smallest x+y
        rect[2] = pts[np.argmax(s)]        # bottom-right has largest x+y
        rect[1] = pts[np.argmin(diff)]     # top-right has smallest (x - y)
        rect[3] = pts[np.argmax(diff)]     # bottom-left has largest (x - y)
        return rect

    def detect_and_flatten_dial(self, meter_image_bgr: np.ndarray, debug: bool = True) -> np.ndarray:
        """
        Detect the dial region (class 0) using OBB output, then warp-perspective transform
        that quadrilateral into a flat, upright rectangle. Return the flattened dial crop.
        """
        # Convert to RGB (if your model expects RGB)
        rgb = cv2.cvtColor(meter_image_bgr, cv2.COLOR_BGR2RGB)
        results = self.dial_detector(rgb, verbose=False)
        if not results or results[0] is None or not hasattr(results[0], "obb"):
            raise RuntimeError("Dial detector returned no oriented detections")

        obb = results[0].obb

        cls_ids = obb.cls.int().cpu().numpy()
        confs = obb.conf.cpu().numpy()
        polys = obb.xyxyxyxy.cpu().numpy()  # each is 8 numbers = 4 corners

        # Filter for class = 0 (dial)
        mask = (cls_ids == 0)
        if not mask.any():
            raise RuntimeError("No dial (class id 0) found in meter image")

        idx = int(np.argmax(confs * mask))
        poly8 = polys[idx]  # shape (8,)
        pts = poly8.reshape(4, 2).astype(np.float32)  # (4,2)

        # Order points
        rect = self.order_points_clockwise(pts)  # tl, tr, br, bl

        # Optionally, you may want to add a small margin (padding) to those points
        # to avoid cropping edges of digits. For example, expand rect slightly outward:
        # (you need to be careful when adding margin; ensure still convex shape)
        # e.g. margin = 5 pixels:
        # margin = 5
        # for i in range(4):
        #     if i in (0, 3):  # left side
        #         rect[i,0] -= margin
        #     if i in (1, 2):  # right side
        #         rect[i,0] += margin
        #     if i in (0, 1):  # top side
        #         rect[i,1] -= margin
        #     if i in (2, 3):  # bottom side
        #         rect[i,1] += margin

        # Compute width and height for the new “flat” image
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # In case maxWidth or maxHeight is zero (degenerate), bail out
        if maxWidth <= 0 or maxHeight <= 0:
            raise RuntimeError(f"Invalid warp target size (w={maxWidth}, h={maxHeight})")

        # Destination rectangle: points in the flattened image
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute homography (perspective transform) from rect → dst
        M = cv2.getPerspectiveTransform(rect, dst)
        # Warp the full image (or a region) to get the flattened dial region
        warped = cv2.warpPerspective(meter_image_bgr, M, (maxWidth, maxHeight),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        # Optionally, for debugging, draw the quadrilateral on the original image
        if True:
            vis = meter_image_bgr.copy()
            pts_int = pts.astype(int)
            for i in range(4):
                pt1 = tuple(pts_int[i])
                pt2 = tuple(pts_int[(i + 1) % 4])
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            # Save this visualization
            results_dir = "/home/islam/repon/smart-gas-meter-reader/results"
            os.makedirs(results_dir, exist_ok=True)
            cv2.imwrite(os.path.join(results_dir, "dial_quad_vis.jpg"), vis)
            cv2.imwrite(os.path.join(results_dir, "dial_flattened.jpg"), warped)

        return warped



    def classify_digits(self, tiles: List[np.ndarray]) -> str:
        """Classify each digit tile using the YOLO classifier.

        Uses the same approach as the single‑stage reader.
        """
        digits: List[str] = []
        for tile in tiles:
            rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            res = self.classifier(rgb_tile, verbose=False)
            try:
                cls_idx = int(res[0].probs.top1)
            except AttributeError:
                scores = res[0].probs.data
                cls_idx = int(np.argmax(scores))
            digits.append(str(cls_idx))
        return ''.join(digits)

    def analyze(self, image: np.ndarray) -> Tuple[str, np.ndarray, List[np.ndarray]]:
        """Run the two‑stage pipeline: detect meter, balance, detect dial, preprocess and classify digits.

        In this variant the balancing step is applied *after* cropping the
        meter.  This ensures that the Hough‑transform based rotation
        correction acts on the meter region itself rather than the
        entire scene.
        """
        # Stage 1: detect and crop the meter from the raw image
        meter_crop = self.detect_meter(image)
        print(meter_crop)
        # Balance the meter crop to reduce tilt
        meter_grey = cv2.cvtColor(meter_crop, cv2.COLOR_BGR2GRAY)
        # balanced_meter = balancing_tilted_image(meter_crop.copy(), meter_grey.copy(), self.balancing_cycles)
        # Stage 2: detect and crop the dial region from the balanced meter
        dial_crop = self.detect_and_flatten_dial(meter_crop.copy())
        
        sharpened = resize_and_sharpen_image(
        dial_crop,
        self.resize_dim,
        self.tb_w,
        self.tb_th,
        self.tb_blur_size,
        self.tb_blur_sigma,
        )
        
        # Slice into 8 digit tiles using predefined x-axis ratios
        tiles = self.slice_digits_by_ratio(dial_crop, self.digit_ranges, img_width=28, img_height=28)
        # Classify
        # # 3. threshold and find contours / tiles
        thresh = adaptive_threshold_and_median_blur(sharpened, self.block_size, self.k)
        # tiles, bbox_info = find_contours(
        #     thresh,
        #     sharpened,
        #     img_width=28,
        #     img_height=28,
        #     h_min=self.h_min,
        #     w_min=self.w_min,
        #     w_max=self.w_max,
        #     x_min=self.x_min,
        #     x_max=self.x_max,
        #     h_w_ratio_max=self.h_w_ratio_max,
        #     h_w_ratio_min=self.h_w_ratio_min,
        #     y_min=self.y_min,
        #     y_max=self.y_max,
        # )

        # 4. classify digits
        pred_str = self.classify_digits(tiles)
        save_root =  "/home/islam/repon/smart-gas-meter-reader/results"

        # 5. Save the images if a save_root is provided
        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)
            # Save sharpened dial
            dial_fname = os.path.join(save_root, "dial_sharpened.jpg")
            cv2.imwrite(dial_fname, sharpened)

            # Save threshold image (optional)
            thresh_fname = os.path.join(save_root, "dial_thresh.jpg")
            cv2.imwrite(thresh_fname, thresh)

            # Save each tile into a subfolder `tiles`
            tiles_dir = os.path.join(save_root, "tiles")
            os.makedirs(tiles_dir, exist_ok=True)
            # bbox_info might carry bounding box or positional data for each tile
            for i, tile in enumerate(tiles):
                # tile is a small 28×28 patch (or similar) as a numpy array (grayscale or BGR)
                tile_fname = os.path.join(tiles_dir, f"tile_{i:03d}.png")
                # If tile is grayscale or single channel, cv2.imwrite works too
                cv2.imwrite(tile_fname, tile)

            # Optionally, you might want to save an overlay / annotated version
            # e.g. bounding boxes on sharpened image
            # if bbox_info is not None:
            #     vis = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB) if sharpened.ndim == 3 else cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            #     for i, (x, y, w, h) in enumerate(bbox_info):
            #         cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #         cv2.putText(vis, str(i), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            #     vis_fname = os.path.join(save_root, "tiles_overlay.jpg")
            #     cv2.imwrite(vis_fname, vis)

        # # Resize and sharpen to a fixed dimension
        # sharpened = resize_and_sharpen_image(
        #     dial_crop,
        #     self.resize_dim,
        #     self.tb_w,
        #     self.tb_th,
        #     self.tb_blur_size,
        #     self.tb_blur_sigma,
        # )
        # # Threshold and find contours
        # thresh = adaptive_threshold_and_median_blur(sharpened, self.block_size, self.k)
        # tiles, _ = find_contours(
        #     thresh,
        #     sharpened,
        #     img_width=28,
        #     img_height=28,
        #     h_min=self.h_min,
        #     w_min=self.w_min,
        #     w_max=self.w_max,
        #     x_min=self.x_min,
        #     x_max=self.x_max,
        #     h_w_ratio_max=self.h_w_ratio_max,
        #     h_w_ratio_min=self.h_w_ratio_min,
        #     y_min=self.y_min,
        #     y_max=self.y_max,
        # )
        # pred_str = self.classify_digits(tiles)
        return pred_str, sharpened, tiles

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
    parser.add_argument("--meter-detector", type=str, required=True, help="Path to the meter detector .pt file")
    parser.add_argument("--dial-detector", type=str, required=True, help="Path to the digit dial detector .pt file")
    parser.add_argument("--classifier", type=str, required=True, help="Path to the digit classifier .pt file")
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
    reader = YOLOTwoStageMeterReader(args.meter_detector, args.dial_detector, args.classifier)

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
        pred_str, processed_img, tiles = reader.analyze(img_bgr)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    timestamp = int(datetime.datetime.now().timestamp())
    seq = len([name for name in os.listdir(args.results_dir) if name.lower().endswith(".jpg")])
    filename = f"{seq}_{pred_str[:-3]}_{pred_str[-3:]}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(args.results_dir, filename), processed_img)
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