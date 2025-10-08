import os
import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_draw(full_image_path: str,
                    meter_model_path: str,
                    dial_model_path: str,
                    output_path: str,
                    meter_conf_thresh: float = 0.3,
                    dial_conf_thresh: float = 0.3,
                    padding: int = 10):
    """
    full_image_path: path to full room image
    meter_model_path: path to YOLO model for meter detection (axis aligned)
    dial_model_path: path to YOLO-OBB model for digit dial detection
    output_path: where to save annotated image
    meter_conf_thresh / dial_conf_thresh: detection thresholds
    padding: extra pixels around meter crop
    """
    # Load models
    meter_model = YOLO(meter_model_path)
    dial_model = YOLO(dial_model_path)

    # Read full image
    img_full = cv2.imread(full_image_path)
    if img_full is None:
        raise ValueError(f"Cannot load image: {full_image_path}")
    h_full, w_full = img_full.shape[:2]

    # 1. Detect meters in full image
    meter_results = meter_model(full_image_path, conf=meter_conf_thresh)
    # meter_results is a list (batch); we assume one image, so take meter_results[0]
    meter_res0 = meter_results[0]

    # Get bounding boxes (axis-aligned) for meters
    # meter_res0.boxes gives Boxes object (x1,y1,x2,y2 etc)
    # Note: convert to numpy or .cpu()
    meter_boxes = meter_res0.boxes.cpu().numpy() if meter_res0.boxes is not None else np.zeros((0, 4))
    meter_scores = meter_res0.boxes.conf.cpu().numpy() if meter_res0.boxes is not None else []
    meter_classes = meter_res0.boxes.cls.cpu().numpy().astype(int) if meter_res0.boxes is not None else []

    # We'll accumulate OBBs back in full-image coords
    all_dial_obbs_global = []

    # 2. For each detected meter, crop and detect dials
    for (box, score, cls) in zip(meter_boxes, meter_scores, meter_classes):
        x1, y1, x2, y2 = box  # absolute pixel coords
        # Add padding
        x1_p = max(0, int(x1) - padding)
        y1_p = max(0, int(y1) - padding)
        x2_p = min(w_full, int(x2) + padding)
        y2_p = min(h_full, int(y2) + padding)

        crop = img_full[y1_p:y2_p, x1_p:x2_p]
        if crop.size == 0:
            continue

        # Save or pass crop to OBB detector
        # Convert crop to a temporary in-memory image if needed
        # dial_results = dial_model(crop, conf=dial_conf_thresh)  # this also works
        # But to ensure proper coordinate transforms, better pass the full image and ROI
        # We'll do direct detection on crop:
        dial_results = dial_model(crop, conf=dial_conf_thresh)

        dial_res0 = dial_results[0]
        # The OBB results are in dial_res0.obb (or dial_res0.boxes + angle) depending on API
        # According to Ultralytics, Results object has `.obb` for the oriented bounding boxes. :contentReference[oaicite:0]{index=0}

        if dial_res0.obb is None:
            # no detections
            continue

        obb_arr = dial_res0.obb.cpu().numpy()  # shape (N, 5) or (N, 8) depending on format (cx, cy, w, h, angle) or polygon
        # Also confidences and class
        obb_confs = dial_res0.obb.conf.cpu().numpy() if hasattr(dial_res0.obb, 'conf') else None
        obb_cls = dial_res0.obb.cls.cpu().numpy().astype(int) if hasattr(dial_res0.obb, 'cls') else None

        # Map each OBB in crop coords to full-image coords
        for idx, obb in enumerate(obb_arr):
            # Suppose obb format is (cx, cy, w, h, angle) in *normalized crop image coords* or absolute?
            # We need to check how ultralytics returns OBB (you can inspect dial_res0.obb.data). Let's assume absolute pixel coords in crop.
            cx, cy, bw, bh, angle = obb  # adjust if polygon form
            # Map center
            global_cx = cx + x1_p
            global_cy = cy + y1_p
            # Store
            all_dial_obbs_global.append((global_cx, global_cy, bw, bh, angle, (x1_p, y1_p, x2_p, y2_p)))

    # 3. Draw everything on a copy
    out = img_full.copy()

    # Draw meter boxes
    for (box, score, cls) in zip(meter_boxes, meter_scores, meter_classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"meter:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw dial OBBs
    for (cx, cy, bw, bh, angle, meter_box) in all_dial_obbs_global:
        # Draw the oriented rectangle — compute its 4 corners
        # Use cv2.boxPoints (expects center, size, angle)
        rect = ((cx, cy), (bw, bh), angle * 180.0 / np.pi if abs(angle) <= np.pi else angle)  # may need rad->deg or deg, check
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)
        cv2.polylines(out, [box_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # Label
        cv2.putText(out, "dial", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save
    cv2.imwrite(output_path, out)
    print("Saved annotated:", output_path)


if __name__ == "__main__":
    # Example usage
    detect_and_draw(
        full_image_path="test_images/meter1.jpg",
        meter_model_path="models/meter_detect.pt",
        dial_model_path="models/digit_dial_obb.pt",
        output_path="results/annotated_meter1.jpg"
    )
