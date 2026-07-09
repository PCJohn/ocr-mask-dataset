"""OCR backend — detection-only mode for mask cleaning.

KEY INSIGHT: for cleaning (we only need WHERE text is, not WHAT it says),
we use EasyOCR in DETECTOR-ONLY mode:
  reader = easyocr.Reader(['en'], recognizer=False)
  boxes, _ = reader.detect(image_array)

The CRAFT detector is completely script-agnostic — it finds text in any
language without a recogniser model. This means:
  - Only ONE reader needed (no per-script groups)
  - ~2x faster: no recogniser forward pass
  - ~half the VRAM: no recogniser model loaded
  - No language compatibility errors

Images are resized to MAX_OCR_SIDE before detection to avoid GPU OOM.
Batch size is tunable for GPU throughput.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image

MAX_OCR_SIDE = 1024  # resize long edge to this before CRAFT detection


def _resize_for_ocr(img: Image.Image) -> Tuple[np.ndarray, float]:
    """Resize to MAX_OCR_SIDE on long side. Returns (rgb_array, scale)."""
    w, h = img.size
    scale = min(1.0, MAX_OCR_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
        )
    return np.array(img.convert("RGB")), scale


def get_reader(gpu: bool = False):
    """Return a single EasyOCR reader in detector-only mode.

    recognizer=False: skips loading all per-script recogniser models.
    The CRAFT detector works for all scripts with just ['en'].
    """
    import easyocr

    print("  Loading EasyOCR detector (detector-only, no recogniser)...")
    return easyocr.Reader(
        ["en"], gpu=gpu, recognizer=False, verbose=False, download_enabled=True
    )


def detect_boxes(
    reader,
    img: Image.Image,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
) -> Tuple[List[np.ndarray], float]:
    """Run CRAFT detector on a PIL image.

    Returns (polys_in_original_coords, scale).
    polys: list of (4,2) float32 arrays (word bounding quads).
    scale: factor applied before detection (divide to get original coords).

    text_threshold / low_text / link_threshold: CRAFT parameters.
      Lower text_threshold → more detections (more recall, less precision).
      Raise it to reduce false positives on non-text regions.
    """
    arr, scale = _resize_for_ocr(img)

    # reader.detect() returns (horizontal_boxes, free_boxes)
    # horizontal_boxes: list of [[x_min, x_max, y_min, y_max]] per image
    # free_boxes: list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] quads per image
    result = reader.detect(
        arr,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
    )

    polys = []
    if result and len(result) >= 2:
        free_boxes = result[1]  # arbitrary quads (better for rotated text)
        horiz_boxes = result[0]

        # prefer free_boxes (quads); fall back to horizontal boxes
        boxes_to_use = free_boxes[0] if free_boxes and free_boxes[0] else None
        if boxes_to_use is None and horiz_boxes and horiz_boxes[0]:
            # horiz boxes are [x_min, x_max, y_min, y_max]
            for b in horiz_boxes[0]:
                x0, x1, y0, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                polys.append(
                    np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                    / scale
                )
        elif boxes_to_use:
            for b in boxes_to_use:
                pts = np.array(b, dtype=np.float32) / scale
                polys.append(pts)

    return polys, scale


def polys_to_mask(polys: List[np.ndarray], size_wh: Tuple[int, int]) -> np.ndarray:
    """Rasterise polygon list into a binary 0/255 uint8 mask (H,W)."""
    import cv2

    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [p.astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two binary (0/255) masks."""
    a = mask_a > 127
    b = mask_b > 127
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 1.0
