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


def detect_text(
    reader,
    img_or_array,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
) -> Tuple[List[np.ndarray], float]:
    """Run CRAFT detector. Accepts PIL Image or pre-resized numpy array.

    Returns (polys_in_original_coords, scale).
    If img_or_array is a PIL Image it will be resized; if it's already a numpy
    array pass scale explicitly via _detect_raw.
    """
    if isinstance(img_or_array, Image.Image):
        arr, scale = _resize_for_ocr(img_or_array)
    else:
        arr, scale = img_or_array, 1.0
    return _detect_raw(arr, scale, reader, text_threshold, low_text, link_threshold)


def _detect_raw(
    arr: np.ndarray,
    scale: float,
    reader,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
) -> Tuple[List[np.ndarray], float]:
    """Run CRAFT on a pre-resized numpy array. Scales boxes back by 1/scale.
    Returns (polys_in_original_coords, scale).
    """
    polys = []
    try:
        result = reader.detect(
            arr,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
        )
        if result and len(result) >= 2:
            free_boxes = result[1]
            horiz_boxes = result[0]
            boxes = free_boxes[0] if free_boxes and free_boxes[0] else None
            if boxes is None and horiz_boxes and horiz_boxes[0]:
                for b in horiz_boxes[0]:
                    x0, x1, y0, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    polys.append(
                        np.array(
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
                        )
                        / scale
                    )
            elif boxes:
                for b in boxes:
                    polys.append(np.array(b, dtype=np.float32) / scale)
    except Exception:
        pass
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
