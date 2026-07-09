"""Thin wrapper around EasyOCR (or PaddleOCR) for text detection.

KEY CONSTRAINT: EasyOCR requires languages to be grouped by script family --
you cannot mix e.g. Devanagari and CJK in one Reader. This module creates
multiple specialised Readers and runs them in sequence, merging results.

Supported language coverage by script family:
  Latin:      en, id (Indonesian), de, fr, es, pt, nl, it, pl, ...
  CJK:        ch_sim, ch_tra, ja, ko
  Devanagari: hi, mr, ne
  Cyrillic:   ru, bg, uk, be, mn, rs_cyrillic
  Arabic:     ar, fa, ur, ug
  Tamil:      ta
  Telugu:     te
  Kannada:    kn
  Bengali:    bn, as (Assamese)

NOT supported (no model released): ml (Malayalam), pa (Punjabi/Gurmukhi),
or (Odia), gu (Gujarati). Text in these scripts will be detected at the
bounding-box level by the shared CRAFT detector (which is script-agnostic)
but character recognition will be wrong -- good enough for mask generation
since we only need *where* text is, not *what* it says.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image

# ── Script family groupings (EasyOCR constraint) ─────────────────────────────
# Each group can be loaded into a single easyocr.Reader together.
# 'en' is compatible with every group so we always include it.
EASYOCR_SCRIPT_GROUPS = [
    [
        "en",
        "id",
        "de",
        "fr",
        "es",
        "pt",
        "nl",
        "it",
        "pl",
        "ru",
        "bg",
        "uk",
        "be",
        "mn",
        "rs_cyrillic",
    ],  # Latin + Cyrillic share a detector in EasyOCR
    ["ch_sim", "ch_tra", "ja", "ko", "en"],  # CJK
    ["hi", "mr", "ne", "en"],  # Devanagari (Hindi, Marathi, Nepali)
    ["ta", "en"],  # Tamil
    ["te", "en"],  # Telugu
    ["kn", "en"],  # Kannada
    ["bn", "as", "en"],  # Bengali / Assamese
    ["ar", "fa", "ur", "ug", "en"],  # Arabic script
]


def get_reader(backend: str = "easyocr", gpu: bool = False, langs: list = None):
    """Return an initialised OCR reader (or list of readers for EasyOCR multi-script).

    For EasyOCR, returns a list of (reader, lang_group) tuples.
    For PaddleOCR, returns a single reader object.
    Pass the return value to boxes_from_image() with the same backend.
    """
    if backend == "easyocr":
        import easyocr

        if langs:
            # single custom group -- caller knows what they're doing
            return [
                (
                    easyocr.Reader(
                        langs, gpu=gpu, verbose=False, download_enabled=True
                    ),
                    langs,
                )
            ]
        readers = []
        for group in EASYOCR_SCRIPT_GROUPS:
            try:
                r = easyocr.Reader(group, gpu=gpu, verbose=False, download_enabled=True)
                readers.append((r, group))
            except Exception as e:
                print(f"[ocr_backend] Warning: could not load group {group}: {e}")
        return readers

    elif backend == "paddleocr":
        from paddleocr import PaddleOCR

        return PaddleOCR(
            use_angle_cls=True, lang="multilingual", use_gpu=gpu, show_log=False
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'easyocr' or 'paddleocr'.")


def boxes_from_image(
    reader, img: Image.Image, backend: str = "easyocr", min_confidence: float = 0.3
) -> List[np.ndarray]:
    """Run OCR on a PIL image, return list of (N,2) float32 polygon arrays.

    `reader` should be the object returned by get_reader().
    For EasyOCR this is a list of (reader, group) tuples.
    """
    arr = np.array(img.convert("RGB"))
    polys = []
    seen_boxes = set()  # deduplicate boxes across script groups

    if backend == "easyocr":
        reader_list = reader if isinstance(reader, list) else [reader]
        for r, _ in reader_list:
            try:
                results = r.readtext(arr, detail=1, paragraph=False)
            except Exception:
                continue
            for bbox, text, conf in results:
                if conf < min_confidence:
                    continue
                pts = np.array(bbox, dtype=np.float32)  # shape (4,2)
                # deduplicate by top-left corner rounded to 5px grid
                key = (round(pts[0, 0] / 5) * 5, round(pts[0, 1] / 5) * 5)
                if key not in seen_boxes:
                    seen_boxes.add(key)
                    polys.append(pts)

    elif backend == "paddleocr":
        results = reader.ocr(arr, cls=True)
        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                if conf < min_confidence:
                    continue
                polys.append(np.array(bbox, dtype=np.float32))

    return polys


def polys_to_mask(polys: List[np.ndarray], size_wh: Tuple[int, int]) -> np.ndarray:
    """Rasterise polygon list into a binary 0/255 uint8 mask of shape (H,W)."""
    import cv2

    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [p.astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask
