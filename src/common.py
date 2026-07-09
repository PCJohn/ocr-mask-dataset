"""Shared helpers: polygon rasterization, resizing, stat aggregation, I/O."""
import io
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_IMAGE_SIDE = 2048      # cap resized image longest side to this many px
MASK_SCALE = 0.5           # masks stored at this fraction of the resized image size
JPEG_QUALITY = 90

# BSTD and other archival datasets have legitimately large images (>89MP).
# Raise PIL's decompression-bomb guard so they load instead of erroring.
Image.MAX_IMAGE_PIXELS = 300_000_000


@dataclass
class Sample:
    """One raw (pre-processing) training sample from any dataset.

    `image` may be either:
      - a PIL.Image.Image already in memory (HF datasets that return images),
      - or None, in which case `image_loader` must be set to a zero-arg
        callable that returns a PIL.Image. This lazy form is used for
        file-backed datasets (NAF, TextOCR, BSTD) so that the shuffle buffer
        never holds full uncompressed bitmaps in RAM.
    """
    sample_id: str
    image: Optional[Image.Image]
    polygons: List[np.ndarray] = field(default_factory=list)
    coarse: bool = False
    image_loader: Optional[Callable[[], Image.Image]] = field(default=None, repr=False)

    def load_image(self) -> Image.Image:
        if self.image is not None:
            return self.image
        if self.image_loader is not None:
            return self.image_loader()
        raise ValueError(f"Sample {self.sample_id} has neither image nor image_loader set")


def resize_keep_aspect(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return img, scale


def rasterize_mask(polygons: Sequence[np.ndarray], size_wh: Tuple[int, int]) -> np.ndarray:
    """polygons: list of (N,2) arrays in pixel coords matching size_wh=(W,H). Returns uint8 0/255 mask."""
    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = [p.astype(np.int32).reshape(-1, 1, 2) for p in polygons if p is not None and len(p) >= 3]
    # boxes given as 2 points (x0,y0,x1,y1) rectangles are expanded by caller before this
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask


def _refine_block_to_strokes(gray_crop: np.ndarray) -> np.ndarray:
    """Given a grayscale crop covering one coarse text-block box, shrink it down to
    the actual dark/light text strokes inside via Otsu thresholding, instead of
    marking the whole block (incl. its background whitespace) as text. Assumes
    text pixels are the minority class within the crop -- a reasonable
    assumption for a text paragraph/title block, less so for a solid list of
    dense text lines with little whitespace, but still tighter than the full box.
    Falls back to the full crop (no refinement) if the crop is degenerate.
    """
    if gray_crop.size == 0:
        return gray_crop
    _, otsu = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_frac = float((otsu == 255).mean())
    if fg_frac > 0.5:
        otsu = 255 - otsu
    return otsu


def rasterize_mask_coarse_refined(image_rgb: Image.Image, polygons: Sequence[np.ndarray],
                                   size_wh: Tuple[int, int]) -> np.ndarray:
    """Like rasterize_mask, but for block/element-level boxes: within each box,
    keep only the pixels that look like actual text strokes (via Otsu on that
    crop) rather than flood-filling the entire block including its whitespace
    and, for slide/document layouts, any non-text content that slipped past
    category filtering.
    """
    w, h = size_wh
    mask = np.zeros((h, w), dtype=np.uint8)
    gray_full = cv2.cvtColor(np.array(image_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    for p in polygons:
        if p is None or len(p) < 3:
            continue
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if x1i <= x0i or y1i <= y0i:
            continue
        local_poly = (p - [x0i, y0i]).astype(np.int32).reshape(-1, 1, 2)
        local_poly_mask = np.zeros((y1i - y0i, x1i - x0i), dtype=np.uint8)
        cv2.fillPoly(local_poly_mask, [local_poly], 255)

        gray_crop = gray_full[y0i:y1i, x0i:x1i]
        refined = _refine_block_to_strokes(gray_crop)
        combined = cv2.bitwise_and(local_poly_mask, refined)
        mask[y0i:y1i, x0i:x1i] = np.maximum(mask[y0i:y1i, x0i:x1i], combined)
    return mask


def downscale_mask(mask: np.ndarray, scale: float = MASK_SCALE) -> np.ndarray:
    h, w = mask.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)


def box_to_polygon(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def mask_contour_stats(mask: np.ndarray) -> Tuple[int, float]:
    """Returns (num_contours, text_area_fraction) for a binary 0/255 mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_frac = float((mask > 0).sum()) / float(mask.size) if mask.size else 0.0
    return len(contours), area_frac


def save_sample(out_dir: str, dataset_name: str, sample: Sample, refine_coarse: bool = True) -> dict:
    """Resize image, rasterize+downscale mask, write both, return a meta.jsonl record.

    If sample.coarse is True (block/element-level boxes, e.g. PubLayNet, SynSlides)
    and refine_coarse is True (default), uses intensity-based refinement to shrink
    each box down to its actual text strokes instead of flood-filling the whole
    block. Pass refine_coarse=False to get the old flood-fill-the-whole-box
    behavior back (e.g. for debugging/comparison).
    """
    img_dir = os.path.join(out_dir, dataset_name, "images")
    mask_dir = os.path.join(out_dir, dataset_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img = sample.load_image()
    orig_w, orig_h = img.size
    resized_img, scale = resize_keep_aspect(img)
    rw, rh = resized_img.size

    scaled_polys = [ (p * scale) for p in sample.polygons ]
    if sample.coarse and refine_coarse:
        full_res_mask = rasterize_mask_coarse_refined(resized_img, scaled_polys, (rw, rh))
    else:
        full_res_mask = rasterize_mask(scaled_polys, (rw, rh))
    small_mask = downscale_mask(full_res_mask, MASK_SCALE)
    mh, mw = small_mask.shape[:2]

    num_contours, area_frac = mask_contour_stats(full_res_mask)  # measured at full res for accuracy

    img_path = os.path.join("images", f"{sample.sample_id}.jpg")
    mask_path = os.path.join("masks", f"{sample.sample_id}.png")

    resized_img.convert("RGB").save(os.path.join(out_dir, dataset_name, img_path), quality=JPEG_QUALITY)
    Image.fromarray(small_mask).save(os.path.join(out_dir, dataset_name, mask_path))

    return {
        "id": f"{dataset_name}_{sample.sample_id}",
        "dataset": dataset_name,
        "image_path": img_path,
        "mask_path": mask_path,
        "orig_width": orig_w, "orig_height": orig_h,
        "resized_width": rw, "resized_height": rh,
        "mask_width": mw, "mask_height": mh,
        "mask_scale": MASK_SCALE,
        "num_polygons": len(sample.polygons),
        "text_area_frac": area_frac,
        "num_contours": num_contours,
        "coarse_source": sample.coarse,
    }


def write_jsonl(path: str, records: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
