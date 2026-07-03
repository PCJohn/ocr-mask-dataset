"""Shared helpers: polygon rasterization, resizing, stat aggregation, I/O."""
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_IMAGE_SIDE = 1024      # cap resized image longest side to this many px
MASK_SCALE = 0.5           # masks stored at this fraction of the resized image size
JPEG_QUALITY = 90


@dataclass
class Sample:
    """One raw (pre-processing) training sample from any dataset."""
    sample_id: str
    image: Image.Image
    polygons: List[np.ndarray] = field(default_factory=list)  # list of (N,2) int arrays, pixel coords in `image`


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


def save_sample(out_dir: str, dataset_name: str, sample: Sample) -> dict:
    """Resize image, rasterize+downscale mask, write both, return a meta.jsonl record."""
    img_dir = os.path.join(out_dir, dataset_name, "images")
    mask_dir = os.path.join(out_dir, dataset_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    orig_w, orig_h = sample.image.size
    resized_img, scale = resize_keep_aspect(sample.image)
    rw, rh = resized_img.size

    scaled_polys = [ (p * scale) for p in sample.polygons ]
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
    }


def write_jsonl(path: str, records: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
