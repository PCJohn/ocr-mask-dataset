"""CORD: Indonesian receipts. Auto-downloaded via HuggingFace `datasets`
(naver-clova-ix/cord-v2), CC BY 4.0, no login required.

VERTICAL EDGE BANNER FILTER:
CORD receipts often have vertical brand/watermark text running along the left
and right edges (e.g. "VIETNAM PHO NOODLE" printed sideways). This text is
annotated in valid_line since it's part of the receipt image, but it has no
semantic content for OCR and is not the kind of text a general detector should
focus on. More importantly, the quads for these words have their long axis
oriented vertically rather than horizontally.

Heuristic: a word quad is treated as vertical-edge banner text and dropped if:
  (a) its height-to-width ratio > VERTICAL_RATIO_THRESH (taller than wide), AND
  (b) its centroid x is within EDGE_MARGIN_FRAC of the left or right image border.

Both conditions must hold -- (a) alone would drop legitimate tall narrow
characters; (b) alone would drop vertical text in the middle of the page
(which could be legitimate, e.g. a rotated label). Together they reliably
target only the sideways edge banners.
"""

import argparse
import json
import os

import numpy as np

from src.common import Sample

HF_NAME = "naver-clova-ix/cord-v2"
CACHE_DIR = "data/raw/cord_hf_cache"

VERTICAL_RATIO_THRESH = 1.5  # height/width > this → considered vertical
EDGE_MARGIN_FRAC = 0.12  # centroid within this fraction of left/right edge


def download(cache_dir: str = CACHE_DIR):
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {HF_NAME} via HuggingFace datasets...")
    load_dataset(HF_NAME, cache_dir=cache_dir)
    print("CORD ready (cached).")


def _is_vertical_edge_banner(pts: np.ndarray, img_w: int) -> bool:
    """Return True if the quad looks like a vertical edge banner to skip."""
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    width = float(x_coords.max() - x_coords.min())
    height = float(y_coords.max() - y_coords.min())
    if width < 1:
        return True  # degenerate quad
    ratio = height / width
    if ratio <= VERTICAL_RATIO_THRESH:
        return False  # not vertical enough
    cx = float(x_coords.mean())
    margin = EDGE_MARGIN_FRAC * img_w
    return cx < margin or cx > (img_w - margin)


def _quads_from_ground_truth(gt_str: str, img_w: int):
    polys = []
    try:
        gt = json.loads(gt_str)
    except (json.JSONDecodeError, TypeError):
        return polys
    for line in gt.get("valid_line", []):
        for word in line.get("words", []):
            quad = word.get("quad")
            if not quad:
                continue
            pts = np.array(
                [
                    [quad["x1"], quad["y1"]],
                    [quad["x2"], quad["y2"]],
                    [quad["x3"], quad["y3"]],
                    [quad["x4"], quad["y4"]],
                ],
                dtype=np.float32,
            )
            if _is_vertical_edge_banner(pts, img_w):
                continue
            polys.append(pts)
    return polys


def iter_samples(cache_dir: str = CACHE_DIR):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, cache_dir=cache_dir)
    for split in ds.keys():
        for i, ex in enumerate(ds[split]):
            img = ex["image"].convert("RGB")
            img_w, _ = img.size
            polys = _quads_from_ground_truth(ex["ground_truth"], img_w)
            yield Sample(sample_id=f"{split}_{i:05d}", image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
