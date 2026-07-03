"""CORD: Indonesian receipts. Auto-downloaded via HuggingFace `datasets`
(naver-clova-ix/cord-v2), CC BY 4.0, no login required.
"""
import argparse
import json
import os

import numpy as np

from src.common import Sample

HF_NAME = "naver-clova-ix/cord-v2"
CACHE_DIR = "data/raw/cord_hf_cache"


def download(cache_dir: str = CACHE_DIR):
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {HF_NAME} via HuggingFace datasets (this triggers on first iter_samples call too)...")
    load_dataset(HF_NAME, cache_dir=cache_dir)
    print("CORD ready (cached).")


def _quads_from_ground_truth(gt_str: str):
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
            pts = np.array([
                [quad["x1"], quad["y1"]],
                [quad["x2"], quad["y2"]],
                [quad["x3"], quad["y3"]],
                [quad["x4"], quad["y4"]],
            ], dtype=np.float32)
            polys.append(pts)
    return polys


def iter_samples(cache_dir: str = CACHE_DIR):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, cache_dir=cache_dir)
    for split in ds.keys():
        for i, ex in enumerate(ds[split]):
            img = ex["image"].convert("RGB")
            polys = _quads_from_ground_truth(ex["ground_truth"])
            yield Sample(sample_id=f"{split}_{i:05d}", image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
