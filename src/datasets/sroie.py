"""SROIE: scanned receipts (ICDAR2019). Manual download required (Kaggle login).

Expected layout after you unzip into data/raw/sroie/:
    data/raw/sroie/img/*.jpg
    data/raw/sroie/box/*.txt      # each line: x1,y1,x2,y2,x3,y3,x4,y4,transcript
See README.md "SROIE manual steps" for the download link + instructions.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from src.common import Sample

RAW_DIR = "data/raw/sroie"


def _parse_box_file(path: str):
    polys = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
            pts = np.array(coords, dtype=np.float32).reshape(4, 2)
            polys.append(pts)
    return polys


def iter_samples(raw_dir: str = RAW_DIR):
    img_dir = os.path.join(raw_dir, "img")
    box_dir = os.path.join(raw_dir, "box")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"{img_dir} not found. Follow the SROIE manual download steps in README.md first."
        )
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        base = os.path.splitext(os.path.basename(img_path))[0]
        box_path = os.path.join(box_dir, base + ".txt")
        if not os.path.exists(box_path):
            continue
        polys = _parse_box_file(box_path)
        img = Image.open(img_path)
        yield Sample(sample_id=base, image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="just verify raw data is present")
    args = p.parse_args()
    if args.check:
        n = sum(1 for _ in iter_samples())
        print(f"Found {n} SROIE samples.")
