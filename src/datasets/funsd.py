"""FUNSD: noisy scanned forms. Direct zip download, no login required.
https://guillaumejaume.github.io/FUNSD/
"""
import argparse
import glob
import json
import os
import zipfile

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.common import Sample, box_to_polygon

URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
RAW_DIR = "data/raw/funsd"


def download(raw_dir: str = RAW_DIR):
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "dataset.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading FUNSD from {URL} ...")
        r = requests.get(URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    extract_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    print(f"FUNSD ready at {extract_dir}")


def iter_samples(raw_dir: str = RAW_DIR):
    """Yields Sample objects for both training_data and testing_data splits."""
    extract_dir = os.path.join(raw_dir, "extracted", "dataset")
    for split in ("training_data", "testing_data"):
        ann_dir = os.path.join(extract_dir, split, "annotations")
        img_dir = os.path.join(extract_dir, split, "images")
        if not os.path.isdir(ann_dir):
            continue
        for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
            base = os.path.splitext(os.path.basename(ann_path))[0]
            img_path = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_path):
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            polys = []
            for item in ann.get("form", []):
                box = item.get("box")  # [x0,y0,x1,y1]
                if box and len(box) == 4:
                    polys.append(box_to_polygon(*box))
                for word in item.get("words", []):
                    wb = word.get("box")
                    if wb and len(wb) == 4:
                        polys.append(box_to_polygon(*wb))
            img = Image.open(img_path)
            yield Sample(sample_id=f"{split}_{base}", image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
