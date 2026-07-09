"""NAF Dataset: US National Archives form images, text/field boxes.
https://github.com/herobd/NAF_dataset -- CC BY 4.0 (commercial use OK w/ attribution).
Fully automatic: images from a GitHub release tar.gz, annotations via git clone.
"""
import argparse
import glob
import json
import os
import subprocess
import tarfile

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.common import Sample

RAW_DIR = "data/raw/naf"
IMAGES_URL = "https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz"
REPO_URL = "https://github.com/herobd/NAF_dataset.git"

# annotation "type" values that denote actual text (as opposed to field/graphic/comment boxes)
TEXT_TYPE_PREFIXES = ("text",)


def download(raw_dir: str = RAW_DIR):
    os.makedirs(raw_dir, exist_ok=True)

    # 1. images
    tgz_path = os.path.join(raw_dir, "labeled_images.tar.gz")
    img_dir = os.path.join(raw_dir, "imgs")
    if not os.path.exists(img_dir):
        if not os.path.exists(tgz_path):
            print(f"Downloading NAF images from {IMAGES_URL} ...")
            r = requests.get(IMAGES_URL, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tgz_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Extracting images...")
        with tarfile.open(tgz_path) as tf:
            tf.extractall(raw_dir)
        # the tarball extracts to labeled_images/ ; normalize to imgs/
        extracted = os.path.join(raw_dir, "labeled_images")
        if os.path.isdir(extracted) and not os.path.exists(img_dir):
            os.rename(extracted, img_dir)

    # 2. annotations (json files with poly_points), via shallow git clone
    ann_repo_dir = os.path.join(raw_dir, "NAF_dataset_repo")
    groups_dir = os.path.join(raw_dir, "groups")
    if not os.path.exists(groups_dir):
        if not os.path.exists(ann_repo_dir):
            print("Cloning annotation repo...")
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, ann_repo_dir], check=True)
        os.rename(os.path.join(ann_repo_dir, "groups"), groups_dir)

    print(f"NAF ready at {raw_dir}")


def _polys_from_json(ann_path: str):
    with open(ann_path) as f:
        ann = json.load(f)
    polys = []
    for key in ("textBBs", "fieldBBs"):
        for item in ann.get(key, []):
            if not isinstance(item, dict):
                continue
            t = item.get("type", "")
            if not str(t).startswith(TEXT_TYPE_PREFIXES):
                continue
            pts = item.get("poly_points")
            if pts and len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.float32))
    return polys


def iter_samples(raw_dir: str = RAW_DIR):
    groups_dir = os.path.join(raw_dir, "groups")
    img_dir = os.path.join(raw_dir, "imgs")
    if not os.path.isdir(groups_dir):
        raise FileNotFoundError(f"{groups_dir} not found. Run `python -m src.datasets.naf --download` first.")
    for ann_path in sorted(glob.glob(os.path.join(groups_dir, "*", "*.json"))):
        base = os.path.splitext(os.path.basename(ann_path))[0]
        # images may live in nested group subdirs matching the json's group name
        candidates = glob.glob(os.path.join(img_dir, "*", base + ".jpg")) or \
                     glob.glob(os.path.join(img_dir, base + ".jpg"))
        if not candidates:
            continue
        img_path = candidates[0]
        polys = _polys_from_json(ann_path)
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        yield Sample(sample_id=base, image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
