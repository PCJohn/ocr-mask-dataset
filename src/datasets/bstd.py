"""Bharat Scene Text Dataset (BSTD): real photographed scene text (signboards,
billboards, bus stops, railway stations, ATMs) across 11 Indian languages
(Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia,
Punjabi, Tamil, Telugu) + English. 6,582 images, 126k word-level polygons.
https://github.com/Bhashini-IITJ/BharatSceneTextDataset

LICENSE NOTE: the repo/annotations are Apache-2.0, but the underlying images
are sourced from Wikimedia Commons under CC BY-SA 4.0 (share-alike). Apache-2.0
alone would be unambiguously fine commercially; CC BY-SA on the images is the
same share-alike caveat as HierText elsewhere in this project -- commercial
use is permitted, but read what share-alike means for your specific product
before shipping. This is meaningfully the best real (non-synthetic),
multilingual, Indian-language scene-text dataset with an open license and a
scriptable download I could find.

SIZE NOTE: detection.zip is ~17GB, hosted on Google Drive (public link, no
login) -- much bigger than the other datasets in this repo. It's a single
archive so there's no partial/streamed download; budget disk + time
accordingly. Use --limit in build_dataset.py to cap how many *processed*
images you keep once it's unpacked.
"""
import argparse
import json
import os
import zipfile

import numpy as np
from PIL import Image

from src.common import Sample

RAW_DIR = "data/raw/bstd"
GDRIVE_FILE_ID = "1S7KUYfB-lQvbu6GtvZxxDPOD5ZZ080M0"  # detection.zip


def download(raw_dir: str = RAW_DIR):
    import gdown
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "detection.zip")
    extract_dir = os.path.join(raw_dir, "Detection")
    if not os.path.exists(zip_path) and not os.path.isdir(extract_dir):
        print("Downloading BSTD detection.zip (~17GB, this will take a while)...")
        gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)
    if not os.path.isdir(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
    print(f"BSTD ready at {extract_dir}")


def iter_samples(raw_dir: str = RAW_DIR):
    detection_dir = os.path.join(raw_dir, "Detection")
    json_candidates = [f for f in os.listdir(detection_dir) if f.startswith("BSTD_") and f.endswith(".json")] \
        if os.path.isdir(detection_dir) else []
    if not json_candidates:
        raise FileNotFoundError(
            f"No BSTD_*.json found under {detection_dir}. Run `python -m src.datasets.bstd --download` first."
        )
    json_path = os.path.join(detection_dir, json_candidates[0])
    with open(json_path) as f:
        data = json.load(f)

    for key, entry in data.items():
        image_name = entry.get("image_name")
        if not image_name:
            continue
        img_path = os.path.join(detection_dir, image_name)
        if not os.path.exists(img_path):
            continue
        polys = []
        for poly_entry in (entry.get("annotations") or {}).values():
            coords = poly_entry.get("coordinates")
            if coords and len(coords) >= 3:
                polys.append(np.array(coords, dtype=np.float32))
        img = Image.open(img_path)
        yield Sample(sample_id=key, image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
