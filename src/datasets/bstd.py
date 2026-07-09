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
from PIL import Image, ImageOps

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
    json_candidates = (
        [
            f
            for f in os.listdir(detection_dir)
            if f.startswith("BSTD_") and f.endswith(".json")
        ]
        if os.path.isdir(detection_dir)
        else []
    )
    if not json_candidates:
        raise FileNotFoundError(
            f"No BSTD_*.json found under {detection_dir}. Run `python -m src.datasets.bstd --download` first."
        )
    json_path = os.path.join(detection_dir, json_candidates[0])
    with open(json_path) as f:
        data = json.load(f)


def _exif_orientation(img_path: str) -> int:
    """Return the EXIF Orientation tag value (1–8), or 1 if absent."""
    try:
        with Image.open(img_path) as im:
            exif = im._getexif()
            if exif:
                # tag 274 = Orientation
                return exif.get(274, 1)
    except Exception:
        pass
    return 1


def _rotate_poly(
    pts: np.ndarray, orientation: int, img_w: int, img_h: int
) -> np.ndarray:
    """Transform polygon coordinates to match what ImageOps.exif_transpose() does."""
    # EXIF orientation → (transpose_method, affects_dims)
    # We need to map from original pixel coords → coords in transposed image.
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    if orientation == 1:
        pass  # no-op
    elif orientation == 2:
        x = img_w - 1 - x  # flip horizontal
    elif orientation == 3:
        x = img_w - 1 - x  # rotate 180
        y = img_h - 1 - y
    elif orientation == 4:
        y = img_h - 1 - y  # flip vertical
    elif orientation == 5:
        x, y = y, x  # transpose (flip over main diagonal)
    elif orientation == 6:
        x, y = img_h - 1 - y, x  # rotate 90 CW
    elif orientation == 7:
        x, y = y, img_w - 1 - x  # transverse
    elif orientation == 8:
        x, y = y, img_w - 1 - x  # rotate 90 CCW
        x, y = img_h - 1 - x, y  # (two-step for 8)
    return np.stack([x, y], axis=1).astype(np.float32)


def iter_samples(raw_dir: str = RAW_DIR):
    detection_dir = os.path.join(raw_dir, "Detection")
    json_candidates = (
        [
            f
            for f in os.listdir(detection_dir)
            if f.startswith("BSTD_") and f.endswith(".json")
        ]
        if os.path.isdir(detection_dir)
        else []
    )
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

        # Read image dimensions and EXIF orientation eagerly (metadata-only, fast).
        # We need these to correctly rotate polygon coords to match the
        # post-EXIF-transpose pixel layout.
        try:
            with Image.open(img_path) as _im:
                raw_w, raw_h = _im.size
        except Exception:
            continue
        orientation = _exif_orientation(img_path)

        polys = []
        for poly_entry in (entry.get("annotations") or {}).values():
            coords = poly_entry.get("coordinates")
            if coords and len(coords) >= 3:
                pts = np.array(coords, dtype=np.float32)
                if orientation != 1:
                    pts = _rotate_poly(pts, orientation, raw_w, raw_h)
                polys.append(pts)

        yield Sample(
            sample_id=key,
            image=None,
            polygons=polys,
            image_loader=lambda p=img_path: ImageOps.exif_transpose(
                Image.open(p)
            ).convert("RGB"),
        )


def _debug(raw_dir=RAW_DIR, n_samples: int = 3):
    import json, os, glob

    detection_dir = os.path.join(raw_dir, "Detection")
    json_candidates = (
        [
            f
            for f in os.listdir(detection_dir)
            if f.startswith("BSTD_") and f.endswith(".json")
        ]
        if os.path.isdir(detection_dir)
        else []
    )
    if not json_candidates:
        print(f"No BSTD_*.json in {detection_dir}  -- run --download first")
        return
    json_path = os.path.join(detection_dir, json_candidates[0])
    with open(json_path) as f:
        data = json.load(f)
    print(f"Top-level type: {type(data).__name__}")
    items = list(data.items())[:n_samples]
    for key, entry in items:
        image_name = entry.get("image_name", "MISSING")
        anns = entry.get("annotations") or {}
        ann_vals = list(anns.values()) if isinstance(anns, dict) else anns
        print(f"\n  key={key}  image_name={image_name}  annotations={len(ann_vals)}")
        if ann_vals:
            a0 = ann_vals[0]
            print(f"    ann[0] keys: {list(a0.keys())}")
            coords = a0.get("coordinates")
            lang = a0.get("script_language", a0.get("language", "MISSING"))
            print(
                f"    coordinates type={type(coords).__name__}  len={len(coords) if coords else 0}  first_point={coords[0] if coords else 'MISSING'}"
            )
            print(f"    language field: {lang}")
    print(
        "\nExpected: coordinates is a list of [x,y] points (>=3 for a valid polygon)."
    )
    print("If coordinates is MISSING or empty, check the JSON key name.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument(
        "--debug", action="store_true", help="inspect raw schema of first few samples"
    )
    args = p.parse_args()
    if args.download:
        download()
    if args.debug:
        _debug()
