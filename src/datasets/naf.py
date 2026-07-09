"""NAF Dataset: US National Archives form images, text/field boxes.
https://github.com/herobd/NAF_dataset -- CC BY 4.0 (commercial use OK w/ attribution).

KEY SCHEMA FACTS (from the official README):
  textBBs: pre-printed text labels on the form (type starts with "text...")
  fieldBBs: places where a response was written/typed. Each has:
    isBlank: 0=handwritten text, 1=handwriting, 2=print/stamp, 3=blank, 4=signature

  The previous version only pulled pre-printed text from textBBs (type="text*")
  and explicitly excluded field boxes -- missing ALL handwritten/typed fill-in
  content. Fixed: we now include fieldBBs where isBlank != 3 (i.e. not blank),
  which captures handwritten responses, typed/stamped text, and signatures.
  We exclude isBlank=3 (blank fields) and graphic/fieldRegion/fieldCol/fieldRow
  types that don't represent text at all.

  Note: "comment" type (any added writing not in a field) is also included,
  since it represents real written text that should be detected.
"""

import argparse
import glob
import json
import os
import subprocess
import tarfile

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm import tqdm

from src.common import Sample

RAW_DIR = "data/raw/naf"
IMAGES_URL = (
    "https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz"
)
REPO_URL = "https://github.com/herobd/NAF_dataset.git"

# textBBs types to include (all "text*" types = pre-printed text of various kinds)
TEXT_TYPE_PREFIXES = ("text",)

# fieldBBs types to exclude (non-text layout elements)
FIELD_NON_TEXT_TYPES = {
    "graphic",
    "fieldRegion",
    "fieldCol",
    "fieldRow",
    "fieldCircle",
    "fieldCheckBox",
}

# isBlank values that mean there IS content we want to detect
# 0=handwritten text, 1=handwriting, 2=print/stamp, 4=signature
# 3=blank -- skip
FIELD_BLANK_VALUE = 3


def download(raw_dir: str = RAW_DIR):
    os.makedirs(raw_dir, exist_ok=True)

    tgz_path = os.path.join(raw_dir, "labeled_images.tar.gz")
    img_dir = os.path.join(raw_dir, "imgs")
    if not os.path.exists(img_dir):
        if not os.path.exists(tgz_path):
            print(f"Downloading NAF images from {IMAGES_URL} ...")
            r = requests.get(IMAGES_URL, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tgz_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Extracting images...")
        with tarfile.open(tgz_path) as tf:
            tf.extractall(raw_dir)
        extracted = os.path.join(raw_dir, "labeled_images")
        if os.path.isdir(extracted) and not os.path.exists(img_dir):
            os.rename(extracted, img_dir)

    ann_repo_dir = os.path.join(raw_dir, "NAF_dataset_repo")
    groups_dir = os.path.join(raw_dir, "groups")
    if not os.path.exists(groups_dir):
        if not os.path.exists(ann_repo_dir):
            print("Cloning annotation repo...")
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, ann_repo_dir], check=True
            )
        os.rename(os.path.join(ann_repo_dir, "groups"), groups_dir)

    print(f"NAF ready at {raw_dir}")


def _polys_from_json(ann_path: str):
    with open(ann_path) as f:
        ann = json.load(f)
    polys = []

    # ── pre-printed text labels (textBBs) ───────────────────────────────────
    for item in ann.get("textBBs", []):
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        if not str(t).startswith(TEXT_TYPE_PREFIXES):
            continue
        pts = item.get("poly_points")
        if pts and len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.float32))

    # ── filled-in fields: handwritten, typed, stamped, signed (fieldBBs) ───
    for item in ann.get("fieldBBs", []):
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        if t in FIELD_NON_TEXT_TYPES:
            continue  # layout structure, not text
        is_blank = item.get("isBlank", FIELD_BLANK_VALUE)
        if is_blank == FIELD_BLANK_VALUE:
            continue  # genuinely empty field
        pts = item.get("poly_points")
        if pts and len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.float32))

    return polys


def iter_samples(raw_dir: str = RAW_DIR):
    groups_dir = os.path.join(raw_dir, "groups")
    img_dir = os.path.join(raw_dir, "imgs")
    if not os.path.isdir(groups_dir):
        raise FileNotFoundError(
            f"{groups_dir} not found. Run `python -m src.datasets.naf --download` first."
        )
    for ann_path in sorted(glob.glob(os.path.join(groups_dir, "*", "*.json"))):
        base = os.path.splitext(os.path.basename(ann_path))[0]
        candidates = glob.glob(os.path.join(img_dir, "*", base + ".jpg")) or glob.glob(
            os.path.join(img_dir, base + ".jpg")
        )
        if not candidates:
            continue
        img_path = candidates[0]
        polys = _polys_from_json(ann_path)
        yield Sample(
            sample_id=base,
            image=None,
            polygons=polys,
            image_loader=lambda p=img_path: ImageOps.exif_transpose(
                Image.open(p)
            ).convert("RGB"),
        )


def _debug(raw_dir=RAW_DIR, n_samples: int = 3):
    import glob, json, os

    groups_dir = os.path.join(raw_dir, "groups")
    img_dir = os.path.join(raw_dir, "imgs")
    if not os.path.isdir(groups_dir):
        print(f"Groups dir not found: {groups_dir}  -- run --download first")
        return
    ann_paths = sorted(glob.glob(os.path.join(groups_dir, "*", "*.json")))[:n_samples]
    for ann_path in ann_paths:
        with open(ann_path) as f:
            ann = json.load(f)
        text_bbs = ann.get("textBBs", [])
        field_bbs = ann.get("fieldBBs", [])
        print(f"\n  {os.path.basename(ann_path)}")
        print(f"    textBBs:  {len(text_bbs)} entries")
        if text_bbs:
            t = text_bbs[0]
            print(
                f"      [0] keys={list(t.keys())}  type={t.get('type')}  poly_points={t.get('poly_points', 'MISSING')[:1] if t.get('poly_points') else 'MISSING'}"
            )
        print(f"    fieldBBs: {len(field_bbs)} entries")
        if field_bbs:
            f0 = field_bbs[0]
            print(
                f"      [0] keys={list(f0.keys())}  type={f0.get('type')}  isBlank={f0.get('isBlank')}  poly_points present={bool(f0.get('poly_points'))}"
            )
        polys = _polys_from_json(ann_path)
        print(f"    → _polys_from_json yields {len(polys)} polygons")
    print("\nExpected: polygons > 0. isBlank values should be 0/1/2/3/4.")
    print("If poly_points=MISSING or type names differ, the schema has changed.")


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
