"""TextOCR: ~29k natural images (from OpenImages) with arbitrary-shape polygon
word annotations -- signboards, product text, street scenes, varying shapes
and sizes. License: CC BY 4.0 (commercial use OK).

Fully automatic direct download, no login: images + annotations are hosted on
Meta's public file server (dl.fbaipublicfiles.com), same source used by
mmocr's own dataset-prep docs.

NOTE: the image zip is the whole dataset (~6.5GB) since it isn't split into
streamable shards -- it's a one-time download, but it's not itself "small".
Use --limit in build_dataset.py to cap how many of the *processed* (resized,
masked) images you keep once it's unpacked.
"""

import argparse
import json
import os
import zipfile

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from src.common import Sample

RAW_DIR = "data/raw/textocr"
IMAGES_ZIP_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
TRAIN_JSON_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json"
)
VAL_JSON_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json"
)


def _download_file(url: str, dest: str):
    if os.path.exists(dest):
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))


def download(raw_dir: str = RAW_DIR):
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "train_val_images.zip")
    _download_file(IMAGES_ZIP_URL, zip_path)
    img_root = os.path.join(raw_dir, "train_images")
    if not os.path.isdir(img_root):
        print("Extracting images (this is a big zip, may take a while)...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)

    _download_file(TRAIN_JSON_URL, os.path.join(raw_dir, "TextOCR_0.1_train.json"))
    _download_file(VAL_JSON_URL, os.path.join(raw_dir, "TextOCR_0.1_val.json"))
    print(f"TextOCR ready at {raw_dir}")


def _iter_split(raw_dir: str, split: str, json_name: str):
    json_path = os.path.join(raw_dir, json_name)
    if not os.path.exists(json_path):
        return
    with open(json_path) as f:
        data = json.load(f)
    imgs = data["imgs"]  # img_id -> {"file_name": ..., "width":..., "height":...}
    anns = data["anns"]  # ann_id -> {"image_id":..., "points": [x1,y1,...]}
    img_to_anns = data.get("imgToAnns", {})

    for img_id, img_info in imgs.items():
        file_name = img_info.get("file_name") or f"{img_id}.jpg"
        img_path = (
            os.path.join(raw_dir, file_name)
            if os.path.isabs(file_name) is False
            and os.path.exists(os.path.join(raw_dir, file_name))
            else os.path.join(raw_dir, "train_images", os.path.basename(file_name))
        )
        if not os.path.exists(img_path):
            # some releases store paths already relative to train_images/
            alt = os.path.join(raw_dir, file_name)
            img_path = alt if os.path.exists(alt) else img_path
        if not os.path.exists(img_path):
            continue

        ann_ids = img_to_anns.get(img_id, [])
        polys = []
        for aid in ann_ids:
            ann = anns.get(aid)
            if not ann:
                continue
            pts = ann.get("points")
            if pts and len(pts) >= 6:
                polys.append(np.array(pts, dtype=np.float32).reshape(-1, 2))

        yield Sample(
            sample_id=f"{split}_{img_id}",
            image=None,
            polygons=polys,
            image_loader=lambda p=img_path: Image.open(p).convert("RGB"),
        )


def iter_samples(raw_dir: str = RAW_DIR):
    yield from _iter_split(raw_dir, "train", "TextOCR_0.1_train.json")
    yield from _iter_split(raw_dir, "val", "TextOCR_0.1_val.json")


def _debug(raw_dir=RAW_DIR, n_samples: int = 3):
    import json, os

    json_path = os.path.join(raw_dir, "TextOCR_0.1_train.json")
    if not os.path.exists(json_path):
        print(f"Annotation file not found: {json_path}  -- run --download first")
        return
    with open(json_path) as f:
        data = json.load(f)
    print(f"Top-level keys: {list(data.keys())}")
    imgs = data.get("imgs", {})
    anns = data.get("anns", {})
    img_to_anns = data.get("imgToAnns", {})
    print(f"imgs={len(imgs)}  anns={len(anns)}  imgToAnns entries={len(img_to_anns)}")
    for j, (img_id, img_info) in enumerate(list(imgs.items())[:n_samples]):
        ann_ids = img_to_anns.get(img_id, [])
        sample_anns = [anns[aid] for aid in ann_ids if aid in anns]
        print(
            f"\n  img_id={img_id}  file_name={img_info.get('file_name')}  w={img_info.get('width')}  h={img_info.get('height')}"
        )
        print(f"    ann_ids={len(ann_ids)}  resolved_anns={len(sample_anns)}")
        if sample_anns:
            a0 = sample_anns[0]
            print(f"    ann[0] keys: {list(a0.keys())}")
            pts = a0.get("points")
            print(
                f"    ann[0] points type={type(pts).__name__}  len={len(pts) if pts else 0}  first4={pts[:4] if pts else 'MISSING'}"
            )
            print(f"    ann[0] utf8_string={repr(a0.get('utf8_string', ''))[:40]}")
        if j + 1 >= n_samples:
            break
    print("\nExpected: points is a flat list of numbers [x1,y1,x2,y2,...] len>=6.")
    print("If points is MISSING or empty, the key name may differ from 'points'.")


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
