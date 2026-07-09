"""scripts/clean_dataset.py
Bulk-clean the processed dataset by comparing each stored mask against
what an off-the-shelf OCR detector (EasyOCR by default) finds in the same
image.  Samples are deleted when the two masks disagree badly:

  BAD MASK (false-empty):  EasyOCR text coverage >> stored mask coverage
                            → mask missed real text; sample is noisy ground-truth
  INVERTED MASK:           stored mask coverage is very high but EasyOCR finds
                            almost nothing → likely an inversion artefact
                            (seen in DocLayNet table regions, PubLayNet coarse Otsu)

The metric is straightforward:
  ocr_frac  = fraction of image pixels EasyOCR marks as text
  mask_frac = fraction stored mask marks as text
  Δ         = ocr_frac - mask_frac

A sample is DELETED if:
  Δ > --max-miss-delta  (mask missed too much OCR text)  [default 0.35]
  OR
  Δ < --max-extra-delta (mask has far more text than OCR found) [default -0.40]

Both thresholds are intentionally conservative -- we'd rather keep a few
noisy samples than delete good ones.

Usage:
    # dry run -- print what WOULD be deleted without deleting anything
    python -m scripts.clean_dataset --dry-run

    # delete bad samples across all datasets, GPU off
    python -m scripts.clean_dataset --datasets cord naf publaynet

    # custom thresholds, GPU on
    python -m scripts.clean_dataset --max-miss-delta 0.25 --max-extra-delta -0.30 --gpu

    # restrict to a specific dataset
    python -m scripts.clean_dataset --datasets doclaynet
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import read_jsonl
from scripts.ocr_backend import get_reader, boxes_from_image, polys_to_mask


def load_mask(mask_path: str, target_wh) -> np.ndarray:
    """Load stored (downscaled) mask and upscale to target_wh=(W,H)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((target_wh[1], target_wh[0]), dtype=np.uint8)
    return cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST)


def mask_frac(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / float(mask.size) if mask.size else 0.0


def check_sample(
    rec: dict,
    processed_dir: str,
    reader,
    backend: str,
    min_conf: float,
    max_miss: float,
    max_extra: float,
    dry_run: bool,
) -> str:
    """Returns 'kept', 'deleted-miss', 'deleted-extra', or 'error'."""
    ds = rec["dataset"]
    base = os.path.join(processed_dir, ds)
    img_path = os.path.join(base, rec["image_path"])
    mask_path = os.path.join(base, rec["mask_path"])

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return "error"

    w, h = img.size
    stored_mask = load_mask(mask_path, (w, h))
    stored_frac = mask_frac(stored_mask)

    ocr_polys = boxes_from_image(reader, img, backend=backend, min_confidence=min_conf)
    ocr_mask = polys_to_mask(ocr_polys, (w, h))
    ocr_frac = mask_frac(ocr_mask)

    delta = ocr_frac - stored_frac  # positive = OCR found more than stored mask

    reason = None
    if delta > max_miss:
        reason = "deleted-miss"  # stored mask missed too much
    elif delta < max_extra:
        reason = "deleted-extra"  # stored mask has far more text than OCR found (likely inversion)

    if reason and not dry_run:
        for path in (img_path, mask_path):
            try:
                os.remove(path)
            except OSError:
                pass

    return reason or "kept"


def update_manifests(processed_dir: str, datasets: list[str]):
    """Rewrite manifest.jsonl and per-dataset meta.jsonl to remove deleted files."""
    manifest_path = os.path.join(processed_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return

    all_records = read_jsonl(manifest_path)
    surviving = []
    for rec in all_records:
        ds = rec["dataset"]
        img_path = os.path.join(processed_dir, ds, rec["image_path"])
        if os.path.exists(img_path):
            surviving.append(rec)

    with open(manifest_path, "w") as f:
        for r in surviving:
            f.write(json.dumps(r) + "\n")

    # per-dataset meta
    for ds in datasets:
        meta_path = os.path.join(processed_dir, ds, "meta.jsonl")
        if not os.path.exists(meta_path):
            continue
        recs = read_jsonl(meta_path)
        kept = [
            r
            for r in recs
            if os.path.exists(os.path.join(processed_dir, ds, r["image_path"]))
        ]
        with open(meta_path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="restrict to these dataset names (default: all in manifest)",
    )
    ap.add_argument("--backend", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="minimum OCR confidence to count a word as found (default 0.3)",
    )
    ap.add_argument(
        "--max-miss-delta",
        type=float,
        default=0.35,
        help="delete if OCR coverage - stored mask coverage > this (default 0.35)",
    )
    ap.add_argument(
        "--max-extra-delta",
        type=float,
        default=-0.40,
        help="delete if OCR coverage - stored mask coverage < this (default -0.40)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be deleted without actually deleting",
    )
    ap.add_argument(
        "--report",
        default="data/processed/clean_report.jsonl",
        help="path to write per-sample decisions",
    )
    args = ap.parse_args()

    manifest_path = os.path.join(args.processed, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(
            f"No manifest.jsonl found at {manifest_path}. Run build_dataset.py first."
        )
        sys.exit(1)

    all_records = read_jsonl(manifest_path)
    if args.datasets:
        records = [r for r in all_records if r["dataset"] in args.datasets]
    else:
        records = all_records

    print(
        f"Initialising {args.backend} (gpu={args.gpu}) -- this may download models on first run..."
    )
    reader = get_reader(backend=args.backend, gpu=args.gpu)

    counts = {"kept": 0, "deleted-miss": 0, "deleted-extra": 0, "error": 0}
    report_rows = []

    for rec in tqdm(records, desc="cleaning"):
        result = check_sample(
            rec,
            args.processed,
            reader,
            args.backend,
            min_conf=args.min_conf,
            max_miss=args.max_miss_delta,
            max_extra=args.max_extra_delta,
            dry_run=args.dry_run,
        )
        counts[result] = counts.get(result, 0) + 1
        report_rows.append(
            {"id": rec["id"], "dataset": rec["dataset"], "result": result}
        )

    if not args.dry_run:
        datasets = args.datasets or list({r["dataset"] for r in records})
        update_manifests(args.processed, datasets)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        for row in report_rows:
            f.write(json.dumps(row) + "\n")

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Results:")
    print(f"  kept:             {counts['kept']}")
    print(f"  deleted (missed): {counts['deleted-miss']}")
    print(f"  deleted (extra):  {counts['deleted-extra']}")
    print(f"  errors:           {counts['error']}")
    print(f"  report:           {args.report}")


if __name__ == "__main__":
    main()
