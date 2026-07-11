"""scripts/clean_dataset.py
Bulk-clean the processed dataset using EasyOCR's CRAFT text detector
(detector-only mode — no recogniser, no per-script model groups, works
for all languages with a single reader).

IoU(stored_mask, detector_mask) < --min-iou → sample deleted.
One threshold, one reader, one pass.

Usage:
    python -m scripts.clean_dataset --dry-run
    python -m scripts.clean_dataset --gpu
    python -m scripts.clean_dataset --gpu --min-iou 0.10
    python -m scripts.clean_dataset --gpu --datasets cord naf
    python -m scripts.clean_dataset --gpu --resume
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import read_jsonl
from scripts.ocr_backend import (
    get_reader,
    _detect_raw,
    polys_to_mask,
    mask_iou,
    _resize_for_ocr,
    MAX_OCR_SIDE,
)


def _load_mask_at_ocr_size(mask_path: str, ocr_wh: tuple) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((ocr_wh[1], ocr_wh[0]), dtype=np.uint8)
    return cv2.resize(mask, ocr_wh, interpolation=cv2.INTER_NEAREST)


def _flush_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


# ── Pass 0: corruption (CPU, no OCR) ─────────────────────────────────────────


def _corruption_check(records, processed_dir, decisions, dry_run):
    print("\nPass 0: corruption check (CPU, no OCR)...")
    n = 0
    for rec in tqdm(records, desc="corruption check"):
        if rec["id"] in decisions:
            continue
        ds = rec["dataset"]
        base = os.path.join(processed_dir, ds)
        img_path = os.path.join(base, rec["image_path"])
        mask_path = os.path.join(base, rec["mask_path"])

        reason = None
        if not os.path.exists(img_path):
            reason = "corrupt-missing-image"
        elif not os.path.exists(mask_path):
            reason = "corrupt-missing-mask"
        else:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                reason = "corrupt-bad-header"
            if not reason:
                try:
                    with Image.open(img_path) as im:
                        im.load()
                except Exception:
                    reason = "corrupt-truncated"
            if not reason:
                try:
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None or m.size == 0:
                        reason = "corrupt-bad-mask"
                except Exception:
                    reason = "corrupt-bad-mask"

        if reason:
            decisions[rec["id"]] = reason
            n += 1
            if not dry_run:
                for path in (img_path, mask_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        print(f"  Warning: could not delete {path}: {e}")

    suffix = (
        " (dry-run, files NOT deleted)"
        if dry_run
        else f" — image + mask files deleted from data/processed/"
    )
    print(f"  {n} corrupt samples{suffix}")
    return n


# ── Pass 1: IoU scoring with detector-only CRAFT ─────────────────────────────


def _score_sample(
    rec, processed_dir, reader, text_threshold, low_text, link_threshold
) -> float | None:
    ds = rec["dataset"]
    base = os.path.join(processed_dir, ds)
    img_path = os.path.join(base, rec["image_path"])
    mask_path = os.path.join(base, rec["mask_path"])

    try:
        img = Image.open(img_path).convert("RGB")
        arr, scale = _resize_for_ocr(img)
        ocr_w, ocr_h = arr.shape[1], arr.shape[0]
        img.close()
        del img

        stored_mask = _load_mask_at_ocr_size(mask_path, (ocr_w, ocr_h))

        polys, _ = _detect_raw(
            arr, scale, reader, text_threshold, low_text, link_threshold
        )

        det_mask = polys_to_mask(polys, (ocr_w, ocr_h))
        iou = mask_iou(stored_mask, det_mask)
        del arr, stored_mask, det_mask, polys
        return iou
    except Exception as e:
        return None


# ── checkpoint / manifest ────────────────────────────────────────────────────


def _ckpt_path(report):
    return report.replace(".jsonl", "_checkpoint.json")


def _load_ckpt(p):
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_ckpt(p, d):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f)


def _update_manifests(processed_dir, datasets):
    mp = os.path.join(processed_dir, "manifest.jsonl")
    if not os.path.exists(mp):
        return
    alive = [
        r
        for r in read_jsonl(mp)
        if os.path.exists(os.path.join(processed_dir, r["dataset"], r["image_path"]))
    ]
    with open(mp, "w") as f:
        for r in alive:
            f.write(json.dumps(r) + "\n")
    for ds in datasets:
        dp = os.path.join(processed_dir, ds, "meta.jsonl")
        if not os.path.exists(dp):
            continue
        kept = [
            r
            for r in read_jsonl(dp)
            if os.path.exists(os.path.join(processed_dir, ds, r["image_path"]))
        ]
        with open(dp, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--min-iou",
        type=float,
        default=0.15,
        help="delete if IoU(stored_mask, detector_mask) < this. "
        "Start with 0.10 and look at clean_report.jsonl to calibrate. "
        "(default 0.15)",
    )
    ap.add_argument(
        "--text-threshold",
        type=float,
        default=0.7,
        help="CRAFT text confidence threshold (default 0.7). "
        "Lower → more detections / more recall.",
    )
    ap.add_argument(
        "--low-text",
        type=float,
        default=0.4,
        help="CRAFT low-bound score (default 0.4)",
    )
    ap.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        help="CRAFT link threshold (default 0.4)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", default="data/processed/clean_report.jsonl")
    args = ap.parse_args()

    mp = os.path.join(args.processed, "manifest.jsonl")
    if not os.path.exists(mp):
        print(f"No manifest.jsonl at {mp}")
        sys.exit(1)

    all_rec = read_jsonl(mp)
    records = [r for r in all_rec if not args.datasets or r["dataset"] in args.datasets]
    print(
        f"{len(records)} samples across "
        f"{len(set(r['dataset'] for r in records))} dataset(s)"
    )

    ckpt = _ckpt_path(args.report)
    decisions = _load_ckpt(ckpt) if args.resume else {}
    if decisions:
        print(f"Resuming: {len(decisions)} already decided")

    # Pass 0: corruption
    _corruption_check(records, args.processed, decisions, args.dry_run)
    _save_ckpt(ckpt, decisions)

    # Pass 1: detector-only IoU scoring (single reader, all scripts)
    undecided = [r for r in records if r["id"] not in decisions]
    if undecided:
        print(
            f"\nPass 1: IoU scoring with CRAFT detector ({len(undecided)} samples)..."
        )
        reader = get_reader(gpu=args.gpu)
        low_iou = 0
        for rec in tqdm(undecided, desc="IoU scoring"):
            iou = _score_sample(
                rec,
                args.processed,
                reader,
                args.text_threshold,
                args.low_text,
                args.link_threshold,
            )
            if iou is None:
                decisions[rec["id"]] = "error"
            elif iou < args.min_iou:
                decisions[rec["id"]] = f"low-iou-{iou:.3f}"
                low_iou += 1
            else:
                decisions[rec["id"]] = "ok"

        del reader
        _flush_gpu()
        _save_ckpt(ckpt, decisions)
        print(f"  {low_iou} samples below IoU threshold {args.min_iou}")

    # fill any remaining
    for rec in records:
        decisions.setdefault(rec["id"], "ok")

    # Pass 2: delete low-iou/error files and write report
    # Note: corrupt files were already deleted during Pass 0.
    # This pass handles low-iou and error verdicts only.
    counts: dict[str, int] = {}
    report_rows = []
    n_deleted_files = 0

    for rec in records:
        v = decisions.get(rec["id"], "ok")
        bucket = (
            "ok"
            if v == "ok"
            else (
                "corrupt"
                if v.startswith("corrupt")
                else "low-iou" if v.startswith("low-iou") else v
            )
        )
        counts[bucket] = counts.get(bucket, 0) + 1
        report_rows.append({"id": rec["id"], "dataset": rec["dataset"], "result": v})

        # Only delete here for non-corrupt verdicts (corrupt files deleted in Pass 0)
        if bucket in ("low-iou", "error") and not args.dry_run:
            ds = rec["dataset"]
            base = os.path.join(args.processed, ds)
            for fld in ("image_path", "mask_path"):
                fpath = os.path.join(base, rec[fld])
                try:
                    os.remove(fpath)
                    n_deleted_files += 1
                except FileNotFoundError:
                    pass  # already gone (shouldn't happen for low-iou, but be safe)
                except OSError as e:
                    print(f"  Warning: could not delete {fpath}: {e}")

    if not args.dry_run:
        print(
            f"\nDeleted {n_deleted_files} files from data/processed/ "
            f"(images + masks for low-iou and error samples)."
        )
        print("Raw source data in data/raw/ is untouched.")

    if not args.dry_run:
        dsets = args.datasets or list({r["dataset"] for r in records})
        _update_manifests(args.processed, dsets)
        try:
            os.remove(ckpt)
        except OSError:
            pass

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        for row in report_rows:
            f.write(json.dumps(row) + "\n")

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Results:")
    print(f"  kept:    {counts.get('ok', 0)}")
    print(f"  low-iou: {counts.get('low-iou', 0)}")
    print(f"  corrupt: {counts.get('corrupt', 0)}")
    print(f"  error:   {counts.get('error', 0)}")
    print(f"  report:  {args.report}")


if __name__ == "__main__":
    main()
