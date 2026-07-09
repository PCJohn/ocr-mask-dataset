"""scripts/clean_dataset.py
Bulk-clean the processed dataset by comparing each stored mask against
what an off-the-shelf OCR detector (EasyOCR by default) finds in the same
image.  Samples are deleted when the two masks disagree badly.

MEMORY STRATEGY:
  - EasyOCR readers are loaded ONE SCRIPT-GROUP AT A TIME and explicitly
    deleted + GPU cache flushed between groups, so peak RAM/VRAM is one
    reader's worth, not all 8 simultaneously.
  - Each PIL image is explicitly closed after processing.
  - Results from pass 1 (decision per sample) are stored as a lightweight
    list of strings; deletion and manifest rewriting happen in pass 2.
  - A checkpoint file is written after each script group so a crash mid-run
    can be resumed with --resume.

Decision metric:
  ocr_frac  = fraction of image pixels EasyOCR marks as text
  mask_frac = fraction stored mask marks as text
  delta     = ocr_frac - mask_frac

A sample is flagged MISS  if delta >  --max-miss-delta   (mask missed real text)
A sample is flagged EXTRA if delta <  --max-extra-delta  (mask has far more text than OCR)
A sample that any group flags MISS or EXTRA is deleted.

Usage:
    python -m scripts.clean_dataset --dry-run
    python -m scripts.clean_dataset --gpu
    python -m scripts.clean_dataset --gpu --datasets cord naf
    python -m scripts.clean_dataset --gpu --resume          # continue after crash
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import read_jsonl
from scripts.ocr_backend import EASYOCR_SCRIPT_GROUPS, polys_to_mask

# ── helpers ───────────────────────────────────────────────────────────────────


def _load_mask_upscaled(mask_path: str, target_wh) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((target_wh[1], target_wh[0]), dtype=np.uint8)
    return cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST)


def _frac(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / float(mask.size) if mask.size else 0.0


def _flush_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


def _load_reader(group: list, gpu: bool):
    import easyocr

    return easyocr.Reader(group, gpu=gpu, verbose=False, download_enabled=True)


def _unload_reader(reader) -> None:
    try:
        del reader.detector
    except Exception:
        pass
    try:
        del reader.recognizer
    except Exception:
        pass
    del reader
    _flush_gpu()


# ── OCR one image with one reader, return delta ───────────────────────────────


def _ocr_delta(
    img_path: str,
    mask_path: str,
    processed_dir: str,
    rec: dict,
    reader,
    min_conf: float,
) -> Optional[float]:
    """Return (ocr_frac - stored_mask_frac), or None on error."""
    ds = rec["dataset"]
    base = os.path.join(processed_dir, ds)
    full_img_path = os.path.join(base, rec["image_path"])
    full_mask_path = os.path.join(base, rec["mask_path"])

    try:
        img = Image.open(full_img_path).convert("RGB")
        w, h = img.size
        arr = np.array(img)
        img.close()
        del img
    except Exception:
        return None

    try:
        stored_mask = _load_mask_upscaled(full_mask_path, (w, h))
        stored_frac = _frac(stored_mask)
        del stored_mask

        results = reader.readtext(arr, detail=1, paragraph=False)
        del arr

        polys = []
        for bbox, text, conf in results:
            if conf >= min_conf:
                polys.append(np.array(bbox, dtype=np.float32))

        ocr_mask = polys_to_mask(polys, (w, h))
        ocr_frac = _frac(ocr_mask)
        del ocr_mask, polys

        return ocr_frac - stored_frac
    except Exception as e:
        return None


# ── checkpoint helpers ────────────────────────────────────────────────────────


def _checkpoint_path(report_path: str) -> str:
    return report_path.replace(".jsonl", "_checkpoint.json")


def _load_checkpoint(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}  # sample_id -> "miss" | "extra" | "ok"


def _save_checkpoint(ckpt_path: str, decisions: dict):
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump(decisions, f)


# ── manifest rewriting ────────────────────────────────────────────────────────


def _update_manifests(processed_dir: str, datasets: list[str]):
    manifest_path = os.path.join(processed_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return
    all_records = read_jsonl(manifest_path)
    surviving = [
        r
        for r in all_records
        if os.path.exists(os.path.join(processed_dir, r["dataset"], r["image_path"]))
    ]
    with open(manifest_path, "w") as f:
        for r in surviving:
            f.write(json.dumps(r) + "\n")
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


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument(
        "--backend",
        default="easyocr",
        choices=["easyocr"],  # paddleocr handles memory differently
        help="only easyocr supported here (paddleocr: use --backend paddleocr at your own risk)",
    )
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument(
        "--max-miss-delta",
        type=float,
        default=0.35,
        help="flag if OCR coverage - mask coverage > this  (default 0.35)",
    )
    ap.add_argument(
        "--max-extra-delta",
        type=float,
        default=-0.40,
        help="flag if OCR coverage - mask coverage < this  (default -0.40)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load checkpoint and skip already-processed samples",
    )
    ap.add_argument("--report", default="data/processed/clean_report.jsonl")
    args = ap.parse_args()

    manifest_path = os.path.join(args.processed, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        print(f"No manifest.jsonl at {manifest_path}. Run build_dataset.py first.")
        sys.exit(1)

    all_records = read_jsonl(manifest_path)
    records = [
        r for r in all_records if not args.datasets or r["dataset"] in args.datasets
    ]
    print(
        f"{len(records)} samples to process across "
        f"{len(set(r['dataset'] for r in records))} dataset(s)"
    )

    ckpt_path = _checkpoint_path(args.report)
    decisions: dict[str, str] = _load_checkpoint(ckpt_path) if args.resume else {}
    if decisions:
        print(f"Resuming: {len(decisions)} samples already in checkpoint")

    # Pass 0: corruption check -- fast, no OCR, no GPU.
    # Catches truncated files, bad headers, unreadable masks before we
    # waste time loading any OCR model.
    print("\nPass 0: checking for corrupted / unreadable files...")
    corrupt_count = 0
    for rec in tqdm(records, desc="corruption check"):
        if rec["id"] in decisions:
            continue
        ds = rec["dataset"]
        base = os.path.join(args.processed, ds)
        img_path = os.path.join(base, rec["image_path"])
        mask_path = os.path.join(base, rec["mask_path"])

        reason = None

        # 1. files must exist
        if not os.path.exists(img_path):
            reason = "corrupt-missing-image"
        elif not os.path.exists(mask_path):
            reason = "corrupt-missing-mask"
        else:
            # 2. image must be openable and have valid pixel data
            try:
                with Image.open(img_path) as im:
                    im.verify()  # checks header / EXIF without decoding pixels
            except Exception:
                reason = "corrupt-bad-image-header"

            if reason is None:
                try:
                    # verify() resets the file; need a fresh open to decode pixels
                    with Image.open(img_path) as im:
                        im.load()  # fully decode -- catches truncated data
                except Exception:
                    reason = "corrupt-truncated-image"

            # 3. mask must be a readable single-channel PNG
            if reason is None:
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        reason = "corrupt-bad-mask"
                    elif mask.size == 0:
                        reason = "corrupt-empty-mask"
                except Exception:
                    reason = "corrupt-bad-mask"

        if reason:
            decisions[rec["id"]] = reason
            corrupt_count += 1
            if not args.dry_run:
                for path in (img_path, mask_path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    print(
        f"  Found {corrupt_count} corrupt / unreadable samples"
        + (" (not deleted, dry-run)" if args.dry_run else " -- deleted")
    )

    # Pass 1: run each EasyOCR script group one at a time
    # For each group, scan every sample and accumulate a "worst" verdict per sample.
    for g_idx, group in enumerate(EASYOCR_SCRIPT_GROUPS):
        remaining = [r for r in records if r["id"] not in decisions]
        if not remaining:
            break

        print(
            f"\n[{g_idx+1}/{len(EASYOCR_SCRIPT_GROUPS)}] "
            f"Loading reader for group: {group}"
        )
        try:
            reader = _load_reader(group, args.gpu)
        except Exception as e:
            print(f"  Failed to load: {e}  -- skipping group")
            continue

        for rec in tqdm(remaining, desc=f"group {g_idx+1}"):
            if rec["id"] in decisions:
                continue  # already flagged by an earlier group

            delta = _ocr_delta(
                rec["image_path"],
                rec["mask_path"],
                args.processed,
                rec,
                reader,
                args.min_conf,
            )
            if delta is None:
                decisions[rec["id"]] = "error"
            elif delta > args.max_miss_delta:
                decisions[rec["id"]] = "miss"
            elif delta < args.max_extra_delta:
                decisions[rec["id"]] = "extra"
            # else: don't write "ok" yet -- another group might flag it

        # explicitly unload reader and flush GPU before next group
        print(f"  Unloading reader group {g_idx+1} and flushing memory...")
        _unload_reader(reader)
        _save_checkpoint(ckpt_path, decisions)
        print(f"  Checkpoint saved ({len(decisions)} decisions so far)")

    # fill any remaining undecided samples as "ok"
    for rec in records:
        if rec["id"] not in decisions:
            decisions[rec["id"]] = "ok"

    # Pass 2: delete flagged samples
    counts: dict[str, int] = {}
    report_rows = []

    for rec in records:
        verdict = decisions.get(rec["id"], "ok")
        counts[verdict] = counts.get(verdict, 0) + 1
        report_rows.append(
            {"id": rec["id"], "dataset": rec["dataset"], "result": verdict}
        )

        is_corrupt = verdict.startswith("corrupt-")
        is_bad_mask = verdict in ("miss", "extra")

        if (is_bad_mask or is_corrupt) and not args.dry_run:
            # corruption pass already deleted files; only delete here for OCR verdicts
            if is_bad_mask:
                ds = rec["dataset"]
                base = os.path.join(args.processed, ds)
                for field in ("image_path", "mask_path"):
                    try:
                        os.remove(os.path.join(base, rec[field]))
                    except OSError:
                        pass

    if not args.dry_run:
        datasets = args.datasets or list({r["dataset"] for r in records})
        _update_manifests(args.processed, datasets)
        # clean up checkpoint now that we're done
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        for row in report_rows:
            f.write(json.dumps(row) + "\n")

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}Results:")
    print(f"  kept:                    {counts.get('ok', 0)}")
    print(f"  deleted (missed text):   {counts.get('miss', 0)}")
    print(f"  deleted (extra/inverted):{counts.get('extra', 0)}")
    corrupt_total = sum(v for k, v in counts.items() if k.startswith("corrupt-"))
    if corrupt_total:
        print(f"  deleted (corrupt):       {corrupt_total}")
        for k, v in sorted(counts.items()):
            if k.startswith("corrupt-") and v:
                print(f"    {k}: {v}")
    print(f"  errors (OCR failed):     {counts.get('error', 0)}")
    print(f"  report:                  {args.report}")


if __name__ == "__main__":
    main()
