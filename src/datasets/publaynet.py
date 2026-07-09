"""PubLayNet: PubMed Central article page images w/ layout boxes+polygons.
License: CDLA-Permissive-1.0 (commercial use OK).

Uses the `jordanparker6/publaynet` mirror (plain Parquet, no custom loading
script, streamed so we only pull what we need).

KEY SCHEMA (from paper + HF mirror):
  annotations[i]["segmentation"]: List of block-outline polygons (flat
    [x1,y1,x2,y2,...]), derived from textline positions. BLOCK-level, not
    per-line. The paper says: "Segmentation is a regular polygon, consisting
    of only horizontal and vertical edges."
  annotations[i]["bbox"]: [x, y, w, h] COCO format.
  annotations[i]["category_id"]: 1=text, 2=title, 3=list, 4=table, 5=figure.

COARSE/OTSU REMOVED:
  Previous versions marked these samples coarse=True and applied Otsu
  threshold refinement inside each block box, intending to tighten the mask
  to actual text strokes. On clean black-on-white scientific papers (the vast
  majority of PubLayNet) this backfired badly: in narrow paragraph crops,
  text pixels can be the MINORITY, causing Otsu to invert and mark background
  as text, producing the jagged diagonal stripe artifacts seen in practice.
  PubLayNet's block polygons already tightly trace paragraph outlines with
  very little whitespace, so flood-filling the polygon IS a reasonable mask.
  We now use the polygons/bboxes directly (coarse=False, no Otsu).

FIGURE SKIPPING:
  Pages containing any figure annotation are skipped entirely, because text
  inside figures (axis labels, tick marks, legends) is never separately
  annotated in PubLayNet -- keeping such pages would silently miss that text.
  Set SKIP_SAMPLES_WITH_FIGURE=False to disable.

KNOWN UPSTREAM LIMITATION:
  Title-page miscellaneous info (running headers, some metadata) is missed
  by the auto-annotation algorithm -- upstream data limitation, not a bug.
"""

import argparse

import numpy as np

from src.common import Sample, box_to_polygon

HF_NAME = "jordanparker6/publaynet"
DEFAULT_LIMIT = 1500

TEXT_CATEGORY_IDS = {1, 2, 3}  # text, title, list
INCLUDE_TABLE = False
if INCLUDE_TABLE:
    TEXT_CATEGORY_IDS.add(4)

FIGURE_CATEGORY_ID = 5
SKIP_SAMPLES_WITH_FIGURE = True


def iter_samples(limit: int = DEFAULT_LIMIT, split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split=split, streaming=True)
    yielded = 0
    for ex in ds:
        if limit and yielded >= limit:
            break
        anns = ex.get("annotations", [])

        if SKIP_SAMPLES_WITH_FIGURE:
            if any(ann.get("category_id") == FIGURE_CATEGORY_ID for ann in anns):
                continue

        img = ex["image"].convert("RGB")
        polys = []
        for ann in anns:
            if ann.get("category_id") not in TEXT_CATEGORY_IDS:
                continue
            # Prefer polygon segmentation; fall back to bbox.
            seg = ann.get("segmentation")
            if seg:
                for poly in seg:
                    if len(poly) >= 6:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        polys.append(pts)
            else:
                bbox = ann.get("bbox")  # [x, y, w, h]
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        polys.append(box_to_polygon(x, y, x + w, y + h))

        # Block-level polygons used directly -- no Otsu refinement.
        yield Sample(
            sample_id=f"{split}_{yielded:06d}", image=img, polygons=polys, coarse=False
        )
        yielded += 1


def _debug(n_samples: int = 3):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split="train", streaming=True)
    for i, ex in enumerate(ds):
        anns = ex.get("annotations") or []
        cats = [a.get("category_id") for a in anns]
        has_fig = FIGURE_CATEGORY_ID in cats
        text_anns = [a for a in anns if a.get("category_id") in TEXT_CATEGORY_IDS]
        print(
            f"\n  sample {i}: image={ex['image'].size}  total_anns={len(anns)}  has_figure={has_fig}"
        )
        print(f"    category_ids present: {sorted(set(cats))}")
        print(f"    text/title/list anns: {len(text_anns)}")
        if anns:
            a0 = anns[0]
            print(f"    ann[0] keys: {list(a0.keys())}")
            print(f"    ann[0] bbox: {a0.get('bbox')}  (should be [x,y,w,h])")
            seg = a0.get("segmentation")
            print(
                f"    ann[0] segmentation present: {bool(seg)}  type={type(seg).__name__}  len={len(seg) if seg else 0}"
            )
            if seg:
                print(
                    f"      seg[0] type={type(seg[0]).__name__}  first5={list(seg[0])[:5] if hasattr(seg[0],'__iter__') else seg[0]}"
                )
        if i + 1 >= n_samples:
            break
    print("\nExpected: text_anns > 0 on most non-figure pages.")
    print(
        "samples with has_figure=True are SKIPPED during build. If most pages have figures,"
    )
    print("set SKIP_SAMPLES_WITH_FIGURE=False in publaynet.py.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--download",
        action="store_true",
        help="alias for --check; dataset is streamed on demand",
    )
    p.add_argument(
        "--debug", action="store_true", help="inspect raw schema of first few samples"
    )
    args = p.parse_args()
    if args.debug:
        _debug()
    elif args.check or args.download:
        n = sum(1 for _ in iter_samples(limit=min(args.limit, 5)))
        print(f"Streamed {n} PubLayNet samples OK.")
