"""PubLayNet: PubMed Central article page images w/ layout boxes+polygons.
License: CDLA-Permissive-1.0 (commercial use OK).

Uses the `jordanparker6/publaynet` mirror, which is stored as plain Parquet
(auto-converted by HF, no custom loading script / trust_remote_code needed).
This avoids a known crash in some other mirrors caused by a stale
dataset_infos.json that doesn't deserialize cleanly against newer `datasets`
versions (TypeError: must be called with a dataclass type or instance).

Loaded in STREAMING mode so we only pull as many examples as we need
(`--limit`), instead of the full ~104GB dataset.
"""
import argparse

import numpy as np

from src.common import Sample

HF_NAME = "jordanparker6/publaynet"
DEFAULT_LIMIT = 1500  # keep the default pull small; override with --limit

# Standard PubLayNet category ids: 1=text, 2=title, 3=list, 4=table, 5=figure.
# "figure" is pure image content -- never text, always excluded.
# "table" contains text but as one whole-table box (not per-cell), which is
# arguably more "graphic region" than "text region" for a mask like ours --
# excluded by default; flip INCLUDE_TABLE to True if you'd rather keep it.
TEXT_CATEGORY_IDS = {1, 2, 3}  # text, title, list
INCLUDE_TABLE = False
if INCLUDE_TABLE:
    TEXT_CATEGORY_IDS.add(4)


def iter_samples(limit: int = DEFAULT_LIMIT, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        img = ex["image"].convert("RGB")
        polys = []
        for ann in ex.get("annotations", []):
            if ann.get("category_id") not in TEXT_CATEGORY_IDS:
                continue  # skip figure/table (or whatever's excluded) -- these are not text
            seg = ann.get("segmentation")
            if not seg:
                continue
            # segmentation is a list of polygons, each a flat [x1,y1,x2,y2,...] list
            for poly in seg:
                if len(poly) >= 6:  # need >= 3 points
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    polys.append(pts)
        # PubLayNet boxes are whole paragraph/title/list blocks, not per-line --
        # mark coarse so save_sample refines them down to actual text strokes
        # instead of flood-filling the whole block (see src/common.py).
        yield Sample(sample_id=f"{split}_{i:06d}", image=img, polygons=polys, coarse=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--check", action="store_true", help="stream a handful of samples to verify everything works")
    p.add_argument("--download", action="store_true",
                    help="alias for --check; this dataset is streamed on demand, there's nothing to pre-download")
    args = p.parse_args()
    if args.check or args.download:
        n = sum(1 for _ in iter_samples(limit=min(args.limit, 5)))
        print(f"Streamed {n} PubLayNet samples OK.")
