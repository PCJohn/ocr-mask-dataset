"""PubLayNet: PubMed Central article page images w/ layout boxes+polygons.
License: CDLA-Permissive-1.0 (commercial use OK).
Loaded via HF `datasets` in STREAMING mode so we only pull as many examples as
we need (`--limit`), instead of the full ~96GB dataset.

Note: this dataset's HF loading script executes remote code (a `datasets`
loading script), which is why `trust_remote_code=True` is required below.
Only do this because we trust the `creative-graphic-design/PubLayNet` repo;
inspect https://huggingface.co/datasets/creative-graphic-design/PubLayNet
yourself before running if you want to be sure.
"""
import argparse

import numpy as np

from src.common import Sample, box_to_polygon

HF_NAME = "creative-graphic-design/PubLayNet"
DEFAULT_LIMIT = 1500  # keep the default pull small; override with --limit


def iter_samples(limit: int = DEFAULT_LIMIT, split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split, streaming=True, trust_remote_code=True)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        img = ex["image"].convert("RGB")
        polys = []
        for ann in ex.get("annotations", []):
            bbox = ann.get("bbox")  # [x, y, w, h]
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                polys.append(box_to_polygon(x, y, x + w, y + h))
        yield Sample(sample_id=f"{split}_{i:06d}", image=img, polygons=polys)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        n = sum(1 for _ in iter_samples(limit=min(args.limit, 5)))
        print(f"Streamed {n} PubLayNet samples OK.")
