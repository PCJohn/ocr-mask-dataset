"""DocLayNet (base/small mirror): real scanned/digital document pages --
financial reports, manuals, scientific articles, laws/regulations, patents,
government tenders. Manually annotated (not auto-derived like PubLayNet),
with genuinely useful LINE-LEVEL boxes via the `bboxes_line` field -- this is
actually a step up in granularity from PubLayNet/SynSlides, not just another
block-level dataset. License: CDLA-Permissive-1.0 (commercial use OK).

Uses the `pierreguillou/DocLayNet-base` mirror (~10% sample, ~691 train
images -- small on purpose). Source: https://huggingface.co/datasets/pierreguillou/DocLayNet-base

CAVEAT: this HF repo uses a custom dataset loading script (`trust_remote_code=True`
required), which is the same category of thing that broke on an earlier
PubLayNet mirror in this project (a stale dataset_infos.json that didn't
deserialize against a newer `datasets` version). I could not test this one
live from this environment. Run `python -m src.datasets.doclaynet --check`
first; if it crashes with a schema/dataclass error like the earlier PubLayNet
issue, try `pip install -U datasets huggingface_hub` first, and if that
doesn't fix it, this dataset isn't safe to rely on until the upstream repo
is patched -- fall back to `pierreguillou/DocLayNet-small` (even smaller) or
skip it.
"""
import argparse

import numpy as np

from src.common import Sample, box_to_polygon

HF_NAME = "pierreguillou/DocLayNet-base"

# DocLayNet's 11 category names. "Picture" is pure image content, excluded
# always. "Table" contains text but as a whole-table region -- excluded by
# default for the same reason as PubLayNet's table category; flip
# INCLUDE_TABLE to True if you'd rather keep it.
EXCLUDE_CATEGORIES = {"Picture"}
INCLUDE_TABLE = False
if not INCLUDE_TABLE:
    EXCLUDE_CATEGORIES.add("Table")


def iter_samples(split: str = "train"):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split, trust_remote_code=True)
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB")
        polys = []
        categories = ex.get("categories", [])
        line_boxes = ex.get("bboxes_line", [])
        for cat, box in zip(categories, line_boxes):
            if cat in EXCLUDE_CATEGORIES:
                continue
            if box and len(box) == 4:
                x0, y0, x1, y1 = box
                polys.append(box_to_polygon(x0, y0, x1, y1))
        # bboxes_line is already line-level, not a whole paragraph/element block --
        # so unlike publaynet/synslides this one is NOT marked coarse.
        yield Sample(sample_id=f"{split}_{i:05d}", image=img, polygons=polys, coarse=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--download", action="store_true", help="alias for --check")
    args = p.parse_args()
    if args.check or args.download:
        n = 0
        for s in iter_samples():
            n += 1
            if n == 1:
                print(f"First sample: {s.sample_id}, {len(s.polygons)} polygons, image size {s.image.size}")
            if n >= 5:
                break
        print(f"Loaded {n} DocLayNet samples OK.")
