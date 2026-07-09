"""DocLayNet (base mirror): real scanned/digital document pages.
License: CDLA-Permissive-1.0 (commercial use OK).
https://huggingface.co/datasets/pierreguillou/DocLayNet-base

KEY SCHEMA (verified from HF dataset viewer + Pierre Guillou's notebooks):
  `categories`  : List[int]  -- one category id per BLOCK.
  `bboxes_block`: List[box]  -- one [x,y,w,h] bounding box per block.
  `bboxes_line` : List[List[box]] -- one LIST of [x,y,w,h] boxes per block,
                  i.e. bboxes_line[i] is the list of line-level boxes for
                  the i-th block (same index as categories[i]).

  Previous bug: the code treated bboxes_line[i] as a single [x,y,w,h] box
  instead of a list of boxes, causing most lines to be silently skipped and
  producing near-empty or garbage masks on pages with formulas, figures, etc.

  Category ids (integers, NOT strings):
    1=Caption, 2=Footnote, 3=Formula, 4=List-item, 5=Page-footer,
    6=Page-header, 7=Picture, 8=Section-header, 9=Table, 10=Text, 11=Title

  All coordinates are in [x, y, w, h] COCO format at coco_width×coco_height
  scale (1025×1025 px).

  We use bboxes_line (line-level) for fine-grained masks. Falls back to
  bboxes_block (block-level) when a block has no line boxes.
  NOT marked coarse -- line boxes don't need Otsu refinement.
"""

import argparse

from src.common import Sample, box_to_polygon

HF_NAME = "pierreguillou/DocLayNet-base"

# Category ids to EXCLUDE from the text mask.
# 7=Picture: pure image content, always excluded.
# 9=Table: whole-table region, no per-cell breakdown; excluded by default.
EXCLUDE_CATEGORY_IDS = {7}
INCLUDE_TABLE = False
if not INCLUDE_TABLE:
    EXCLUDE_CATEGORY_IDS.add(9)


def _box_to_poly(box):
    """[x, y, w, h] -> polygon. Returns None if degenerate."""
    if not box:
        return None
    if not hasattr(box, "__len__") or len(box) != 4:
        return None
    try:
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return box_to_polygon(x, y, x + w, y + h)


def _parse_line_boxes(raw):
    """bboxes_line[i] can be shaped either as:
      - a list of [x,y,w,h] sub-lists  → [[x,y,w,h], [x,y,w,h], ...]
      - a flat single box               → [x, y, w, h]  (ints, not nested)
    Detect by checking whether the first element is itself a list/sequence.
    """
    if not raw:
        return []
    first = raw[0]
    if hasattr(first, "__len__"):
        # nested: each element is one [x,y,w,h] box
        return raw
    else:
        # flat: the whole thing is one [x,y,w,h] box
        return [raw]


def iter_samples(split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split=split, trust_remote_code=True)
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB")

        categories = ex.get("categories") or []
        bboxes_line = ex.get("bboxes_line") or []
        bboxes_block = ex.get("bboxes_block") or []

        polys = []
        for idx, cat_id in enumerate(categories):
            if cat_id in EXCLUDE_CATEGORY_IDS:
                continue

            # bboxes_line[idx] may be nested [[x,y,w,h],...] or flat [x,y,w,h]
            line_boxes = _parse_line_boxes(
                bboxes_line[idx] if idx < len(bboxes_line) else []
            )

            if line_boxes:
                for box in line_boxes:
                    p = _box_to_poly(box)
                    if p is not None:
                        polys.append(p)
            else:
                # fallback: use the block box when line boxes are absent
                block_box = bboxes_block[idx] if idx < len(bboxes_block) else None
                p = _box_to_poly(block_box)
                if p is not None:
                    polys.append(p)

        # Line-level boxes: NOT coarse, no Otsu refinement needed.
        yield Sample(
            sample_id=f"{split}_{i:05d}", image=img, polygons=polys, coarse=False
        )


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
                print(
                    f"First sample: {s.sample_id}, {len(s.polygons)} polygons, image {s.image.size}"
                )
                if s.polygons:
                    print(f"  First polygon: {s.polygons[0]}")
            if n >= 5:
                break
        print(f"Loaded {n} DocLayNet samples OK.")
