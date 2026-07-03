# ocr-seg-dataset

Scripts to build a small, diverse, **commercially-usable** OCR text-segmentation
dataset (binary text-vs-background masks, later extendable to boxes) for
real-time OCR targeting: podcasts/lecture video frames, university lecture
recordings, receipts, screenshots of PDFs/webpages/tables, and scanned
documents.

## ⚠️ On "free for commercial use"

Every dataset below carries a license that **explicitly permits commercial
use** (CC BY 4.0 or CDLA-Permissive-1.0) — no "research/non-commercial only"
clauses. A lot of the classic OCR benchmarks people reach for first
(**FUNSD, SROIE, IAM, ICDAR RRC challenge sets, LectureVideoDB, GNHK**) are
explicitly **non-commercial/research-only** and have been dropped from this
repo for that reason — see "Excluded datasets" at the bottom for details.
CC BY / CDLA-Permissive still require attribution (keep the citation/notice),
and CC BY-SA (noted below) is share-alike — read what that means for you
before shipping a commercial model trained on it.

I'm not a lawyer and license terms/hosting can change — re-check the source
before you ship a commercial product, especially for anything CC BY-SA.

## Dataset checklist

| # | Dataset | Domain | Size | Annotation type | License | Auto-download? |
|---|---|---|---|---|---|---|
| 1 | [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | receipts | ~1–2 GB | word/line quads | **CC BY 4.0** | ✅ via `datasets` lib |
| 2 | [NAF Dataset](https://github.com/herobd/NAF_dataset) | forms, printed+some handwritten | few hundred MB | quad/polygon boxes | **CC BY 4.0** | ✅ git clone + release tar.gz |
| 3 | [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) | PDF/document pages (stand-in for "screenshots of PDFs", tables) | full set ~96GB, but we **stream** a capped sample | rect boxes + polygon segmentation | **CDLA-Permissive-1.0** | ✅ via HF `datasets` streaming, capped by `--limit` |
| 4 | [DocLayNet](https://github.com/DS4SD/DocLayNet) | diverse real documents: financial reports, manuals, patents, laws, tenders (great stand-in for scanned docs / PDF screenshots) | full ~31.8 GB, `-base` HF mirror is a 10% sample (~3GB) | rect boxes, 11 classes | **CDLA-Permissive-1.0** | ⚠️ manual (see below) — not yet wired into `build_dataset.py` |
| 5 | [TextOCR](https://huggingface.co/datasets/yunusserhat/TextOCR-Dataset) | photographed/scene text (stand-in for photographed receipts, signage) | ~6–8 GB full (images from OpenImages); stream+cap to go small | arbitrary-shape polygons | **CC BY 4.0** | ⚠️ manual / streaming, not yet wired in |
| 6 | [HierText](https://github.com/google-research-datasets/hiertext) | dense scene text + document-like images, word/line/paragraph level | ~a few GB | polygons, hierarchical | **CC BY-SA 4.0** ⚠️ share-alike, read the license | ⚠️ manual (AWS S3, no-sign-request), not yet wired in |

Implemented in this repo **right now** (`src/build_dataset.py --datasets ...`):
`cord`, `naf`, `publaynet`. Rows 4–6 are documented so you can add a processor
file the same way (`src/datasets/<name>.py` exposing `iter_samples()`); they
just aren't wired into the default run yet.

For **webpage/PDF screenshots** specifically: there is no small, license-clean,
purpose-built public dataset for this (most webpage-UI datasets are unlicensed
scrapes or research-only). Best path is self-generating this slice — render
webpages/PDFs you have the rights to (your own site, open-source docs,
Creative-Commons content) and extract ground-truth boxes straight from the
HTML DOM or the PDF text layer, which gives you perfect, free-to-use labels.
That's a good next script to add here (`src/datasets/selfgen_screenshots.py`,
not yet written).

## Unified output format

```
data/processed/
  <dataset_name>/
    images/<id>.jpg          # resized, longest side capped (default 1024px)
    masks/<id>.png           # binary mask (0/255), saved at mask_scale of the resized image (default 0.5x)
    meta.jsonl                # one JSON record per image (see below)
  manifest.jsonl              # concatenation of all datasets' meta.jsonl, +dataset field
  stats/
    per_dataset_stats.json
    combined_stats.json
    stats_report.md
```

Each line of `meta.jsonl`:
```json
{
  "id": "cord_train_00012",
  "dataset": "cord",
  "image_path": "images/cord_train_00012.jpg",
  "mask_path": "masks/cord_train_00012.png",
  "orig_width": 762, "orig_height": 1000,
  "resized_width": 780, "resized_height": 1024,
  "mask_width": 390, "mask_height": 512,
  "mask_scale": 0.5,
  "num_polygons": 42,
  "text_area_frac": 0.081,
  "num_contours": 39
}
```

At training time, load the mask and `cv2.resize`/`F.interpolate` it back up to
the image resolution (nearest-neighbor, since it's a binary mask). See
`src/torch_dataset.py` for a minimal example.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. CORD - fully automatic (uses HF `datasets`)
python -m src.datasets.cord --download

# 2. NAF - fully automatic (git clone + GitHub release tarball)
python -m src.datasets.naf --download

# 3. PubLayNet - fully automatic, streamed & capped (no manual download)
python -m src.datasets.publaynet --check   # optional smoke test

# Build the unified processed dataset + stats for everything
python -m src.build_dataset --datasets cord naf publaynet --limit 1500
```

Drop `--limit` (or raise it) once you've confirmed the pipeline works and want
more data — `publaynet` in particular will happily stream as much as you ask
for, so start small and dial it up.

### DocLayNet manual steps (optional, not yet wired in)
1. Use the **10%-sample HF mirror** to stay small:
   `pierreguillou/DocLayNet-base` (~3GB) instead of the full 31.8GB release.
2. `pip install datasets` then `load_dataset("pierreguillou/DocLayNet-base")`.
3. Write a `src/datasets/doclaynet.py` following `src/datasets/publaynet.py`
   as a template — same COCO-style bbox structure.

### TextOCR / HierText manual steps (optional, not yet wired in)
- TextOCR: `load_dataset("yunusserhat/TextOCR-Dataset", streaming=True)`,
  polygons are in the `points` field per annotation.
- HierText: no HF mirror; pull directly from the public, no-auth-required S3
  bucket per the repo's README (`aws s3 --no-sign-request cp
  s3://open-images-dataset/ocr/train.tgz .`), then parse the accompanying
  JSONL (word/line/paragraph polygons).

## Notes on masks vs boxes

For now every processor rasterizes word/line/box **polygons into filled
binary masks** (`cv2.fillPoly`). This is what `build_dataset.py` produces.
The raw polygon/box coordinates are not discarded — they're cheap to keep —
but for this first pass we only persist the rasterized mask + summary stats
(`num_polygons`), not the raw box list, to keep the unified format simple.
When you're ready for detection-style boxes, re-run the processors and extend
`meta.jsonl` with a `boxes` field (the per-dataset extraction code already
computes them, see each `src/datasets/*.py`).

## Excluded datasets (research/non-commercial license — do not use commercially)

- **FUNSD** — explicitly "non-commercial, research and educational purposes"
  per its own license page.
- **SROIE** — ICDAR2019 competition data, standard research-use terms.
- **IAM Handwriting DB** — registration terms restrict to non-commercial research.
- **ICDAR RRC challenge sets** (Born-Digital Images, Focused Scene Text) —
  competition data, research-use terms.
- **LectureVideoDB**, **GNHK** — no clear commercial grant; treat as research-only
  until/unless the maintainers say otherwise.

If you already have a legal review process that clears one of these (e.g. you
license it separately, or you're doing pure research/internal eval, not a
shipped commercial product), you can still add them back using the same
`iter_samples()` pattern — just don't fold them into anything commercial.
