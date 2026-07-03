# ocr-seg-dataset

Scripts to build a small, diverse, **license-clean** OCR text-segmentation dataset
(binary text-vs-background masks, later extendable to boxes) for real-time OCR
targeting: podcasts/lecture video frames, university lecture recordings, receipts,
screenshots of PDFs/webpages/tables, and scanned documents.

All component datasets below are free for research use with no paywall/registration
fee. A couple require a free account (Kaggle / ICDAR site) to click "download" —
this is normal and not a licensing concern, just noted per-dataset below.

## Dataset checklist

| # | Dataset | Domain | Size | Annotation type | Auto-download? | License |
|---|---|---|---|---|---|---|
| 1 | [FUNSD](https://guillaumejaume.github.io/FUNSD/) | scanned forms / documents | ~16 MB | word boxes (rect) | ✅ direct zip | Research use (public) |
| 2 | [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | receipts | ~1–2 GB | word/line quads | ✅ via `datasets` lib | CC BY 4.0 |
| 3 | [SROIE](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) | scanned receipts | <1 GB | quad boxes | ⚠️ manual (Kaggle login) | Research use (ICDAR2019 competition) |
| 4 | [ICDAR13 Born-Digital Images](https://rrc.cvc.uab.es/?ch=1) | screenshots / born-digital text | ~40 MB | quad boxes + **pixel masks** | ⚠️ manual (free RRC account) | Research use |
| 5 | [ICDAR13 Focused Scene Text](https://rrc.cvc.uab.es/?ch=2) | signage/photographed text (stand-in for "in the wild" prints, e.g. receipts photographed) | ~250 MB | quad boxes + **pixel masks** | ⚠️ manual (free RRC account) | Research use |
| 6 | [NAF Dataset](https://github.com/herobd/NAF_dataset) | forms, printed+handwritten | few hundred MB | quad boxes | ✅ git clone | CC BY 4.0 |
| 7 | [LectureVideoDB](https://ieeexplore.ieee.org/document/8563242) | lecture video frames / slides | ~2.3 GB | word boxes | ⚠️ manual (author request) | Research use |
| 8 | [IAM Handwriting DB](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | handwritten lines/words | ~1.5 GB | line/word boxes | ⚠️ manual (free registration) | Research use, non-commercial |
| 9 | [GNHK](https://www.goodnotes.com/gnhk) | handwritten notes/receipts | small | quad boxes | ⚠️ manual | Research use |
|10 | Self-collected: webpage/PDF screenshots | screenshots of PDFs/webpages/tables | build your own | boxes from HTML/PDF text layer | ✅ scriptable (see `src/datasets/selfgen_notes.md`, TODO) | your own captures |

Implemented in this repo **right now** (auto or semi-auto with a processor script):
`FUNSD`, `CORD`, `SROIE`. The rest are listed for the checklist / next iteration —
add a new file under `src/datasets/<name>.py` following the same interface
(`iter_samples() -> (image, list_of_polygons)`) and register it in `build_dataset.py`.

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
  "id": "funsd_00012",
  "dataset": "funsd",
  "image_path": "images/funsd_00012.jpg",
  "mask_path": "masks/funsd_00012.png",
  "orig_width": 762, "orig_height": 1000,
  "resized_width": 780, "resized_height": 1024,
  "mask_width": 390, "mask_height": 512,
  "mask_scale": 0.5,
  "num_polygons": 42,
  "text_area_frac": 0.081,
  "num_contours": 39
}
```

At training time, load the mask and `cv2.resize`/`F.interpolate` it back up to the
image resolution (nearest-neighbor, since it's a binary mask).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. FUNSD - fully automatic
python -m src.datasets.funsd --download

# 2. CORD - fully automatic (uses HF `datasets`)
python -m src.datasets.cord --download

# 3. SROIE - manual download required first, see instructions below

# Build the unified processed dataset + stats for everything you've downloaded
python -m src.build_dataset --datasets funsd cord sroie
```

### SROIE manual steps
1. Create a free Kaggle account.
2. Download https://www.kaggle.com/datasets/urbikn/sroie-datasetv2 (or the original
   ICDAR2019 SROIE task1/2 zip).
3. Unzip into `data/raw/sroie/` so you have `data/raw/sroie/img/*.jpg` and
   `data/raw/sroie/box/*.txt` (each line: `x1,y1,x2,y2,x3,y3,x4,y4,text`).
4. Run `python -m src.build_dataset --datasets sroie` (or include it alongside others).

### ICDAR13 Born-Digital / Focused Scene Text manual steps
1. Register (free) at https://rrc.cvc.uab.es/.
2. Download the task-1 (text localization) train+test images and ground truth
   for the "Born-Digital Images" and/or "Focused Scene Text" challenges.
   These are the only two datasets here that ship pixel-level segmentation
   masks directly (`*_GT.bmp` / `*_GT.png`) rather than boxes — useful if you
   want a sanity check against your rasterized-from-boxes masks elsewhere.
3. Not yet wired into `build_dataset.py` — add a processor following
   `src/datasets/funsd.py` as a template (parse the GT mask images directly
   instead of rasterizing polygons).

## Notes on masks vs boxes

For now every processor rasterizes word/line **polygons or boxes into filled
binary masks** (`cv2.fillPoly`). This is what `build_dataset.py` produces.
The raw polygon/box coordinates are *not* discarded — they're cheap to keep —
but for this first pass we only persist the rasterized mask + summary stats
(`num_polygons`), not the raw box list, to keep the unified format simple.
When you're ready for detection-style boxes, re-run the processors and extend
`meta.jsonl` with a `boxes` field (the per-dataset extraction code already
computes them, see each `src/datasets/*.py`).
