# ocr-seg-dataset

Scripts to build a diverse, **commercially-usable**, **fully-automatically
downloadable** OCR text-segmentation dataset (binary text-vs-background
masks) for real-time OCR targeting: printed/scanned docs, receipts, forms,
and scene text (signboards, varying shapes/sizes).

Every dataset here downloads with **zero manual steps** — no login, no
clicking "I agree," no Kaggle/RRC account. Run one script and you have raw
data on disk.

## Dataset checklist

| # | Dataset | Domain | Size | Annotation type | License | Download |
|---|---|---|---|---|---|---|
| 1 | [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | receipts | ~1–2 GB | word/line quads | CC BY 4.0 | HF `datasets`, auto |
| 2 | [NAF Dataset](https://github.com/herobd/NAF_dataset) | forms, printed + some handwritten | few hundred MB | quad/polygon boxes | CC BY 4.0 | GitHub release + git clone, auto |
| 3 | [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) (via [jordanparker6/publaynet](https://huggingface.co/datasets/jordanparker6/publaynet) mirror) | PDF/document pages — stand-in for scanned docs, screenshots-of-PDFs, tables | full ~104GB, but **streamed + capped** so effectively however much you ask for | rect boxes + polygon segmentation | CDLA-Permissive-1.0 | HF `datasets` streaming, auto, no pre-download needed |
| 4 | [TextOCR](https://textvqa.org/textocr/) | natural scene text: signboards, product labels, street scenes — arbitrary shapes/sizes | ~6.5 GB (single zip, not shardable) | arbitrary-shape polygons | CC BY 4.0 | direct HTTPS zip from Meta's public file server, auto |
| 5 | [SynSlides](https://huggingface.co/datasets/NerdyVisky/SynSlides) ([paper](https://arxiv.org/abs/2506.23605)) | **synthetic lecture slides** — closest automatic/commercial-clean stand-in for slideshows/lecture-video-frames | ~544 MB | element bounding boxes (title, body text, bullets, equations, tables, etc.) | MIT | HF `datasets`, auto |
| 6 | [BSTD (Bharat Scene Text Dataset)](https://github.com/Bhashini-IITJ/BharatSceneTextDataset) ([paper](https://arxiv.org/abs/2511.23071)) | real photographed scene text: signboards, billboards, bus stops, ATMs — **11 Indian languages + English** | ~17 GB, single zip (no partial download) | word-level polygons | Apache-2.0 repo/annotations; images are CC BY-SA 4.0 (Wikimedia Commons) ⚠️ share-alike | Google Drive public link via `gdown`, auto — **opt-in only**, not in the default dataset list (size) |
| 7 | [DocLayNet](https://github.com/DS4SD/DocLayNet) (via [pierreguillou/DocLayNet-base](https://huggingface.co/datasets/pierreguillou/DocLayNet-base) mirror) | real scanned/digital documents: financial reports, manuals, scientific articles, laws, patents, tenders | ~691 train images (small on purpose) | **line-level boxes** (`bboxes_line`) — a step up in granularity from PubLayNet/SynSlides | CDLA-Permissive-1.0 | HF `datasets`, auto, but uses a custom loading script (`trust_remote_code=True`) — **opt-in only**, unverified from this sandboxed environment |
| 8 | [NVIDIA OCR-Synthetic-Multilingual-v1](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1) | **synthetic** rendered text — **Japanese, Korean, Russian, Chinese (Simplified + Traditional), English** | full dataset is 5.45TB across millions of samples/language, but we only pull one small shard per language (a few thousand samples each) via direct HF file downloads | genuine **word- and line-level quad polygons** (not block-level) | CC BY 4.0 | HF Hub direct file download (`hf_hub_download`), auto — **opt-in only** (synthetic, not real photos/scans, and you should pick which languages you actually want) |

## Language Coverage

- Indonesian (CORD)
- English (NAF, PubLayNet, TextOCR, SynSlides, DocLayNet, BSTD, NVIDIA multilingual)
- Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu (BSTD)
- Japanese, Korean, Russian, Chinese (Simplified + Traditional) (NVIDIA multilingual)

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

Everything below is designed to be run top-to-bottom after a fresh clone.
Copy-paste each block in order.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium   # one-time: downloads headless browser for Wikipedia mining
```

```bash
# 2. Download pre-labeled datasets
python -m src.download_all   # CORD + NAF + TextOCR (~8GB) + SynSlides
# TextOCR is ~6.5GB; skip it with: --datasets cord naf synslides
```

```bash
# 3. Build unified processed dataset (images + binary masks + stats)
#    Shuffle is on by default so you get a representative spread from each dataset.
python -m src.build_dataset \
    --datasets cord naf publaynet textocr bstd doclaynet nvidia_multilingual \
    --limit 20000
```

```bash
# 4. Mine additional masks from free online sources
#    Each command is independent — run whichever you want.
pip install easyocr   # one-time: OCR backend used for Wikipedia/YouTube sources

python -m scripts.mine_masks --sources arxiv      --n 100   # arXiv PDFs (exact PDF text layer)
python -m scripts.mine_masks --sources wikipedia  --n 100   # Wikipedia screenshots, 54 languages
python -m scripts.mine_masks --sources synthetic  --n 100   # Pillow-rendered text, varied fonts
python -m scripts.mine_masks --sources pubmed     --n 100   # PubMed Central OA PDFs
python -m scripts.mine_masks --sources gutenberg  --n 100   # Project Gutenberg books
python -m scripts.mine_masks --sources openalex   --n 100   # OpenAlex OA PDFs across disciplines

# YouTube: provide your own channel or video URLs
python -m scripts.mine_masks --sources youtube --n 100 \
    --youtube-urls https://www.youtube.com/@3blue1brown \
                   https://www.youtube.com/@lexfridman \
                   https://www.youtube.com/@TED

# Mine masks from your own local images (photos, screenshots, etc.)
python -m scripts.mine_masks --sources local \
    --local-dir /path/to/your/images \
    --dataset-name my_images \
    --n 100
```

```bash
# 5. Clean: remove samples where mask quality is poor
#    Corrupt files are always removed (Pass 0). Pass 1 uses EasyOCR's CRAFT
#    detector to verify mask coverage — delete if IoU < threshold.
python -m scripts.clean_dataset --gpu --min-iou 0.9 --dry-run   # preview first
python -m scripts.clean_dataset --gpu --min-iou 0.9              # delete for real
```

```bash
# 6. Browse and manually review samples
python visualize.py                          # all datasets, shuffled
python visualize.py --datasets cord naf      # filter to specific datasets
```

`--limit` caps how many processed images are kept per dataset. Shuffle is on
by default so samples are drawn from across each dataset rather than the first
N in file order. Pass `--no-shuffle` to disable.

## Mask granularity & known limitations

| Dataset | Granularity | Non-text categories excluded? |
|---|---|---|
| CORD | word-level | n/a — pure text dataset |
| NAF | word/line-level | yes — filters annotation `type` to `text*`, drops field/graphic/comment boxes |
| TextOCR | word-level, arbitrary shape | n/a — pure text dataset |
| BSTD | word-level | n/a — pure text dataset |
| PubLayNet | **block/paragraph-level, not line-level** | yes — filters to `text`/`title`/`list` category ids, excludes `figure` always, excludes `table` by default (see `TEXT_CATEGORY_IDS` in `src/datasets/publaynet.py`) |
| SynSlides | **element-level, not line-level** | yes — filters annotation category *names* against an include/exclude keyword list (`src/datasets/synslides.py`), since this dataset's exact category schema wasn't independently verified from this environment |
| DocLayNet | **line-level** (genuinely fine-grained, not a heuristic fix) | yes — excludes `Picture` always, excludes `Table` by default |
| NVIDIA multilingual | **word/line-level quads** (genuinely fine-grained) | n/a — pure text dataset, but synthetic not real |

## Notes on masks vs boxes

Every processor rasterizes word/line/box **polygons into filled binary masks**
(`cv2.fillPoly`). That's what `build_dataset.py` produces. The raw
polygon/box coordinates aren't discarded — they're cheap to keep — but for
this first pass we only persist the rasterized mask + summary stats
(`num_polygons`), not the raw box list, to keep the unified format simple.
When you're ready for detection-style boxes, extend `meta.jsonl` with a
`boxes` field (the per-dataset extraction code already computes them, see
each `src/datasets/*.py`).