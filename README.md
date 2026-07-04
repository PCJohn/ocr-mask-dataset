# ocr-seg-dataset

Scripts to build a diverse, **commercially-usable**, **fully-automatically
downloadable** OCR text-segmentation dataset (binary text-vs-background
masks) for real-time OCR targeting: printed/scanned docs, receipts, forms,
and scene text (signboards, varying shapes/sizes).

Every dataset here downloads with **zero manual steps** — no login, no
clicking "I agree," no Kaggle/RRC account. Run one script and you have raw
data on disk.

## License note

All four datasets carry a license that **explicitly permits commercial use**
(CC BY 4.0 or CDLA-Permissive-1.0). CC BY / CDLA-Permissive still require you
to keep attribution/citation — that's a "don't strip the notice" obligation,
not a blocker. I'm not a lawyer; re-verify before shipping a commercial
product, licenses/hosting can change.

Common research benchmarks (FUNSD, SROIE, IAM, ICDAR RRC challenge sets,
IAM, GNHK, LectureVideoDB) are **excluded** — they're non-commercial/research
-only licensed and/or require a manual login, both of which are out of scope
here.

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

### One I looked at and rejected: IndicDLP

[IndicDLP](https://indicdlp.github.io/) would have been an excellent fit on
paper — 119,806 manually-annotated document images across **11 Indic
languages + English**, spanning 12 domains including forms, newspapers,
notices, and textbooks (real overlap with your "scanned documents" and
"forms" use cases, and multilingual to boot). It doesn't make the cut here
because its HF mirrors (`IndicDLP/IndicDLP-dataset`, `ai4bharat/indicdlp`)
are **gated** — "Log in or Sign Up to review the conditions and access this
dataset content" — which fails the "fully automatic, no login" bar, and its
license isn't stated plainly enough on the gated page for me to confirm
commercial use is clearly permitted. Worth revisiting if the gating/license
situation changes; not included for now.

## Language coverage — none of these are multilingual except BSTD

You asked specifically about multilingual/Indian-language coverage, so to be
direct about what's actually in each dataset:

| Dataset | Language(s) |
|---|---|
| CORD | Indonesian only |
| NAF | English only |
| PubLayNet | English only (PubMed Central scientific articles) |
| TextOCR | Predominantly English/Latin scenes (sourced from OpenImages). Non-English/illegible words *are* still polygon-annotated for location, just transcribed as a placeholder `"."` instead of real text — so it's usable for pure text-*location* masks on non-English text, just don't expect transcription value from it. |
| SynSlides | English only (LLM-generated) |
| **BSTD** | **Real multilingual**: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, English — this is the one dataset here actually built for Indian-language coverage. |

If your product needs balanced multilingual/Indian-language representation,
**BSTD is the one to prioritize** despite its size — everything else in this
repo is effectively monolingual. I looked for smaller multilingual
alternatives (MLT-19, IIIT-ILST, ICDAR RRC multilingual sets) and they're all
research-license-only, which is why they're not here — see "Excluded" notes
throughout. I also didn't find a small (<1GB) multilingual scene-text dataset
that's both commercially licensed and auto-downloadable; BSTD's size is the
tradeoff for actually having that coverage. `--limit` in `build_dataset.py`
still lets you cap how much of it you process into the final unified dataset,
even though the initial download itself is all-or-nothing.

## About the podcast/video-lecture domain specifically

I looked for a dataset that's simultaneously (a) real lecture/podcast/video
footage, (b) commercially licensed, and (c) fully automatic to download, and
didn't find one — see "what's not here" below. **SynSlides** (#5) is the
closest automatic + commercial-clean alternative I could find: it's
LLM-generated synthetic slide images with automated bounding-box annotations
for slide elements (titles, body text, bullet lists, equations, tables).
It's not real lecture footage — no talking-head frames, no camera artifacts,
no projector glare/angle — but it covers the **slide-content half** of that
domain (dense on-slide text, presentation-style layouts) with a clean MIT
license and zero manual steps.

⚠️ One honest caveat on SynSlides: I could not verify its exact bounding-box
annotation schema from this environment (the HF dataset viewer didn't fully
render the annotation column for me). `src/datasets/synslides.py` inspects
the dataset's actual columns at load time and tries several likely field
names; if none match, it fails loudly and tells you which columns *were*
found so you can fix one line. **Run `python -m src.datasets.synslides
--check` first** before including it in a full build.

## What's still not here, and why

Real **podcast / video-lecture / university-lecture-recording** footage
(actual video frames, not synthetic slides) remains out of scope:
- Public "lecture video" OCR datasets that exist (e.g. LectureVideoDB) are
  research-license-only or require an author request — excluded by this
  repo's rules.
- General video-text datasets are either research-only (ICDAR video text
  challenges) or aren't slide/lecture-domain.

For that remaining slice — actual camera/screen-capture footage, talking
heads, projector artifacts — your best bet is still self-generating it:
sample frames from lecture/podcast videos you have rights to (your own
recordings, or CC-licensed content — check the specific license per source,
a lot of OpenCourseWare-type material is CC BY-NC which wouldn't clear your
commercial-use bar), then either hand-label a small set or bootstrap labels
by running a detector (e.g. one trained on the datasets above) and
spot-correcting. That's a good next script to add
(`src/datasets/selfgen_video_frames.py`, not yet written) but it's
fundamentally a data-collection task, not a "here's a URL" task.

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

# One script downloads everything that needs a pre-download step
# (publaynet is streamed on demand, it isn't included here)
python -m src.download_all
# or a subset:
python -m src.download_all --datasets cord naf   # skip the ~6.5GB textocr zip

# Build the unified processed dataset + stats
python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 1500
```

`--limit` caps how many *processed* images are kept per dataset — start small
(e.g. 500) to sanity check the pipeline, then raise it or drop the flag for a
full run once you're happy with the output.

**By default `--limit` does not shuffle** — it takes the first N samples in
whatever order each dataset's `iter_samples()` yields them, which for e.g.
CORD means "the first N receipts in the train split," not a random spread.
Pass `--shuffle` to get a uniform random sample instead (reservoir sampling,
one pass over the data, seeded via `--seed`, default 42):

```bash
python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 500 --shuffle
```

Note `--shuffle` still walks the *entire* underlying dataset once to sample
from it fairly — cheap for already-downloaded local data, but for a streamed
dataset (publaynet) it means pulling the whole remote stream over the network
just to end up keeping `limit` of them. Fine at these dataset sizes, just
don't expect it to be instant.

## Mask granularity & known limitations

You should know this going in, since it affects what your trained segmenter
will learn:

| Dataset | Granularity | Non-text categories excluded? |
|---|---|---|
| CORD | word-level | n/a — pure text dataset |
| NAF | word/line-level | yes — filters annotation `type` to `text*`, drops field/graphic/comment boxes |
| TextOCR | word-level, arbitrary shape | n/a — pure text dataset |
| BSTD | word-level | n/a — pure text dataset |
| PubLayNet | **block/paragraph-level, not line-level** | yes — filters to `text`/`title`/`list` category ids, excludes `figure` always, excludes `table` by default (see `TEXT_CATEGORY_IDS` in `src/datasets/publaynet.py`) |
| SynSlides | **element-level, not line-level** | yes — filters annotation category *names* against an include/exclude keyword list (`src/datasets/synslides.py`), since this dataset's exact category schema wasn't independently verified from this environment |
| DocLayNet | **line-level** (genuinely fine-grained, not a heuristic fix) | yes — excludes `Picture` always, excludes `Table` by default |

Two distinct problems were found and fixed here:
1. **A real bug**: both PubLayNet and SynSlides annotate *all* layout elements
   (including images/figures/charts), and the original processors here
   included every annotation regardless of category — so image/infographic
   regions were being rasterized into the "text" mask. Both now filter by
   category before building polygons.
2. **A real limitation, mitigated but not eliminated**: PubLayNet and
   SynSlides annotate whole text *blocks* (a full paragraph, a full title),
   not individual lines — that's inherent to how those datasets were labeled,
   not something a processing script can invent finer boxes for. To reduce
   the effect, samples from these two datasets are marked `coarse=True`, and
   `save_sample()` (see `src/common.py`) applies an **intensity-based
   refinement**: within each coarse box, it keeps only the pixels that Otsu
   thresholding identifies as actual text strokes, instead of flood-filling
   the entire block including its background whitespace. This meaningfully
   tightens the mask but is still a heuristic, not ground truth — expect it
   to occasionally under- or over-segment on unusual backgrounds (e.g. a
   title block with a colored banner behind it). Disable it with
   `--no-refine-coarse-masks` if you want to compare against the old
   flood-fill behavior or roll your own refinement.

Every processed sample records whether it came from a coarse source
(`coarse_source: true/false` in `meta.jsonl`), so you can filter, upweight,
or exclude coarse-sourced samples per your training needs — e.g. if you care
a lot about fine-grained masks, you could train primarily on
CORD/NAF/TextOCR/BSTD and use PubLayNet/SynSlides only for scale/diversity,
or exclude them from an eval set to keep your metrics honest.

## Notes on masks vs boxes

Every processor rasterizes word/line/box **polygons into filled binary masks**
(`cv2.fillPoly`). That's what `build_dataset.py` produces. The raw
polygon/box coordinates aren't discarded — they're cheap to keep — but for
this first pass we only persist the rasterized mask + summary stats
(`num_polygons`), not the raw box list, to keep the unified format simple.
When you're ready for detection-style boxes, extend `meta.jsonl` with a
`boxes` field (the per-dataset extraction code already computes them, see
each `src/datasets/*.py`).
