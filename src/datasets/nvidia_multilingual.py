"""NVIDIA OCR-Synthetic-Multilingual-v1: large-scale SYNTHETIC OCR data
(SynthDoG-based rendering) covering Japanese, Korean, Russian, Chinese
(Simplified + Traditional), and English. This is the dataset that fills the
CJK + Russian gap in this repo -- nothing else here covers those scripts.
License: CC BY 4.0 (commercial use OK).
https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1

It's synthetic (rendered text on backgrounds), not real photographed/scanned
content -- so it's a good complement to, not a replacement for, the
real-world datasets elsewhere in this repo. Genuinely fine-grained though:
real word-level AND line-level quad polygons (not block-level boxes), so no
coarse-refinement heuristic needed here.

SIZE NOTE: the full dataset is 5.45TB across millions of samples per
language, split into many .h5 shard files. We only ever download a handful
of individual shards (not the whole repo) via direct HF file downloads --
by default the smallest ("test") split's first shard per language, which is
still a few thousand samples each. Use --languages and --limit-per-shard
to control how much you pull.
"""

import argparse
import io
import json
import os

import numpy as np
from PIL import Image

from src.common import Sample

HF_NAME = "nvidia/OCR-Synthetic-Multilingual-v1"
RAW_DIR = "data/raw/nvidia_multilingual"

# All languages in the dataset; "en" is included for completeness but you
# probably don't need it given CORD/NAF/TextOCR/etc. already cover English.
ALL_LANGUAGES = ["ja", "ko", "ru", "zh_hans", "zh_hant", "en"]
DEFAULT_LANGUAGES = ["ja", "ko", "ru", "zh_hans", "zh_hant"]  # the actual gap-fillers
DEFAULT_SPLIT = "test"  # smallest split -- fewest samples per shard file
DEFAULT_SHARD_INDEX = 0  # just the first shard per language


def _shard_filename(lang: str, split: str, shard_index: int) -> str:
    return f"{lang}/{split}/{split}_{shard_index:03d}.h5"


def download(
    languages=None,
    split: str = DEFAULT_SPLIT,
    shard_index: int = DEFAULT_SHARD_INDEX,
    raw_dir: str = RAW_DIR,
):
    from huggingface_hub import hf_hub_download

    languages = languages or DEFAULT_LANGUAGES
    os.makedirs(raw_dir, exist_ok=True)
    for lang in languages:
        rel_path = _shard_filename(lang, split, shard_index)
        print(f"Downloading {rel_path} ...")
        hf_hub_download(
            repo_id=HF_NAME, repo_type="dataset", filename=rel_path, local_dir=raw_dir
        )
    print(f"NVIDIA multilingual shards ready at {raw_dir}")


def iter_samples(
    languages=None,
    split: str = DEFAULT_SPLIT,
    shard_index: int = DEFAULT_SHARD_INDEX,
    raw_dir: str = RAW_DIR,
    limit_per_shard: int = None,
):
    import h5py

    languages = languages or DEFAULT_LANGUAGES
    for lang in languages:
        h5_path = os.path.join(raw_dir, _shard_filename(lang, split, shard_index))
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"{h5_path} not found. Run `python -m src.datasets.nvidia_multilingual --download` first "
                f"(or pass --languages to restrict to what you've actually downloaded)."
            )
        with h5py.File(h5_path, "r") as f:
            n = len(f["images"])
            if limit_per_shard:
                n = min(n, limit_per_shard)
            for i in range(n):
                img_bytes = f["images"][i]
                img = Image.open(io.BytesIO(img_bytes.tobytes())).convert("RGB")
                ann = json.loads(f["annotations"][i])
                polys = []
                # word-level quads are the finest granularity available; use those directly
                for w in ann.get("word_bboxes", []):
                    quad = w.get("quad")
                    if quad and len(quad) >= 3:
                        polys.append(np.array(quad, dtype=np.float32))
                yield Sample(
                    sample_id=f"{lang}_{split}_{shard_index:03d}_{i:06d}",
                    image=img,
                    polygons=polys,
                    coarse=False,
                )


def _debug(
    languages=None,
    split=DEFAULT_SPLIT,
    shard_index=DEFAULT_SHARD_INDEX,
    raw_dir=RAW_DIR,
    n_samples: int = 3,
):
    import h5py, json, os

    languages = languages or DEFAULT_LANGUAGES[:2]  # just first 2 to be fast
    for lang in languages:
        h5_path = os.path.join(raw_dir, _shard_filename(lang, split, shard_index))
        if not os.path.exists(h5_path):
            print(f"  {lang}: file not found: {h5_path}  -- run --download first")
            continue
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            n = len(f["images"])
            print(f"\n  lang={lang}  shard={h5_path}")
            print(f"    HDF5 keys: {keys}  n_images={n}")
            for i in range(min(n_samples, n)):
                ann_raw = f["annotations"][i]
                try:
                    ann = json.loads(ann_raw.tobytes())
                except Exception as e:
                    print(f"    sample {i}: annotation parse error: {e}")
                    continue
                ann_keys = list(ann.keys())
                word_bboxes = ann.get("word_bboxes", [])
                line_bboxes = ann.get("line_bboxes", [])
                print(f"    sample {i}: ann keys={ann_keys}")
                print(
                    f"      word_bboxes={len(word_bboxes)}  line_bboxes={len(line_bboxes)}"
                )
                if word_bboxes:
                    wb0 = word_bboxes[0]
                    print(
                        f"      word_bboxes[0] keys={list(wb0.keys()) if isinstance(wb0, dict) else type(wb0).__name__}"
                    )
                    if isinstance(wb0, dict):
                        print(
                            f"        quad={wb0.get('quad')}  text={repr(wb0.get('text',''))[:30]}"
                        )
    print("\nExpected: word_bboxes > 0, quad is a list of [x,y] points.")
    print(
        "If quad is missing or has different shape, update iter_samples in nvidia_multilingual.py."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGUAGES, choices=ALL_LANGUAGES
    )
    p.add_argument(
        "--split", default=DEFAULT_SPLIT, choices=["train", "test", "validation"]
    )
    p.add_argument("--shard-index", type=int, default=DEFAULT_SHARD_INDEX)
    p.add_argument("--download", action="store_true")
    p.add_argument(
        "--check",
        action="store_true",
        help="download if needed, then verify a few samples parse",
    )
    p.add_argument(
        "--debug", action="store_true", help="inspect raw schema of first few samples"
    )
    args = p.parse_args()
    if args.download or args.check:
        download(
            languages=args.languages, split=args.split, shard_index=args.shard_index
        )
    if args.debug:
        _debug(languages=args.languages, split=args.split, shard_index=args.shard_index)
    elif args.check:
        n = 0
        for s in iter_samples(
            languages=args.languages,
            split=args.split,
            shard_index=args.shard_index,
            limit_per_shard=3,
        ):
            n += 1
            print(
                f"{s.sample_id}: {len(s.polygons)} polygons, image size {s.image.size}"
            )
        print(f"Loaded {n} samples OK across {len(args.languages)} language(s).")
