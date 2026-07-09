"""Single entry point: process selected datasets into the unified
images+masks format, then compute per-dataset and combined statistics.

Usage:
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 200
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 200 --no-shuffle
"""

import argparse
import json
import os
import random
import traceback

from tqdm import tqdm

from src.common import save_sample, write_jsonl, read_jsonl
from src.stats import compute_stats, render_markdown

REGISTRY = {}

# Datasets that are already on disk after download -- can be shuffled cheaply
# by collecting all samples first, then shuffling, then slicing.
# Streamed datasets (publaynet) can't do this -- they use a shuffle buffer instead.
LOCAL_DATASETS = {
    "cord",
    "naf",
    "textocr",
    "synslides",
    "bstd",
    "doclaynet",
    "nvidia_multilingual",
}
SHUFFLE_BUFFER = 2000  # samples buffered before random draw for streamed datasets


def _lazy_registry():
    if REGISTRY:
        return REGISTRY
    from src.datasets import (
        cord,
        naf,
        publaynet,
        textocr,
        synslides,
        bstd,
        doclaynet,
        nvidia_multilingual,
    )

    REGISTRY["cord"] = cord
    REGISTRY["naf"] = naf
    REGISTRY["publaynet"] = publaynet
    REGISTRY["textocr"] = textocr
    REGISTRY["synslides"] = synslides
    REGISTRY["bstd"] = bstd
    REGISTRY["doclaynet"] = doclaynet
    REGISTRY["nvidia_multilingual"] = nvidia_multilingual
    return REGISTRY


def process_dataset(
    name: str,
    out_dir: str,
    limit: int = None,
    shuffle: bool = True,
    seed: int = 42,
    refine_coarse: bool = True,
) -> str:
    mod = _lazy_registry()[name]
    records = []
    rng = random.Random(seed)

    if name in LOCAL_DATASETS:
        # Local datasets: collect everything (or up to a generous cap to avoid
        # OOM on huge datasets like textocr), shuffle, then slice to limit.
        # Cap collection at max(limit*10, 5000) so we sample from a good spread
        # without loading all 28k TextOCR images into memory.
        collect_cap = max(limit * 10, 5000) if limit else None
        all_samples = []
        for sample in tqdm(mod.iter_samples(), desc=f"collecting {name}"):
            all_samples.append(sample)
            if collect_cap and len(all_samples) >= collect_cap:
                break
        if shuffle:
            rng.shuffle(all_samples)
        samples = all_samples[:limit] if limit else all_samples
    else:
        # Streamed datasets (publaynet): use a shuffle buffer -- fill a buffer,
        # draw randomly from it, refill. Avoids streaming the whole dataset.
        if shuffle and limit:
            buf = []
            samples = []
            for sample in tqdm(
                mod.iter_samples(), desc=f"processing {name} (buffered shuffle)"
            ):
                buf.append(sample)
                if len(buf) >= SHUFFLE_BUFFER:
                    idx = rng.randrange(len(buf))
                    samples.append(buf.pop(idx))
                    if len(samples) >= limit:
                        break
            # drain remainder if we still need more
            if len(samples) < (limit or 0) and buf:
                rng.shuffle(buf)
                samples.extend(buf[: limit - len(samples)])
        else:
            samples = []
            for sample in tqdm(mod.iter_samples(), desc=f"processing {name}"):
                samples.append(sample)
                if limit and len(samples) >= limit:
                    break

    for sample in tqdm(samples, desc=f"saving {name}", leave=False):
        try:
            rec = save_sample(out_dir, name, sample, refine_coarse=refine_coarse)
            records.append(rec)
        except Exception as e:
            print(f"[{name}] skipping {sample.sample_id}: {e}")
            traceback.print_exc()

    meta_path = os.path.join(out_dir, name, "meta.jsonl")
    write_jsonl(meta_path, records)
    print(f"[{name}] wrote {len(records)} samples -> {meta_path}")
    return meta_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["cord", "naf", "publaynet", "textocr", "synslides"],
        choices=[
            "cord",
            "naf",
            "publaynet",
            "textocr",
            "synslides",
            "bstd",
            "doclaynet",
            "nvidia_multilingual",
        ],
        help="bstd (~17GB), doclaynet (custom loading script, unverified), and "
        "nvidia_multilingual (synthetic, not real photos/scans) aren't in the default "
        "set -- opt in explicitly",
    )
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument(
        "--limit", type=int, default=None, help="cap number of samples per dataset"
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="take the first --limit samples in dataset order instead of shuffling "
        "(default is to shuffle when --limit is set)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed for shuffle")
    p.add_argument(
        "--no-refine-coarse-masks",
        action="store_true",
        help="disable intensity-based stroke refinement for block-level datasets "
        "(publaynet, synslides) -- without it, their masks flood-fill the whole "
        "text block/element box instead of just the text strokes inside it",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_dataset_records = {}
    for name in args.datasets:
        meta_path = process_dataset(
            name,
            args.out_dir,
            limit=args.limit,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            refine_coarse=not args.no_refine_coarse_masks,
        )
        per_dataset_records[name] = read_jsonl(meta_path)

    # combined manifest
    all_records = [r for recs in per_dataset_records.values() for r in recs]
    write_jsonl(os.path.join(args.out_dir, "manifest.jsonl"), all_records)

    # stats
    stats_dir = os.path.join(args.out_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    per_dataset_stats = {
        name: compute_stats(recs) for name, recs in per_dataset_records.items()
    }
    combined_stats = compute_stats(all_records)

    with open(os.path.join(stats_dir, "per_dataset_stats.json"), "w") as f:
        json.dump(per_dataset_stats, f, indent=2)
    with open(os.path.join(stats_dir, "combined_stats.json"), "w") as f:
        json.dump(combined_stats, f, indent=2)

    all_stats_for_md = dict(per_dataset_stats)
    all_stats_for_md["COMBINED"] = combined_stats
    report = render_markdown(all_stats_for_md)
    with open(os.path.join(stats_dir, "stats_report.md"), "w") as f:
        f.write(report)

    print(
        f"\nDone. {len(all_records)} total images across {len(args.datasets)} dataset(s)."
    )
    print(f"Manifest: {os.path.join(args.out_dir, 'manifest.jsonl')}")
    print(f"Stats report: {os.path.join(stats_dir, 'stats_report.md')}")


if __name__ == "__main__":
    main()
