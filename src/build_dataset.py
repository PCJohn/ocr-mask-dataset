"""Single entry point: process selected datasets into the unified
images+masks format, then compute per-dataset and combined statistics.

Usage:
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 200   # quick smoke test
    python -m src.build_dataset --datasets cord naf publaynet textocr synslides --limit 200 --shuffle
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


def _lazy_registry():
    if REGISTRY:
        return REGISTRY
    from src.datasets import cord, naf, publaynet, textocr, synslides, bstd, doclaynet, nvidia_multilingual
    REGISTRY["cord"] = cord
    REGISTRY["naf"] = naf
    REGISTRY["publaynet"] = publaynet
    REGISTRY["textocr"] = textocr
    REGISTRY["synslides"] = synslides
    REGISTRY["bstd"] = bstd
    REGISTRY["doclaynet"] = doclaynet
    REGISTRY["nvidia_multilingual"] = nvidia_multilingual
    return REGISTRY


def process_dataset(name: str, out_dir: str, limit: int = None, shuffle: bool = False, seed: int = 42,
                     refine_coarse: bool = True) -> str:
    mod = _lazy_registry()[name]
    records = []

    if limit and shuffle:
        # Reservoir sampling (Algorithm R): one pass over the full stream, uniform
        # random sample of `limit` items without needing to know the stream length
        # up front. Note this still walks the *entire* underlying dataset once --
        # for a fully-local/already-downloaded dataset that's cheap, but for a
        # streamed one (e.g. publaynet) it means pulling the whole remote stream
        # over the network just to end up keeping `limit` of them. Fine for
        # smaller datasets, worth knowing about for the big streamed ones.
        rng = random.Random(seed)
        reservoir = []
        for i, sample in enumerate(tqdm(mod.iter_samples(), desc=f"processing {name} (reservoir sampling)")):
            if i < limit:
                reservoir.append(sample)
            else:
                j = rng.randint(0, i)
                if j < limit:
                    reservoir[j] = sample
        samples = reservoir
    else:
        samples = []
        for sample in tqdm(mod.iter_samples(), desc=f"processing {name}"):
            samples.append(sample)
            if limit and len(samples) >= limit:
                break

    for sample in samples:
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
    p.add_argument("--datasets", nargs="+", default=["cord", "naf", "publaynet", "textocr", "synslides"],
                    choices=["cord", "naf", "publaynet", "textocr", "synslides", "bstd", "doclaynet",
                             "nvidia_multilingual"],
                    help="bstd (~17GB), doclaynet (custom loading script, unverified), and "
                         "nvidia_multilingual (synthetic, not real photos/scans) aren't in the default "
                         "set -- opt in explicitly")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--limit", type=int, default=None,
                    help="cap number of samples per dataset (useful for a quick smoke test)")
    p.add_argument("--shuffle", action="store_true",
                    help="randomly sample `--limit` items instead of taking the first N in dataset order "
                         "(via reservoir sampling -- one full pass over each dataset, cheap for local data, "
                         "slower for streamed ones like publaynet since it still reads the whole remote stream)")
    p.add_argument("--seed", type=int, default=42, help="random seed for --shuffle")
    p.add_argument("--no-refine-coarse-masks", action="store_true",
                    help="disable intensity-based stroke refinement for block-level datasets "
                         "(publaynet, synslides) -- without it, their masks flood-fill the whole "
                         "text block/element box instead of just the text strokes inside it")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_dataset_records = {}
    for name in args.datasets:
        meta_path = process_dataset(name, args.out_dir, limit=args.limit, shuffle=args.shuffle, seed=args.seed,
                                     refine_coarse=not args.no_refine_coarse_masks)
        per_dataset_records[name] = read_jsonl(meta_path)

    # combined manifest
    all_records = [r for recs in per_dataset_records.values() for r in recs]
    write_jsonl(os.path.join(args.out_dir, "manifest.jsonl"), all_records)

    # stats
    stats_dir = os.path.join(args.out_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    per_dataset_stats = {name: compute_stats(recs) for name, recs in per_dataset_records.items()}
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

    print(f"\nDone. {len(all_records)} total images across {len(args.datasets)} dataset(s).")
    print(f"Manifest: {os.path.join(args.out_dir, 'manifest.jsonl')}")
    print(f"Stats report: {os.path.join(stats_dir, 'stats_report.md')}")


if __name__ == "__main__":
    main()
