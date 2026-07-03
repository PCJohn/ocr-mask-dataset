"""Single entry point: process selected datasets into the unified
images+masks format, then compute per-dataset and combined statistics.

Usage:
    python -m src.build_dataset --datasets funsd cord sroie
    python -m src.build_dataset --datasets funsd cord sroie --limit 200   # quick smoke test
"""
import argparse
import json
import os
import traceback

from tqdm import tqdm

from src.common import save_sample, write_jsonl, read_jsonl
from src.stats import compute_stats, render_markdown

REGISTRY = {}


def _lazy_registry():
    if REGISTRY:
        return REGISTRY
    from src.datasets import funsd, cord, sroie
    REGISTRY["funsd"] = funsd
    REGISTRY["cord"] = cord
    REGISTRY["sroie"] = sroie
    return REGISTRY


def process_dataset(name: str, out_dir: str, limit: int = None) -> str:
    mod = _lazy_registry()[name]
    records = []
    n = 0
    for sample in tqdm(mod.iter_samples(), desc=f"processing {name}"):
        try:
            rec = save_sample(out_dir, name, sample)
            records.append(rec)
        except Exception as e:
            print(f"[{name}] skipping {sample.sample_id}: {e}")
            traceback.print_exc()
        n += 1
        if limit and n >= limit:
            break
    meta_path = os.path.join(out_dir, name, "meta.jsonl")
    write_jsonl(meta_path, records)
    print(f"[{name}] wrote {len(records)} samples -> {meta_path}")
    return meta_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["funsd", "cord", "sroie"],
                    choices=["funsd", "cord", "sroie"])
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--limit", type=int, default=None,
                    help="cap number of samples per dataset (useful for a quick smoke test)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_dataset_records = {}
    for name in args.datasets:
        meta_path = process_dataset(name, args.out_dir, limit=args.limit)
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
