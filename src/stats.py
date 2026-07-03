"""Compute min/mean/std/p50/p90/max style stats over dataset meta records."""
from typing import Dict, List

import numpy as np


def _dist(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: None for k in ("min", "mean", "std", "p50", "p90", "max")}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


FIELDS_TO_SUMMARIZE = [
    ("orig_width", "orig_width"),
    ("orig_height", "orig_height"),
    ("resized_width", "resized_width"),
    ("resized_height", "resized_height"),
    ("mask_width", "mask_width"),
    ("mask_height", "mask_height"),
    ("num_polygons", "num_polygons"),
    ("num_contours", "num_contours"),
    ("text_area_frac", "text_area_frac"),
]


def compute_stats(records: List[dict]) -> dict:
    out = {"num_images": len(records)}
    for label, field in FIELDS_TO_SUMMARIZE:
        out[label] = _dist([r[field] for r in records if field in r])
    # image aspect ratio distribution, useful for augmentation/resize decisions
    aspects = [r["orig_width"] / r["orig_height"] for r in records if r.get("orig_height")]
    out["aspect_ratio"] = _dist(aspects)
    return out


def render_markdown(all_stats: Dict[str, dict]) -> str:
    lines = ["# Dataset statistics report\n"]
    for name, stats in all_stats.items():
        lines.append(f"## {name}\n")
        lines.append(f"- num_images: **{stats['num_images']}**\n")
        for label, _ in FIELDS_TO_SUMMARIZE + [("aspect_ratio", "aspect_ratio")]:
            d = stats.get(label)
            if not d or d.get("mean") is None:
                continue
            lines.append(
                f"- {label}: min={d['min']:.3f} mean={d['mean']:.3f} std={d['std']:.3f} "
                f"p50={d['p50']:.3f} p90={d['p90']:.3f} max={d['max']:.3f}"
            )
        lines.append("")
    return "\n".join(lines)
