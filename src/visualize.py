"""visualize.py — interactive dataset sample browser.

Browse processed samples (image + mask side-by-side) in random order,
filter by dataset, and delete samples you want to drop (removes images,
masks, and updates manifest.jsonl + per-dataset meta.jsonl in-place).

Usage:
    python visualize.py                          # all datasets, random order
    python visualize.py --processed data/processed
    python visualize.py --datasets cord naf      # filter to subset
"""
import argparse
import json
import os
import random
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# ─── data helpers ────────────────────────────────────────────────────────────

def load_manifest(processed_dir):
    path = os.path.join(processed_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No manifest.jsonl found at {path}. Run build_dataset.py first.")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_manifest(processed_dir, records):
    path = os.path.join(processed_dir, "manifest.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def save_per_dataset_meta(processed_dir, dataset, records):
    path = os.path.join(processed_dir, dataset, "meta.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def delete_sample(processed_dir, record, all_records):
    """Delete image + mask files, remove from manifest + per-dataset meta."""
    dataset = record["dataset"]
    base = os.path.join(processed_dir, dataset)

    for field in ("image_path", "mask_path"):
        fpath = os.path.join(base, record[field])
        if os.path.exists(fpath):
            os.remove(fpath)

    # remove from global manifest
    remaining = [r for r in all_records if r["id"] != record["id"]]
    save_manifest(processed_dir, remaining)

    # remove from per-dataset meta
    ds_records = [r for r in remaining if r["dataset"] == dataset]
    save_per_dataset_meta(processed_dir, dataset, ds_records)

    return remaining


# ─── image helpers ───────────────────────────────────────────────────────────

def load_pair(processed_dir, record):
    dataset = record["dataset"]
    base = os.path.join(processed_dir, dataset)

    img = cv2.imread(os.path.join(base, record["image_path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {record['image_path']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(base, record["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {record['mask_path']}")
    mask = cv2.resize(mask, (record["resized_width"], record["resized_height"]),
                       interpolation=cv2.INTER_NEAREST)

    # overlay: green tint on mask pixels, alpha-blended onto image
    mask_rgb = np.zeros_like(img)
    mask_rgb[mask > 127] = [0, 220, 0]
    overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

    return img, mask, overlay


def make_panel(img_rgb, mask_bw, overlay_rgb, panel_h=600):
    """Stack image | mask | overlay side by side, scaled to panel_h."""
    mask_3ch = cv2.cvtColor(mask_bw, cv2.COLOR_GRAY2RGB)
    panels = [img_rgb, mask_3ch, overlay_rgb]
    scaled = []
    for p in panels:
        h, w = p.shape[:2]
        scale = panel_h / h
        new_w = max(1, int(w * scale))
        r = cv2.resize(p, (new_w, panel_h), interpolation=cv2.INTER_LINEAR)
        scaled.append(r)
    combined = np.concatenate(scaled, axis=1)
    return Image.fromarray(combined)


# ─── GUI ─────────────────────────────────────────────────────────────────────

class Viewer:
    PANEL_H = 560

    def __init__(self, root, processed_dir, initial_records, all_records):
        self.root = root
        self.processed_dir = processed_dir
        self.all_records = all_records          # always the full live list (updated on delete)
        self.queue = list(initial_records)      # display order for current filter
        self.index = 0
        self.current_record = None

        root.title("OCR Seg Dataset Viewer")
        root.resizable(True, True)

        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(top, text="Filter dataset:").pack(side=tk.LEFT)
        datasets = sorted({r["dataset"] for r in all_records})
        self.ds_var = tk.StringVar(value="all")
        ds_combo = ttk.Combobox(top, textvariable=self.ds_var, state="readonly", width=22,
                                 values=["all"] + datasets)
        ds_combo.pack(side=tk.LEFT, padx=4)
        ds_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.shuffle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Shuffle", variable=self.shuffle_var,
                        command=self._on_filter_change).pack(side=tk.LEFT, padx=8)

        self.counter_lbl = tk.Label(top, text="")
        self.counter_lbl.pack(side=tk.RIGHT, padx=8)

        # ── info bar ─────────────────────────────────────────────────────────
        info = tk.Frame(root)
        info.pack(fill=tk.X, padx=8)
        self.info_lbl = tk.Label(info, text="", anchor="w", font=("Courier", 10))
        self.info_lbl.pack(fill=tk.X)

        # ── image panel ──────────────────────────────────────────────────────
        self.canvas = tk.Label(root, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ── bottom bar ───────────────────────────────────────────────────────
        bot = tk.Frame(root)
        bot.pack(fill=tk.X, padx=8, pady=6)

        btn = lambda text, cmd, **kw: tk.Button(bot, text=text, command=cmd,
                                                  width=14, **kw)
        btn("◀  Prev", self.prev).pack(side=tk.LEFT, padx=4)
        btn("Next  ▶", self.next).pack(side=tk.LEFT, padx=4)
        btn("🗑  Delete", self._delete, bg="#c0392b", fg="white").pack(side=tk.RIGHT, padx=4)

        root.bind("<Left>", lambda _: self.prev())
        root.bind("<Right>", lambda _: self.next())
        root.bind("<Delete>", lambda _: self._delete())

        self._apply_filter()
        self._show()

    # ── filter ───────────────────────────────────────────────────────────────

    def _apply_filter(self):
        ds = self.ds_var.get()
        pool = self.all_records if ds == "all" else [r for r in self.all_records if r["dataset"] == ds]
        self.queue = list(pool)
        if self.shuffle_var.get():
            random.shuffle(self.queue)
        self.index = 0

    def _on_filter_change(self, *_):
        self._apply_filter()
        self._show()

    # ── navigation ───────────────────────────────────────────────────────────

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self._show()

    def next(self):
        if self.index < len(self.queue) - 1:
            self.index += 1
            self._show()

    # ── display ──────────────────────────────────────────────────────────────

    def _show(self):
        if not self.queue:
            self.info_lbl.config(text="No samples to display.")
            self.canvas.config(image="")
            self.counter_lbl.config(text="0 / 0")
            return

        rec = self.queue[self.index]
        self.current_record = rec
        self.counter_lbl.config(text=f"{self.index + 1} / {len(self.queue)}")

        try:
            img, mask, overlay = load_pair(self.processed_dir, rec)
        except FileNotFoundError as e:
            self.info_lbl.config(text=f"[missing] {e}")
            self.canvas.config(image="")
            return

        panel_h = max(200, self.root.winfo_height() - 160)
        panel_img = make_panel(img, mask, overlay, panel_h=panel_h)

        tk_img = ImageTk.PhotoImage(panel_img)
        self.canvas.config(image=tk_img)
        self.canvas.image = tk_img  # keep reference

        h, w = img.shape[:2]
        self.info_lbl.config(
            text=(f"id={rec['id']}  |  {w}×{h}  |  "
                  f"polygons={rec.get('num_polygons', '?')}  |  "
                  f"text_area={rec.get('text_area_frac', 0):.1%}  |  "
                  f"coarse={rec.get('coarse_source', '?')}")
        )

    # ── delete ───────────────────────────────────────────────────────────────

    def _delete(self):
        if not self.current_record:
            return
        rec = self.current_record
        if not messagebox.askyesno("Delete sample",
                                    f"Permanently delete:\n{rec['id']}\n\n"
                                    "This removes the image and mask files and updates "
                                    "manifest.jsonl + meta.jsonl."):
            return
        self.all_records = delete_sample(self.processed_dir, rec, self.all_records)
        self.queue.pop(self.index)
        if self.index >= len(self.queue):
            self.index = max(0, len(self.queue) - 1)
        self._show()


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed",
                     help="path to the processed data directory (default: data/processed)")
    ap.add_argument("--datasets", nargs="+", default=None,
                     help="restrict to these dataset names (default: all)")
    ap.add_argument("--no-shuffle", action="store_true",
                     help="show samples in dataset order instead of shuffled")
    ap.add_argument("--seed", type=int, default=None,
                     help="random seed for shuffle")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    all_records = load_manifest(args.processed)
    if args.datasets:
        initial = [r for r in all_records if r["dataset"] in args.datasets]
    else:
        initial = list(all_records)

    if not args.no_shuffle:
        random.shuffle(initial)

    root = tk.Tk()
    root.geometry("1400x700")
    Viewer(root, args.processed, initial, all_records)
    root.mainloop()


if __name__ == "__main__":
    main()
