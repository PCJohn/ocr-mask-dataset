"""visualize.py — interactive dataset sample browser.

Browse processed samples (image + mask side-by-side) in random order,
filter by dataset, and delete samples you want to drop (removes images,
masks, and updates manifest.jsonl + per-dataset meta.jsonl in-place).

Controls:
  Mouse wheel        zoom in / out (centred on cursor)
  Click + drag       pan
  Left / Right       previous / next sample
  Delete key         delete current sample
  R                  reset zoom / pan

Usage:
    python visualize.py
    python visualize.py --processed data/processed
    python visualize.py --datasets cord naf
"""

import argparse
import json
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ─── data helpers ─────────────────────────────────────────────────────────────


def load_manifest(processed_dir):
    path = os.path.join(processed_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No manifest.jsonl at {path}. Run build_dataset.py first."
        )
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_manifest(processed_dir, records):
    with open(os.path.join(processed_dir, "manifest.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def save_per_dataset_meta(processed_dir, dataset, records):
    with open(os.path.join(processed_dir, dataset, "meta.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def delete_sample(processed_dir, record, all_records):
    dataset = record["dataset"]
    base = os.path.join(processed_dir, dataset)
    for field in ("image_path", "mask_path"):
        fpath = os.path.join(base, record[field])
        if os.path.exists(fpath):
            os.remove(fpath)
    remaining = [r for r in all_records if r["id"] != record["id"]]
    save_manifest(processed_dir, remaining)
    save_per_dataset_meta(
        processed_dir, dataset, [r for r in remaining if r["dataset"] == dataset]
    )
    return remaining


# ─── image helpers ────────────────────────────────────────────────────────────


def load_pair(processed_dir, record):
    base = os.path.join(processed_dir, record["dataset"])
    img = cv2.imread(os.path.join(base, record["image_path"]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(record["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(base, record["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(record["mask_path"])
    mask = cv2.resize(
        mask,
        (record["resized_width"], record["resized_height"]),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_rgb = np.zeros_like(img)
    mask_rgb[mask > 127] = [0, 220, 0]
    overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
    return img, mask, overlay


def make_panel(img_rgb, mask_bw, overlay_rgb):
    """Return a single wide PIL image: original | mask | overlay at native resolution."""
    mask_3ch = cv2.cvtColor(mask_bw, cv2.COLOR_GRAY2RGB)
    combined = np.concatenate([img_rgb, mask_3ch, overlay_rgb], axis=1)
    return Image.fromarray(combined)


# ─── zoomable canvas ──────────────────────────────────────────────────────────


class ZoomCanvas(tk.Frame):
    """A tk.Canvas with mouse-wheel zoom (centred on cursor) and drag-to-pan."""

    ZOOM_STEP = 1.25
    MIN_ZOOM = 0.05
    MAX_ZOOM = 20.0

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._canvas = tk.Canvas(
            self, bg="#1e1e1e", cursor="crosshair", highlightthickness=0
        )
        hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._canvas.xview)
        vbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        hbar.grid(row=1, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._pil_img = None  # full-resolution PIL source image
        self._zoom = 1.0
        self._img_id = None
        self._tk_img = None  # keep reference
        # anchor: canvas coords of image top-left corner
        self._ox = 0
        self._oy = 0

        # drag state
        self._drag_start = None

        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<MouseWheel>", self._on_wheel)  # Windows / macOS
        self._canvas.bind("<Button-4>", self._on_wheel)  # Linux scroll up
        self._canvas.bind("<Button-5>", self._on_wheel)  # Linux scroll down

    # ── public API ───────────────────────────────────────────────────────────

    def set_image(self, pil_img: Image.Image):
        self._pil_img = pil_img
        self.reset_view()

    def reset_view(self):
        if self._pil_img is None:
            return
        cw = self._canvas.winfo_width() or 1
        ch = self._canvas.winfo_height() or 1
        iw, ih = self._pil_img.size
        self._zoom = min(
            cw / iw, ch / ih, 1.0
        )  # fit-to-window, never upscale beyond 1×
        self._ox = max(0, (cw - iw * self._zoom) / 2)
        self._oy = max(0, (ch - ih * self._zoom) / 2)
        self._redraw()

    # ── internal ─────────────────────────────────────────────────────────────

    def _redraw(self):
        if self._pil_img is None:
            return
        iw, ih = self._pil_img.size
        nw = max(1, int(iw * self._zoom))
        nh = max(1, int(ih * self._zoom))
        # Use NEAREST at high zoom so you can see individual pixels;
        # LANCZOS when zoomed out for quality.
        resample = Image.NEAREST if self._zoom >= 2.0 else Image.LANCZOS
        resized = self._pil_img.resize((nw, nh), resample)
        self._tk_img = ImageTk.PhotoImage(resized)

        cx = max(nw, self._canvas.winfo_width())
        cy = max(nh, self._canvas.winfo_height())
        self._canvas.configure(scrollregion=(0, 0, cx, cy))

        if self._img_id is None:
            self._img_id = self._canvas.create_image(
                int(self._ox), int(self._oy), anchor="nw", image=self._tk_img
            )
        else:
            self._canvas.coords(self._img_id, int(self._ox), int(self._oy))
            self._canvas.itemconfig(self._img_id, image=self._tk_img)

    def _on_wheel(self, event):
        # determine zoom direction
        if event.num == 5 or event.delta < 0:
            factor = 1 / self.ZOOM_STEP
        else:
            factor = self.ZOOM_STEP

        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        # zoom centred on the cursor position in image-space
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        # image coords under cursor before zoom
        ix = (cx - self._ox) / self._zoom
        iy = (cy - self._oy) / self._zoom
        self._zoom = new_zoom
        # keep same image point under cursor
        self._ox = cx - ix * self._zoom
        self._oy = cy - iy * self._zoom
        self._redraw()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y, self._ox, self._oy)

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        sx, sy, ox0, oy0 = self._drag_start
        self._ox = ox0 + (event.x - sx)
        self._oy = oy0 + (event.y - sy)
        self._redraw()


# ─── main viewer ──────────────────────────────────────────────────────────────


class Viewer:
    def __init__(self, root, processed_dir, initial_records, all_records):
        self.root = root
        self.processed_dir = processed_dir
        self.all_records = all_records
        self.queue = list(initial_records)
        self.index = 0
        self.current_record = None

        root.title("OCR Seg Dataset Viewer")
        root.resizable(True, True)

        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(top, text="Dataset:").pack(side=tk.LEFT)
        datasets = sorted({r["dataset"] for r in all_records})
        self.ds_var = tk.StringVar(value="all")
        ds_combo = ttk.Combobox(
            top,
            textvariable=self.ds_var,
            state="readonly",
            width=22,
            values=["all"] + datasets,
        )
        ds_combo.pack(side=tk.LEFT, padx=4)
        ds_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.shuffle_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            top,
            text="Shuffle",
            variable=self.shuffle_var,
            command=self._on_filter_change,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Reset zoom  [R]", command=self._reset_zoom, width=14).pack(
            side=tk.LEFT, padx=10
        )

        self.zoom_lbl = tk.Label(top, text="zoom: 100%", width=12)
        self.zoom_lbl.pack(side=tk.LEFT)

        self.counter_lbl = tk.Label(top, text="")
        self.counter_lbl.pack(side=tk.RIGHT, padx=8)

        # ── info bar ─────────────────────────────────────────────────────────
        info = tk.Frame(root)
        info.pack(fill=tk.X, padx=8)
        self.info_lbl = tk.Label(info, text="", anchor="w", font=("Courier", 10))
        self.info_lbl.pack(fill=tk.X)

        # ── zoom canvas ──────────────────────────────────────────────────────
        self.zcanvas = ZoomCanvas(root, bg="#1e1e1e")
        self.zcanvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        # patch _redraw to update zoom label
        _orig_redraw = self.zcanvas._redraw

        def _patched_redraw():
            _orig_redraw()
            self.zoom_lbl.config(text=f"zoom: {self.zcanvas._zoom*100:.0f}%")

        self.zcanvas._redraw = _patched_redraw

        # ── bottom bar ───────────────────────────────────────────────────────
        bot = tk.Frame(root)
        bot.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(bot, text="◀  Prev", command=self.prev, width=14).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(bot, text="Next  ▶", command=self.next, width=14).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(
            bot,
            text="🗑  Delete",
            command=self._delete,
            width=14,
            bg="#c0392b",
            fg="white",
        ).pack(side=tk.RIGHT, padx=4)

        root.bind("<Left>", lambda _: self.prev())
        root.bind("<Right>", lambda _: self.next())
        root.bind("<Delete>", lambda _: self._delete())
        root.bind("<r>", lambda _: self._reset_zoom())
        root.bind("<R>", lambda _: self._reset_zoom())

        self._apply_filter()
        # wait for canvas to have real dimensions before first draw
        root.update_idletasks()
        self._show()

    # ── filter ───────────────────────────────────────────────────────────────

    def _apply_filter(self):
        ds = self.ds_var.get()
        pool = (
            self.all_records
            if ds == "all"
            else [r for r in self.all_records if r["dataset"] == ds]
        )
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

    def _reset_zoom(self):
        self.zcanvas.reset_view()

    # ── display ──────────────────────────────────────────────────────────────

    def _show(self):
        if not self.queue:
            self.info_lbl.config(text="No samples to display.")
            self.counter_lbl.config(text="0 / 0")
            return

        rec = self.queue[self.index]
        self.current_record = rec
        self.counter_lbl.config(text=f"{self.index + 1} / {len(self.queue)}")

        try:
            img, mask, overlay = load_pair(self.processed_dir, rec)
        except FileNotFoundError as e:
            self.info_lbl.config(text=f"[missing file] {e}")
            return

        panel = make_panel(img, mask, overlay)
        self.zcanvas.set_image(panel)  # set_image calls reset_view → fit-to-window

        h, w = img.shape[:2]
        self.info_lbl.config(
            text=(
                f"id={rec['id']}  |  {w}×{h}  |  "
                f"polygons={rec.get('num_polygons','?')}  |  "
                f"text_area={rec.get('text_area_frac',0):.1%}  |  "
                f"coarse={rec.get('coarse_source','?')}"
            )
        )

    # ── delete ───────────────────────────────────────────────────────────────

    def _delete(self):
        if not self.current_record:
            return
        rec = self.current_record
        if not messagebox.askyesno(
            "Delete sample",
            f"Permanently delete:\n{rec['id']}\n\n"
            "Removes image + mask files and updates "
            "manifest.jsonl + meta.jsonl.",
        ):
            return
        self.all_records = delete_sample(self.processed_dir, rec, self.all_records)
        self.queue.pop(self.index)
        if self.index >= len(self.queue):
            self.index = max(0, len(self.queue) - 1)
        self._show()


# ─── entry point ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    all_records = load_manifest(args.processed)
    initial = (
        [r for r in all_records if r["dataset"] in args.datasets]
        if args.datasets
        else list(all_records)
    )
    if not args.no_shuffle:
        random.shuffle(initial)

    root = tk.Tk()
    root.geometry("1500x760")
    Viewer(root, args.processed, initial, all_records)
    root.mainloop()


if __name__ == "__main__":
    main()
