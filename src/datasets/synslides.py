"""SynSlides (SynLecSlideGen / SynDet subset): LLM-generated synthetic lecture
slide images with automated element-detection annotations (titles, body text,
bullet lists, equations, tables, etc). Closest available stand-in for
"lecture recordings / slideshows" -- it's synthetic, not real lecture footage,
but it's slide-layout-realistic and, being AI-generated, carries no
underlying-copyright risk the way scraped real slide decks would.

License: MIT (commercial use OK). Source:
https://huggingface.co/datasets/NerdyVisky/SynSlides
Paper: https://arxiv.org/abs/2506.23605

IMPORTANT: `datasets.load_dataset()`'s default imagefolder auto-inference on
this repo only picks up the two image zips and silently drops the annotation
file, which lives elsewhere in the repo as a separate COCO-format JSON (per
the paper: "we construct a ground truth file compatible with the COCO object
detection annotation format"). So this script instead pulls the *entire*
repo with `snapshot_download`, extracts any zip files it finds, and
auto-discovers any JSON file that looks like COCO format (has "images" and
"annotations" keys) rather than hardcoding a path we can't verify from here.
Run `python -m src.datasets.synslides --check` first to confirm it finds
something before trusting it in a full build.
"""

import argparse
import json
import os
import zipfile

from PIL import Image

from src.common import Sample, box_to_polygon

HF_NAME = "NerdyVisky/SynSlides"
RAW_DIR = "data/raw/synslides"

# We don't have a verified fixed category-id scheme for this dataset (its exact
# schema wasn't inspectable from this environment), so filtering is done by
# matching category *names* against keyword lists instead of hardcoded ids.
# Anything matching an EXCLUDE keyword wins even if it also matches an INCLUDE
# keyword (e.g. "image caption" would still be excluded -- err toward excluding
# ambiguous visual-sounding categories rather than polluting the text mask).
INCLUDE_KEYWORDS = (
    "text",
    "title",
    "body",
    "bullet",
    "list",
    "caption",
    "equation",
    "footer",
    "header",
    "heading",
    "paragraph",
)
EXCLUDE_KEYWORDS = (
    "image",
    "figure",
    "chart",
    "photo",
    "logo",
    "graphic",
    "infographic",
    "icon",
    "diagram",
    "picture",
)


def _is_text_category(name: str) -> bool:
    name = (name or "").lower()
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in name for k in INCLUDE_KEYWORDS)


def download(raw_dir: str = RAW_DIR):
    from huggingface_hub import snapshot_download

    os.makedirs(raw_dir, exist_ok=True)
    print(f"Downloading full {HF_NAME} repo (~544MB)...")
    snapshot_download(repo_id=HF_NAME, repo_type="dataset", local_dir=raw_dir)

    for root, _, files in os.walk(raw_dir):
        for fn in files:
            if fn.lower().endswith(".zip"):
                zip_path = os.path.join(root, fn)
                out_dir = os.path.join(root, fn[:-4])
                if not os.path.isdir(out_dir):
                    print(f"Extracting {fn}...")
                    with zipfile.ZipFile(zip_path) as z:
                        z.extractall(out_dir)
    print(f"SynSlides ready at {raw_dir}")


def _find_coco_jsons(raw_dir: str):
    found = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                found.append(path)
    return found


def _index_images(raw_dir: str):
    idx = {}
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                idx.setdefault(fn, os.path.join(root, fn))
    return idx


def iter_samples(raw_dir: str = RAW_DIR):
    coco_jsons = _find_coco_jsons(raw_dir)
    if not coco_jsons:
        raise RuntimeError(
            f"No COCO-format annotation JSON found anywhere under {raw_dir}. "
            "Run `python -m src.datasets.synslides --download` first, or if you already "
            "have, browse https://huggingface.co/datasets/NerdyVisky/SynSlides/tree/main "
            "yourself to find the actual annotation file path/format."
        )
    img_index = _index_images(raw_dir)

    for jp in coco_jsons:
        with open(jp) as f:
            d = json.load(f)
        images_by_id = {im["id"]: im for im in d.get("images", [])}
        cat_id_to_name = {c["id"]: c.get("name", "") for c in d.get("categories", [])}
        anns_by_img = {}
        for ann in d.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        tag = os.path.splitext(os.path.basename(jp))[0]
        for img_id, im in images_by_id.items():
            fname = os.path.basename(im.get("file_name", ""))
            path = img_index.get(fname)
            if not path:
                continue
            polys = []
            for ann in anns_by_img.get(img_id, []):
                cat_name = cat_id_to_name.get(ann.get("category_id"), "")
                if cat_id_to_name and not _is_text_category(cat_name):
                    continue  # skip image/figure/chart/etc elements -- not text
                bbox = ann.get("bbox")  # COCO format: [x, y, w, h]
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    polys.append(box_to_polygon(x, y, x + w, y + h))
            with Image.open(path) as im:
                img = im.convert("RGB")
            # SynSlides boxes are whole element blocks (a title block, a body-text
            # block), not per-line -- mark coarse so save_sample refines them down
            # to actual text strokes instead of flood-filling the whole block.
            yield Sample(
                sample_id=f"{tag}_{img_id}", image=img, polygons=polys, coarse=True
            )


def _debug(raw_dir=RAW_DIR, n_samples: int = 3):
    coco_jsons = _find_coco_jsons(raw_dir)
    if not coco_jsons:
        print(f"No COCO JSON found under {raw_dir}  -- run --download first")
        return
    print(f"Found COCO JSON files: {[os.path.basename(j) for j in coco_jsons]}")
    import json

    for jp in coco_jsons[:1]:
        with open(jp) as f:
            d = json.load(f)
        cats = {c["id"]: c.get("name", "?") for c in d.get("categories", [])}
        print(f"\n  {os.path.basename(jp)}: categories={cats}")
        imgs = d.get("images", [])[:n_samples]
        anns_by_img = {}
        for ann in d.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)
        img_index = _index_images(raw_dir)
        for im in imgs:
            fname = os.path.basename(im.get("file_name", ""))
            path = img_index.get(fname)
            anns = anns_by_img.get(im["id"], [])
            cat_names = [cats.get(a.get("category_id"), "?") for a in anns]
            print(f"\n  img_id={im['id']}  file={fname}  found_on_disk={bool(path)}")
            print(f"    {len(anns)} annotations  category_names={list(set(cat_names))}")
            if anns:
                a0 = anns[0]
                print(f"    ann[0] keys: {list(a0.keys())}")
                print(f"    ann[0] bbox: {a0.get('bbox')}  (should be [x,y,w,h])")
                print(
                    f"    ann[0] category_id={a0.get('category_id')}  name={cats.get(a0.get('category_id'), '?')}"
                )
    text_cats = {i: n for i, n in cats.items() if _is_text_category(n)}
    excl_cats = {i: n for i, n in cats.items() if not _is_text_category(n)}
    print(f"\n  INCLUDED categories: {text_cats}")
    print(f"  EXCLUDED categories: {excl_cats}")
    print(
        "\nExpected: bbox present and [x,y,w,h]. INCLUDED should contain text/title/body types."
    )
    print(
        "If included/excluded split looks wrong, update INCLUDE_KEYWORDS/EXCLUDE_KEYWORDS in synslides.py."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    if args.download or args.check or args.debug:
        download()
    if args.debug:
        _debug()
    elif args.check:
        n = 0
        for s in iter_samples():
            n += 1
            if n == 1:
                print(
                    f"First sample: {s.sample_id}, {len(s.polygons)} polygons, image size {s.image.size}"
                )
            if n >= 5:
                break
        print(f"Loaded {n} SynSlides samples OK.")
