"""scripts/mine_masks.py
Self-contained data-mining engine: fetches diverse unlabeled images from
multiple free online sources, runs EasyOCR to derive text masks, and writes
output in the unified processed format (same as build_dataset.py).

Sources (all free, no login/API key required unless noted):
  --source arxiv        Random arXiv papers rendered as page images (PDF)
  --source wikipedia    Random Wikipedia article screenshots
  --source youtube      Video frame samples from provided channel/video URLs
  --source synthetic    Synthetic text images (varied fonts, sizes, layouts)
  --source pubmed       PubMed Central open-access PDFs (dense scientific docs)
  --source gutenberg    Project Gutenberg plain-text pages rendered as images

Usage:
    pip install arxiv pypdfium2 wikipedia-api playwright yt-dlp faker
    playwright install chromium   # one-time: downloads headless browser

    # mine 200 images from each source
    python -m scripts.mine_masks --sources arxiv wikipedia synthetic --n 200

    # YouTube: pass channel or video URLs
    python -m scripts.mine_masks --sources youtube --n 200 \\
        --youtube-urls https://www.youtube.com/@3blue1brown \\
                       https://www.youtube.com/@lexfridman

    # all sources at once
    python -m scripts.mine_masks --sources all --n 100 \\
        --youtube-urls https://www.youtube.com/@3blue1brown

    # dry run (OCR report, no output written)
    python -m scripts.mine_masks --sources arxiv --n 10 --dry-run
"""

from __future__ import annotations
import argparse
import io
import json
import os
import random
import string
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import (
    MAX_IMAGE_SIDE,
    MASK_SCALE,
    JPEG_QUALITY,
    resize_keep_aspect,
    downscale_mask,
    mask_contour_stats,
)
from scripts.ocr_backend import get_reader, _detect_raw, polys_to_mask, _resize_for_ocr

# ── helpers ───────────────────────────────────────────────────────────────────


def _save(
    img: Image.Image,
    ocr_polys: list,
    out_dir: Path,
    dataset_name: str,
    sample_id: str,
    meta_rows: list,
    dry_run: bool,
) -> None:
    """Resize, mask, write to disk, append meta row."""
    orig_w, orig_h = img.size
    resized, scale = resize_keep_aspect(img, MAX_IMAGE_SIDE)
    rw, rh = resized.size

    full_mask = polys_to_mask(ocr_polys, (rw, rh))
    area_frac = float((full_mask > 0).sum()) / full_mask.size

    if dry_run:
        print(
            f"  [{sample_id}] {orig_w}x{orig_h} -> {rw}x{rh}  "
            f"words={len(ocr_polys)}  text_area={area_frac:.1%}"
        )
        return

    if not ocr_polys:
        return  # skip blank pages

    small_mask = downscale_mask(full_mask, MASK_SCALE)
    mh, mw = small_mask.shape[:2]
    num_contours, _ = mask_contour_stats(full_mask)

    img_dir = out_dir / dataset_name / "images"
    mask_dir = out_dir / dataset_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    safe_id = sample_id.replace("/", "_").replace("\\", "_")[:80]
    img_rel = f"images/{safe_id}.jpg"
    mask_rel = f"masks/{safe_id}.png"

    resized.convert("RGB").save(
        str(out_dir / dataset_name / img_rel), quality=JPEG_QUALITY
    )
    Image.fromarray(small_mask).save(str(out_dir / dataset_name / mask_rel))

    meta_rows.append(
        {
            "id": f"{dataset_name}_{safe_id}",
            "dataset": dataset_name,
            "image_path": img_rel,
            "mask_path": mask_rel,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "resized_width": rw,
            "resized_height": rh,
            "mask_width": mw,
            "mask_height": mh,
            "mask_scale": MASK_SCALE,
            "num_polygons": len(ocr_polys),
            "text_area_frac": area_frac,
            "num_contours": num_contours,
            "coarse_source": False,
        }
    )


# ── Source: arXiv ─────────────────────────────────────────────────────────────


def _random_arxiv_queries():
    """Rotate through diverse arXiv categories to get visual variety."""
    cats = [
        "cs.CV",
        "cs.LG",
        "cs.CL",
        "cs.AI",
        "cs.RO",
        "math.ST",
        "physics.med-ph",
        "q-bio.NC",
        "econ.GN",
        "stat.ML",
        "cs.HC",
        "cs.SE",
        "cs.SY",
        "astro-ph.GA",
    ]
    while True:
        yield random.choice(cats)


def _pdf_page_to_image_and_mask(page, scale: float = 2.0):
    """Render a pypdfium2 page to an image and build an exact text mask from
    the PDF text layer using per-character bounding boxes.

    Returns (pil_image, mask_array) or (pil_image, None) if no text layer.
    mask_array is uint8 0/255, same size as pil_image.
    """
    import numpy as np

    bitmap = page.render(scale=scale)
    pil_img = bitmap.to_pil().convert("RGB")
    img_w, img_h = pil_img.size

    try:
        textpage = page.get_textpage()
        n_chars = textpage.count_chars()
        if n_chars == 0:
            textpage.close()
            return pil_img, None

        pdf_w = page.get_width()
        pdf_h = page.get_height()
        sx = img_w / pdf_w
        sy = img_h / pdf_h

        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # Use get_charbox() per character -- simpler and more reliable than
        # count_rects/get_rect. Returns (left, bottom, right, top) in PDF
        # canvas units (origin bottom-left, y increases upward).
        for i in range(n_chars):
            try:
                # loose=True gives a slightly larger box covering the full
                # font cell, which means adjacent characters merge into
                # connected regions -- better for segmentation masks
                l, b, r, t = textpage.get_charbox(i, loose=True)
            except Exception:
                continue
            if r <= l or t <= b:
                continue
            # Convert PDF coords (bottom-left origin) → pixel coords (top-left origin)
            x0 = max(0, int(l * sx))
            x1 = min(img_w, int(r * sx) + 1)
            y0 = max(0, int((pdf_h - t) * sy))
            y1 = min(img_h, int((pdf_h - b) * sy) + 1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255

        textpage.close()

        if mask.max() == 0:
            return pil_img, None
        return pil_img, mask

    except Exception as e:
        return pil_img, None


def source_arxiv(
    n: int, reader, text_threshold: float, out_dir: Path, meta_rows: list, dry_run: bool
):
    """Fetch random arXiv PDFs and extract text masks from the PDF text layer.
    No OCR needed -- PDFs have embedded character bounding boxes."""
    import arxiv, pypdfium2

    print(f"\n[arxiv] Fetching {n} page images from random arXiv papers...")
    client = arxiv.Client()
    yielded = 0
    for cat in _random_arxiv_queries():
        if yielded >= n:
            break
        try:
            results = list(
                client.results(
                    arxiv.Search(
                        query=f"cat:{cat}",
                        max_results=3,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                    )
                )
            )
            if not results:
                continue
            paper = random.choice(results)
            r = requests.get(paper.pdf_url, timeout=30)
            if r.status_code != 200:
                continue
            doc = pypdfium2.PdfDocument(r.content)
            n_pages = len(doc)
            pages = random.sample(range(n_pages), min(3, n_pages))
            for page_idx in pages:
                if yielded >= n:
                    break
                page = doc[page_idx]
                pil_img, mask = _pdf_page_to_image_and_mask(page, scale=2.0)
                sample_id = f"arxiv_{paper.entry_id.split('/')[-1]}_p{page_idx}"
                if mask is not None:
                    # exact mask from PDF text layer
                    _save_with_mask(
                        pil_img,
                        mask,
                        out_dir,
                        "mined_arxiv",
                        sample_id,
                        meta_rows,
                        dry_run,
                    )
                    nc = int((mask > 0).sum() / mask.size * 100)
                    print(
                        f"  [{yielded+1}/{n}] {sample_id} ✓ {nc}% text (PDF layer)",
                        flush=True,
                    )
                else:
                    # fallback to CRAFT for scanned/image-only PDFs
                    arr, sc = _resize_for_ocr(pil_img)
                    ocr_polys = []
                    for tt, lt, lk in [
                        (0.3, 0.3, 0.3),
                        (0.2, 0.2, 0.2),
                        (0.1, 0.1, 0.15),
                    ]:
                        ocr_polys, _ = _detect_raw(arr, sc, reader, tt, lt, lk)
                        if len(ocr_polys) >= 5:
                            break
                    del arr
                    _save(
                        pil_img,
                        ocr_polys,
                        out_dir,
                        "mined_arxiv",
                        sample_id,
                        meta_rows,
                        dry_run,
                    )
                    print(
                        f"  [{yielded+1}/{n}] {sample_id} ✓ {len(ocr_polys)} boxes (CRAFT fallback)",
                        flush=True,
                    )
                yielded += 1
            doc.close()
        except Exception as e:
            print(f"  [arxiv] error: {e}", flush=True)
            time.sleep(1)
    print(f"[arxiv] done: {yielded} pages")


# ── Source: Wikipedia ─────────────────────────────────────────────────────────


def _random_wiki_titles(lang: str = "en", n: int = 50):
    """Fetch n random Wikipedia article titles."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(n, 20),
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("random", [])
        return [p["title"] for p in pages]
    except Exception as e:
        print(f"({e})", end=" ", flush=True)
        return []


def source_wikipedia(
    n: int,
    reader,
    text_threshold: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
    langs: list = None,
):
    """Screenshot random Wikipedia pages at varied scroll positions."""
    try:
        from playwright.sync_api import sync_playwright, Error as PlaywrightError
    except ImportError:
        print(
            "[wikipedia] playwright not installed. Run: pip install playwright && playwright install chromium"
        )
        return

    wiki_langs = langs or ["en", "de", "fr", "ja", "ko", "zh", "ru", "hi", "ar"]

    with sync_playwright() as p:
        print("[wikipedia] Launching Chromium...", flush=True)
        try:
            browser = p.chromium.launch(headless=True, timeout=30000)
        except Exception as e:
            print(f"[wikipedia] Failed to launch Chromium: {e}")
            print("  Make sure you ran: python -m playwright install chromium")
            return

        print(
            f"[wikipedia] Ready. Screenshotting {n} pages "
            f"({len(wiki_langs)} languages in pool)...",
            flush=True,
        )

        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        yielded = 0

        yielded = 0
        attempts = 0
        max_attempts = n * 5

        while yielded < n and attempts < max_attempts:
            lang = random.choice(wiki_langs)
            attempts += 1
            # Navigate directly to Special:Random — no separate API call needed,
            # Wikipedia redirects the browser to a random article automatically.
            random_url = f"https://{lang}.wikipedia.org/wiki/Special:Random"
            try:
                print(f"  [{yielded+1}/{n}] {lang}: navigating...", end=" ", flush=True)
                page.goto(random_url, timeout=15000, wait_until="domcontentloaded")
                page.wait_for_timeout(600)

                # get the actual article title from the final URL after redirect
                final_url = page.url
                from urllib.parse import unquote

                title = unquote(final_url.split("/wiki/")[-1]).replace("_", " ")[:50]
                print(f"{title}", end=" ", flush=True)

                # Hide UI chrome (sidebars, nav, settings panels) and maximise
                # text contrast before screenshotting. This dramatically helps
                # CRAFT detect dense CJK/Indic text on rendered web pages.
                page.evaluate("""() => {
                    // hide sidebar panels, nav, footer, floating elements
                    const hide = [
                        '.vector-page-toolbar', '#mw-navigation', '#footer',
                        '.mw-portlet', '#p-tb', '#p-lang', '.vector-column-end',
                        '.mw-table-of-contents-container', '#vector-toc-collapsed-button',
                        '.vector-appearance-landmark', '#vector-appearance',
                        '#p-views', '.vector-sticky-header', '#siteNotice',
                        '.mw-indicators', '#catlinks',
                    ];
                    hide.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => el.style.display = 'none');
                    });
                    // force black-on-white, remove background images
                    document.body.style.cssText += ';background:#fff!important;color:#000!important;';
                    document.querySelectorAll('a').forEach(a => a.style.color = '#000');
                    document.querySelectorAll('img, figure, .thumb').forEach(img => img.style.display = 'none');
                }""")
                page.wait_for_timeout(200)

                scroll_height = page.evaluate("document.body.scrollHeight")
                scroll_y = random.randint(0, max(0, scroll_height - 900))
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                page.wait_for_timeout(200)

                img_bytes = page.screenshot(type="png")
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                # For web screenshots, detect at full resolution (1280x900 is
                # already reasonable) — downscaling to 1024 makes rendered
                # body text (~11px) too small for CRAFT to resolve reliably.
                arr = np.array(pil_img)
                scale = 1.0
                # Progressive threshold fallback: start aggressive, retry lower
                # if too few boxes found. RTL (Farsi/Arabic/Hebrew) and many
                # Indic scripts score lower in CRAFT's confidence map because
                # the model was pretrained mostly on Latin/CJK scene text.
                for tt, lt, lk in [(0.2, 0.2, 0.3), (0.1, 0.1, 0.2), (0.05, 0.05, 0.1)]:
                    ocr_polys, _ = _detect_raw(arr, scale, reader, tt, lt, lk)
                    if len(ocr_polys) >= 5:
                        break
                del arr

                safe_title = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in title
                )[:40]
                sample_id = f"wiki_{lang}_{safe_title}_{scroll_y}"
                _save(
                    pil_img,
                    ocr_polys,
                    out_dir,
                    "mined_wikipedia",
                    sample_id,
                    meta_rows,
                    dry_run,
                )
                print(f"✓ {len(ocr_polys)} boxes", flush=True)
                yielded += 1
            except Exception as e:
                print(f"✗ {e}", flush=True)

        if attempts >= max_attempts and yielded < n:
            print(
                f"[wikipedia] gave up after {attempts} attempts ({yielded} collected)"
            )

        browser.close()
    print(f"[wikipedia] done: {yielded} screenshots")


# ── Source: YouTube frames ────────────────────────────────────────────────────


def source_youtube(
    n: int,
    reader,
    text_threshold: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
    youtube_urls: list = None,
):
    """Sample frames from YouTube videos using yt-dlp (no API key needed)."""
    try:
        import yt_dlp
    except ImportError:
        print("[youtube] yt-dlp not installed. Run: pip install yt-dlp")
        return
    import subprocess, shutil

    if not youtube_urls:
        print("[youtube] No --youtube-urls provided, skipping.")
        return
    if not shutil.which("ffmpeg"):
        print("[youtube] ffmpeg not found in PATH. Install ffmpeg to extract frames.")
        return

    print(f"\n[youtube] Mining {n} frames from {len(youtube_urls)} URL(s)...")
    yielded = 0
    frames_per_video = max(1, n // max(len(youtube_urls), 1))

    for url in youtube_urls:
        if yielded >= n:
            break
        try:
            # resolve channel/playlist to individual video URLs
            ydl_opts_list = {
                "quiet": True,
                "extract_flat": True,
                "playlistend": 20,
                "ignoreerrors": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
                info = ydl.extract_info(url, download=False)
            entries = info.get("entries") or [info]
            videos = [
                e.get("url") or e.get("webpage_url") or e.get("id")
                for e in entries
                if e
            ]
            videos = [v for v in videos if v]
            random.shuffle(videos)

            for vid_url in videos:
                if yielded >= n:
                    break
                with tempfile.TemporaryDirectory() as tmpdir:
                    frame_path = os.path.join(tmpdir, "frame_%04d.jpg")
                    # download a 30-second random clip and extract frames at 1fps
                    ydl_opts = {
                        "quiet": True,
                        "outtmpl": os.path.join(tmpdir, "vid.%(ext)s"),
                        "format": "bestvideo[height<=480][ext=mp4]/best[height<=480]",
                        "external_downloader": "ffmpeg",
                        "external_downloader_args": {"ffmpeg_i": ["-t", "30"]},
                        "ignoreerrors": True,
                    }
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([vid_url])
                        vid_files = [
                            f for f in os.listdir(tmpdir) if f.startswith("vid.")
                        ]
                        if not vid_files:
                            continue
                        vid_file = os.path.join(tmpdir, vid_files[0])
                        # extract 1 frame per second
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-i",
                                vid_file,
                                "-vf",
                                "fps=1",
                                frame_path,
                                "-loglevel",
                                "error",
                            ],
                            check=True,
                            capture_output=True,
                        )
                        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
                        random.shuffle(frame_files)
                        vid_id = (
                            vid_url.split("=")[-1][:12]
                            if "=" in vid_url
                            else vid_url[-12:]
                        )
                        for i, f in enumerate(frame_files[:frames_per_video]):
                            if yielded >= n:
                                break
                            pil_img = Image.open(f).convert("RGB")
                            arr, scale = _resize_for_ocr(pil_img)
                            ocr_polys = []
                            for tt, lt, lk in [
                                (0.3, 0.3, 0.3),
                                (0.2, 0.2, 0.2),
                                (0.1, 0.1, 0.15),
                            ]:
                                ocr_polys, _ = _detect_raw(
                                    arr, scale, reader, tt, lt, lk
                                )
                                if len(ocr_polys) >= 5:
                                    break
                            del arr
                            sample_id = f"yt_{vid_id}_{i:04d}"
                            _save(
                                pil_img,
                                ocr_polys,
                                out_dir,
                                "mined_youtube",
                                sample_id,
                                meta_rows,
                                dry_run,
                            )
                            yielded += 1
                    except Exception as e:
                        print(f"  [youtube] {vid_url}: {e}")
        except Exception as e:
            print(f"  [youtube] {url}: {e}")

    print(f"[youtube] done: {yielded} frames")


# ── Source: PubMed Central ────────────────────────────────────────────────────


def _europepmc_search(topic: str, page_size: int = 10) -> list[dict]:
    """Search Europe PMC, returning articles with direct PDF URLs.
    Uses resultType=core to get fullTextUrlList in one call."""
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": (
                    f"{topic} OPEN_ACCESS:Y HAS_PDF:Y "
                    f"PUB_TYPE:Journal Article FIRST_PDATE:[2010-01-01 TO 2023-12-31]"
                ),
                "format": "json",
                "pageSize": page_size,
                "resultType": "core",
                "cursorMark": "*",
            },
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        results = r.json().get("resultList", {}).get("result", [])
        out = []
        for res in results:
            pmcid = res.get("pmcid")
            if not pmcid:
                continue
            # Extract PDF URL from fullTextUrlList
            pdf_url = None
            for item in (res.get("fullTextUrlList") or {}).get("fullTextUrl", []):
                if (
                    item.get("documentStyle") == "pdf"
                    and item.get("availabilityCode") == "OA"
                ):
                    pdf_url = item.get("url")
                    break
            if pdf_url:
                out.append({"pmcid": pmcid, "pdf_url": pdf_url})
        return out
    except Exception as e:
        print(f"    (europepmc search error: {e})", flush=True)
        return []


def source_pubmed(
    n: int, reader, text_threshold: float, out_dir: Path, meta_rows: list, dry_run: bool
):
    """Open-access PDFs via Europe PMC REST API (no login, no API key needed).
    Uses resultType=core to get direct PDF download URLs in one call."""
    import pypdfium2

    topics = [
        "machine learning medicine",
        "radiology imaging",
        "climate change ecology",
        "genomics sequencing",
        "epidemiology infectious disease",
        "surgery outcomes",
        "neuroscience brain",
        "cancer treatment",
        "pharmacology drug",
        "pediatrics children",
        "cardiology heart",
        "diabetes metabolism",
    ]
    print(f"\n[pubmed] Fetching {n} page images via Europe PMC OA PDFs...")
    yielded = 0

    for topic in topics * 5:
        if yielded >= n:
            break
        print(f"  [pubmed] searching: {topic}...", end=" ", flush=True)
        articles = _europepmc_search(topic, page_size=8)
        print(f"{len(articles)} articles with PDFs", flush=True)
        random.shuffle(articles)

        for art in articles:
            if yielded >= n:
                break
            pmcid = art["pmcid"]
            pdf_url = art["pdf_url"]
            print(f"    {pmcid}...", end=" ", flush=True)

            try:
                pdf_r = requests.get(
                    pdf_url,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0"},
                    allow_redirects=True,
                )
                if pdf_r.status_code != 200 or pdf_r.content[:4] != b"%PDF":
                    print(f"bad response ({pdf_r.status_code})", flush=True)
                    continue

                doc = pypdfium2.PdfDocument(pdf_r.content)
                pages = random.sample(range(len(doc)), min(2, len(doc)))
                for pg in pages:
                    if yielded >= n:
                        break
                    pil_img, mask = _pdf_page_to_image_and_mask(doc[pg], scale=2.0)
                    sample_id = f"{pmcid}_p{pg}"
                    if mask is not None:
                        _save_with_mask(
                            pil_img,
                            mask,
                            out_dir,
                            "mined_pubmed",
                            sample_id,
                            meta_rows,
                            dry_run,
                        )
                        nc = int((mask > 0).sum() / mask.size * 100)
                        print(f"✓ {nc}% text", flush=True)
                    else:
                        arr, sc = _resize_for_ocr(pil_img)
                        ocr_polys = []
                        for tt, lt, lk in [
                            (0.3, 0.3, 0.3),
                            (0.2, 0.2, 0.2),
                            (0.1, 0.1, 0.15),
                        ]:
                            ocr_polys, _ = _detect_raw(arr, sc, reader, tt, lt, lk)
                            if len(ocr_polys) >= 5:
                                break
                        del arr
                        _save(
                            pil_img,
                            ocr_polys,
                            out_dir,
                            "mined_pubmed",
                            sample_id,
                            meta_rows,
                            dry_run,
                        )
                        print(f"✓ {len(ocr_polys)} boxes (CRAFT)", flush=True)
                    yielded += 1
                doc.close()
                time.sleep(0.3)

            except Exception as e:
                print(f"error: {e}", flush=True)

        time.sleep(0.5)

    print(f"[pubmed] done: {yielded} pages")


# ── Source: Project Gutenberg ─────────────────────────────────────────────────

# A curated list of well-known, always-available PG book IDs spanning
# multiple languages. All are out of copyright and freely downloadable
# from gutenberg.org without rate limits or API keys.
_GUTENBERG_BOOK_IDS = [
    # English
    1342,
    11,
    1661,
    98,
    2701,
    1952,
    84,
    1400,
    2600,
    174,
    5200,
    43,
    1232,
    1184,
    76,
    2554,
    4300,
    766,
    2148,
    345,
    2542,
    844,
    1260,
    # French
    13951,
    17489,
    4650,
    3400,
    2650,
    14155,
    # German
    2591,
    5765,
    146,
    2229,
    22367,
    # Spanish
    2000,
    15728,
    14420,
    # Portuguese
    996,
    3678,
    # Italian
    24736,
    4823,
    # Finnish
    7000,
    11302,
    # Dutch
    11024,
    13715,
    # Chinese (Simplified & Traditional) -- stored as UTF-8
    23950,  # 水滸傳 (Water Margin)
    24264,  # 三國演義 (Romance of the Three Kingdoms)
    25421,  # 西遊記 (Journey to the West)
    23962,  # 紅樓夢 (Dream of the Red Chamber)
    # Japanese -- stored as UTF-8
    56935,  # 吾輩は猫である (I Am a Cat) - Natsume Soseki
    57865,  # 坊っちゃん (Botchan) - Natsume Soseki
    # Hungarian
    36034,
    40583,
    # Russian (transliterated/Latin-script editions available)
    2600,  # War and Peace (already above as English translation)
    # Swedish
    4028,
    7170,
    # Czech
    37616,
]


def _fetch_gutenberg_text(book_id: int) -> str | None:
    """Try the two common URL patterns for a PG plain-text file."""
    for url in [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except Exception:
            continue
    return None


def source_gutenberg(
    n: int, reader, text_threshold: float, out_dir: Path, meta_rows: list, dry_run: bool
):
    """Render random pages of Project Gutenberg books as images.
    Uses direct gutenberg.org URLs — no third-party API needed."""
    print(f"\n[gutenberg] Rendering {n} book-page images from Project Gutenberg...")
    yielded = 0
    book_ids = list(_GUTENBERG_BOOK_IDS)
    random.shuffle(book_ids)

    for book_id in book_ids * 3:
        if yielded >= n:
            break
        print(f"  [gutenberg] fetching book {book_id}...", end=" ", flush=True)
        raw = _fetch_gutenberg_text(book_id)
        if not raw:
            print("not found, skipping", flush=True)
            continue
        print(f"{len(raw)//1000}k chars", flush=True)

        # strip PG header/footer boilerplate, take content from the middle
        stripped = raw[3000 : min(len(raw) - 3000, 200000)]
        chunks = [stripped[i : i + 2000] for i in range(0, len(stripped), 2000)]
        random.shuffle(chunks)

        for chunk in chunks[: max(1, n - yielded + 2)]:
            if yielded >= n:
                break
            chunk = chunk.strip()
            if len(chunk) < 100:
                continue
            pil_img, exact_mask = _render_text_page(chunk, width=900, height=1200)
            if pil_img is None:
                continue
            _save_with_mask(
                pil_img,
                exact_mask,
                out_dir,
                "mined_gutenberg",
                f"gut{book_id}_{yielded:04d}",
                meta_rows,
                dry_run,
            )
            nc = int((exact_mask > 0).sum() / max(exact_mask.size, 1) * 100)
            print(
                f"  [gutenberg] page {yielded+1}/{n}: book {book_id} ✓ {nc}% text",
                flush=True,
            )
            yielded += 1

    print(f"[gutenberg] done: {yielded} pages")


# ── Source: Synthetic ─────────────────────────────────────────────────────────


def _try_system_fonts() -> list[str]:
    """Return a list of available TTF/OTF/TTC font paths on the current system."""
    import glob

    candidates = []
    search_paths = [
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.ttc",
        "/usr/share/fonts/**/*.otf",
        "/usr/local/share/fonts/**/*.ttf",
        "/usr/local/share/fonts/**/*.ttc",
        os.path.expanduser("~/.fonts/**/*.ttf"),
        os.path.expanduser("~/.fonts/**/*.ttc"),
        "C:/Windows/Fonts/*.ttf",
        "C:/Windows/Fonts/*.otf",
        "C:/Windows/Fonts/*.ttc",  # CJK fonts on Windows are mostly .ttc
        "/System/Library/Fonts/**/*.ttf",
        "/System/Library/Fonts/**/*.ttc",
        "/Library/Fonts/**/*.ttf",
        "/Library/Fonts/**/*.ttc",
    ]
    for pattern in search_paths:
        candidates.extend(glob.glob(pattern, recursive=True))
    return candidates


def _render_text_page(
    text: str,
    width: int = 900,
    height: int = 1200,
    font_path: str = None,
    font_size: int = None,
    bg_color=None,
    text_color=None,
):
    """Render a page of text that fills the canvas using Pillow.

    Returns (pil_image, mask_array) where mask_array is a binary 0/255 uint8
    array with text pixels marked. This is exact -- no CRAFT needed for
    synthetic text since we know exactly where we drew each line.
    """
    import textwrap
    import cv2

    bg = bg_color or (random.randint(248, 255),) * 3
    tc = text_color or (random.randint(0, 20),) * 3
    fs = font_size or random.randint(16, 26)
    margin_x = random.randint(50, 100)
    margin_y = random.randint(50, 80)
    usable_w = width - 2 * margin_x
    usable_h = height - 2 * margin_y

    # Pick a random font
    all_fonts = _try_system_fonts()
    font = None
    if all_fonts:
        candidates = (
            [font_path]
            if font_path
            else random.sample(all_fonts, min(len(all_fonts), 10))
        )
        for fp in candidates:
            if not fp:
                continue
            try:
                font = ImageFont.truetype(fp, fs)
                break
            except Exception:
                try:
                    font = ImageFont.truetype(fp, fs, index=0)  # for .ttc collections
                    break
                except Exception:
                    continue

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    if font is None:
        try:
            font = ImageFont.load_default(size=fs)
        except TypeError:
            font = ImageFont.load_default()

    # Measure character width and line height with actual font
    try:
        bbox = draw.textbbox((0, 0), "x" * 40, font=font)
        avg_char_w = (bbox[2] - bbox[0]) / 40
    except Exception:
        avg_char_w = fs * 0.55

    chars_per_line = max(10, int(usable_w / max(avg_char_w, 1)))

    try:
        lbbox = draw.textbbox((0, 0), "Ay", font=font)
        line_h = int((lbbox[3] - lbbox[1]) * random.uniform(1.3, 1.7))
    except Exception:
        line_h = int(fs * 1.5)

    if line_h < 1:
        line_h = max(int(fs * 1.4), 1)

    lines = textwrap.wrap(text, width=chars_per_line)
    repeat_limit = 10
    while len(lines) * line_h < usable_h * 0.7 and repeat_limit > 0:
        lines = lines + textwrap.wrap(text, width=chars_per_line)
        repeat_limit -= 1

    # Draw lines and record exact bounding boxes for the mask
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    y = margin_y
    for line in lines:
        if y + line_h > height - margin_y:
            break
        draw.text((margin_x, y), line, fill=tc, font=font)
        # Record tight bounding box of this line in the mask
        try:
            tb = draw.textbbox((margin_x, y), line, font=font)
            x0, y0, x1, y1 = int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])
            # small padding to catch descenders/ascenders
            pad = max(2, fs // 8)
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(width, x1 + pad)
            y1 = min(height, y1 + pad)
            mask[y0:y1, x0:x1] = 255
        except Exception:
            # fallback: mark the whole line strip
            mask[
                max(0, y) : min(height, y + line_h), margin_x : margin_x + usable_w
            ] = 255
        y += line_h

    return img, mask


def _save_with_mask(
    img: Image.Image,
    mask_arr,
    out_dir: Path,
    dataset_name: str,
    sample_id: str,
    meta_rows: list,
    dry_run: bool,
) -> None:
    """Like _save() but takes a precomputed binary mask array instead of OCR polys."""
    import cv2
    import numpy as np
    from src.common import (
        resize_keep_aspect,
        downscale_mask,
        mask_contour_stats,
        MASK_SCALE,
        JPEG_QUALITY,
    )

    orig_w, orig_h = img.size
    resized, scale = resize_keep_aspect(img, MAX_IMAGE_SIDE)
    rw, rh = resized.size

    # Resize the precomputed mask to match the resized image
    full_mask = cv2.resize(
        mask_arr.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST
    )
    area_frac = float((full_mask > 0).sum()) / float(full_mask.size)

    if dry_run:
        print(
            f"  [{sample_id}] {orig_w}x{orig_h} -> {rw}x{rh}  text_area={area_frac:.1%}"
        )
        return

    small_mask = downscale_mask(full_mask, MASK_SCALE)
    mh, mw = small_mask.shape[:2]
    num_contours, _ = mask_contour_stats(full_mask)

    safe_id = sample_id.replace("/", "_").replace("\\", "_")[:80]
    img_rel = f"images/{safe_id}.jpg"
    mask_rel = f"masks/{safe_id}.png"

    img_dir = out_dir / dataset_name / "images"
    mask_dir = out_dir / dataset_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    resized.convert("RGB").save(
        str(out_dir / dataset_name / img_rel), quality=JPEG_QUALITY
    )
    Image.fromarray(small_mask).save(str(out_dir / dataset_name / mask_rel))

    meta_rows.append(
        {
            "id": f"{dataset_name}_{safe_id}",
            "dataset": dataset_name,
            "image_path": img_rel,
            "mask_path": mask_rel,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "resized_width": rw,
            "resized_height": rh,
            "mask_width": mw,
            "mask_height": mh,
            "mask_scale": MASK_SCALE,
            "num_polygons": num_contours,  # contours as proxy for "boxes"
            "text_area_frac": area_frac,
            "num_contours": num_contours,
            "coarse_source": False,
        }
    )
    """Return a list of available TTF font paths on the current system."""
    import glob

    candidates = []
    search_paths = [
        "/usr/share/fonts/**/*.ttf",
        "/usr/local/share/fonts/**/*.ttf",
        os.path.expanduser("~/.fonts/**/*.ttf"),
        "C:/Windows/Fonts/*.ttf",
        "C:/Windows/Fonts/*.otf",
        "/System/Library/Fonts/**/*.ttf",
        "/Library/Fonts/**/*.ttf",
    ]
    for pattern in search_paths:
        candidates.extend(glob.glob(pattern, recursive=True))
    return candidates


def source_synthetic(
    n: int, reader, text_threshold: float, out_dir: Path, meta_rows: list, dry_run: bool
):
    """Generate synthetic text images with varied fonts, sizes, and layouts."""
    try:
        from faker import Faker
    except ImportError:
        print("[synthetic] faker not installed. Run: pip install faker")
        return

    print(f"\n[synthetic] Generating {n} synthetic text images...")
    fakers = {
        "en": Faker("en_US"),
        "de": Faker("de_DE"),
        "fr": Faker("fr_FR"),
        "es": Faker("es_ES"),
        "ja": Faker("ja_JP"),
        "ko": Faker("ko_KR"),
        "zh": Faker("zh_CN"),
        "ru": Faker("ru_RU"),
        "ar": Faker("ar_AA"),
        "hi": Faker("hi_IN"),
    }
    fonts = _try_system_fonts() or [None]
    layouts = ["paragraph", "multicolumn", "heading+body", "list", "table_like"]
    yielded = 0

    while yielded < n:
        try:
            faker_lang, fk = random.choice(list(fakers.items()))
            layout = random.choice(layouts)
            w = random.choice([640, 800, 1024, 1280])
            h = random.choice([480, 600, 800, 1024])
            bg_rgb = tuple(random.randint(220, 255) for _ in range(3))
            text_rgb = tuple(random.randint(0, 40) for _ in range(3))
            font_path = random.choice(fonts)
            font_size = random.randint(12, 36)

            if layout == "paragraph":
                text = "\n\n".join(fk.paragraph() for _ in range(random.randint(2, 5)))
            elif layout == "heading+body":
                text = (
                    fk.sentence().upper()
                    + "\n\n"
                    + "\n\n".join(fk.paragraph() for _ in range(3))
                )
            elif layout == "list":
                items = [f"• {fk.sentence()}" for _ in range(random.randint(4, 10))]
                text = fk.sentence().upper() + "\n\n" + "\n".join(items)
            elif layout == "multicolumn":
                text = (
                    "  |  ".join(fk.word() for _ in range(8))
                    + "\n"
                    + "\n".join(fk.sentence() for _ in range(6))
                )
            else:  # table_like
                rows = [[fk.word() for _ in range(4)] for _ in range(6)]
                text = "\n".join("  |  ".join(r) for r in rows)

            pil_img, exact_mask = _render_text_page(
                text, w, h, font_path, font_size, bg_rgb, text_rgb
            )
            if pil_img is None:
                continue
            _save_with_mask(
                pil_img,
                exact_mask,
                out_dir,
                "mined_synthetic",
                f"syn_{faker_lang}_{layout}_{yielded:05d}",
                meta_rows,
                dry_run,
            )
            yielded += 1
        except Exception as e:
            print(f"  [synthetic] {e}")
    print(f"[synthetic] done: {yielded} images")


# ── Source: OpenAlex / Semantic Scholar PDFs ──────────────────────────────────


def source_openalex(
    n: int, reader, text_threshold: float, out_dir: Path, meta_rows: list, dry_run: bool
):
    """Open-access PDFs from OpenAlex (no API key needed for basic use).
    Great diversity: STEM, humanities, social sciences, multilingual."""
    import pypdfium2

    print(f"\n[openalex] Fetching {n} page images from OpenAlex OA papers...")
    concepts = [
        "artificial intelligence",
        "education",
        "climate change",
        "medicine",
        "history",
        "economics",
        "chemistry",
        "law",
    ]
    yielded = 0
    for concept in concepts * 10:
        if yielded >= n:
            break
        try:
            r = requests.get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"title.search:{concept},open_access.is_oa:true,has_fulltext:true",
                    "select": "id,title,open_access",
                    "per-page": 5,
                    "sample": 5,
                    "mailto": "miner@example.com",
                },
                timeout=10,
            )
            works = r.json().get("results", [])
            for work in works:
                if yielded >= n:
                    break
                pdf_url = (work.get("open_access") or {}).get("oa_url")
                if not pdf_url or not pdf_url.endswith(".pdf"):
                    continue
                try:
                    pdf_r = requests.get(
                        pdf_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if pdf_r.status_code != 200:
                        continue
                    content = pdf_r.content
                    if not content[:4] == b"%PDF":
                        continue
                    doc = pypdfium2.PdfDocument(content)
                    pages = random.sample(range(len(doc)), min(2, len(doc)))
                    work_id = work["id"].split("/")[-1]
                    for pg in pages:
                        if yielded >= n:
                            break
                        pil_img, mask = _pdf_page_to_image_and_mask(doc[pg], scale=2.0)
                        sample_id = f"oa{work_id}_p{pg}"
                        if mask is not None:
                            _save_with_mask(
                                pil_img,
                                mask,
                                out_dir,
                                "mined_openalex",
                                sample_id,
                                meta_rows,
                                dry_run,
                            )
                        else:
                            arr, sc = _resize_for_ocr(pil_img)
                            ocr_polys = []
                            for tt, lt, lk in [
                                (0.3, 0.3, 0.3),
                                (0.2, 0.2, 0.2),
                                (0.1, 0.1, 0.15),
                            ]:
                                ocr_polys, _ = _detect_raw(arr, sc, reader, tt, lt, lk)
                                if len(ocr_polys) >= 5:
                                    break
                            del arr
                            _save(
                                pil_img,
                                ocr_polys,
                                out_dir,
                                "mined_openalex",
                                sample_id,
                                meta_rows,
                                dry_run,
                            )
                        yielded += 1
                    doc.close()
                except Exception as e:
                    print(f"  [openalex] pdf error: {e}")
        except Exception as e:
            print(f"  [openalex] {e}")
        time.sleep(0.2)
    print(f"[openalex] done: {yielded} pages")


# ── Source: local folder ──────────────────────────────────────────────────────

_LOCAL_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".avif",
    ".heif",
    ".heic",
    ".jfif",
}


def source_local(
    n: int,
    reader,
    text_threshold: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
    local_dir: str = None,
    dataset_name: str = "mined_local",
    recursive: bool = True,
):
    """Mine masks from a local folder of images you manually sourced.

    Walks `local_dir`, picks up to `n` images (shuffled), runs CRAFT on each
    to detect text, and writes to data/processed/<dataset_name>/ in the
    same unified format as every other source.

    Use this for:
      - Screenshots you took yourself
      - Photos of lecture boards, signage, receipts, etc.
      - Any images you have rights to use but that don't come from an automated source
    """
    if not local_dir:
        print("[local] --local-dir not specified, skipping.")
        return

    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"[local] directory not found: {local_dir}")
        return

    # find all images
    if recursive:
        all_imgs = [p for p in local_path.rglob("*") if p.suffix.lower() in _LOCAL_EXTS]
    else:
        all_imgs = [p for p in local_path.iterdir() if p.suffix.lower() in _LOCAL_EXTS]

    if not all_imgs:
        print(f"[local] no images found in {local_dir}")
        return

    random.shuffle(all_imgs)
    to_process = all_imgs[:n]
    print(
        f"\n[local] Processing {len(to_process)} images from {local_dir} "
        f"(found {len(all_imgs)} total, dataset_name='{dataset_name}')..."
    )

    yielded = 0
    for img_path in to_process:
        try:
            print(
                f"  [{yielded+1}/{len(to_process)}] {img_path.name}...",
                end=" ",
                flush=True,
            )
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)  # handle EXIF rotation for phone photos
            arr, scale = _resize_for_ocr(img)
            ocr_polys = []
            for tt, lt, lk in [(0.3, 0.3, 0.3), (0.2, 0.2, 0.2), (0.1, 0.1, 0.15)]:
                ocr_polys, _ = _detect_raw(arr, scale, reader, tt, lt, lk)
                if len(ocr_polys) >= 3:
                    break
            del arr

            sample_id = img_path.stem[:60].replace(" ", "_")
            _save(img, ocr_polys, out_dir, dataset_name, sample_id, meta_rows, dry_run)
            print(f"✓ {len(ocr_polys)} boxes", flush=True)
            yielded += 1
        except Exception as e:
            print(f"✗ {e}", flush=True)

    print(f"[local] done: {yielded} images")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_SOURCES = [
    "arxiv",
    "wikipedia",
    "youtube",
    "synthetic",
    "pubmed",
    "gutenberg",
    "openalex",
    "local",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv", "synthetic"],
        choices=ALL_SOURCES + ["all"],
        help="which sources to mine (default: arxiv synthetic)",
    )
    ap.add_argument(
        "--n", type=int, default=100, help="target images PER SOURCE (default: 100)"
    )
    ap.add_argument(
        "--out-dir",
        default="data/processed",
        help="output root (default: data/processed)",
    )
    # backend is now always easyocr detector-only (see ocr_backend.py)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument(
        "--text-threshold",
        type=float,
        default=0.7,
        help="CRAFT text detection confidence (default 0.7, lower = more recall)",
    )
    ap.add_argument(
        "--youtube-urls",
        nargs="*",
        default=[],
        help="YouTube channel or video URLs for --source youtube",
    )
    ap.add_argument(
        "--wiki-langs",
        nargs="*",
        default=[
            # European
            "en",
            "de",
            "fr",
            "es",
            "pt",
            "it",
            "nl",
            "pl",
            "sv",
            "da",
            "no",
            "fi",
            "cs",
            "sk",
            "ro",
            "hu",
            "el",
            "bg",
            "hr",
            "sr",
            "uk",
            "ru",
            "be",
            "lt",
            "lv",
            "et",
            "ca",
            "eu",
            "gl",
            "oc",
            # Indian
            "hi",
            "bn",
            "te",
            "mr",
            "ta",
            "ur",
            "gu",
            "kn",
            "ml",
            "pa",
            "or",
            "as",
            "sa",
            "ne",
            "si",
            # CJK
            "zh",
            "ja",
            "ko",
            # Arabic / Persian / other widely-spoken scripts
            "ar",
            "fa",
            "id",
            "ms",
            "vi",
            "th",
        ],
        help="Wikipedia language codes to sample (default: broad multilingual set)",
    )
    ap.add_argument(
        "--local-dir",
        default=None,
        help="path to a local folder of images for --sources local",
    )
    ap.add_argument(
        "--dataset-name",
        default="mined_local",
        help="dataset name used as subfolder for --sources local (default: mined_local)",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="don't recurse into subdirectories for --sources local",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run OCR and print stats without writing output",
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    sources = ALL_SOURCES if "all" in args.sources else args.sources
    out_dir = Path(args.out_dir)

    print(f"Initialising CRAFT detector (detector-only, gpu={args.gpu})...")
    reader = get_reader(gpu=args.gpu)

    meta_rows: list = []

    dispatch = {
        "arxiv": lambda: source_arxiv(
            args.n, reader, args.text_threshold, out_dir, meta_rows, args.dry_run
        ),
        "wikipedia": lambda: source_wikipedia(
            args.n,
            reader,
            args.text_threshold,
            out_dir,
            meta_rows,
            args.dry_run,
            langs=args.wiki_langs,
        ),
        "youtube": lambda: source_youtube(
            args.n,
            reader,
            args.text_threshold,
            out_dir,
            meta_rows,
            args.dry_run,
            youtube_urls=args.youtube_urls,
        ),
        "synthetic": lambda: source_synthetic(
            args.n, reader, args.text_threshold, out_dir, meta_rows, args.dry_run
        ),
        "pubmed": lambda: source_pubmed(
            args.n, reader, args.text_threshold, out_dir, meta_rows, args.dry_run
        ),
        "gutenberg": lambda: source_gutenberg(
            args.n, reader, args.text_threshold, out_dir, meta_rows, args.dry_run
        ),
        "openalex": lambda: source_openalex(
            args.n, reader, args.text_threshold, out_dir, meta_rows, args.dry_run
        ),
        "local": lambda: source_local(
            args.n,
            reader,
            args.text_threshold,
            out_dir,
            meta_rows,
            args.dry_run,
            local_dir=args.local_dir,
            dataset_name=args.dataset_name,
            recursive=not args.no_recursive,
        ),
    }

    for src in sources:
        dispatch[src]()

    if args.dry_run:
        print(f"\nDry run complete. Would have written {len(meta_rows)} samples.")
        return

    if not meta_rows:
        print("\nNo samples collected.")
        return

    # group by dataset and write per-dataset meta.jsonl
    by_dataset: dict[str, list] = {}
    for row in meta_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for ds_name, rows in by_dataset.items():
        meta_path = out_dir / ds_name / "meta.jsonl"
        with open(meta_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # append to global manifest
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(meta_rows)} total samples written.")
    for ds_name, rows in by_dataset.items():
        print(f"  {ds_name}: {len(rows)} samples")
    print(f"  Manifest appended: {manifest_path}")


if __name__ == "__main__":
    main()
