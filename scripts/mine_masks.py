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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import (
    MAX_IMAGE_SIDE,
    MASK_SCALE,
    JPEG_QUALITY,
    resize_keep_aspect,
    downscale_mask,
    mask_contour_stats,
)
from scripts.ocr_backend import get_reader, boxes_from_image, polys_to_mask

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


def source_arxiv(
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
):
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
            pdf_url = paper.pdf_url
            r = requests.get(pdf_url, timeout=30)
            if r.status_code != 200:
                continue
            pdf_bytes = r.content
            doc = pypdfium2.PdfDocument(pdf_bytes)
            n_pages = len(doc)
            # sample up to 3 random pages per paper
            pages = random.sample(range(n_pages), min(3, n_pages))
            for page_idx in pages:
                if yielded >= n:
                    break
                page = doc[page_idx]
                # render at 150 DPI equivalent (scale=2.0 gives ~144 DPI from 72 DPI base)
                bitmap = page.render(scale=2.0)
                pil_img = bitmap.to_pil().convert("RGB")
                ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
                sample_id = f"arxiv_{paper.entry_id.split('/')[-1]}_p{page_idx}"
                _save(
                    pil_img,
                    ocr_polys,
                    out_dir,
                    "mined_arxiv",
                    sample_id,
                    meta_rows,
                    dry_run,
                )
                yielded += 1
            doc.close()
        except Exception as e:
            print(f"  [arxiv] error: {e}")
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
        pages = r.json().get("query", {}).get("random", [])
        return [p["title"] for p in pages]
    except Exception:
        return []


def source_wikipedia(
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
    langs: list = None,
):
    """Screenshot random Wikipedia pages at varied scroll positions."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "[wikipedia] playwright not installed. Run: pip install playwright && playwright install chromium"
        )
        return

    wiki_langs = langs or ["en", "de", "fr", "ja", "ko", "zh", "ru", "hi", "ar"]
    print(f"\n[wikipedia] Screenshotting {n} random Wikipedia pages...")
    yielded = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        while yielded < n:
            lang = random.choice(wiki_langs)
            titles = _random_wiki_titles(lang=lang, n=5)
            for title in titles:
                if yielded >= n:
                    break
                url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                try:
                    page.goto(url, timeout=15000)
                    page.wait_for_timeout(1000)
                    # scroll to random position on the page
                    scroll_height = page.evaluate("document.body.scrollHeight")
                    max_scroll = max(0, scroll_height - 900)
                    scroll_y = random.randint(0, max_scroll)
                    page.evaluate(f"window.scrollTo(0, {scroll_y})")
                    page.wait_for_timeout(300)

                    img_bytes = page.screenshot(type="png")
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
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
                    yielded += 1
                except Exception as e:
                    print(f"  [wikipedia] {title}: {e}")

        browser.close()
    print(f"[wikipedia] done: {yielded} screenshots")


# ── Source: YouTube frames ────────────────────────────────────────────────────


def source_youtube(
    n: int,
    reader,
    backend: str,
    min_conf: float,
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
                            ocr_polys = boxes_from_image(
                                reader, pil_img, backend, min_conf
                            )
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


def source_pubmed(
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
):
    """Open-access PDFs from PubMed Central (OA subset, no login)."""
    import pypdfium2

    # PMC FTP open-access PDF list sampled via NCBI E-utilities (no API key for basic use)
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    topics = [
        "machine learning",
        "radiology imaging",
        "climate",
        "economics",
        "protein structure",
        "language model",
        "surgery",
        "ecology",
    ]
    print(f"\n[pubmed] Fetching {n} page images from PubMed Central OA papers...")
    yielded = 0
    for topic in topics * 5:
        if yielded >= n:
            break
        try:
            r = requests.get(
                ESEARCH,
                params={
                    "db": "pmc",
                    "term": f"{topic}[Title] AND open+access[filter]",
                    "retmax": 10,
                    "retmode": "json",
                    "tool": "ocr_miner",
                    "email": "miner@example.com",
                },
                timeout=10,
            )
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            pmc_id = random.choice(ids)
            # fetch PDF link
            fetch_r = requests.get(
                EFETCH,
                params={
                    "db": "pmc",
                    "id": pmc_id,
                    "rettype": "pdf",
                    "retmode": "asn1",
                    "tool": "ocr_miner",
                    "email": "miner@example.com",
                },
                timeout=5,
            )
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
            pdf_r = requests.get(
                pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
            )
            if pdf_r.status_code != 200 or not pdf_r.content[:4] == b"%PDF":
                continue
            doc = pypdfium2.PdfDocument(pdf_r.content)
            pages = random.sample(range(len(doc)), min(2, len(doc)))
            for pg in pages:
                if yielded >= n:
                    break
                bitmap = doc[pg].render(scale=2.0)
                pil_img = bitmap.to_pil().convert("RGB")
                ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
                _save(
                    pil_img,
                    ocr_polys,
                    out_dir,
                    "mined_pubmed",
                    f"pmc{pmc_id}_p{pg}",
                    meta_rows,
                    dry_run,
                )
                yielded += 1
            doc.close()
        except Exception as e:
            print(f"  [pubmed] {e}")
        time.sleep(0.5)  # be polite to NCBI
    print(f"[pubmed] done: {yielded} pages")


# ── Source: Project Gutenberg ─────────────────────────────────────────────────


def source_gutenberg(
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
):
    """Render random pages of Project Gutenberg books as images."""
    import textwrap

    CATALOG_URL = "https://gutendex.com/books/?languages=en,de,fr,es,pt&mime_type=text"
    print(f"\n[gutenberg] Rendering {n} book-page images from Project Gutenberg...")
    yielded = 0
    page_num = random.randint(1, 50)
    try:
        cat = requests.get(f"{CATALOG_URL}&page={page_num}", timeout=15).json()
        books = cat.get("results", [])
    except Exception as e:
        print(f"  [gutenberg] catalog fetch failed: {e}")
        return

    random.shuffle(books)
    for book in books * 5:
        if yielded >= n:
            break
        try:
            txt_url = next(
                (
                    v
                    for k, v in book.get("formats", {}).items()
                    if "plain" in k and "zip" not in k
                ),
                None,
            )
            if not txt_url:
                continue
            r = requests.get(txt_url, timeout=20)
            if r.status_code != 200:
                continue
            raw = r.text
            # split into chunks of ~500 chars (roughly a page)
            chunks = [raw[i : i + 500] for i in range(0, min(len(raw), 50000), 500)]
            random.shuffle(chunks)
            for chunk in chunks[:4]:
                if yielded >= n:
                    break
                pil_img = _render_text_page(chunk.strip(), width=900, height=1200)
                if pil_img is None:
                    continue
                ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
                book_id = str(book.get("id", ""))
                _save(
                    pil_img,
                    ocr_polys,
                    out_dir,
                    "mined_gutenberg",
                    f"gut{book_id}_{yielded:04d}",
                    meta_rows,
                    dry_run,
                )
                yielded += 1
        except Exception as e:
            print(f"  [gutenberg] {e}")
    print(f"[gutenberg] done: {yielded} pages")


# ── Source: Synthetic ─────────────────────────────────────────────────────────


def _render_text_page(
    text: str,
    width: int = 900,
    height: int = 1200,
    font_path: str = None,
    font_size: int = None,
    bg_color=None,
    text_color=None,
) -> Image.Image | None:
    """Render a page of text onto a plain background using Pillow."""
    import textwrap

    bg = bg_color or (random.randint(230, 255),) * 3
    tc = text_color or (random.randint(0, 40),) * 3
    fs = font_size or random.randint(14, 28)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, fs)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    margin = random.randint(40, 120)
    line_spacing = int(fs * random.uniform(1.2, 1.8))
    chars_per_line = max(10, (width - 2 * margin) // max(fs // 2, 1))
    wrapped = textwrap.fill(text, width=chars_per_line)
    draw.text((margin, margin), wrapped, fill=tc, font=font)
    return img


def _try_system_fonts() -> list[str]:
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
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
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

            pil_img = _render_text_page(
                text, w, h, font_path, font_size, bg_rgb, text_rgb
            )
            if pil_img is None:
                continue
            ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
            _save(
                pil_img,
                ocr_polys,
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
    n: int,
    reader,
    backend: str,
    min_conf: float,
    out_dir: Path,
    meta_rows: list,
    dry_run: bool,
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
                        bitmap = doc[pg].render(scale=2.0)
                        pil_img = bitmap.to_pil().convert("RGB")
                        ocr_polys = boxes_from_image(reader, pil_img, backend, min_conf)
                        _save(
                            pil_img,
                            ocr_polys,
                            out_dir,
                            "mined_openalex",
                            f"oa{work_id}_p{pg}",
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


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_SOURCES = [
    "arxiv",
    "wikipedia",
    "youtube",
    "synthetic",
    "pubmed",
    "gutenberg",
    "openalex",
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
    ap.add_argument("--backend", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument(
        "--youtube-urls",
        nargs="*",
        default=[],
        help="YouTube channel or video URLs for --source youtube",
    )
    ap.add_argument(
        "--wiki-langs",
        nargs="*",
        default=["en", "de", "fr", "ja", "ko", "zh", "ru", "hi", "ar"],
        help="Wikipedia language codes to sample",
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

    print(f"Initialising OCR backend ({args.backend}, gpu={args.gpu})...")
    reader = get_reader(backend=args.backend, gpu=args.gpu)

    meta_rows: list = []

    dispatch = {
        "arxiv": lambda: source_arxiv(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
        ),
        "wikipedia": lambda: source_wikipedia(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
            langs=args.wiki_langs,
        ),
        "youtube": lambda: source_youtube(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
            youtube_urls=args.youtube_urls,
        ),
        "synthetic": lambda: source_synthetic(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
        ),
        "pubmed": lambda: source_pubmed(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
        ),
        "gutenberg": lambda: source_gutenberg(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
        ),
        "openalex": lambda: source_openalex(
            args.n,
            reader,
            args.backend,
            args.min_conf,
            out_dir,
            meta_rows,
            args.dry_run,
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
