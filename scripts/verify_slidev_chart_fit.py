#!/usr/bin/env python3
"""Verify chart images fit inside Slidev slides (no clip/overflow)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRES = ROOT / "presentation"
DIST = PRES / "dist"
SLIDES_MD = PRES / "slides.md"


def chart_slide_numbers() -> list[int]:
    """1-based slide numbers with class chart-slide."""
    import re

    text = SLIDES_MD.read_text(encoding="utf-8")
    # Drop deck frontmatter (first --- ... --- block).
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)
    chunks = re.split(r"\n---\s*\n", text)
    nums: list[int] = []
    slide_no = 0
    i = 0
    while i < len(chunks):
        chunk = chunks[i].strip()
        if not chunk:
            i += 1
            continue
        # Slide frontmatter block: layout/class lines only, body follows after next ---.
        if chunk.startswith("layout:") or chunk.startswith("class:"):
            header = chunk
            body = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
            i += 2
        else:
            header = ""
            body = chunk
            i += 1
        slide_no += 1
        if "class: chart-slide" in header:
            nums.append(slide_no)
    return nums


def _wait_server(url: str, timeout: float = 30.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.4)
    raise RuntimeError(f"Server not ready: {url}")


def verify(base_url: str, tolerance_px: float = 2.0) -> list[str]:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    chart_slides = chart_slide_numbers()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        for n in chart_slides:
            page.goto(f"{base_url}/{n}", wait_until="networkidle")
            page.wait_for_timeout(500)
            slide = None
            pages = page.locator(".slidev-page")
            for i in range(pages.count()):
                candidate = pages.nth(i)
                if candidate.is_visible():
                    slide = candidate
                    break
            if slide is None:
                errors.append(f"Slide {n}: no visible .slidev-page")
                continue
            img = slide.locator(".chart-frame img").first
            if not img.count():
                errors.append(f"Slide {n}: no chart image on active slide")
                continue
            sb = page.locator("#slide-content").first.bounding_box()
            ib = img.bounding_box()
            if not sb or not ib:
                errors.append(f"Slide {n}: could not measure boxes")
                continue
            for edge, val, limit in [
                ("left", ib["x"], sb["x"]),
                ("top", ib["y"], sb["y"]),
                ("right", ib["x"] + ib["width"], sb["x"] + sb["width"]),
                ("bottom", ib["y"] + ib["height"], sb["y"] + sb["height"]),
            ]:
                if edge in ("left", "top") and val < limit - tolerance_px:
                    errors.append(f"Slide {n}: image {edge} {val:.1f} < slide {limit:.1f}")
                if edge in ("right", "bottom") and val > limit + tolerance_px:
                    errors.append(f"Slide {n}: image {edge} {val:.1f} > slide {limit:.1f}")
        browser.close()
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--port", type=int, default=3031)
    parser.add_argument("--serve", action="store_true", help="Build and serve dist locally")
    args = parser.parse_args()

    base_url = (args.url or f"http://127.0.0.1:{args.port}").rstrip("/")
    proc = None
    if args.serve:
        if not DIST.is_dir():
            subprocess.run(["npm", "run", "build"], cwd=PRES, check=True, shell=True)
        proc = subprocess.Popen(
            f"npx --yes serve dist -l {args.port}",
            cwd=PRES,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_server(base_url, timeout=90.0)

    try:
        errors = verify(base_url)
    finally:
        if proc:
            proc.terminate()

    if errors:
        print("FAIL chart fit check:")
        for e in errors:
            print(" -", e)
        return 1
    print(f"OK all {len(chart_slide_numbers())} chart slides fit inside canvas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
