#!/usr/bin/env python
"""Rasterize a PDF into per-page PNG images so the Read tool can ingest figures
it cannot read from the original PDF (size limit, complex embedded images, etc.).

Usage:
    python scripts/pdf_to_images.py <pdf_path> [--out OUT_DIR] [--dpi 150]
                                    [--pages 1-10] [--prefix page]

Output:
    OUT_DIR/<prefix>_001.png, ..., one PNG per page, 0-padded to match page count.

Defaults:
    OUT_DIR = <pdf_dir>/<pdf_stem>_img
    DPI     = 150  (good compromise: readable text, ~200-400 KB per page for
                    typical two-column papers; raise to 200 for small fonts,
                    lower to 110 if the Read tool rejects a page)

Also writes OUT_DIR/pages.txt plain-text extraction for cross-reference, when
available, so the reader can correlate page N text with page N PNG.

Keep this script small and dependency-light (PyMuPDF only, pypdf optional).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def parse_page_range(spec: str | None, n_pages: int) -> list[int]:
    if not spec:
        return list(range(n_pages))
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            lo = max(1, int(a))
            hi = min(n_pages, int(b))
            for p in range(lo, hi + 1):
                out.add(p - 1)
        else:
            p = int(chunk)
            if 1 <= p <= n_pages:
                out.add(p - 1)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to the input PDF")
    ap.add_argument("--out", default=None, help="output directory (default: <pdf>_img)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pages", default=None, help='e.g. "1-5,9,12-14"; default: all')
    ap.add_argument("--prefix", default="page")
    ap.add_argument("--skip-text", action="store_true", help="skip pages.txt extraction")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"error: {pdf_path} not found", file=sys.stderr)
        return 2

    out_dir = Path(args.out).resolve() if args.out else pdf_path.with_name(pdf_path.stem + "_img")
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    n = doc.page_count
    pages = parse_page_range(args.pages, n)
    width = max(3, len(str(n)))
    zoom = args.dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    print(f"pdf: {pdf_path}")
    print(f"pages_total: {n}; rendering: {len(pages)} @ {args.dpi} DPI")
    print(f"out_dir: {out_dir}")

    text_chunks: list[str] = []
    for idx in pages:
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_path = out_dir / f"{args.prefix}_{idx+1:0{width}d}.png"
        pix.save(png_path)
        size_kb = png_path.stat().st_size / 1024
        print(f"  {png_path.name}  {pix.width}x{pix.height}  {size_kb:.0f} KB")
        if not args.skip_text:
            try:
                text_chunks.append(f"\n===== PAGE {idx+1} =====\n{page.get_text()}")
            except Exception as e:  # noqa: BLE001
                text_chunks.append(f"\n===== PAGE {idx+1} (text extract failed: {e}) =====\n")

    if not args.skip_text:
        (out_dir / "pages.txt").write_text("".join(text_chunks), encoding="utf-8")
        print(f"  pages.txt  ({sum(len(c) for c in text_chunks)} chars)")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
