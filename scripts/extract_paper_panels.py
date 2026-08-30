"""Crop the published figure panels from the two SHAARP manuscript PDFs into
docs/_static/paper_panels/ so the reproduction notebooks can show SHAARP.py results BESIDE the
actual paper panels (directive: comparable results against the paper, not just "looks similar").

Each figure image sits above its "Figure N." caption; we take the largest image on the figure's
page and crop its bbox at high DPI. ML Fig 3's caption flows to the top of p17, so its image is on
p16 -- handled explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
EM = ROOT.parent.parent  # directory holding the published PDFs, two levels above the repo
OUT = ROOT / "docs" / "_static" / "paper_panels"
OUT.mkdir(parents=True, exist_ok=True)

SI_PDF = EM / "Papers/Manuscript_Linear/EM Code/SHAARP.si/Manuscript_si/Submission/Arxiv/V2_SHAARP_MainText.pdf"
ML_PDF = EM / "Papers/Manuscript_Linear/EM Code/SHAARP_SLAB/Manuscript/Submission/Arxiv/SHAARP_ml_Arxiv_2.pdf"

# (tag, pdf, figure-number, page-index-holding-the-IMAGE)
JOBS = [
    ("si", SI_PDF, 4, 19), ("si", SI_PDF, 5, 21), ("si", SI_PDF, 6, 23), ("si", SI_PDF, 7, 27),
    ("ml", ML_PDF, 3, 16), ("ml", ML_PDF, 4, 21), ("ml", ML_PDF, 5, 24), ("ml", ML_PDF, 6, 27),
    ("ml", ML_PDF, 7, 31),
]


def largest_image_rect(page):
    """Return the bbox of the largest raster image on the page (the figure)."""
    best, best_area = None, 0.0
    for info in page.get_image_info():
        bb = fitz.Rect(info["bbox"])
        if bb.get_area() > best_area:
            best, best_area = bb, bb.get_area()
    return best


def main() -> None:
    for tag, pdf, fig, page_idx in JOBS:
        doc = fitz.open(pdf)
        page = doc[page_idx]
        rect = largest_image_rect(page)
        if rect is None:
            print(f"  !! {tag} Fig {fig}: no image on page {page_idx}")
            continue
        # small padding so panel labels/axes are not clipped
        pad = 4
        clip = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=220)
        out = OUT / f"{tag}_fig{fig}.png"
        pix.save(out)
        print(f"  {tag} Fig {fig}: page {page_idx} bbox {rect.width:.0f}x{rect.height:.0f} -> {out.name} "
              f"({pix.width}x{pix.height}px)")
        doc.close()


if __name__ == "__main__":
    main()
