"""PDF -> per-page text. PyMuPDF is imported lazily so the rest of the package works without it.

UNVERIFIED IN THIS SANDBOX (PyMuPDF could not be installed here). The table / header-strip logic is adapted from the
working v3 notebook and is covered by tests in tests/test_pdf_parsing.py, which skip when PyMuPDF is absent.
Docling / PaddleOCR / Camelot are intentionally not wired in yet; see "Known limitations" in the README.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from .clean import clean_text, strip_headers_footers, table_to_markdown
from .models import ExtractedDoc, PageText


def _fitz():
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as e:
            raise ImportError("PyMuPDF is required to read PDFs: pip install pymupdf") from e
    return fitz


def _ocr_page(page, dpi: int = 200) -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    if pix.width * pix.height > 60_000_000:  # decompression-bomb guard
        raise ValueError("page image too large for OCR")
    return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))


def _page_text(fitz, p, tables: bool) -> str:
    """Ruled tables become Markdown rows (numbers keep row/column context); other pages use sorted text blocks."""
    if tables:
        try:
            tabs = [t for t in p.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
        except Exception:
            tabs = []
        if tabs:
            rects = [fitz.Rect(t.bbox) for t in tabs]
            items = []
            for b in p.get_text("blocks", sort=True):
                if b[6] != 0:
                    continue
                c = fitz.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if not any(c in r for r in rects) and b[4].strip():
                    items.append((b[1], b[4].strip()))
            for t, r in zip(tabs, rects):
                md = table_to_markdown(t.extract())
                if md:
                    items.append((r.y0, md))
            items.sort(key=lambda x: x[0])
            return "\n".join(x[1] for x in items)
    return p.get_text("text", sort=True) or ""


def extract_pdf(path: str | Path, ocr: bool = False, tables: bool = True, strip_hf: bool = True,
                ocr_min_chars: int = 30) -> ExtractedDoc:
    fitz = _fitz()
    t0 = time.perf_counter()
    doc = fitz.open(str(path))
    warnings: list[str] = []
    try:
        if doc.needs_pass:
            raise ValueError("PDF is password-protected")
        raw, flags = [], []
        for p in doc:
            t, used = _page_text(fitz, p, tables), False
            if ocr and len(t.strip()) < ocr_min_chars:
                try:
                    t, used = _ocr_page(p), True
                except Exception as e:
                    warnings.append(f"OCR unavailable/failed on page {p.number + 1}: {type(e).__name__}")
            raw.append(t)
            flags.append(used)
    finally:
        doc.close()
    if strip_hf:
        raw = strip_headers_footers(raw)
    pages = [PageText(i + 1, clean_text(t), flags[i]) for i, t in enumerate(raw)]
    empty = sum(1 for p in pages if not p.text.strip())
    if empty:
        warnings.append(f"{empty} page(s) have no extractable text (scanned? enable OCR)")
    return ExtractedDoc(pages=pages, n_pages=len(pages), ocr_pages=sum(flags),
                        seconds=time.perf_counter() - t0, warnings=warnings)
