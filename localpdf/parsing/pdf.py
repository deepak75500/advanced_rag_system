"""PDF -> per-page text. PyMuPDF is imported lazily so the rest of the package works without it.

Scanned pages (little or no text layer) are OCR'd with RapidOCR (offline ONNX models) or, if that is not installed,
Tesseract. extract_pdf(ocr="auto") turns OCR on whenever an OCR engine is available.
"""
from __future__ import annotations

import io
import re
import unicodedata
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


_ENGINES: dict = {}
# RapidOCR keeps detection settings from earlier calls, so each setting gets its own engine instance.
# "loose" boxes stop ascenders/descenders being clipped (which made whole lines unreadable); the default, tighter
# boxes do better on dense pipe-separated table rows. A page is read with "loose" first and re-read with "default"
# only when confidence is low; the more confident reading wins.
_DET = {"loose": {"box_thresh": 0.3, "unclip_ratio": 2.0}, "default": {}}


def _rapidocr(kind: str = "loose"):
    """RapidOCR (pip install rapidocr_onnxruntime): ONNX models bundled in the wheel, CPU-only, no system binary."""
    if kind not in _ENGINES:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINES[kind] = RapidOCR()
    return _ENGINES[kind]


def _lines_from_boxes(result) -> str:
    """RapidOCR returns [box, text, score] per text box; rebuild reading order (top-to-bottom, then left-to-right)."""
    items = []
    for box, text, score in result or []:
        if not text.strip() or float(score) < 0.5:
            continue
        ys, xs = [pt[1] for pt in box], [pt[0] for pt in box]
        items.append((min(ys), max(ys), min(xs), text.strip()))
    items.sort(key=lambda b: (b[0], b[2]))
    lines, cur, cur_bottom = [], [], None
    for top, bottom, left, text in items:
        if cur and top > cur_bottom - 0.5 * (cur_bottom - cur[0][0]):  # starts below the middle of the current line
            lines.append(cur)
            cur = []
        cur.append((top, left, text))
        cur_bottom = bottom if len(cur) == 1 else max(cur_bottom, bottom)
    if cur:
        lines.append(cur)
    return "\n".join(" ".join(t for _, _, t in sorted(line, key=lambda x: x[1])) for line in lines)


_BAR_LOOKALIKES = str.maketrans({"丨": "|", "│": "|", "｜": "|", "¦": "|"})
_CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
_ZERO_IN_WORD = re.compile(r"\b(?=[A-Za-z0]*[A-Za-z]{2})(?=[A-Za-z0]*0)[A-Za-z0]{3,}\b")


def _normalize_ocr(text: str) -> str:
    """Undo typical recogniser confusions on Latin-script pages: full-width punctuation, CJK look-alikes of '|',
    and the digit 0 inside words ('Bo0st' -> 'Boost'). CJK text is left alone when the page is mostly CJK."""
    text = unicodedata.normalize("NFKC", text).translate(_BAR_LOOKALIKES)
    if len(_CJK.findall(text)) < 0.2 * max(1, len(re.findall(r"[A-Za-z]", text))):
        text = _CJK.sub("", text)
        text = _ZERO_IN_WORD.sub(lambda m: m.group(0).replace("0", "o"), text)
    return text


def _render(page, dpi: int) -> bytes:
    pix = page.get_pixmap(dpi=dpi)
    if pix.width * pix.height > 60_000_000:  # decompression-bomb guard
        raise ValueError("page image too large for OCR")
    return pix.tobytes("png")


def _median_box_height(result) -> float:
    hs = sorted(max(p[1] for p in box) - min(p[1] for p in box) for box, _, _ in result or [])
    return hs[len(hs) // 2] if hs else 0.0


def _mean_confidence(result) -> float:
    chars = sum(len(t) for _, t, _ in result or [])
    return sum(float(sc) * len(t) for _, t, sc in result or []) / chars if chars else 0.0


def _rapid_read(kind: str, page, dpi: int | None):
    """The recogniser is most accurate (and keeps inter-word spaces) when text lines are ~15-20 px tall. The first
    pass at 100 dpi measures line height; if it is off, the page is re-rendered once at the dpi that gives ~17 px."""
    engine, d = _rapidocr(kind), dpi or 100
    result, _ = engine(_render(page, d), **_DET[kind])
    h = _median_box_height(result)
    if dpi is None and result and not 12 <= h <= 22:
        d = max(50, min(400, round(d * 17 / h)))
        result, _ = engine(_render(page, d), **_DET[kind])
    return result or []


def _ocr_page(page, dpi: int | None = None) -> str:
    try:
        _rapidocr()
        have_rapid = True
    except ImportError:
        have_rapid = False
    if have_rapid:
        result = _rapid_read("loose", page, dpi)
        if _mean_confidence(result) < 0.9:
            alt = _rapid_read("default", page, dpi)
            if _mean_confidence(alt) > _mean_confidence(result):
                result = alt
        return _normalize_ocr(_lines_from_boxes(result))
    png = _render(page, dpi or 300)
    import pytesseract  # fallback: needs the tesseract binary on PATH
    from PIL import Image

    return pytesseract.image_to_string(Image.open(io.BytesIO(png)))


def ocr_available() -> bool:
    try:
        _rapidocr()
        return True
    except ImportError:
        pass
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


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


def extract_pdf(path: str | Path, ocr: bool | str = "auto", tables: bool = True, strip_hf: bool = True,
                ocr_min_chars: int = 30) -> ExtractedDoc:
    fitz = _fitz()
    if ocr == "auto":
        ocr = ocr_available()
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
