"""Upload validation, safe file names and the prompt-injection guard.

PDFs are untrusted input twice over: the *file* may be malformed/hostile, and its *text* may try to instruct the model.
"""
from __future__ import annotations

import re
from pathlib import Path

MAX_FILE_MB = 200
MAX_PAGES = 3000
WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UploadRejected(ValueError):
    pass


def safe_filename(name: str) -> str:
    """Never trust a client file name: strip directories and odd characters."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^\w.\-]+", "_", base).strip("._") or "document.pdf"
    return base[:120]


def clean_collection(name: str | None) -> str:
    c = re.sub(r"[^\w\- ]", "", (name or "").strip()).strip().replace(" ", "_")[:60]
    return c or "default"


def valid_workspace_id(ws: str) -> str:
    if not WORKSPACE_RE.match(ws or ""):
        raise UploadRejected("invalid workspace id")
    return ws


def validate_upload(path: str | Path, max_mb: int = MAX_FILE_MB, max_pages: int = MAX_PAGES) -> None:
    """Cheap, parser-free checks first (size, magic bytes), then structural checks with PyMuPDF if installed."""
    p = Path(path)
    if not p.is_file():
        raise UploadRejected("file not found")
    size = p.stat().st_size
    if size == 0:
        raise UploadRejected("empty file")
    if size > max_mb * 1024 * 1024:
        raise UploadRejected(f"file larger than {max_mb} MB")
    with open(p, "rb") as f:
        head = f.read(1024)
    if b"%PDF-" not in head:
        raise UploadRejected("not a PDF (missing %PDF header)")
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError:
            return  # structural checks need PyMuPDF; ingestion will fail loudly later without it
    try:
        doc = fitz.open(str(p))
    except Exception as e:  # malformed / truncated / hostile structure
        raise UploadRejected(f"cannot open PDF: {type(e).__name__}") from e
    try:
        if doc.needs_pass:
            raise UploadRejected("password-protected PDF")
        if len(doc) > max_pages:
            raise UploadRejected(f"more than {max_pages} pages")
        if len(doc) == 0:
            raise UploadRejected("PDF has no pages")
    finally:
        doc.close()


# ---------------------------------------------------------------- prompt-injection guard
INJECTION_RE = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b[^.\n]{0,30}"
    r"\b(instructions?|prompts?|rules|guidelines)\b"
    r"|\b(reveal|print|show|repeat|output|display)\b[^.\n]{0,15}\byour\s+(system\s+|hidden\s+|initial\s+)?(prompt|instructions?)\b"
    r"|\bnew\s+instructions?\s*:"
    r"|\byou\s+must\s+now\s+(act|behave|respond)\b"
    r"|</?(excerpt|system|assistant)\b", re.I)


def is_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text or ""))


def sanitize_hits(hits: list) -> tuple[list, int]:
    """Drop retrieved passages that try to instruct the model. Returns (clean_hits, n_dropped)."""
    clean = [h for h in hits if not is_injection(h.context) and not is_injection(h.text)]
    return clean, len(hits) - len(clean)


def neutralise(text: str) -> str:
    """Stop document text from spoofing our citation tags or closing our excerpt wrapper."""
    t = re.sub(r"\[\s*#", "[ #", text)
    return t.replace("</excerpt>", "< /excerpt>")
