"""Text clean-up: de-hyphenation, whitespace, running headers/footers and page numbers."""
from __future__ import annotations

import re


def clean_text(t: str) -> str:
    t = re.sub(r"-\n(?=[a-z])", "", t)                 # de-hyphenate line breaks
    t = re.sub(r"(?<![\n|])\n(?!\n)(?!\|)", " ", t)    # single newline -> space (table rows keep their own line)
    t = re.sub(r"[ \t  ]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def _norm_line(l: str) -> str:
    return re.sub(r"\d+", "#", l.strip().lower())      # "Page 3 of 10" == "Page 4 of 10"


def strip_headers_footers(raw_pages: list[str], min_pages: int = 4, ratio: float = 0.4) -> list[str]:
    """Remove short lines at page edges that repeat on many pages (running headers, footers, page numbers)."""
    n = len(raw_pages)
    if n < min_pages:
        return raw_pages
    split = [t.split("\n") for t in raw_pages]
    counts: dict[str, int] = {}
    for lines in split:
        nz = [l for l in lines if l.strip()]
        for k in {_norm_line(l) for l in nz[:3] + nz[-3:] if len(l.strip()) <= 120}:
            counts[k] = counts.get(k, 0) + 1
    bad = {k for k, v in counts.items() if v >= max(3, ratio * n)}
    if not bad:
        return raw_pages
    out = []
    for lines in split:
        nz_idx = [i for i, l in enumerate(lines) if l.strip()]
        edge = set(nz_idx[:3] + nz_idx[-3:])
        out.append("\n".join(l for i, l in enumerate(lines) if not (i in edge and _norm_line(l) in bad)))
    return out


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Rows -> Markdown table, one row per line, so numbers keep their row/column context."""
    rows = [[re.sub(r"(\w) -(?=\w)", r"\1-", (c or "").replace("\n", " ")).strip() for c in r]
            for r in rows if r and any((c or "").strip() for c in r)]
    if len(rows) < 2:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join("---" for _ in range(w)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)
