"""Page-aware parent/child chunking.

* Children (~chunk_chars) are what gets embedded and keyword-indexed: small, precise matches.
* Parents (~parent_chars) are what the LLM reads: more surrounding context per hit.
* Neither crosses a page boundary, so every chunk has one exact page number to cite.
* Markdown table rows are atomic pieces, and the table header row is re-attached to every chunk that starts with a
  data row (including on the following pages of a multi-page table), so numbers keep their column names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..util import split_sentences
from .models import PageText

_SEP_RE = re.compile(r"^\|\s*:?-{3,}")


@dataclass
class Child:
    text: str
    page: int
    idx: int = 0


@dataclass
class Parent:
    page: int
    idx: int
    text: str
    children: list[Child] = field(default_factory=list)


def _join(pieces: list[str]) -> str:
    return "\n".join(pieces) if any(p.startswith("|") for p in pieces) else " ".join(pieces)


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [line] if line.startswith("|") else split_sentences(line) or [line]
        for s in parts:
            if len(s) > max_chars and not s.startswith("|"):
                out += [s[i:i + max_chars] for i in range(0, len(s), max_chars)]
            else:
                out.append(s)
    return out


def _group(pieces: list[str], max_chars: int, overlap_chars: int = 0) -> list[list[str]]:
    groups, cur, n = [], [], 0
    for p in pieces:
        if n + len(p) > max_chars and cur:
            groups.append(cur)
            keep, m = [], 0
            for x in reversed(cur):  # sentence-level overlap (never for table rows)
                if x.startswith("|") or m + len(x) > overlap_chars:
                    break
                keep.insert(0, x)
                m += len(x)
            cur, n = keep, m
        cur.append(p)
        n += len(p) + 1
    if cur:
        groups.append(cur)
    return groups


def chunk_pages(pages: list[PageText], chunk_chars: int = 900, overlap: int = 150, parent_chars: int = 2400,
                min_chars: int = 25) -> list[Parent]:
    parents: list[Parent] = []
    header: tuple[str, str] | None = None  # (header row, separator row) of the table currently open
    for pg in pages:
        pieces = _pieces(pg.text, chunk_chars)
        if not pieces:
            continue
        if not pieces[0].startswith("|"):
            header = None  # a table cannot continue across a page that starts with prose
        for pi, pgroup in enumerate(_group(pieces, parent_chars, 0)):
            parent = Parent(page=pg.page, idx=len(parents), text=_join(pgroup))
            for cgroup in _group(pgroup, chunk_chars, overlap):
                for i, p in enumerate(cgroup):  # remember the most recent table header
                    if p.startswith("|") and not _SEP_RE.match(p) and i + 1 < len(cgroup) and _SEP_RE.match(cgroup[i + 1]):
                        header = (p, cgroup[i + 1])
                text = _join(cgroup)
                first = text.split("\n", 1)[0]
                if header and first.startswith("|") and first != header[0]:
                    text = (header[0] + "\n" + text) if _SEP_RE.match(first) else (header[0] + "\n" + header[1] + "\n" + text)
                if len(text) >= min_chars:
                    parent.children.append(Child(text=text, page=pg.page, idx=0))
            if parent.children:
                parents.append(parent)
    n = 0
    for p in parents:
        for c in p.children:
            c.idx = n
            n += 1
    return parents
