"""Small text helpers shared by retrieval, verification and metrics (no heavy dependencies)."""
from __future__ import annotations

import re

STOP = set(
    "a an the of to in on for and or is are was were be been being by with as at from that this these those it its "
    "what which who whom how when where why do does did can could should would will shall may might about into than "
    "then there their they them he she we you i not no".split()
)

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles: used for exact match, duplicate detection and F1."""
    t = re.sub(r"[^\w\s.%/-]", " ", (text or "").lower())
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # drop sentence dots, keep decimals
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", (text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) > 1 and t not in STOP]


def stem5(tok: str) -> str:
    """Cheap prefix stem used by the lexical verifier ('parameters' ~ 'parameter')."""
    return tok[:5] if not tok.replace(".", "").isdigit() else tok


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text or "") if s.strip()]


def fts_query(q: str, max_terms: int = 40) -> str:
    toks = content_tokens(q)[:max_terms]
    return " OR ".join('"' + t.replace('"', "") + '"' for t in toks)


def short_name(filename: str, n: int = 28) -> str:
    stem = re.sub(r"\.pdf$", "", filename, flags=re.I)
    return stem if len(stem) <= n else stem[: n - 1] + "…"
