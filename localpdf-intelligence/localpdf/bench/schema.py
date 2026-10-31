"""Benchmark dataset schema, loading, validation against the real corpus, and fingerprinting.

Ground truth is never invented by the engine: items come from the dataset author, and `validate_against_corpus`
refuses items whose supporting evidence cannot be found on the stated page of the stated PDF.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..security import safe_filename
from ..util import content_tokens, normalize

CATEGORIES = [
    "factual", "multi_hop", "cross_page", "table", "numerical", "math", "ocr", "long_document", "ambiguous",
    "contradictory", "unanswerable", "citation", "context_retention", "hallucination", "comparison",
]
DIFFICULTIES = ["easy", "medium", "hard"]
UNANSWERABLE_CATEGORIES = {"unanswerable", "hallucination"}


@dataclass
class GTPage:
    pdf: str
    page: int

    def key(self) -> tuple[str, int]:
        return (self.pdf, self.page)


@dataclass
class BenchItem:
    id: str
    question: str
    expected_answer: str
    category: str
    difficulty: str = "medium"
    gt_pages: list[GTPage] = field(default_factory=list)      # ground-truth (pdf, page) pairs; empty if unanswerable
    evidence: list[str] = field(default_factory=list)         # verbatim passages supporting the answer
    answerable: bool = True
    aliases: list[str] = field(default_factory=list)          # other acceptable answers (for exact match)
    history: list[tuple[str, str]] = field(default_factory=list)   # earlier turns for context-retention items
    forbidden_terms: list[str] = field(default_factory=list)  # unanswerable items: terms that must not occur in the corpus
    ocr_reference: str | None = None                          # ground-truth page text for OCR character-error-rate items
    requires: list[str] = field(default_factory=list)         # e.g. ["ocr"]: skipped (not failed) if unavailable
    split: str | None = None                                  # "dev" | "test": configuration selection uses dev only
    mode: str = "answer"

    @property
    def pdfs(self) -> list[str]:
        return sorted({g.pdf for g in self.gt_pages})


def _norm_gt(raw: dict) -> list[GTPage]:
    """Accept the flat spec format (supporting_pdf + ground_truth_pages) or explicit [{pdf, page}] pairs."""
    pages = raw.get("ground_truth_pages", raw.get("gt_pages", []))
    pdf = raw.get("supporting_pdf")
    out: list[GTPage] = []
    for p in pages or []:
        if isinstance(p, dict):
            out.append(GTPage(safe_filename(p["pdf"]), int(p["page"])))
        else:
            if isinstance(pdf, list):
                raise ValueError("with several supporting PDFs give ground_truth_pages as [{'pdf':..,'page':..}]")
            out.append(GTPage(safe_filename(pdf), int(p)))
    return out


def item_from_dict(raw: dict) -> BenchItem:
    ev = raw.get("supporting_evidence", raw.get("evidence", []))
    it = BenchItem(
        id=str(raw["id"]), question=raw["question"], expected_answer=raw["expected_answer"], category=raw["category"],
        difficulty=raw.get("difficulty", "medium"), gt_pages=_norm_gt(raw),
        evidence=[ev] if isinstance(ev, str) else list(ev or []), answerable=bool(raw.get("answerable", True)),
        aliases=list(raw.get("aliases", [])), history=[tuple(h) for h in raw.get("history", [])],
        forbidden_terms=list(raw.get("forbidden_terms", [])), ocr_reference=raw.get("ocr_reference"),
        requires=list(raw.get("requires", [])), split=raw.get("split"), mode=raw.get("mode", "answer"))
    if not it.answerable or it.category in UNANSWERABLE_CATEGORIES and not it.gt_pages:
        it.answerable = False
    return it


def item_to_dict(it: BenchItem) -> dict:
    return {"id": it.id, "question": it.question, "expected_answer": it.expected_answer, "category": it.category,
            "difficulty": it.difficulty, "ground_truth_pages": [{"pdf": g.pdf, "page": g.page} for g in it.gt_pages],
            "supporting_pdf": it.pdfs, "supporting_evidence": it.evidence, "answerable": it.answerable,
            "aliases": it.aliases, "history": [list(h) for h in it.history], "forbidden_terms": it.forbidden_terms,
            "ocr_reference": it.ocr_reference, "requires": it.requires, "split": it.split, "mode": it.mode}


class Dataset:
    def __init__(self, items: list[BenchItem], name: str = "dataset"):
        self.items, self.name = items, name

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        p = Path(path)
        return cls.from_text(p.read_text(encoding="utf-8"), p.stem)

    @classmethod
    def from_text(cls, text: str, name: str = "dataset") -> "Dataset":
        """Parse JSONL; raises ValueError listing every bad line (used by the upload endpoint)."""
        items, errors = [], []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(item_from_dict(json.loads(line)))
            except Exception as e:
                errors.append(f"line {n}: {type(e).__name__}: {e}")
        if errors:
            raise ValueError("invalid benchmark dataset:\n" + "\n".join(errors[:20]))
        ds = cls(items, name)
        problems = ds.check_structure()
        if problems:
            raise ValueError("invalid benchmark dataset:\n" + "\n".join(problems[:20]))
        return ds

    def save(self, path: str | Path):
        Path(path).write_text("\n".join(json.dumps(item_to_dict(i), ensure_ascii=False) for i in self.items) + "\n",
                              encoding="utf-8")

    def sha256(self) -> str:
        blob = "\n".join(json.dumps(item_to_dict(i), sort_keys=True, ensure_ascii=False) for i in self.items)
        return hashlib.sha256(blob.encode()).hexdigest()

    def check_structure(self) -> list[str]:
        out, ids = [], set()
        for it in self.items:
            if it.id in ids:
                out.append(f"{it.id}: duplicate id")
            ids.add(it.id)
            if it.category not in CATEGORIES:
                out.append(f"{it.id}: unknown category {it.category!r}")
            if it.difficulty not in DIFFICULTIES:
                out.append(f"{it.id}: unknown difficulty {it.difficulty!r}")
            if it.answerable and not it.gt_pages and "ocr" not in it.requires:
                out.append(f"{it.id}: answerable item needs ground-truth page(s)")
            if it.answerable and not it.evidence and not it.ocr_reference:
                out.append(f"{it.id}: answerable item needs supporting evidence")
            if not it.answerable and it.gt_pages:
                out.append(f"{it.id}: unanswerable item must not list ground-truth pages")
        return out

    def validate_against_corpus(self, corpus: dict[str, list[str]]) -> list[str]:
        """corpus: {pdf filename: [page 1 text, page 2 text, ...]} -> list of problems (empty == ground truth is real).

        * every ground-truth (pdf, page) must exist
        * every evidence passage must occur (normalised) on at least one ground-truth page
        * unanswerable items: none of their forbidden terms may occur anywhere in the corpus
        """
        problems = []
        norm_pages = {safe_filename(k): [normalize(t) for t in v] for k, v in corpus.items()}
        everything = " ".join(t for v in norm_pages.values() for t in v)
        for it in self.items:
            bad_gt = False
            for g in it.gt_pages:
                if g.pdf not in norm_pages:
                    problems.append(f"{it.id}: supporting PDF {g.pdf!r} not in corpus")
                    bad_gt = True
                elif not 1 <= g.page <= len(norm_pages[g.pdf]):
                    problems.append(f"{it.id}: page {g.page} out of range for {g.pdf} ({len(norm_pages[g.pdf])} pages)")
                    bad_gt = True
            if it.answerable and not bad_gt:
                gt_text = " ".join(norm_pages.get(g.pdf, [""])[g.page - 1] for g in it.gt_pages
                                   if g.pdf in norm_pages and 1 <= g.page <= len(norm_pages[g.pdf]))
                for ev in it.evidence:
                    if normalize(ev) not in gt_text:
                        problems.append(f"{it.id}: evidence not found on the stated page(s): {ev[:60]!r}")
            else:
                for t in it.forbidden_terms:
                    if normalize(t) in everything:
                        problems.append(f"{it.id}: unanswerable, but forbidden term {t!r} occurs in the corpus")
        return problems

    def summary(self) -> dict:
        by_cat = {c: 0 for c in CATEGORIES}
        by_diff = {d: 0 for d in DIFFICULTIES}
        for it in self.items:
            by_cat[it.category] = by_cat.get(it.category, 0) + 1
            by_diff[it.difficulty] = by_diff.get(it.difficulty, 0) + 1
        return {"items": len(self.items), "by_category": by_cat, "by_difficulty": by_diff,
                "uncovered_categories": [c for c, n in by_cat.items() if n == 0]}

    def split_items(self, split: str | None) -> list[BenchItem]:
        return [i for i in self.items if split is None or i.split == split]
