"""Evaluation metrics, grouped the way the report presents them. All functions are pure and unit-tested.

RETRIEVAL   recall@k, MRR, nDCG@k, context precision (AP-style), context recall (evidence-token)
GENERATION  exact match, token F1, semantic similarity*, answer relevancy*, faithfulness (lexical proxy)
CITATION    citation precision / recall, citation grounding (fabrication check)
EXTRACTION  OCR character error rate, extraction coverage
PERFORMANCE latency, first-token latency, tokens/s, peak RAM/VRAM, indexing time, PDF processing time  (see resources.py)

* embedding-cosine proxies: only as meaningful as the embedder that computes them (the report states which one).
  They are NOT LLM-judge scores. Faithfulness here is a deterministic lexical-support proxy, not RAGAS faithfulness.
"""
from __future__ import annotations

import math

import numpy as np

from ..util import content_tokens, normalize, split_sentences, stem5

Key = tuple[str, int]


# ------------------------------------------------------------------------------------------------- generation
def exact_match(pred: str, gold: str, aliases: list[str] | None = None) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in [gold, *(aliases or [])]))


def contains_answer(pred: str, gold: str, aliases: list[str] | None = None) -> float:
    """Lenient companion to EM: the normalised gold answer occurs inside the prediction (as whole words)."""
    p = f" {normalize(pred)} "
    return float(any(f" {normalize(g)} " in p for g in [gold, *(aliases or [])] if normalize(g)))


def _f1(pred_toks: list[str], gold_toks: list[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    g: dict[str, int] = {}
    for t in gold_toks:
        g[t] = g.get(t, 0) + 1
    overlap = 0
    pc: dict[str, int] = {}
    for t in pred_toks:
        pc[t] = pc.get(t, 0) + 1
    for t, n in pc.items():
        overlap += min(n, g.get(t, 0))
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred_toks), overlap / len(gold_toks)
    return 2 * p * r / (p + r)


def token_f1(pred: str, gold: str, aliases: list[str] | None = None) -> float:
    """SQuAD-style token F1 on normalised text, maximised over gold + aliases."""
    pt = normalize(pred).split()
    return max(_f1(pt, normalize(g).split()) for g in [gold, *(aliases or [])])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def faithfulness(answer_sentences: list[str], contexts: list[str], threshold: float = 0.5) -> float:
    """Share of answer sentences whose content words appear in the retrieved contexts (lexical proxy)."""
    if not answer_sentences:
        return float("nan")
    have = set().union(*[{stem5(t) for t in content_tokens(c)} for c in contexts]) if contexts else set()
    ok = 0
    for s in answer_sentences:
        w = {stem5(t) for t in content_tokens(s)}
        ok += (not w) or len(w & have) / len(w) >= threshold
    return ok / len(answer_sentences)


# ------------------------------------------------------------------------------------------------- retrieval
def recall_at_k(ranked: list[Key], gt: set[Key], k: int) -> float:
    """1 if any ground-truth page is in the top-k, averaged over items = hit-rate@k. Also see page_recall_at_k."""
    return float(bool(set(ranked[:k]) & gt)) if gt else float("nan")


def page_recall_at_k(ranked: list[Key], gt: set[Key], k: int) -> float:
    """Fraction of ALL ground-truth pages found in the top-k (matters for multi-hop / cross-page questions)."""
    return len(set(ranked[:k]) & gt) / len(gt) if gt else float("nan")


def mrr(ranked: list[Key], gt: set[Key]) -> float:
    for i, k in enumerate(ranked, 1):
        if k in gt:
            return 1.0 / i
    return 0.0 if gt else float("nan")


def ndcg_at_k(ranked: list[Key], gt: set[Key], k: int) -> float:
    if not gt:
        return float("nan")
    seen: set[Key] = set()
    dcg = 0.0
    for i, key in enumerate(ranked[:k]):
        if key in gt and key not in seen:  # a page counts once, however many chunks from it were retrieved
            dcg += 1.0 / math.log2(i + 2)
            seen.add(key)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / ideal if ideal else 0.0


def context_precision(ranked: list[Key], gt: set[Key]) -> float:
    """Rank-aware precision of the retrieved contexts: mean of precision@i over the positions i that are relevant."""
    if not gt:
        return float("nan")
    rel = [k in gt for k in ranked]
    if not any(rel):
        return 0.0
    hits, acc = 0, 0.0
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1
            acc += hits / i
    return acc / hits


def context_recall(evidence: list[str], contexts: list[str]) -> float:
    """Share of the ground-truth evidence's content words that appear in the retrieved contexts."""
    ev = {stem5(t) for e in evidence for t in content_tokens(e)}
    if not ev:
        return float("nan")
    have = set().union(*[{stem5(t) for t in content_tokens(c)} for c in contexts]) if contexts else set()
    return len(ev & have) / len(ev)


# ------------------------------------------------------------------------------------------------- citations
def citation_precision(cited: list[Key], gt: set[Key]) -> float:
    return len(set(cited) & gt) / len(set(cited)) if cited else float("nan")


def citation_recall(cited: list[Key], gt: set[Key]) -> float:
    return len(set(cited) & gt) / len(gt) if gt else float("nan")


def citation_grounded(evidence: str, page_text: str) -> bool:
    """A citation is grounded if the snippet we show really occurs on the page we cite (fabrication check)."""
    return bool(normalize(evidence)) and normalize(evidence) in normalize(page_text)


# ------------------------------------------------------------------------------------------------- extraction
def levenshtein(a: str, b: str) -> int:
    """Edit distance, bit-parallel (Myers / Hyyro) so page-sized strings are fast in pure Python."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq: dict[str, int] = {}
    for i, ch in enumerate(b):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask, high = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for ch in a:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & high:
            score += 1
        if mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit distance / reference length, on whitespace-normalised text. Can exceed 1.0."""
    r, h = " ".join(reference.split()), " ".join(hypothesis.split())
    return levenshtein(r, h) / len(r) if r else float("nan")


# ------------------------------------------------------------------------------------------------- aggregation
def nanmean(xs) -> float:
    a = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(a.mean()) if len(a) else float("nan")


def bootstrap_ci(xs, n: int = 2000, seed: int = 0) -> tuple[float, float, float, int]:
    """(mean, lo95, hi95, n) of the non-NaN values. Fixed seed => the report is reproducible."""
    a = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3 + (0,)
    if len(a) == 1:
        return (float(a[0]), float(a[0]), float(a[0]), 1)
    rng = np.random.default_rng(seed)
    means = rng.choice(a, (n, len(a))).mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(a)


def percentile(xs, q: float) -> float:
    a = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.quantile(a, q)) if a else float("nan")
