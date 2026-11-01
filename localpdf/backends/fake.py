"""Deterministic stand-in backends.

These are NOT models of interest. They exist so the pipeline, the store, the metrics and the benchmark engine can be
tested in CI and on machines without model weights. Any benchmark run with them measures the *plumbing*, never
language-model quality, and reports produced with them are labelled accordingly (`Backends.is_stand_in`).
"""
from __future__ import annotations

import math
import re
import time
import zlib
from typing import Callable

import numpy as np

from ..util import content_tokens, stem5
from .base import Backends, Generation

ABSTAIN_TEXT = "The information is not available in the uploaded documents."


class HashingEmbedder:
    """Bag of (stemmed) words + bigrams hashed into a fixed dimension. Deterministic, no weights."""

    def __init__(self, dim: int = 512, name: str = "hashing-512"):
        self.id, self.dim = name, dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = [stem5(t) for t in content_tokens(text)]
        for t in toks:
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        for a, b in zip(toks, toks[1:]):
            v[zlib.crc32((a + "_" + b).encode()) % self.dim] += 0.5
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts, is_query=False):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.ascontiguousarray(np.vstack([self._vec(t) for t in texts]).astype(np.float32))


class LexicalReranker:
    """Query-term coverage (IDF-free) mapped to [0, 1]. Stand-in for a cross-encoder."""

    id = "lexical-coverage"

    def score(self, query, texts):
        q = {stem5(t) for t in content_tokens(query)}
        if not q:
            return np.zeros(len(texts))
        out = []
        for t in texts:
            d = {stem5(x) for x in content_tokens(t)}
            out.append(len(q & d) / len(q))
        return np.asarray(out, dtype=np.float64)


class ExtractiveLLM:
    """Answers by copying the best-matching excerpt sentences and tagging them [#n]. Never invents anything.

    It follows the same prompt protocol as a real LLM (reads `<excerpt id=..>` blocks, abstains when nothing
    matches), so the whole pipeline - prompts, citation parsing, verification, abstention - runs end to end.
    """

    id = "extractive-stand-in"
    _EX = re.compile(r'<excerpt id="(\d+)"[^>]*>\n(.*?)\n</excerpt>', re.S)

    def generate(self, messages, max_new_tokens=300, on_text: Callable[[str], None] | None = None) -> Generation:
        t0 = time.perf_counter()
        user = messages[-1]["content"]
        m = re.search(r"Question:\s*(.*?)\nAnswer:", user, re.S)
        question = m.group(1).strip() if m else user
        q = {stem5(t) for t in content_tokens(question)}
        lines = []
        for eid, body in self._EX.findall(user):
            for line in re.split(r"(?<=[.!?])\s+|\n", body):
                line = line.strip()
                if len(line) >= 12:
                    lines.append((int(eid), line, {stem5(t) for t in content_tokens(line)}))
        # IDF over the candidate sentences: words that appear everywhere ("farm") matter less than rare ones ("capacity")
        n = max(len(lines), 1)
        w = {t: math.log(1 + n / max(1, sum(t in d for _, _, d in lines))) for t in q}
        scored = []
        for eid, line, d in lines:
            if q and d:
                scored.append((sum(w[t] for t in q & d) / sum(w.values()), eid, line))
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] < 0.4:
            text = ABSTAIN_TEXT
        else:
            best = scored[0][0]
            picked = [s for s in scored if s[0] >= max(0.4, best - 0.15)][:2]
            text = " ".join(f"{line} [#{eid}]" for _, eid, line in picked)
        words = text.split(" ")
        acc = ""
        for i, w in enumerate(words):
            acc = (acc + " " + w).strip()
            if on_text:
                on_text(acc)
        total = time.perf_counter() - t0
        return Generation(text=text, first_token_s=min(total, 1e-4), total_s=total, n_tokens=len(words))


def stand_in_backends() -> Backends:
    return Backends(HashingEmbedder(), LexicalReranker(), ExtractiveLLM(), is_stand_in=True)
