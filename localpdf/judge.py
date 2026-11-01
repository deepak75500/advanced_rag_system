"""LLM-as-judge checks, run with a local model (no cloud calls).

Used in two places:
  * the pipeline (RAGConfig.llm_verify): after the lexical check, each answer sentence is shown to the LLM together
    with the passage it cites; sentences the judge says are not supported are dropped.
  * the benchmark (--judge): LLM-judged faithfulness and citation support, reported next to the lexical proxies.

A small judge model is itself error-prone; pass a larger one (e.g. --judge-llm Qwen/Qwen2.5-1.5B-Instruct) when you can.
"""
from __future__ import annotations

import re

from .backends.base import LLM

_SYSTEM = ("You are a strict fact checker. You are given a PASSAGE from a document and a CLAIM. "
           "Reply YES if every fact in the claim (names, numbers, units, dates) is stated in or directly follows from "
           "the passage. Reply NO otherwise. Reply with exactly one word: YES or NO.")
_YES = re.compile(r"^\W*(yes|supported|true)\b", re.I)
_NO = re.compile(r"^\W*(no|not|unsupported|false)\b", re.I)


class LLMJudge:
    def __init__(self, llm: LLM, max_passage_chars: int = 2500):
        self.llm, self.max_chars = llm, max_passage_chars
        self.calls = 0

    @property
    def id(self) -> str:
        return self.llm.id

    def supports(self, claim: str, passages: list[str]) -> bool | None:
        """True / False, or None when the judge's reply could not be parsed (callers treat None as 'unknown')."""
        passage = "\n---\n".join(passages)[: self.max_chars]
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"PASSAGE:\n{passage}\n\nCLAIM: {claim}\n\nIs the claim supported? YES or NO:"}]
        self.calls += 1
        out = self.llm.generate(msgs, max_new_tokens=4).text.strip()
        if _YES.match(out):
            return True
        if _NO.match(out):
            return False
        return None

    def faithfulness(self, sentences: list[str], contexts: list[str]) -> float:
        """Share of answer sentences the judge says are supported by the retrieved contexts (nan if none judged)."""
        votes = [self.supports(s, contexts) for s in sentences if s.strip()]
        votes = [v for v in votes if v is not None]
        return sum(votes) / len(votes) if votes else float("nan")
