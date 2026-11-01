"""Interfaces every model backend implements. The pipeline only ever talks to these."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class Generation:
    text: str
    first_token_s: float  # time from request to first generated token
    total_s: float
    n_tokens: int

    @property
    def tokens_per_s(self) -> float:
        dt = max(self.total_s - self.first_token_s, 1e-9)
        return max(self.n_tokens - 1, 0) / dt if self.n_tokens > 1 else 0.0


class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """L2-normalised float32 matrix [len(texts), dim]."""


class Reranker(Protocol):
    id: str

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Relevance probability in [0, 1] for each text."""


class LLM(Protocol):
    id: str

    def generate(self, messages: list[dict], max_new_tokens: int = 300,
                 on_text: Callable[[str], None] | None = None) -> Generation:
        """Greedy, deterministic generation. `on_text` receives the full text so far while streaming."""


@dataclass
class Backends:
    embedder: Embedder
    reranker: Reranker
    llm: LLM
    is_stand_in: bool = False  # True => hashing embedder / lexical reranker / extractive LLM (pipeline tests only)
