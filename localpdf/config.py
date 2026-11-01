"""Hardware detection, model profiles and the versioned RAG configuration."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    llm_id: str
    embed_id: str
    rerank_id: str
    max_new_tokens: int
    ctx_chars: int
    quantization: str = "none"  # none | int8 (CPU dynamic quantisation) | 4bit (GPU, bitsandbytes)


# NOTE: model IDs are the defaults the code *tries to load*. None of them has been benchmarked in this repo yet;
# use `python -m localpdf.bench.cli sweep` on your hardware to measure them (see README).
PROFILES: dict[str, HardwareProfile] = {
    "gpu": HardwareProfile("gpu", "Qwen/Qwen2.5-3B-Instruct", "BAAI/bge-m3", "BAAI/bge-reranker-base", 350, 8000),
    "gpu-4bit-7b": HardwareProfile("gpu-4bit-7b", "Qwen/Qwen2.5-7B-Instruct", "BAAI/bge-m3",
                                   "BAAI/bge-reranker-base", 350, 8000, "4bit"),
    "cpu-quality": HardwareProfile("cpu-quality", "Qwen/Qwen2.5-3B-Instruct", "BAAI/bge-small-en-v1.5",
                                   "BAAI/bge-reranker-base", 300, 4500),
    "cpu-balanced": HardwareProfile("cpu-balanced", "Qwen/Qwen2.5-1.5B-Instruct", "BAAI/bge-small-en-v1.5",
                                    "BAAI/bge-reranker-base", 260, 3600),
    "cpu-fast": HardwareProfile("cpu-fast", "Qwen/Qwen2.5-0.5B-Instruct", "BAAI/bge-small-en-v1.5",
                                "cross-encoder/ms-marco-MiniLM-L-6-v2", 220, 3200),
}


def detect_hardware() -> dict:
    """Device, VRAM, RAM and cores. Works without torch installed (reports CPU)."""
    info = {"device": "cpu", "gpu_name": None, "vram_gb": 0.0, "cpu_count": os.cpu_count() or 2, "ram_gb": None}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(device="cuda", gpu_name=p.name, vram_gb=round(p.total_memory / 1e9, 1))
    except Exception:
        pass
    return info


def select_profile(hw: dict | None = None, force: str | None = None) -> HardwareProfile:
    """Pick the largest profile that fits. Never assumes a model fits: VRAM/cores/RAM decide, `force` overrides."""
    force = force or os.environ.get("LOCALPDF_PROFILE")
    if force:
        if force not in PROFILES:
            raise ValueError(f"unknown profile {force!r}; choose from {sorted(PROFILES)}")
        return PROFILES[force]
    hw = hw or detect_hardware()
    if hw["device"] == "cuda":
        return PROFILES["gpu-4bit-7b"] if hw["vram_gb"] >= 14 else PROFILES["gpu"] if hw["vram_gb"] >= 6 else PROFILES["cpu-balanced"]
    ram = hw.get("ram_gb") or 8
    if hw["cpu_count"] >= 8 and ram >= 12:
        return PROFILES["cpu-balanced"]
    return PROFILES["cpu-fast"]


def default_min_relevance(rerank_id: str) -> float:
    """Reranker-probability threshold below which nothing in the corpus resembles the question."""
    if "MiniLM" in rerank_id:
        return 0.01
    if "bge-reranker" in rerank_id:
        return 0.02
    return 0.15  # lexical stand-in reranker


@dataclass(frozen=True)
class RAGConfig:
    """Everything that changes system behaviour. Its fingerprint is stored with every experiment."""

    name: str = "default"
    chunk_chars: int = 900
    chunk_overlap: int = 150
    parent_chars: int = 2400
    dense_k: int = 30
    bm25_k: int = 30
    use_dense: bool = True
    use_bm25: bool = True
    rerank: bool = True
    multi_query: bool = False
    parent_expand: bool = True
    top_k: int = 5
    rrf_k: int = 60
    min_relevance: float = 0.15
    rewrite_followups: bool = True
    verify_citations: bool = True
    llm_verify: bool = False         # also ask the local LLM whether each kept sentence is supported (slower)
    support_threshold: float = 0.5   # share of an answer sentence's content words that must appear in its cited passage

    def fingerprint(self) -> str:
        d = asdict(self)
        d.pop("name")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def but(self, **kw) -> "RAGConfig":
        return replace(self, **kw)
