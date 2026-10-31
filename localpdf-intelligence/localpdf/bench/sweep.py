"""Model-comparison laboratory: run the SAME dataset against several configurations, select the best on the dev split,
report it on the held-out test split, and say whether differences are statistically distinguishable."""
from __future__ import annotations

import copy
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..backends.base import Backends
from ..config import RAGConfig
from ..rag import RAGPipeline
from ..store import KnowledgeBase
from . import metrics as M
from .runner import BenchResult, ItemResult, aggregate, ingest_corpus, run_benchmark
from .schema import Dataset

# page_recall@5 saturates (~1.0) on small corpora and cannot separate retrieval strategies, so a rank-sensitive term (MRR)
# is part of the objective. Weights are an explicit, documented choice, overridable per run.
TIE_EPS = 0.005
DEFAULT_WEIGHTS = {"f1": 0.30, "page_recall@5": 0.15, "mrr": 0.10, "correct_abstention": 0.20, "citation_precision": 0.15,
                   "faithfulness": 0.10}


@dataclass
class Experiment:
    name: str
    cfg: RAGConfig
    backends_factory: Callable[[], Backends]
    ctx_chars: int = 6000
    max_new_tokens: int = 300
    notes: str = ""


def ablation_grid(base: RAGConfig | None = None) -> list[RAGConfig]:
    """Retrieval ablations around a base config (each differs from the base in one or two switches)."""
    b = base or RAGConfig()
    return [
        b.but(name="hybrid+rerank+parent (default)"),
        b.but(name="dense only", use_bm25=False, rerank=False, parent_expand=False),
        b.but(name="bm25 only", use_dense=False, rerank=False, parent_expand=False),
        b.but(name="hybrid, no rerank", rerank=False, parent_expand=False),
        b.but(name="hybrid+rerank, no parent expand", parent_expand=False),
        b.but(name="hybrid+rerank+parent, multi-query", multi_query=True),
        b.but(name="chunks 400/parent 1200", chunk_chars=400, chunk_overlap=60, parent_chars=1200),
        b.but(name="chunks 1500/parent 3600", chunk_chars=1500, chunk_overlap=200, parent_chars=3600),
    ]


def objective(agg: dict, weights: dict[str, float] | None = None) -> float:
    """Weighted mean of headline metrics (missing metrics count as 0). Documented in the report."""
    w = weights or DEFAULT_WEIGHTS
    g, r, c, s = agg["generation"], agg["retrieval"], agg["citation"], agg["safety"]
    vals = {"f1": g["f1"]["mean"], "page_recall@5": r["page_recall@5"]["mean"], "mrr": r["mrr"]["mean"],
            "correct_abstention": s["correct_abstention_rate"]["mean"], "citation_precision": c["precision"]["mean"],
            "faithfulness": g["faithfulness"]["mean"]}
    return float(sum(w[k] * (0.0 if np.isnan(vals[k]) else vals[k]) for k in w) / sum(w.values()))


def subset(res: BenchResult, split: str | None, k: int = 5) -> BenchResult:
    """Same run, restricted to one split, re-aggregated (cheap: nothing is re-executed)."""
    if split is None:
        return res
    sub = copy.copy(res)
    sub.items = [i for i in res.items if i.split == split]
    sub.aggregate = aggregate(sub, k)
    return sub


def paired_diff(a: BenchResult, b: BenchResult, metric: str = "f1", n: int = 4000, seed: int = 0) -> dict:
    """Item-matched bootstrap of mean(a) - mean(b). 'distinguishable' only if the 95% CI excludes 0."""
    def per_item(r: BenchResult) -> dict[str, float]:
        out = {}
        for i in r.items:
            if i.status != "ok":
                continue
            if metric == "f1" and i.answerable:
                out[i.id] = i.generation.get("f1", float("nan"))
            elif metric == "page_recall@5" and i.answerable:
                out[i.id] = i.retrieval.get("page_recall@5", float("nan"))
            elif metric == "correct_abstention" and not i.answerable:
                out[i.id] = i.safety.get("correct_abstention", float("nan"))
        return {k: v for k, v in out.items() if not np.isnan(v)}

    xa, xb = per_item(a), per_item(b)
    ids = sorted(set(xa) & set(xb))
    if len(ids) < 2:
        return {"metric": metric, "n": len(ids), "diff": float("nan"), "ci95": [float("nan")] * 2, "distinguishable": False}
    d = np.array([xa[i] - xb[i] for i in ids])
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d))).mean(axis=1)
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"metric": metric, "n": len(ids), "diff": float(d.mean()), "ci95": [lo, hi], "distinguishable": bool(lo > 0 or hi < 0)}


@dataclass
class SweepResult:
    results: list[BenchResult]
    dev: list[BenchResult]
    test: list[BenchResult]
    scores: dict[str, float]            # objective on the selection split, by experiment name
    winner: str | None
    disqualified: dict[str, str]
    weights: dict[str, float]
    select_split: str | None
    comparisons: list[dict] = field(default_factory=list)
    tied: list[str] = field(default_factory=list)       # configurations within TIE_EPS of the best objective
    tie_note: str = ""


def run_experiment(exp: Experiment, corpus: dict, dataset: Dataset, k: int = 5, can_ocr: bool = False,
                   progress=None) -> BenchResult:
    backends = exp.backends_factory()
    kb = KnowledgeBase(":memory:", backends.embedder, exp.cfg)
    try:
        ing = ingest_corpus(kb, corpus)
        pipe = RAGPipeline(kb, backends, exp.cfg, ctx_chars=exp.ctx_chars, max_new_tokens=exp.max_new_tokens)
        return run_benchmark(exp.name, pipe, kb, backends, dataset, ingest=ing, k=k, can_ocr=can_ocr, progress=progress)
    finally:
        kb.close()


def run_sweep(experiments: list[Experiment], corpus: dict, dataset: Dataset, weights: dict[str, float] | None = None,
              max_latency_s: float | None = None, k: int = 5, can_ocr: bool = False, log: "ExperimentLog | None" = None,
              verbose: bool = True) -> SweepResult:
    weights = weights or DEFAULT_WEIGHTS
    has_split = any(i.split == "dev" for i in dataset.items) and any(i.split == "test" for i in dataset.items)
    sel_split = "dev" if has_split else None
    results = []
    for n, exp in enumerate(experiments, 1):
        t0 = time.perf_counter()
        r = run_experiment(exp, corpus, dataset, k, can_ocr)
        results.append(r)
        if verbose:
            print(f"[{n}/{len(experiments)}] {exp.name}: {time.perf_counter() - t0:.1f}s")
        if log:
            log.record(exp, dataset.sha256(), r)
    dev = [subset(r, sel_split, k) for r in results]
    test = [subset(r, "test", k) if has_split else r for r in results]
    scores, dq = {}, {}
    for r, d in zip(results, dev):
        scores[r.name] = objective(d.aggregate, weights)
        if max_latency_s is not None and d.aggregate["performance"]["latency_mean_s"] > max_latency_s:
            dq[r.name] = f"mean latency {d.aggregate['performance']['latency_mean_s']:.2f}s exceeds {max_latency_s}s"
    eligible = {n: s for n, s in scores.items() if n not in dq}
    winner, tied, tie_note = None, [], ""
    if eligible:
        best = max(eligible.values())
        tied = [n for n, s in eligible.items() if best - s <= TIE_EPS]
        lat = {d.name: d.aggregate["performance"]["latency_mean_s"] for d in dev}
        winner = min(tied, key=lambda n: (lat[n], [e.name for e in experiments].index(n)))
        if len(tied) > 1:
            tie_note = (f"{len(tied)} configurations are tied within {TIE_EPS} on the objective; "
                        f"`{winner}` was picked only because it has the lowest mean latency. This is NOT evidence that it is better.")
    comps = []
    if winner:
        wi = next(i for i, r in enumerate(results) if r.name == winner)
        for i, r in enumerate(results):
            if i != wi:
                for m in ("f1", "page_recall@5", "correct_abstention"):
                    comps.append({"a": winner, "b": r.name, "split": "test" if has_split else "all",
                                  **paired_diff(test[wi], test[i], m)})
    return SweepResult(results, dev, test, scores, winner, dq, weights, sel_split, comps, tied, tie_note)


# ------------------------------------------------------------------------------------------ experiment history
class ExperimentLog:
    """SQLite log of every experiment: config fingerprint + dataset hash + metrics => repeatable, comparable runs."""

    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS experiments(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, config_fp TEXT, config_json TEXT, dataset_sha TEXT,
            backend_json TEXT, stand_in INTEGER, metrics_json TEXT, environment_json TEXT, notes TEXT)""")
        self.db.commit()

    def record(self, exp: Experiment, dataset_sha: str, res: BenchResult) -> int:
        from dataclasses import asdict

        cur = self.db.execute(
            "INSERT INTO experiments(ts,name,config_fp,config_json,dataset_sha,backend_json,stand_in,metrics_json,environment_json,notes) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), exp.name, exp.cfg.fingerprint(), json.dumps(asdict(exp.cfg)),
             dataset_sha, json.dumps(res.backend), int(res.backend["stand_in"]), json.dumps(res.aggregate, default=str),
             json.dumps(res.environment, default=str), exp.notes))
        self.db.commit()
        return cur.lastrowid

    def list(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM experiments ORDER BY id DESC")]
