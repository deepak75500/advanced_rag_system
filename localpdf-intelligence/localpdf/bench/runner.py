"""Run one configuration over a dataset and compute every metric group. Nothing here talks to a specific model."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..backends.base import Backends
from ..rag import RAGPipeline
from ..store import IngestResult, KnowledgeBase
from ..util import split_sentences
from . import metrics as M
from .resources import ResourceMonitor
from .schema import CATEGORIES, BenchItem, Dataset

K_VALUES = (1, 3, 5)


@dataclass
class ItemResult:
    id: str
    category: str
    difficulty: str
    split: str | None
    answerable: bool
    question: str
    answer: str
    status: str = "ok"                 # ok | skipped
    skip_reason: str = ""
    abstained: bool = False
    route: str = ""
    retrieval: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    citation: dict = field(default_factory=dict)
    extraction: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    cited: list = field(default_factory=list)


@dataclass
class BenchResult:
    name: str
    items: list[ItemResult]
    resources: dict
    ingest: dict
    config: dict
    backend: dict
    dataset: dict
    environment: dict
    aggregate: dict = field(default_factory=dict)


def environment() -> dict:
    from ..config import detect_hardware

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "hardware": detect_hardware()}
    for mod in ("numpy", "pandas", "torch", "transformers", "sentence_transformers", "fitz"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", getattr(m, "VersionBind", "?"))
        except Exception:
            env[mod] = None
    return env


def ingest_corpus(kb: KnowledgeBase, corpus: dict[str, list]) -> dict:
    """corpus: {filename: [PageText...]}. Returns summed ingest timings (the 'PDF processing' / 'indexing' metrics)."""
    rs: list[IngestResult] = []
    for name, doc in corpus.items():  # doc: list[PageText] or an ExtractedDoc (whose extraction time counts as parse time)
        pages, parse_s = (doc.pages, doc.seconds) if hasattr(doc, "pages") else (doc, 0.0)
        rs.append(kb.ingest_pages(name, pages, parse_s=parse_s))
    return {"documents": len(rs), "chunks": sum(r.chunks for r in rs), "pages": sum(r.pages for r in rs),
            "processing_s": sum(r.processing_s for r in rs), "indexing_s": sum(r.indexing_s for r in rs),
            "parse_s": sum(r.parse_s for r in rs), "chunk_s": sum(r.chunk_s for r in rs),
            "embed_s": sum(r.embed_s for r in rs), "index_s": sum(r.index_s for r in rs),
            "duplicate_chunks_dropped": sum(r.duplicate_chunks_dropped for r in rs)}


def evaluate_item(pipe: RAGPipeline, kb: KnowledgeBase, backends: Backends, it: BenchItem, k: int,
                  can_ocr: bool = False) -> ItemResult:
    base = dict(id=it.id, category=it.category, difficulty=it.difficulty, split=it.split, answerable=it.answerable,
                question=it.question)
    if "ocr" in it.requires and not can_ocr:
        return ItemResult(**base, answer="", status="skipped", skip_reason="OCR not available in this run")

    ans = pipe.ask(it.question, history=it.history, mode=it.mode)
    gt = {g.key() for g in it.gt_pages}
    ranked = [(h.filename, h.page) for h in ans.hits]
    ctxs = [h.context for h in ans.hits]
    res = ItemResult(**base, answer=ans.answer, abstained=ans.abstained, route=ans.route,
                     cited=[(c.doc, c.page) for c in ans.citations])

    if it.answerable:
        # ---- retrieval (independent of whether the LLM answered or abstained)
        res.retrieval = {f"hit@{kk}": M.recall_at_k(ranked, gt, kk) for kk in K_VALUES if kk <= max(k, 5)}
        res.retrieval.update({f"page_recall@{kk}": M.page_recall_at_k(ranked, gt, kk) for kk in K_VALUES},
                             mrr=M.mrr(ranked, gt), **{f"ndcg@{k}": M.ndcg_at_k(ranked, gt, k)},
                             context_precision=M.context_precision(ranked, gt),
                             context_recall=M.context_recall(it.evidence, ctxs))
        # ---- generation
        if ans.abstained:
            res.generation = {"exact_match": 0.0, "contains_answer": 0.0, "f1": 0.0, "semantic_similarity": 0.0,
                              "answer_relevancy": float("nan"), "faithfulness": float("nan")}
            res.safety = {"false_refusal": 1.0}
        else:
            e = backends.embedder.embed([it.expected_answer, ans.answer, it.question], is_query=False)
            res.generation = {
                "exact_match": M.exact_match(ans.answer, it.expected_answer, it.aliases),
                "contains_answer": M.contains_answer(ans.answer, it.expected_answer, it.aliases),
                "f1": M.token_f1(ans.answer, it.expected_answer, it.aliases),
                "semantic_similarity": M.cosine(e[0], e[1]),
                "answer_relevancy": M.cosine(e[2], e[1]),
                "faithfulness": M.faithfulness(split_sentences(ans.answer), ctxs)}
            res.safety = {"false_refusal": 0.0}
        # ---- citations
        cited = [(c.doc, c.page) for c in ans.citations]
        grounded = [M.citation_grounded(c.evidence, kb.page_text(c.doc_id, c.page)) for c in ans.citations]
        res.citation = {"precision": M.citation_precision(cited, gt), "recall": M.citation_recall(cited, gt),
                        "grounded": float(np.mean(grounded)) if grounded else float("nan")}
    else:
        res.safety = {"correct_abstention": float(ans.abstained), "hallucination": float(not ans.abstained)}
        grounded = [M.citation_grounded(c.evidence, kb.page_text(c.doc_id, c.page)) for c in ans.citations]
        res.citation = {"grounded": float(np.mean(grounded)) if grounded else float("nan")}

    if it.ocr_reference is not None and gt:
        g = next(iter(gt))
        doc = [d for d in kb.list_documents() if d["filename"] == g[0]]
        if doc:
            res.extraction = {"cer": M.character_error_rate(it.ocr_reference, kb.page_text(doc[0]["id"], g[1]))}

    t = ans.timings
    res.performance = {"latency_s": t.get("total_s"), "first_token_s": t.get("first_token_s"),
                       "tokens_per_s": t.get("tokens_per_s"), "n_tokens": t.get("n_tokens"),
                       "retrieve_s": t.get("retrieve_s"), "generate_s": t.get("generate_s")}
    return res


def run_benchmark(name: str, pipe: RAGPipeline, kb: KnowledgeBase, backends: Backends, dataset: Dataset,
                  items: list[BenchItem] | None = None, ingest: dict | None = None, k: int = 5, can_ocr: bool = False,
                  progress: Callable[[int, int], None] | None = None) -> BenchResult:
    items = dataset.items if items is None else items
    results: list[ItemResult] = []
    with ResourceMonitor() as mon:
        for i, it in enumerate(items):
            results.append(evaluate_item(pipe, kb, backends, it, k, can_ocr))
            if progress:
                progress(i + 1, len(items))
    from dataclasses import asdict

    res = BenchResult(name=name, items=results, resources=mon.as_dict(), ingest=ingest or {}, config=asdict(pipe.cfg) | {"fingerprint": pipe.cfg.fingerprint()},
                      backend={"llm": backends.llm.id, "embedder": backends.embedder.id, "reranker": backends.reranker.id,
                               "stand_in": backends.is_stand_in},
                      dataset={"name": dataset.name, "sha256": dataset.sha256(), **dataset.summary(), "evaluated": len(items)},
                      environment=environment())
    res.aggregate = aggregate(res, k)
    return res


# ----------------------------------------------------------------------------------------------- aggregation
def _col(items: list[ItemResult], group: str, key: str) -> list[float]:
    return [getattr(i, group).get(key, float("nan")) for i in items if i.status == "ok" and getattr(i, group)]


def aggregate(res: BenchResult, k: int = 5) -> dict:
    ok = [i for i in res.items if i.status == "ok"]
    ans, una = [i for i in ok if i.answerable], [i for i in ok if not i.answerable]

    def ci(xs):
        m, lo, hi, n = M.bootstrap_ci(xs)
        return {"mean": m, "ci95": [lo, hi], "n": n}

    retrieval_keys = [f"hit@{kk}" for kk in K_VALUES] + [f"page_recall@{kk}" for kk in K_VALUES] + \
                     ["mrr", f"ndcg@{k}", "context_precision", "context_recall"]
    agg = {
        "retrieval": {key: ci(_col(ans, "retrieval", key)) for key in retrieval_keys},
        "generation": {key: ci(_col(ans, "generation", key)) for key in
                       ["exact_match", "contains_answer", "f1", "semantic_similarity", "answer_relevancy", "faithfulness"]},
        "citation": {key: ci(_col(ans, "citation", key)) for key in ["precision", "recall", "grounded"]},
        "safety": {
            "false_refusal_rate": ci(_col(ans, "safety", "false_refusal")),
            "correct_abstention_rate": ci(_col(una, "safety", "correct_abstention")),
            "hallucination_rate": ci(_col(una, "safety", "hallucination")),
        },
        "extraction": {"cer": ci(_col(ok, "extraction", "cer"))},
    }
    lat = [i.performance.get("latency_s") for i in ok]
    ft = [i.performance.get("first_token_s") for i in ok]
    tps = [i.performance.get("tokens_per_s") for i in ok if i.performance.get("tokens_per_s")]
    agg["performance"] = {
        "latency_mean_s": M.nanmean(lat), "latency_p50_s": M.percentile(lat, 0.5), "latency_p95_s": M.percentile(lat, 0.95),
        "first_token_mean_s": M.nanmean(ft), "first_token_p95_s": M.percentile(ft, 0.95),
        "tokens_per_s_mean": M.nanmean(tps), **res.resources,
        "pdf_processing_s": res.ingest.get("processing_s"), "indexing_s": res.ingest.get("indexing_s"),
        "chunks": res.ingest.get("chunks")}
    # per-category view (the same few headline numbers for each of the 15 categories)
    by_cat = {}
    for c in CATEGORIES:
        its = [i for i in res.items if i.category == c]
        oks = [i for i in its if i.status == "ok"]
        a = [i for i in oks if i.answerable]
        u = [i for i in oks if not i.answerable]
        by_cat[c] = {
            "n": len(its), "skipped": len(its) - len(oks),
            "f1": M.nanmean(_col(a, "generation", "f1")), "contains_answer": M.nanmean(_col(a, "generation", "contains_answer")),
            "hit@5": M.nanmean(_col(a, "retrieval", "hit@5")), "page_recall@5": M.nanmean(_col(a, "retrieval", "page_recall@5")),
            "citation_recall": M.nanmean(_col(a, "citation", "recall")),
            "correct_abstention": M.nanmean(_col(u, "safety", "correct_abstention"))}
    agg["by_category"] = by_cat
    agg["counts"] = {"items": len(res.items), "evaluated": len(ok), "skipped": len(res.items) - len(ok),
                     "answerable": len(ans), "unanswerable": len(una)}
    return agg
