"""Markdown / JSON / CSV report generation. Every number comes from a BenchResult; nothing is hand-written."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from .runner import BenchResult
from .sweep import SweepResult


def f(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fci(d: dict, nd=3) -> str:
    if d is None or d["n"] == 0 or math.isnan(d["mean"]):
        return "n/a"
    return f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}-{d['ci95'][1]:.{nd}f}] n={d['n']}"


STAND_IN_BANNER = (
    "> **STAND-IN BACKENDS - PIPELINE CHECK, NOT A MODEL BENCHMARK.** These results were produced with a hashing embedder, "
    "a lexical reranker and an extractive (copy-sentences) generator so the pipeline could run without model weights. "
    "They verify that retrieval, citation verification, abstention and the metric code work end to end. They say nothing "
    "about Qwen/Gemma/Llama/BGE quality. Re-run with `--backend hf` on your hardware for real numbers.\n"
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def metric_sections(results: list[BenchResult]) -> str:
    names = [r.name for r in results]
    A = [r.aggregate for r in results]
    parts = []

    def block(title, note, rows):
        parts.append(f"### {title}\n{note}\n\n" + _table(["metric"] + names, rows))

    block("Retrieval metrics", "Did the right pages come back? Scored on answerable items, independent of what the LLM then said "
          "(page-level: a hit is a retrieved chunk from a ground-truth page).",
          [[k] + [fci(a["retrieval"][k]) for a in A] for k in A[0]["retrieval"]])
    block("Generation metrics", "Is the answer right? `semantic_similarity` / `answer_relevancy` are embedding cosines (only as good as "
          "the embedder); `faithfulness` is a deterministic lexical-support proxy, not an LLM judge. An abstention on an answerable "
          "question scores 0 for EM/F1/similarity.",
          [[k] + [fci(a["generation"][k]) for a in A] for k in A[0]["generation"]])
    block("Citation metrics", "Page-level precision/recall of the cited pages vs. ground truth; `grounded` = the snippet shown for a "
          "citation really occurs on the cited page (fabrication check).",
          [[k] + [fci(a["citation"][k]) for a in A] for k in A[0]["citation"]])
    block("Reliability / abstention", "On unanswerable and hallucination-bait questions the correct behaviour is to say the "
          "information is unavailable.", [[k] + [fci(a["safety"][k]) for a in A] for k in A[0]["safety"]])
    block("Document extraction metrics", "Character error rate needs items with `ocr_reference` (scanned-page ground truth).",
          [["ocr_cer"] + [fci(a["extraction"]["cer"]) for a in A]])
    perf_keys = [("latency_mean_s", 3), ("latency_p50_s", 3), ("latency_p95_s", 3), ("first_token_mean_s", 3), ("first_token_p95_s", 3),
                 ("tokens_per_s_mean", 1), ("peak_ram_mb", 0), ("baseline_ram_mb", 0), ("peak_vram_mb", 0),
                 ("pdf_processing_s", 3), ("indexing_s", 3), ("chunks", 0)]
    block("Performance & resources", "Latency is end-to-end per question (retrieval + generation + verification). Processing = "
          "parse+chunk, indexing = embed+store (summed over the corpus). RAM is process RSS (includes loaded models and "
          "interpreter baseline); VRAM is torch's peak allocation (n/a without CUDA).",
          [[k] + [f(a["performance"].get(k), nd) for a in A] for k, nd in perf_keys])
    return "\n\n".join(parts)


def category_table(res: BenchResult) -> str:
    rows = []
    for c, d in res.aggregate["by_category"].items():
        if d["n"] == 0:
            rows.append([c, "0", "-", "-", "-", "-", "-", "no items (not covered)"])
        else:
            rows.append([c, str(d["n"]), f(d["f1"], 2), f(d["contains_answer"], 2), f(d["page_recall@5"], 2), f(d["citation_recall"], 2),
                         f(d["correct_abstention"], 2), f"{d['skipped']} skipped" if d["skipped"] else ""])
    return _table(["category", "n", "F1", "contains answer", "page recall@5", "citation recall", "correct abstention", "note"], rows)


def failures(res: BenchResult, limit: int = 25) -> str:
    rows = []
    for i in res.items:
        if i.status != "ok":
            continue
        bad = (i.answerable and (i.generation.get("contains_answer", 0) == 0)) or (not i.answerable and i.safety.get("hallucination", 0) == 1)
        if bad:
            rows.append([i.id, i.category, i.question.replace("|", "/"), i.route, i.answer.replace("\n", " ").replace("|", "/")[:110]])
    if not rows:
        return "_No failures on this dataset._"
    return _table(["id", "category", "question", "route", "system answer"], rows[:limit]) + (f"\n\n_{len(rows) - limit} more not shown._" if len(rows) > limit else "")


def render(sweep: SweepResult, title: str = "LocalPDF Intelligence benchmark report") -> str:
    res = sweep.test if sweep.select_split else sweep.results
    first = sweep.results[0]
    ds = first.dataset
    stand = any(r.backend["stand_in"] for r in sweep.results)
    L = [f"# {title}", f"_Generated {datetime.now():%Y-%m-%d %H:%M} · dataset `{ds['name']}` (sha256 `{ds['sha256'][:12]}`) · "
         f"{ds['items']} items · hardware: {first.environment['hardware']['device']}"
         f"{' ' + str(first.environment['hardware']['gpu_name']) if first.environment['hardware']['gpu_name'] else ''}, "
         f"{first.environment['hardware']['cpu_count']} cores_", ""]
    if stand:
        L += [STAND_IN_BANNER]
    unc = ds.get("uncovered_categories")
    if unc:
        L += [f"**Coverage gap:** no items for category: {', '.join(unc)}. Those categories are not evaluated by this dataset.", ""]
    L += ["## 1. Configurations compared", _table(
        ["name", "fingerprint", "dense", "bm25", "rerank", "parent", "multi-query", "chunk/parent chars", "LLM", "embedder", "reranker"],
        [[r.name, r.config["fingerprint"], str(r.config["use_dense"]), str(r.config["use_bm25"]), str(r.config["rerank"]),
          str(r.config["parent_expand"]), str(r.config["multi_query"]), f"{r.config['chunk_chars']}/{r.config['parent_chars']}",
          r.backend["llm"], r.backend["embedder"], r.backend["reranker"]] for r in sweep.results]), ""]

    L += ["## 2. Selection", ""]
    if sweep.winner:
        sp = f"the **{sweep.select_split}** split" if sweep.select_split else "all items (no dev/test split in this dataset, so the selection is optimistic)"
        L += [f"Method: each configuration is scored on {sp} with objective = " + " + ".join(f"{v:.2f}·{k}" for k, v in sweep.weights.items()) +
              " (normalised). Reported numbers below are on the " + ("**held-out test** split." if sweep.select_split else "same items."), ""]
        if sweep.tie_note:
            L += [f"> **Tie.** {sweep.tie_note}", ""]
        L += [_table(["configuration", "objective (selection split)", "status"],
                     [[n, f(s), ("SELECTED (tie-break)" if len(sweep.tied) > 1 else "SELECTED") if n == sweep.winner else
                       ("tied" if n in sweep.tied else (f"disqualified: {sweep.disqualified[n]}" if n in sweep.disqualified else ""))]
                      for n, s in sorted(sweep.scores.items(), key=lambda x: -x[1])]), ""]
        L += [f"### Is `{sweep.winner}` really better? (paired bootstrap on the {sweep.comparisons[0]['split'] if sweep.comparisons else '-'} items)",
              _table(["vs", "metric", "mean diff", "95% CI", "n items", "verdict"],
                     [[c["b"], c["metric"], f(c["diff"]), f"[{f(c['ci95'][0])}, {f(c['ci95'][1])}]", str(c["n"]),
                       "distinguishable" if c["distinguishable"] else "NOT distinguishable (CI includes 0)"] for c in sweep.comparisons]), "",
              "_A configuration is only called better where the interval excludes 0. With a small dataset most differences will not be._", ""]
    else:
        L += ["No configuration satisfied the constraints.", ""]

    L += ["## 3. Metrics (held-out test split)" if sweep.select_split else "## 3. Metrics", "",
          f"Counts - " + ", ".join(f"{r.name}: {r.aggregate['counts']['evaluated']} evaluated / {r.aggregate['counts']['skipped']} skipped" for r in res), "",
          metric_sections(res), ""]
    best = next((r for r in sweep.results if r.name == sweep.winner), sweep.results[0])
    L += [f"## 4. Results by category - `{best.name}` (all {best.aggregate['counts']['evaluated']} items)",
          "_Descriptive only: includes the dev items used for selection, and several categories have just 1-3 items._", "",
          category_table(best), "",
          f"## 5. Failures - `{best.name}` (all items)", failures(best), "",
          "## 6. Caveats",
          "- Item-level bootstrap intervals describe sampling noise on *this* dataset only; they do not cover dataset-selection bias.",
          "- Ground truth in the bundled dataset is synthetic and verified by construction; real-document performance needs your own PDFs "
          "(`--dataset`, `--pdf-dir`).",
          "- Embedding-cosine metrics and the lexical faithfulness proxy are approximations, not LLM-judge scores.",
          "- Latency/RAM are for the hardware listed above and include any other load on the machine.", ""]
    return "\n".join(L)


def write_outputs(sweep: SweepResult, out_dir: str | Path, title: str = "LocalPDF Intelligence benchmark report") -> dict:
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render(sweep, title), encoding="utf-8")
    payload = {"winner": sweep.winner, "scores": sweep.scores, "weights": sweep.weights, "select_split": sweep.select_split,
               "comparisons": sweep.comparisons,
               "results": [{"name": r.name, "config": r.config, "backend": r.backend, "dataset": r.dataset,
                            "environment": r.environment, "aggregate_all": r.aggregate, "aggregate_test": t.aggregate}
                           for r, t in zip(sweep.results, sweep.test)]}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with open(out / "per_item.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "id", "category", "difficulty", "split", "answerable", "status", "abstained", "route", "f1", "contains_answer",
                    "hit@5", "page_recall@5", "mrr", "citation_precision", "citation_recall", "correct_abstention", "latency_s", "answer"])
        for r in sweep.results:
            for i in r.items:
                w.writerow([r.name, i.id, i.category, i.difficulty, i.split, i.answerable, i.status, i.abstained, i.route,
                            i.generation.get("f1"), i.generation.get("contains_answer"), i.retrieval.get("hit@5"),
                            i.retrieval.get("page_recall@5"), i.retrieval.get("mrr"), i.citation.get("precision"),
                            i.citation.get("recall"), i.safety.get("correct_abstention"), i.performance.get("latency_s"),
                            i.answer.replace("\n", " ")[:300]])
    return {"report": str(out / "report.md"), "json": str(out / "results.json"), "csv": str(out / "per_item.csv")}
