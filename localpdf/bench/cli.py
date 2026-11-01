"""Command line entry point:  python -m localpdf.bench.cli <command> [options]

  validate        check a dataset's ground truth against the real PDFs / corpus (refuses fabricated ground truth)
  smoke           full pipeline + metrics + report on the bundled synthetic corpus with STAND-IN backends (no models)
  sweep           retrieval-configuration ablation (one backend, many RAGConfigs)
  compare-models  same retrieval config, several local LLM / embedder / reranker / quantisation combinations (needs torch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..backends import stand_in_backends
from ..config import PROFILES, RAGConfig, default_min_relevance, detect_hardware, select_profile
from .report import write_outputs
from .schema import Dataset
from .sweep import Experiment, ExperimentLog, ablation_grid, run_sweep
from .synthetic import build_corpus, build_dataset, corpus_texts


def load_corpus(pdf_dir: str | None, ocr: bool = False) -> dict:
    if not pdf_dir:
        return build_corpus()
    from ..parsing.pdf import extract_pdf

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found in {pdf_dir}")
    from ..security import safe_filename, validate_upload

    out = {}
    for p in pdfs:
        validate_upload(p)
        out[safe_filename(p.name)] = extract_pdf(p, ocr=ocr)
    return out


def hf_factory(profile_name: str | None, **overrides):
    hw = detect_hardware()
    prof = select_profile(hw, profile_name)

    def make():
        from ..backends.hf import build_hf_backends

        return build_hf_backends(prof, hw["device"], **overrides)

    return make, prof, hw


def judge_factory(a):
    """--judge: judge with the run's own LLM; --judge-llm ID: load that model once and reuse it for every experiment."""
    if not (getattr(a, "judge", False) or getattr(a, "judge_llm", None)):
        return None
    from ..judge import LLMJudge

    cache = {}

    def make(backends):
        if not a.judge_llm:
            return LLMJudge(backends.llm)
        if "llm" not in cache:
            from ..backends.hf import HFLLM

            cache["llm"] = HFLLM(a.judge_llm, detect_hardware()["device"], "none")
        return LLMJudge(cache["llm"])

    return make


def _dataset(args) -> Dataset:
    return Dataset.load(args.dataset) if args.dataset else build_dataset()


def _check_ground_truth(ds: Dataset, corpus) -> None:
    problems = ds.validate_against_corpus(corpus_texts(corpus))
    if problems:
        sys.exit("ground truth does not match the corpus:\n  " + "\n  ".join(problems[:30]))


def cmd_validate(a):
    ds, corpus = _dataset(a), load_corpus(a.pdf_dir)
    s = ds.summary()
    print(json.dumps(s, indent=2))
    problems = ds.check_structure() + ds.validate_against_corpus(corpus_texts(corpus))
    print("OK: every ground-truth page and evidence passage exists in the corpus" if not problems else "\n".join(problems))
    return 1 if problems else 0


def _finish(sw, a, title):
    out = write_outputs(sw, a.out, title)
    print(json.dumps(out, indent=2))
    print(f"selected: {sw.winner}" + (f"  (TIE: {sw.tie_note})" if sw.tie_note else ""))
    return 0


def cmd_smoke(a):
    ds, corpus = build_dataset(), build_corpus()
    _check_ground_truth(ds, corpus)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = ExperimentLog(Path(a.out) / "experiments.sqlite3") if a.log else None
    exps = [Experiment(c.name, c, stand_in_backends) for c in ablation_grid()]
    sw = run_sweep(exps, corpus, ds, log=log)
    return _finish(sw, a, "Pipeline smoke test (stand-in backends)")


def cmd_sweep(a):
    ds, corpus = _dataset(a), load_corpus(a.pdf_dir, a.ocr)
    _check_ground_truth(ds, corpus)
    if a.backend == "stand-in":
        factory, mr = stand_in_backends, RAGConfig().min_relevance
    else:
        factory, prof, _ = hf_factory(a.profile, **_overrides(a))
        mr = default_min_relevance(_overrides(a).get("rerank_id", prof.rerank_id))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = ExperimentLog(Path(a.out) / "experiments.sqlite3")
    jf = None if a.backend == "stand-in" else judge_factory(a)
    exps = [Experiment(c.name, c, factory, judge_factory=jf) for c in ablation_grid(RAGConfig(min_relevance=mr))]
    sw = run_sweep(exps, corpus, ds, max_latency_s=a.max_latency, can_ocr=a.ocr, log=log)
    return _finish(sw, a, f"Retrieval-configuration sweep ({a.backend})")


def _overrides(a) -> dict:
    return {k: v for k, v in dict(llm_id=a.llm, embed_id=a.embed, rerank_id=a.rerank, quantization=a.quant).items() if v}


def cmd_compare_models(a):
    ds, corpus = _dataset(a), load_corpus(a.pdf_dir, a.ocr)
    _check_ground_truth(ds, corpus)
    specs = json.loads(Path(a.models).read_text())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = ExperimentLog(Path(a.out) / "experiments.sqlite3")
    exps = []
    for s in specs:  # [{"name":..., "llm_id":..., "embed_id":..., "rerank_id":..., "quantization": "none|int8|4bit"}]
        ov = {k: s[k] for k in ("llm_id", "embed_id", "rerank_id", "quantization") if k in s}
        factory, prof, _ = hf_factory(s.get("profile", a.profile), **ov)
        cfg = RAGConfig(name=s["name"], min_relevance=default_min_relevance(ov.get("rerank_id", prof.rerank_id)))
        exps.append(Experiment(s["name"], cfg, factory, notes=json.dumps(ov), judge_factory=judge_factory(a)))
    sw = run_sweep(exps, corpus, ds, max_latency_s=a.max_latency, can_ocr=a.ocr, log=log)
    return _finish(sw, a, "Model comparison")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="localpdf-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, models=False):
        p.add_argument("--dataset", help="benchmark JSONL (default: bundled synthetic dataset)")
        p.add_argument("--pdf-dir", help="directory with the supporting PDFs (default: bundled synthetic corpus)")
        p.add_argument("--out", default="reports/latest")
        p.add_argument("--ocr", action="store_true", help="enable OCR when parsing PDFs (needs tesseract)")
        p.add_argument("--max-latency", type=float, default=None, help="disqualify configurations slower than this (mean s/question)")
        p.add_argument("--profile", choices=sorted(PROFILES), default=None)
        p.add_argument("--judge", action="store_true", help="add LLM-judged faithfulness / citation support (uses the run's LLM)")
        p.add_argument("--judge-llm", default=None, help="Hugging Face id of a separate, ideally larger, judge LLM (implies --judge)")

    p = sub.add_parser("validate"); p.add_argument("--dataset"); p.add_argument("--pdf-dir"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("smoke"); p.add_argument("--out", default="reports/smoke_stand_in"); p.add_argument("--log", action="store_true"); p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("sweep"); common(p); p.add_argument("--backend", choices=["stand-in", "hf"], default="hf")
    for f in ("llm", "embed", "rerank", "quant"):
        p.add_argument(f"--{f}", default=None)
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("compare-models"); common(p); p.add_argument("--models", required=True, help="JSON list of model combinations")
    p.set_defaults(fn=cmd_compare_models)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
