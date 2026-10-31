import json

import pytest

from localpdf.backends import stand_in_backends
from localpdf.bench import Dataset, Experiment, ExperimentLog, ablation_grid, run_sweep
from localpdf.bench.report import render, write_outputs
from localpdf.bench.schema import CATEGORIES, BenchItem, GTPage, item_from_dict, item_to_dict
from localpdf.bench.sweep import objective, paired_diff, run_experiment, subset
from localpdf.bench.synthetic import build_corpus, build_dataset, corpus_texts
from localpdf.config import RAGConfig


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_synthetic_ground_truth_is_real(corpus, dataset):
    assert dataset.check_structure() == []
    assert dataset.validate_against_corpus(corpus_texts(corpus)) == []
    s = dataset.summary()
    assert s["uncovered_categories"] == ["ocr"] and s["items"] >= 30
    assert all(any(i.split == sp and i.category == c for i in dataset.items) for c in CATEGORIES if c != "ocr" for sp in ("dev", "test"))


def test_validator_catches_fabricated_ground_truth(corpus):
    texts = corpus_texts(corpus)
    bad = [
        BenchItem("x1", "q", "a", "factual", gt_pages=[GTPage("orion_spec_v1.pdf", 1)], evidence=["a sentence that is not on the page"]),
        BenchItem("x2", "q", "a", "factual", gt_pages=[GTPage("orion_spec_v1.pdf", 99)], evidence=["Kestrel Dynamics"]),
        BenchItem("x3", "q", "a", "factual", gt_pages=[GTPage("nope.pdf", 1)], evidence=["x"]),
        BenchItem("x4", "q", "", "unanswerable", answerable=False, forbidden_terms=["Kestrel"]),
    ]
    problems = Dataset(bad).validate_against_corpus(texts)
    assert len(problems) == 4 and any("not found" in p for p in problems) and any("out of range" in p for p in problems)


def test_schema_roundtrip_flat_spec_format_and_errors(tmp_path):
    flat = {"id": "q1", "question": "Q?", "expected_answer": "A", "supporting_pdf": "My Doc.pdf", "ground_truth_pages": [3],
            "supporting_evidence": "some evidence", "category": "factual", "difficulty": "easy"}
    it = item_from_dict(flat)
    assert it.gt_pages == [GTPage("My_Doc.pdf", 3)] and it.evidence == ["some evidence"]
    assert item_from_dict(item_to_dict(it)) == it
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(flat) + "\n" + json.dumps({**flat, "id": "q2", "category": "bogus"}) + "\n")
    with pytest.raises(ValueError, match="unknown category"):
        Dataset.load(p)
    p.write_text(json.dumps({**flat, "ground_truth_pages": []}) + "\n{not json}\n")
    with pytest.raises(ValueError) as e:
        Dataset.load(p)
    assert "line 2" in str(e.value)


def test_dataset_hash_changes_with_content(dataset):
    a = dataset.sha256()
    b = Dataset([BenchItem(**{**dataset.items[0].__dict__, "expected_answer": "changed"})] + dataset.items[1:]).sha256()
    assert a != b and a == build_dataset().sha256()


def test_full_run_is_deterministic_and_reports_every_group(corpus, dataset):
    exp = Experiment("default", RAGConfig(), stand_in_backends)
    r1, r2 = run_experiment(exp, corpus, dataset), run_experiment(exp, corpus, dataset)
    a = r1.aggregate
    assert set(a) >= {"retrieval", "generation", "citation", "safety", "extraction", "performance", "by_category", "counts"}
    assert a["counts"]["items"] == len(dataset.items) and a["counts"]["skipped"] == 0
    for k in ("hit@1", "hit@5", "mrr", "ndcg@5", "context_precision", "context_recall"):
        assert 0 <= a["retrieval"][k]["mean"] <= 1
    for k in ("latency_mean_s", "first_token_mean_s", "peak_ram_mb", "indexing_s", "pdf_processing_s"):
        assert a["performance"][k] is not None
    assert a["performance"]["peak_vram_mb"] is None  # no CUDA in this sandbox: reported as n/a, not invented
    # identical inputs => identical quality metrics (latency differs)
    strip = lambda agg: {g: {k: v["mean"] for k, v in agg[g].items()} for g in ("retrieval", "generation", "citation", "safety")}
    assert strip(r1.aggregate) == strip(r2.aggregate)


def test_citations_are_never_fabricated(corpus, dataset):
    r = run_experiment(Experiment("d", RAGConfig(), stand_in_backends), corpus, dataset)
    g = r.aggregate["citation"]["grounded"]
    assert g["n"] > 0 and g["mean"] == 1.0   # every cited snippet really occurs on the cited page


def test_ocr_items_are_skipped_not_failed(corpus):
    it = BenchItem("o1", "What does the scan say?", "x", "ocr", requires=["ocr"], gt_pages=[GTPage("orion_spec_v1.pdf", 1)], ocr_reference="abc")
    ds = Dataset([it, build_dataset().items[0]])
    r = run_experiment(Experiment("d", RAGConfig(), stand_in_backends), corpus, ds)
    assert [i.status for i in r.items] == ["skipped", "ok"] and r.aggregate["counts"]["skipped"] == 1


def test_ocr_cer_item_is_scored_against_extracted_text(corpus):
    ref = corpus["orion_spec_v1.pdf"][0].text
    it = BenchItem("o2", "Who manufactures the Orion-7 pump controller?", "Kestrel Dynamics", "ocr", gt_pages=[GTPage("orion_spec_v1.pdf", 1)],
                   evidence=["Kestrel Dynamics"], ocr_reference=ref.replace("Kestrel", "Kestrol"))
    r = run_experiment(Experiment("d", RAGConfig(), stand_in_backends), corpus, Dataset([it]), can_ocr=True)
    assert 0 < r.aggregate["extraction"]["cer"]["mean"] < 0.02


def test_sweep_selects_on_dev_reports_on_test_and_flags_ties(corpus, dataset, tmp_path):
    exps = [Experiment(c.name, c, stand_in_backends) for c in ablation_grid()]
    log = ExperimentLog(tmp_path / "lab.sqlite3")
    sw = run_sweep(exps, corpus, dataset, log=log, verbose=False)
    assert sw.select_split == "dev" and sw.winner in {e.name for e in exps}
    assert all(d.aggregate["counts"]["evaluated"] < len(dataset.items) for d in sw.dev)
    assert {i.split for r in sw.test for i in r.items} == {"test"}
    assert len(log.list()) == len(exps) and log.list()[0]["stand_in"] == 1
    assert log.list()[0]["dataset_sha"] == dataset.sha256()
    if len(sw.tied) > 1:
        assert "NOT evidence" in sw.tie_note
    # the honest-comparison table: every verdict must follow from its CI
    for c in sw.comparisons:
        if c["distinguishable"]:
            assert c["ci95"][0] > 0 or c["ci95"][1] < 0
    out = write_outputs(sw, tmp_path / "rep")
    md = open(out["report"]).read()
    assert "STAND-IN BACKENDS" in md and "held-out test" in md and "Coverage gap" in md and "ocr" in md
    assert json.load(open(out["json"]))["winner"] == sw.winner


def test_latency_constraint_disqualifies(corpus, dataset):
    exps = [Experiment("a", RAGConfig(), stand_in_backends)]
    sw = run_sweep(exps, corpus, dataset, max_latency_s=1e-9, verbose=False)
    assert sw.winner is None and "a" in sw.disqualified


def test_paired_diff_detects_real_difference_and_not_noise(corpus, dataset):
    good = run_experiment(Experiment("good", RAGConfig(), stand_in_backends), corpus, dataset)
    worse = run_experiment(Experiment("worse", RAGConfig(min_relevance=0.99), stand_in_backends), corpus, dataset)  # abstains almost always
    d = paired_diff(good, worse, "f1")
    assert d["diff"] > 0 and d["distinguishable"] and d["ci95"][0] > 0
    same = paired_diff(good, good, "f1")
    assert same["diff"] == 0 and not same["distinguishable"]


def test_objective_weights_and_subset(corpus, dataset):
    r = run_experiment(Experiment("d", RAGConfig(), stand_in_backends), corpus, dataset)
    assert 0 <= objective(r.aggregate) <= 1
    only_f1 = objective(r.aggregate, {"f1": 1, "page_recall@5": 0, "mrr": 0, "correct_abstention": 0, "citation_precision": 0, "faithfulness": 0})
    assert only_f1 == pytest.approx(r.aggregate["generation"]["f1"]["mean"])
    assert subset(r, "dev").aggregate["counts"]["evaluated"] + subset(r, "test").aggregate["counts"]["evaluated"] == len(dataset.items)
