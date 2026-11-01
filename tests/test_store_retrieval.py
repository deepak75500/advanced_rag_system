from localpdf.config import RAGConfig
from localpdf.retrieval import Retriever, expand_queries
from localpdf.store import EmbeddingMismatch, Filter, KnowledgeBase
from localpdf.backends import HashingEmbedder
from conftest import pages


def test_ingest_counts_and_timings(small_kb):
    s = small_kb.stats()
    assert s["documents"] == 2 and s["pages"] == 5 and s["chunks"] >= 5 and s["vectors"] == s["chunks"]


def test_identical_content_is_skipped(small_kb):
    r = small_kb.ingest_pages("copy.pdf", pages(
        "The Zephyr wind farm has 48 turbines and a capacity of 180 megawatts.",
        "Zephyr was commissioned in 2021 and is operated by the Gale company."))
    assert r.status == "skipped"


def test_new_content_same_name_bumps_version(small_kb):
    r = small_kb.ingest_pages("wind_farm.pdf", pages("Zephyr now has 60 turbines after the 2024 expansion of the park."))
    assert r.status == "updated" and r.version == 2
    assert small_kb.stats()["vectors"] == small_kb.stats()["chunks"]


def test_duplicate_chunks_within_document_dropped(kb):
    boiler = "This page intentionally contains the exact same boilerplate sentence repeated."
    r = kb.ingest_pages("dup.pdf", pages(boiler, boiler, "Something unique about turbines and capacity appears here."))
    assert r.duplicate_chunks_dropped >= 1


def test_hybrid_finds_right_page(small_kb, backends, cfg):
    ret = Retriever(small_kb, backends.embedder, backends.reranker, cfg)
    hits = ret.retrieve("What is the capacity of the Helios solar farm?")
    assert hits[0].filename == "solar_farm.pdf" and hits[0].page == 1


def test_bm25_only_and_dense_only_both_work(small_kb, backends, cfg):
    for kw in (dict(use_dense=False), dict(use_bm25=False)):
        ret = Retriever(small_kb, backends.embedder, backends.reranker, cfg.but(**kw))
        assert ret.retrieve("How many turbines does Zephyr have?")[0].filename == "wind_farm.pdf"


def test_metadata_filters_isolate_documents(small_kb, backends, cfg):
    ret = Retriever(small_kb, backends.embedder, backends.reranker, cfg)
    wind = [d for d in small_kb.list_documents() if d["filename"] == "wind_farm.pdf"][0]["id"]
    hits = ret.retrieve("capacity in megawatts", Filter(doc_ids=[wind]))
    assert hits and {h.doc_id for h in hits} == {wind}
    hits = ret.retrieve("capacity in megawatts", Filter(filenames=["solar_farm.pdf"], page_min=2))
    assert all(h.page >= 2 and h.filename == "solar_farm.pdf" for h in hits)
    assert ret.retrieve("capacity", Filter(doc_ids=[])) == []


def test_balanced_retrieval_represents_every_document(small_kb, backends, cfg):
    ret = Retriever(small_kb, backends.embedder, backends.reranker, cfg)
    ids = [d["id"] for d in small_kb.list_documents()]
    hits = ret.retrieve_balanced("capacity of the farm in megawatts", ids)
    assert {h.filename for h in hits} == {"solar_farm.pdf", "wind_farm.pdf"}


def test_delete_cleans_everything(small_kb, backends, cfg):
    d = [d for d in small_kb.list_documents() if d["filename"] == "wind_farm.pdf"][0]["id"]
    assert small_kb.delete_document(d)
    s = small_kb.stats()
    assert s["documents"] == 1 and s["vectors"] == s["chunks"]
    n_fts = small_kb.db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert n_fts == s["chunks"]
    ret = Retriever(small_kb, backends.embedder, backends.reranker, cfg)
    assert all(h.filename != "wind_farm.pdf" for h in ret.retrieve("Zephyr turbines"))
    small_kb.purge()


def test_multi_query_variants():
    v = expand_queries("What is the capacity of the farm?", ["power output of the plant"])
    assert v[0].startswith("What") and len(v) == 3


def test_embedding_model_change_is_detected(tmp_path, backends, cfg):
    db = tmp_path / "kb.sqlite3"
    k = KnowledgeBase(db, backends.embedder, cfg)
    k.ingest_pages("a.pdf", pages("Turbines produce electricity from wind energy at the site."))
    k.close()
    try:
        KnowledgeBase(db, HashingEmbedder(dim=128, name="other"), cfg)
        raise AssertionError("should have raised")
    except EmbeddingMismatch:
        pass
    k2 = KnowledgeBase(db, backends.embedder, cfg)  # same model reloads from disk
    assert k2.stats()["vectors"] == 1
