"""OCR for scanned pages and the LLM judge (pipeline verification + benchmark metrics)."""
from pathlib import Path

import pytest

from localpdf.backends import stand_in_backends
from localpdf.backends.base import Backends, Generation
from localpdf.bench.metrics import character_error_rate
from localpdf.judge import LLMJudge
from localpdf.parsing.pdf import _lines_from_boxes, _normalize_ocr
from localpdf.rag import RAGPipeline

PDFS = Path(__file__).resolve().parents[1] / "datasets" / "pdfs"


# ------------------------------------------------------------------------------------------------ OCR helpers
def test_ocr_lines_rebuilt_in_reading_order():
    box = lambda x, y: [[x, y], [x + 50, y], [x + 50, y + 18], [x, y + 18]]
    result = [[box(120, 52), "world", 0.9], [box(10, 50), "Hello", 0.95], [box(10, 90), "Next line", 0.9],
              [box(10, 130), "noise", 0.2]]  # low-confidence box is dropped
    assert _lines_from_boxes(result) == "Hello world\nNext line"


def test_ocr_normalisation_fixes_lookalikes():
    assert _normalize_ocr("丨Silent丨8丨22丨") == "|Silent|8|22|"
    assert _normalize_ocr("Flow rate（L/min）") == "Flow rate(L/min)"
    assert _normalize_ocr("Bo0st mode at 50 C, model X100") == "Boost mode at 50 C, model X100"


def test_scanned_pdf_is_read_with_ocr(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("rapidocr_onnxruntime")
    from localpdf.parsing.pdf import extract_pdf

    src = pymupdf.open(PDFS / "harbor_report.pdf")
    out = pymupdf.open()
    pix = src[0].get_pixmap(dpi=150)  # image-only page: no text layer at all
    pg = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    pg.insert_image(pg.rect, pixmap=pix)
    f = tmp_path / "scan.pdf"
    out.save(f)

    assert extract_pdf(f, ocr=False).pages[0].text.strip() == ""
    doc = extract_pdf(f, strip_hf=False)  # ocr="auto" switches OCR on because an engine is installed
    assert doc.ocr_pages == 1 and doc.pages[0].ocr
    assert character_error_rate(src[0].get_text().strip(), doc.pages[0].text) < 0.05


# ------------------------------------------------------------------------------------------------ LLM judge
class ScriptedLLM:
    """Answers questions with a fixed sentence; answers judge prompts with `verdict`."""

    id = "scripted"

    def __init__(self, answer: str, verdict: str):
        self.answer, self.verdict, self.judge_calls = answer, verdict, 0

    def generate(self, messages, max_new_tokens=300, on_text=None):
        if "fact checker" in messages[0]["content"]:
            self.judge_calls += 1
            text = self.verdict
        else:
            text = self.answer
        return Generation(text=text, first_token_s=0.0, total_s=0.0, n_tokens=1)


def _backends(llm):
    s = stand_in_backends()
    return Backends(embedder=s.embedder, reranker=s.reranker, llm=llm, is_stand_in=False)


@pytest.mark.parametrize("reply, expected", [("YES", True), (" yes.", True), ("NO", False), ("No, it is not", False),
                                              ("maybe", None)])
def test_judge_parses_verdicts(reply, expected):
    assert LLMJudge(ScriptedLLM("", reply)).supports("claim", ["passage"]) is expected


def test_judge_faithfulness_ignores_unparseable_votes():
    j = LLMJudge(ScriptedLLM("", "YES"))
    assert j.faithfulness(["a b c.", "d e f."], ["ctx"]) == 1.0
    assert str(LLMJudge(ScriptedLLM("", "hmm")).faithfulness(["a b c."], ["ctx"])) == "nan"


def test_llm_verify_drops_sentence_the_judge_rejects(small_kb, cfg):
    answer = "The Helios farm has a capacity of 240 megawatts. [1]"
    yes = ScriptedLLM(answer, "YES")
    a = RAGPipeline(small_kb, _backends(yes), cfg.but(llm_verify=True)).ask("What is the capacity of the Helios farm?")
    assert not a.abstained and "240" in a.answer and yes.judge_calls == 1

    no = ScriptedLLM(answer, "NO")
    a = RAGPipeline(small_kb, _backends(no), cfg.but(llm_verify=True)).ask("What is the capacity of the Helios farm?")
    assert a.abstained and no.judge_calls == 1


def test_llm_verify_off_by_default(small_kb, cfg):
    llm = ScriptedLLM("The Helios farm has a capacity of 240 megawatts. [1]", "NO")
    a = RAGPipeline(small_kb, _backends(llm), cfg).ask("What is the capacity of the Helios farm?")
    assert not a.abstained and llm.judge_calls == 0


def test_benchmark_reports_llm_judge_metrics():
    from localpdf.bench.runner import run_benchmark
    from localpdf.bench.synthetic import build_corpus, build_dataset
    from localpdf.bench.runner import ingest_corpus
    from localpdf.config import RAGConfig
    from localpdf.store import KnowledgeBase

    b = stand_in_backends()
    ds = build_dataset()
    kb = KnowledgeBase(":memory:", b.embedder, RAGConfig())
    try:
        ingest_corpus(kb, build_corpus())
        pipe = RAGPipeline(kb, b, RAGConfig())
        res = run_benchmark("judged", pipe, kb, b, ds, items=ds.items[:6], judge=LLMJudge(ScriptedLLM("", "YES")))
    finally:
        kb.close()
    assert res.aggregate["generation"]["llm_faithfulness"]["mean"] == 1.0
    assert res.aggregate["citation"]["llm_supported"]["mean"] == 1.0
