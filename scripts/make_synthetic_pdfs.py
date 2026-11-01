"""Render the synthetic corpus to real PDFs (datasets/pdfs/) and VERIFY the ground truth against the written files.

Rendering: matplotlib (TrueType fonts embedded, so the text layer is extractable).
Verification (independent of this package's parser): `pdftotext -f N -l N` (poppler) must show every dataset item's
evidence on its ground-truth page, and each PDF must have the expected number of pages.

Note: tables are rendered as pipe-delimited text lines, not ruled tables, so the PDFs exercise page/ground-truth
alignment and the table-row chunker, but NOT PyMuPDF's ruled-table detection (use real PDFs for that).

    python scripts/make_synthetic_pdfs.py [--out datasets/pdfs] [--no-verify]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from localpdf.bench.synthetic import build_corpus, build_dataset, corpus_texts  # noqa: E402
from localpdf.util import normalize  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType: extractable text


def render(corpus, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, pages in corpus.items():
        with PdfPages(out / name, metadata={"Title": name.replace(".pdf", ""), "Creator": "localpdf-intelligence synthetic corpus"}) as pdf:
            for p in pages:
                fig = plt.figure(figsize=(8.27, 11.69))
                lines = []
                for raw in p.text.split("\n"):
                    lines += [raw] if raw.startswith("|") else (textwrap.wrap(raw, 88) or [""])
                y = 0.94
                for ln in lines:
                    fig.text(0.08, y, ln, fontsize=9.5, family="DejaVu Sans Mono" if ln.startswith("|") else "DejaVu Sans", va="top")
                    y -= 0.022
                fig.text(0.5, 0.03, f"{name}  -  page {p.page}", ha="center", fontsize=8, color="gray")
                pdf.savefig(fig)
                plt.close(fig)


def verify(ds, corpus, out: Path) -> list[str]:
    if not shutil.which("pdftotext"):
        return ["pdftotext not installed: PDFs written but NOT verified"]
    problems = []
    texts = {}
    for name, pages in corpus.items():
        n = len(pages)
        pages_txt = []
        for i in range(1, n + 1):
            r = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(out / name), "-"], capture_output=True, text=True)
            pages_txt.append(normalize(r.stdout))
        info = subprocess.run(["pdfinfo", str(out / name)], capture_output=True, text=True).stdout if shutil.which("pdfinfo") else ""
        if info and f"Pages:           {n}" not in info:
            problems.append(f"{name}: expected {n} pages, pdfinfo says: {[l for l in info.splitlines() if l.startswith('Pages')]}")
        texts[name] = pages_txt
    for it in ds.items:
        if not it.answerable:
            continue
        on_pages = " ".join(texts[g.pdf][g.page - 1] for g in it.gt_pages)
        for ev in it.evidence:
            if normalize(ev) not in on_pages:
                problems.append(f"{it.id}: evidence not found by pdftotext on {[(g.pdf, g.page) for g in it.gt_pages]}: {ev[:50]!r}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "datasets" / "pdfs"))
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    corpus, ds = build_corpus(), build_dataset()
    assert not ds.validate_against_corpus(corpus_texts(corpus))
    out = Path(a.out)
    render(corpus, out)
    ds.save(out.parent / "synthetic_v1.jsonl")
    print(f"wrote {len(corpus)} PDFs to {out} and {out.parent / 'synthetic_v1.jsonl'}")
    if not a.no_verify:
        problems = verify(ds, corpus, out)
        print("VERIFIED against the PDF files: every evidence passage is on its ground-truth page" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)
