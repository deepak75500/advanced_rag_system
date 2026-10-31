"""Render docs/workflow.svg and docs/workflow.png: the document->answer pipeline and the benchmark pipeline."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"

def row(ax, y, title, steps, color):
    ax.text(0.2, y + 1.15, title, fontsize=13, weight="bold")
    w, gap = 2.15, 0.25
    for i, s in enumerate(steps):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, 0.95, boxstyle="round,pad=0.02", fc=color, ec="#333", lw=1))
        ax.text(x + w / 2, y + 0.475, s, ha="center", va="center", fontsize=8.5)
        if i:
            ax.annotate("", xy=(x, y + 0.475), xytext=(x - gap, y + 0.475), arrowprops=dict(arrowstyle="->", lw=1.2))

fig, ax = plt.subplots(figsize=(17, 8.2))
ax.set_xlim(0, 17.6); ax.set_ylim(0, 8.2); ax.axis("off")
row(ax, 6.0, "A. Ingestion (per workspace)", ["Upload\nsize / %PDF magic\nstructure checks", "Parse pages\n(PyMuPDF, opt. OCR)", "Clean\nheader/footer strip\ntables -> Markdown", "Parent-child chunks\n(never cross pages)", "Embed children\n+ FTS5 keyword index", "SQLite + vector file\n(isolated dir)"], "#dbe9f6")
row(ax, 3.7, "B. Question answering", ["Follow-up rewrite\nmulti-query", "Dense + BM25\nRRF fusion", "Cross-encoder rerank\ndedupe, relevance gate", "Parent expansion\ninjection sanitising", "Local LLM (Transformers)\nCPU or GPU", "Citation verification\nunsupported -> dropped\nelse abstain"], "#e4f3df")
row(ax, 1.4, "C. Benchmark / comparison lab", ["Dataset (15 categories)\nvalidated vs PDFs", "dev/test split", "Run configs / models\nsame retrieval, same data", "Metrics: retrieval,\ngeneration, extraction,\nperformance", "Select on dev,\nreport on test\npaired bootstrap CI", "Report + experiment log\n(config + dataset hash)"], "#fbefd5")
ax.text(0.2, 0.3, "Stand-in backends (hashing embedder, lexical reranker, extractive LLM) exercise the plumbing only; reports from them are labelled STAND-IN.", fontsize=9, style="italic")
fig.savefig(OUT / "workflow.svg", bbox_inches="tight"); fig.savefig(OUT / "workflow.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "workflow.svg", OUT / "workflow.png")
