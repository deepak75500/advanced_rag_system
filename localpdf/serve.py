"""Start the web app with real local models:

    python -m localpdf.serve                      # hardware profile picked automatically
    python -m localpdf.serve --profile cpu-fast   # smallest models (CPU, ~2.5 GB RAM)
    python -m localpdf.serve --stand-in           # no models: quick UI check only

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse

from .config import PROFILES, RAGConfig, default_min_relevance, detect_hardware, select_profile


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="localpdf.serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None)
    ap.add_argument("--stand-in", action="store_true", help="use stand-in backends (no model downloads)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    import uvicorn

    from .api import create_app

    if a.stand_in:
        from .backends import stand_in_backends

        backends, cfg = stand_in_backends(), RAGConfig()
    else:
        from .backends.hf import build_hf_backends

        hw = detect_hardware()
        prof = select_profile(hw, a.profile)
        print(f"Loading local models for profile '{prof.name}' on {hw['device']} "
              f"(LLM {prof.llm_id}, embedder {prof.embed_id}, reranker {prof.rerank_id}) ...")
        backends = build_hf_backends(prof, hw["device"])
        cfg = RAGConfig(min_relevance=default_min_relevance(prof.rerank_id))
    app = create_app(a.data_dir, backends, cfg=cfg)
    print(f"LocalPDF Intelligence is running at http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
