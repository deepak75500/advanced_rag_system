"""FastAPI service (optional dependency).  Run:  uvicorn localpdf.api:create_app --factory

Endpoints (all scoped to a workspace id; workspaces are isolated on disk):
  POST   /workspaces                          create (returns id)        DELETE /workspaces/{ws}   delete everything
  POST   /workspaces/{ws}/documents           upload a PDF (validated)   GET /workspaces/{ws}/documents
  DELETE /workspaces/{ws}/documents/{id}      delete a document
  GET    /workspaces/{ws}/documents/{id}/pages/{n}   extracted page text (for citation preview)
  POST   /workspaces/{ws}/ask                 JSON answer;  /ask/stream  server-sent events
  GET    /system                              detected hardware and recommended profile
Status: written against the FastAPI API but NOT exercised in the environment it was developed in (no fastapi/httpx
installed); tests/test_api.py runs them when those packages are present.
"""
# No `from __future__ import annotations`: FastAPI must resolve UploadFile/AskBody, which are imported locally.
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .backends import stand_in_backends
from .backends.base import Backends
from .config import detect_hardware, select_profile
from .security import UploadRejected, safe_filename, validate_upload, clean_collection
from .store import Filter
from .workspace import WorkspaceManager


def _answer_json(a) -> dict:
    return {"answer": a.answer, "route": a.route, "abstained": a.abstained, "confidence": a.confidence,
            "verified": a.verified, "unverified_numbers": a.unverified_numbers, "timings": a.timings,
            "resolved_question": a.resolved_question,
            "citations": [{"doc": c.doc, "doc_id": c.doc_id, "page": c.page, "evidence": c.evidence, "score": c.score}
                          for c in a.citations]}


def create_app(data_dir: str = "data", backends: Backends | None = None):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class AskBody(BaseModel):
        question: str
        history: list[tuple[str, str]] = []  # [(user, assistant), ...]
        mode: str = "answer"
        beginner: bool = False
        collection: str | None = None
        doc_ids: list[int] | None = None

    app = FastAPI(title="LocalPDF Intelligence")
    mgr = WorkspaceManager(data_dir, backends or stand_in_backends())

    def ws_of(ws: str):
        try:
            return mgr.get(ws)
        except UploadRejected:
            raise HTTPException(400, "invalid workspace id")
        except KeyError:
            raise HTTPException(404, "unknown workspace")

    def flt_of(b: AskBody):
        return Filter(doc_ids=b.doc_ids, collection=clean_collection(b.collection) if b.collection else None) \
            if (b.doc_ids or b.collection) else None

    @app.post("/workspaces")
    def create_ws():
        return {"id": mgr.create().id}

    @app.delete("/workspaces/{ws}")
    def delete_ws(ws: str):
        try:
            ok = mgr.delete(ws)
        except UploadRejected:
            raise HTTPException(400, "invalid workspace id")
        if not ok:
            raise HTTPException(404, "unknown workspace")
        return {"deleted": True}

    @app.post("/workspaces/{ws}/documents")
    async def upload(ws: str, file: UploadFile = File(...), collection: str = "default"):
        w = ws_of(ws)
        name = safe_filename(file.filename or "upload.pdf")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / name
            with open(p, "wb") as f:  # stream to disk with a hard cap so oversized uploads never sit in memory
                total = 0
                while chunk := await file.read(1 << 20):
                    total += len(chunk)
                    if total > 60 * (1 << 20):
                        raise HTTPException(413, "file too large")
                    f.write(chunk)
            try:
                validate_upload(p)
                r = w.kb.ingest_pdf(p, collection=clean_collection(collection), original_name=name)
            except UploadRejected as e:
                raise HTTPException(400, str(e))
        return asdict(r) if hasattr(r, "__dataclass_fields__") else {"result": str(r)}

    @app.get("/workspaces/{ws}/documents")
    def docs(ws: str):
        return ws_of(ws).kb.list_documents()

    @app.delete("/workspaces/{ws}/documents/{doc_id}")
    def del_doc(ws: str, doc_id: int):
        w = ws_of(ws)
        if not w.kb.delete_document(doc_id):
            raise HTTPException(404, "unknown document")
        w.kb.purge()
        return {"deleted": True}

    @app.get("/workspaces/{ws}/documents/{doc_id}/pages/{n}")
    def page(ws: str, doc_id: int, n: int):
        t = ws_of(ws).kb.page_text(doc_id, n)
        if not t:
            raise HTTPException(404, "no such page")
        return {"doc_id": doc_id, "page": n, "text": t}

    @app.post("/workspaces/{ws}/ask")
    def ask(ws: str, b: AskBody):
        a = ws_of(ws).pipeline.ask(b.question, history=b.history, flt=flt_of(b), mode=b.mode, beginner=b.beginner)
        return _answer_json(a)

    @app.post("/workspaces/{ws}/ask/stream")
    def ask_stream(ws: str, b: AskBody):
        import queue
        import threading

        w, q = ws_of(ws), queue.Queue()

        def work():
            try:
                a = w.pipeline.ask(b.question, history=b.history, flt=flt_of(b), mode=b.mode, beginner=b.beginner,
                                   on_text=lambda t: q.put(("token", t)))
                q.put(("final", _answer_json(a)))  # final payload is the VERIFIED answer; streamed tokens are provisional
            except Exception as e:  # noqa: BLE001
                q.put(("error", str(e)))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def gen():
            while (m := q.get()) is not None:
                yield f"event: {m[0]}\ndata: {json.dumps(m[1])}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/system")
    def system():
        hw = detect_hardware()
        return {"hardware": hw, "recommended_profile": select_profile(hw, None).name}

    return app
