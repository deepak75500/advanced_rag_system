"""Workspaces: isolated user sessions. Each workspace owns its own SQLite database, vector index and stored originals,
so one user's documents can never appear in another's retrieval, and deleting a workspace removes everything."""
from __future__ import annotations

import secrets
import shutil
import threading
from pathlib import Path
from typing import Callable

from .backends.base import Backends
from .config import RAGConfig
from .rag import RAGPipeline
from .security import UploadRejected, valid_workspace_id
from .store import KnowledgeBase


class Workspace:
    def __init__(self, ws_id: str, root: Path, backends: Backends, cfg: RAGConfig, ctx_chars: int, max_new_tokens: int):
        self.id, self.dir = ws_id, root
        self.kb = KnowledgeBase(root / "kb.sqlite3", backends.embedder, cfg, files_dir=root / "files")
        self.pipeline = RAGPipeline(self.kb, backends, cfg, ctx_chars=ctx_chars, max_new_tokens=max_new_tokens)

    def close(self):
        self.kb.close()


class WorkspaceManager:
    def __init__(self, data_dir: str | Path, backends: Backends, cfg: RAGConfig | None = None, ctx_chars: int = 6000,
                 max_new_tokens: int = 300):
        self.root = Path(data_dir) / "workspaces"
        self.root.mkdir(parents=True, exist_ok=True)
        self.backends, self.cfg = backends, cfg or RAGConfig()
        self.ctx_chars, self.max_new_tokens = ctx_chars, max_new_tokens
        self._open: dict[str, Workspace] = {}
        self._lock = threading.Lock()

    def _path(self, ws_id: str) -> Path:
        p = (self.root / valid_workspace_id(ws_id)).resolve()
        if self.root.resolve() not in p.parents:  # belt and braces against traversal
            raise UploadRejected("invalid workspace id")
        return p

    def create(self, ws_id: str | None = None) -> Workspace:
        if ws_id is None:
            ws_id = secrets.token_urlsafe(12).replace("=", "")
        p = self._path(ws_id)
        p.mkdir(parents=True, exist_ok=True)
        return self.get(ws_id)

    def exists(self, ws_id: str) -> bool:
        try:
            return self._path(ws_id).is_dir()
        except UploadRejected:
            return False

    def get(self, ws_id: str) -> Workspace:
        p = self._path(ws_id)
        with self._lock:
            if ws_id in self._open:
                return self._open[ws_id]
            if not p.is_dir():
                raise KeyError(ws_id)
            w = Workspace(ws_id, p, self.backends, self.cfg, self.ctx_chars, self.max_new_tokens)
            self._open[ws_id] = w
            return w

    def delete(self, ws_id: str) -> bool:
        """Remove the workspace completely: database, vectors, FTS index and the stored original PDFs."""
        p = self._path(ws_id)
        with self._lock:
            w = self._open.pop(ws_id, None)
            if w:
                w.close()
        if not p.is_dir():
            return False
        shutil.rmtree(p)
        return True

    def list(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())
