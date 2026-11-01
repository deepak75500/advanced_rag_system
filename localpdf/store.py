"""Persistent knowledge base: SQLite (source of truth, FTS5/BM25) + an exact vector index derived from it.

One KnowledgeBase == one isolated workspace (its own database file, its own index, its own stored originals).
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .backends.base import Embedder
from .config import RAGConfig
from .parsing.chunking import chunk_pages
from .parsing.models import PageText
from .security import clean_collection, safe_filename, validate_upload
from .util import fts_query, normalize


class EmbeddingMismatch(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- vector index
class FlatIndex:
    """Exact inner-product search over L2-normalised vectors (numpy). Fine up to a few hundred thousand chunks;
    swap in FAISS/Qdrant behind the same three methods for more."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, dim), dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def add(self, ids, vecs):
        ids = np.asarray(ids, dtype=np.int64)
        self.ids = np.concatenate([self.ids, ids])
        self.mat = np.vstack([self.mat, np.asarray(vecs, dtype=np.float32).reshape(-1, self.dim)])

    def remove(self, ids):
        keep = ~np.isin(self.ids, np.asarray(list(ids), dtype=np.int64))
        self.ids, self.mat = self.ids[keep], self.mat[keep]

    def search(self, q, k, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if len(self.ids) == 0 or k <= 0:
            return []
        sims = self.mat @ np.asarray(q, dtype=np.float32).reshape(-1)
        if allowed is not None:
            mask = np.isin(self.ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.ids[i]), float(sims[i])) for i in top if np.isfinite(sims[i])]


# ----------------------------------------------------------------------------- filters / results
@dataclass
class Filter:
    """Metadata filter applied inside both dense and BM25 search."""
    doc_ids: list[int] | None = None
    collection: str | None = None
    filenames: list[str] | None = None
    page_min: int | None = None
    page_max: int | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.doc_ids, self.collection, self.filenames, self.page_min, self.page_max))


@dataclass
class IngestResult:
    status: str  # indexed | updated | skipped
    doc_id: int
    filename: str
    pages: int = 0
    parents: int = 0
    chunks: int = 0
    duplicate_chunks_dropped: int = 0
    version: int = 1
    message: str = ""
    # per-stage timings (seconds): these feed the "PDF processing time" and "indexing time" metrics
    parse_s: float = 0.0
    chunk_s: float = 0.0
    embed_s: float = 0.0
    index_s: float = 0.0

    @property
    def processing_s(self) -> float:
        return self.parse_s + self.chunk_s

    @property
    def indexing_s(self) -> float:
        return self.embed_s + self.index_s


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT, collection TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
    title TEXT, pages INTEGER, n_parents INTEGER, n_chunks INTEGER, ocr_pages INTEGER DEFAULT 0, version INTEGER DEFAULT 1,
    stored_path TEXT, added_at TEXT, updated_at TEXT, UNIQUE(collection, filename));
CREATE TABLE IF NOT EXISTS pages(doc_id INTEGER NOT NULL, page INTEGER NOT NULL, text TEXT, ocr INTEGER DEFAULT 0,
    PRIMARY KEY(doc_id, page));
CREATE TABLE IF NOT EXISTS parents(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, page INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, parent_id INTEGER NOT NULL, page INTEGER, idx INTEGER,
    text TEXT NOT NULL, norm_hash TEXT NOT NULL, emb BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(norm_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


class KnowledgeBase:
    def __init__(self, db_path: str | Path = ":memory:", embedder: Embedder | None = None,
                 cfg: RAGConfig | None = None, files_dir: str | Path | None = None):
        if embedder is None:
            raise ValueError("an embedder is required")
        self.embedder, self.cfg = embedder, cfg or RAGConfig()
        self.files_dir = Path(files_dir) if files_dir else None
        if self.files_dir:
            self.files_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript(SCHEMA)
            self.db.commit()
        self.version = 0  # bumped on every change (invalidates answer caches)
        self.index = FlatIndex(embedder.dim)
        self._check_embedding_model()
        self._rebuild_index()

    # ------------------------------------------------------------------ consistency
    def _check_embedding_model(self):
        cur = f"{self.embedder.id}|{self.embedder.dim}"
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key='emb_model'").fetchone()
            n = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if row and row["value"] != cur and n:
                raise EmbeddingMismatch(f"index was built with {row['value']} but embedder is {cur}; "
                                        "call reembed_all() or use a new workspace")
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('emb_model',?)", (cur,))

    def reembed_all(self, batch: int = 128):
        with self.lock:
            rows = self.db.execute("SELECT c.id, c.text, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id").fetchall()
        for i in range(0, len(rows), batch):
            b = rows[i:i + batch]
            vecs = self.embedder.embed([f"{r['title']}\n{r['text']}" for r in b])
            with self.lock, self.db:
                self.db.executemany("UPDATE chunks SET emb=? WHERE id=?", [(v.tobytes(), r["id"]) for v, r in zip(vecs, b)])
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('emb_model',?)",
                            (f"{self.embedder.id}|{self.embedder.dim}",))
        self._rebuild_index()

    def _rebuild_index(self):
        idx = FlatIndex(self.embedder.dim)
        with self.lock:
            cur = self.db.execute("SELECT id, emb FROM chunks")
            while True:
                rows = cur.fetchmany(20000)
                if not rows:
                    break
                idx.add([r["id"] for r in rows], np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows]))
            self.index = idx

    def close(self):
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------------ ingestion
    def ingest_pdf(self, path: str | Path, collection: str = "default", original_name: str | None = None,
                   ocr: bool | str = "auto", force: bool = False) -> IngestResult:
        from .parsing.pdf import extract_pdf

        path = Path(path)
        validate_upload(path)
        name = safe_filename(original_name or path.name)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        t0 = time.perf_counter()
        data = extract_pdf(path, ocr=ocr)
        parse_s = time.perf_counter() - t0
        stored = None
        if self.files_dir:
            stored = self.files_dir / f"{sha[:16]}_{name}"
            if not stored.exists():
                shutil.copy2(path, stored)
        res = self.ingest_pages(name, data.pages, collection, sha256=sha, stored_path=str(stored) if stored else None,
                                parse_s=parse_s, force=force)
        if data.warnings:
            res.message += " | " + "; ".join(data.warnings)
        return res

    def ingest_pages(self, filename: str, pages: list[PageText], collection: str = "default", title: str | None = None,
                     sha256: str | None = None, stored_path: str | None = None, parse_s: float = 0.0,
                     force: bool = False) -> IngestResult:
        """Chunk -> embed -> atomically add or replace. Embedding happens outside the lock so chat is never blocked."""
        collection, filename = clean_collection(collection), safe_filename(filename)
        sha = sha256 or hashlib.sha256("\x00".join(p.text for p in pages).encode()).hexdigest()
        title = title or filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        with self.lock:
            same_name = self.db.execute("SELECT * FROM documents WHERE collection=? AND filename=?", (collection, filename)).fetchone()
            same_sha = self.db.execute("SELECT * FROM documents WHERE collection=? AND sha256=?", (collection, sha)).fetchone()
        if not force and (same_name and same_name["sha256"] == sha or same_sha):
            d = same_name if same_name and same_name["sha256"] == sha else same_sha
            return IngestResult("skipped", d["id"], filename, message="identical content already indexed")

        t = time.perf_counter()
        parents = chunk_pages(pages, self.cfg.chunk_chars, self.cfg.chunk_overlap, self.cfg.parent_chars)
        # duplicate chunk detection (within the document: running text repeated across pages, boilerplate)
        seen: set[str] = set()
        dropped = 0
        for p in parents:
            keep = []
            for c in p.children:
                h = hashlib.sha1(normalize(c.text).encode()).hexdigest()
                c.norm_hash = h  # type: ignore[attr-defined]
                if h in seen:
                    dropped += 1
                    continue
                seen.add(h)
                keep.append(c)
            p.children = keep
        parents = [p for p in parents if p.children]
        children = [c for p in parents for c in p.children]
        if not children:
            raise ValueError("no extractable text (scanned PDF? enable OCR)")
        chunk_s = time.perf_counter() - t

        t = time.perf_counter()
        vecs = self.embedder.embed([f"{title}\n{c.text}" for c in children])
        embed_s = time.perf_counter() - t

        t = time.perf_counter()
        now = _now()
        with self.lock:
            try:
                with self.db:
                    existing = self.db.execute("SELECT * FROM documents WHERE collection=? AND filename=?", (collection, filename)).fetchone()
                    old_ids: list[int] = []
                    if existing:
                        doc_id, version = existing["id"], existing["version"] + 1
                        old_ids = self._purge_doc_rows(doc_id)
                        self.db.execute(
                            "UPDATE documents SET sha256=?,title=?,pages=?,n_parents=?,n_chunks=?,ocr_pages=?,version=?,"
                            "stored_path=?,updated_at=? WHERE id=?",
                            (sha, title, len(pages), len(parents), len(children), sum(p.ocr for p in pages), version,
                             stored_path, now, doc_id))
                    else:
                        version = 1
                        doc_id = self.db.execute(
                            "INSERT INTO documents(collection,filename,sha256,title,pages,n_parents,n_chunks,ocr_pages,version,"
                            "stored_path,added_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (collection, filename, sha, title, len(pages), len(parents), len(children),
                             sum(p.ocr for p in pages), 1, stored_path, now, now)).lastrowid
                    self.db.executemany("INSERT INTO pages(doc_id,page,text,ocr) VALUES(?,?,?,?)",
                                        [(doc_id, p.page, p.text, int(p.ocr)) for p in pages])
                    new_ids, vi = [], 0
                    for par in parents:
                        pid = self.db.execute("INSERT INTO parents(doc_id,page,text) VALUES(?,?,?)",
                                              (doc_id, par.page, par.text)).lastrowid
                        for c in par.children:
                            cid = self.db.execute(
                                "INSERT INTO chunks(doc_id,parent_id,page,idx,text,norm_hash,emb) VALUES(?,?,?,?,?,?,?)",
                                (doc_id, pid, c.page, c.idx, c.text, c.norm_hash, vecs[vi].tobytes())).lastrowid  # type: ignore[attr-defined]
                            self.db.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (cid, c.text))
                            new_ids.append(cid)
                            vi += 1
                    if old_ids:
                        self.index.remove(old_ids)
                    self.index.add(new_ids, vecs)
            except Exception:
                self._rebuild_index()
                raise
            self.version += 1
        index_s = time.perf_counter() - t
        return IngestResult("updated" if existing else "indexed", doc_id, filename, pages=len(pages),
                            parents=len(parents), chunks=len(children), duplicate_chunks_dropped=dropped, version=version,
                            parse_s=parse_s, chunk_s=chunk_s, embed_s=embed_s, index_s=index_s,
                            message=f"{len(pages)} pages, {len(children)} chunks" + (f" (v{version})" if existing else ""))

    def _purge_doc_rows(self, doc_id: int) -> list[int]:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        for t in ("chunks", "parents", "pages"):
            self.db.execute(f"DELETE FROM {t} WHERE doc_id=?", (doc_id,))
        return ids

    def delete_document(self, doc_id: int, delete_original: bool = True) -> bool:
        """Remove a document, its chunks, vectors, keyword entries and (optionally) the stored original file."""
        with self.lock:
            row = self.db.execute("SELECT stored_path FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return False
            try:
                with self.db:
                    ids = self._purge_doc_rows(doc_id)
                    self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                    if ids:
                        self.index.remove(ids)
            except Exception:
                self._rebuild_index()
                raise
            self.version += 1
        if delete_original and row["stored_path"]:
            Path(row["stored_path"]).unlink(missing_ok=True)
        return True

    def purge(self):
        """Reclaim disk space after deletions (rebuilds the FTS segments and vacuums the database)."""
        with self.lock:
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            self.db.commit()
            self.db.execute("VACUUM")

    # ------------------------------------------------------------------ catalogue
    def list_documents(self, flt: Filter | None = None) -> list[dict]:
        where, args = self._where(flt, doc_table=True)
        with self.lock:
            return [dict(r) for r in self.db.execute(f"SELECT * FROM documents {where} ORDER BY collection, filename", args)]

    def get_doc(self, doc_id: int) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def doc_pages(self, doc_id: int) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT page, text FROM pages WHERE doc_id=? ORDER BY page", (doc_id,))]

    def page_text(self, doc_id: int, n: int) -> str:
        with self.lock:
            r = self.db.execute("SELECT text FROM pages WHERE doc_id=? AND page=?", (doc_id, n)).fetchone()
        return r["text"] if r else ""

    def stats(self) -> dict:
        with self.lock:
            d = self.db.execute("SELECT COUNT(*), COALESCE(SUM(pages),0) FROM documents").fetchone()
            c = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": d[0], "pages": d[1], "chunks": c, "vectors": len(self.index)}

    # ------------------------------------------------------------------ search primitives
    @staticmethod
    def _where(flt: Filter | None, doc_table: bool = False, alias: str = "c") -> tuple[str, list]:
        if flt is None or flt.is_empty():
            return "", []
        conds, args = [], []
        did = "id" if doc_table else f"{alias}.doc_id"
        if flt.doc_ids is not None:
            if not flt.doc_ids:
                return "WHERE 0", []
            conds.append(f"{did} IN ({','.join('?' * len(flt.doc_ids))})")
            args += list(flt.doc_ids)
        if flt.collection:
            conds.append(f"{did} IN (SELECT id FROM documents WHERE collection=?)" if not doc_table else "collection=?")
            args.append(flt.collection)
        if flt.filenames:
            conds.append(f"{did} IN (SELECT id FROM documents WHERE filename IN ({','.join('?' * len(flt.filenames))}))"
                         if not doc_table else f"filename IN ({','.join('?' * len(flt.filenames))})")
            args += list(flt.filenames)
        if not doc_table:
            if flt.page_min is not None:
                conds.append(f"{alias}.page>=?")
                args.append(flt.page_min)
            if flt.page_max is not None:
                conds.append(f"{alias}.page<=?")
                args.append(flt.page_max)
        return ("WHERE " + " AND ".join(conds)) if conds else "", args

    def allowed_ids(self, flt: Filter | None) -> set[int] | None:
        where, args = self._where(flt)
        if not where:
            return None
        with self.lock:
            return {r[0] for r in self.db.execute(f"SELECT c.id FROM chunks c {where}", args)}

    def dense(self, qvec, k: int, allowed: set[int] | None) -> list[tuple[int, float]]:
        with self.lock:
            return self.index.search(qvec, k, allowed)

    def bm25(self, query: str, k: int, flt: Filter | None) -> list[tuple[int, float]]:
        fq = fts_query(query)
        if not fq:
            return []
        where, args = self._where(flt)
        sql = ("SELECT f.rowid AS id, bm25(chunks_fts) AS s FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
               "WHERE chunks_fts MATCH ? " + (" AND " + where[6:] if where else "") + " ORDER BY s LIMIT ?")
        with self.lock:
            try:
                return [(r["id"], -float(r["s"])) for r in self.db.execute(sql, [fq] + args + [k])]
            except sqlite3.OperationalError:
                return []

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        out: dict[int, dict] = {}
        with self.lock:
            for i in range(0, len(ids), 500):
                b = ids[i:i + 500]
                rows = self.db.execute(
                    "SELECT c.id, c.doc_id, c.page, c.idx, c.text, c.norm_hash, c.parent_id, p.text AS parent_text, "
                    f"d.filename, d.collection FROM chunks c JOIN parents p ON p.id=c.parent_id JOIN documents d ON d.id=c.doc_id "
                    f"WHERE c.id IN ({','.join('?' * len(b))})", b).fetchall()
                out.update({r["id"]: dict(r) for r in rows})
        return out
