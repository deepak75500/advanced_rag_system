import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const MODES = [
  { id: "answer", label: "Answer" },
  { id: "summary", label: "Summarise" },
  { id: "concepts", label: "Key concepts" },
  { id: "questions", label: "Study questions" },
];

const WS_KEY = "pdfathom.workspace";
const store = {
  get: (k) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* private mode */
    }
  },
};

const shortName = (f) => f.replace(/\.pdf$/i, "");

export default function App() {
  const [ws, setWs] = useState(null);
  const [system, setSystem] = useState(null);
  const [docs, setDocs] = useState([]);
  const [scope, setScope] = useState(new Set()); // empty = all documents
  const [uploads, setUploads] = useState([]); // {name, state: "busy"|"done"|"error", note}
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("answer");
  const [llmCheck, setLlmCheck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null); // {doc_id, doc, page, evidence}
  const [error, setError] = useState("");
  const abortRef = useRef(null);
  const endRef = useRef(null);

  // ---- workspace bootstrap: reuse the one saved in this browser, or create a new one
  useEffect(() => {
    api.system().then(setSystem).catch(() => {});
    (async () => {
      const saved = store.get(WS_KEY);
      if (saved) {
        try {
          setDocs(await api.documents(saved));
          setWs(saved);
          return;
        } catch {
          /* workspace gone: create a new one */
        }
      }
      try {
        const { id } = await api.createWorkspace();
        store.set(WS_KEY, id);
        setWs(id);
      } catch (e) {
        setError(`Cannot reach the local server: ${e.message}`);
      }
    })();
  }, []);

  // Block body on purpose: newer browsers return a Promise from scrollIntoView, which React would treat as a cleanup.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const refreshDocs = useCallback(async () => ws && setDocs(await api.documents(ws)), [ws]);

  async function onFiles(files) {
    for (const file of files) {
      if (!/\.pdf$/i.test(file.name)) {
        setUploads((u) => [...u, { name: file.name, state: "error", note: "not a PDF" }]);
        continue;
      }
      setUploads((u) => [...u, { name: file.name, state: "busy", note: "reading & indexing…" }]);
      try {
        const r = await api.upload(ws, file);
        const note = r.status === "skipped" ? "already indexed" : `${r.pages} pages, ${r.chunks} chunks`;
        setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, state: "done", note } : x)));
      } catch (e) {
        setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, state: "error", note: e.message } : x)));
      }
      await refreshDocs();
    }
    setTimeout(() => setUploads((u) => u.filter((x) => x.state !== "done")), 4000);
  }

  async function removeDoc(d) {
    await api.deleteDocument(ws, d.id);
    setScope((s) => {
      const n = new Set(s);
      n.delete(d.id);
      return n;
    });
    if (preview?.doc_id === d.id) setPreview(null);
    await refreshDocs();
  }

  function toggleScope(id) {
    setScope((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  }

  async function ask(e) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || busy || !ws) return;
    const history = messages
      .filter((m) => m.role === "assistant" && m.final && !m.final.abstained)
      .slice(-3)
      .map((m) => [m.question, m.final.answer]);
    const id = Date.now();
    setMessages((ms) => [...ms, { id: id - 1, role: "user", text: q }, { id, role: "assistant", question: q, text: "", final: null }]);
    setQuestion("");
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const patch = (p) => setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...p } : m)));
    try {
      const final = await api.askStream(
        ws,
        { question: q, history, mode, llm_verify: llmCheck, doc_ids: scope.size ? [...scope] : null },
        { onToken: (t) => patch({ text: t }), signal: ctrl.signal },
      );
      patch({ final, text: final.answer });
    } catch (err) {
      patch({ error: err.name === "AbortError" ? "Stopped." : err.message });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  const ready = docs.length > 0;
  const models = system?.models;

  return (
    <div className={`app ${preview ? "with-preview" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="logo" aria-hidden>
            <svg viewBox="0 0 32 32">
              <path d="M10 6h9l5 5v13a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" />
              <circle cx="16" cy="17" r="3.4" />
              <path d="M18.5 19.5l2.6 2.6" />
            </svg>
          </div>
          <div>
            <h1>PDFathom</h1>
            <p>Local PDF intelligence · nothing leaves this machine</p>
          </div>
        </div>

        <DropZone disabled={!ws} onFiles={onFiles} />
        {uploads.length > 0 && (
          <ul className="uploads">
            {uploads.map((u) => (
              <li key={u.name} className={u.state}>
                <span className="dot" />
                <span className="name">{u.name}</span>
                <span className="note">{u.note}</span>
              </li>
            ))}
          </ul>
        )}

        <div className="section-title">
          Documents <span>{scope.size ? `${scope.size} selected` : "searching all"}</span>
        </div>
        {docs.length === 0 ? (
          <p className="muted small">No documents yet. Drop a PDF above — scanned PDFs are read with OCR.</p>
        ) : (
          <ul className="docs">
            {docs.map((d) => (
              <li key={d.id} className={scope.has(d.id) ? "on" : ""}>
                <label>
                  <input type="checkbox" checked={scope.has(d.id)} onChange={() => toggleScope(d.id)} />
                  <span className="doc-name" title={d.filename}>
                    {d.filename}
                  </span>
                </label>
                <span className="doc-meta">
                  {d.pages} p{d.ocr_pages ? ` · ${d.ocr_pages} OCR` : ""}
                </span>
                <button className="icon" title="Remove document" onClick={() => removeDoc(d)}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="system">
          {models?.llm ? (
            <>
              <div>
                <b>LLM</b> {models.stand_in ? "stand-in (no model)" : models.llm.split("/").pop()}
              </div>
              {!models.stand_in && (
                <>
                  <div>
                    <b>Embedder</b> {models.embedder.split("/").pop()}
                  </div>
                  <div>
                    <b>Reranker</b> {models.reranker.split("/").pop()}
                  </div>
                </>
              )}
            </>
          ) : null}
          {system?.hardware && (
            <div>
              <b>Device</b> {system.hardware.device.toUpperCase()}
              {system.hardware.gpu_name ? ` · ${system.hardware.gpu_name}` : ` · ${system.hardware.ram_gb} GB RAM`}
            </div>
          )}
        </div>
      </aside>

      <main className="chat">
        {error && <div className="banner">{error}</div>}
        <div className="messages">
          {messages.length === 0 ? (
            <Welcome ready={ready} onPick={(q) => setQuestion(q)} />
          ) : (
            messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="msg user">
                  {m.text}
                </div>
              ) : (
                <AssistantMessage key={m.id} m={m} onCite={(c) => setPreview(c)} active={preview} />
              ),
            )
          )}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={ask}>
          <div className="modes" role="tablist">
            {MODES.map((x) => (
              <button
                type="button"
                key={x.id}
                role="tab"
                aria-selected={mode === x.id}
                className={mode === x.id ? "on" : ""}
                onClick={() => setMode(x.id)}
              >
                {x.label}
              </button>
            ))}
            <label className="check" title="Ask the local LLM to confirm every sentence against its cited page (slower)">
              <input type="checkbox" checked={llmCheck} onChange={(e) => setLlmCheck(e.target.checked)} />
              Double-check with LLM
            </label>
          </div>
          <div className="input-row">
            <textarea
              rows={1}
              value={question}
              placeholder={ready ? (mode === "answer" ? "Ask about your PDFs…" : "Optional focus, e.g. 'warranty terms'") : "Upload a PDF to start"}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) ask(e);
              }}
              disabled={!ready}
            />
            {busy ? (
              <button type="button" className="send stop" onClick={() => abortRef.current?.abort()}>
                Stop
              </button>
            ) : (
              <button type="submit" className="send" disabled={!ready || (!question.trim() && mode === "answer")}>
                Ask
              </button>
            )}
          </div>
        </form>
      </main>

      {preview && <PagePreview ws={ws} c={preview} onClose={() => setPreview(null)} />}
    </div>
  );
}

function DropZone({ onFiles, disabled }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  return (
    <div
      className={`drop ${over ? "over" : ""} ${disabled ? "disabled" : ""}`}
      onClick={() => !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) onFiles([...e.dataTransfer.files]);
      }}
    >
      <input ref={input} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(e) => onFiles([...e.target.files])} />
      <strong>Drop PDFs here</strong>
      <span>or click to browse</span>
    </div>
  );
}

function Welcome({ ready, onPick }) {
  const examples = ["Who manufactures the Orion-7 pump controller?", "Compare the supply voltage in revision 1 and revision 2.", "What was approved for Marlow Quay?"];
  return (
    <div className="welcome">
      <h2>Ask your documents anything.</h2>
      <p>
        Answers come only from your PDFs, every sentence is checked against the page it cites, and when the documents don't say,
        PDFathom tells you so instead of guessing.
      </p>
      {ready && (
        <div className="examples">
          {examples.map((q) => (
            <button key={q} onClick={() => onPick(q)}>
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Answer text carries labels like "[orion_spec_v1 p.1; orion_spec_v2 p.3]": turn each into a clickable chip.
function AnswerText({ text, citations, onCite, active }) {
  const parts = text.split(/(\[[^\]]+ p\.\d+[^\]]*\])/g);
  return (
    <div className="answer-text">
      {parts.map((p, i) => {
        const m = /^\[(.*)\]$/.exec(p);
        if (!m) return <span key={i}>{p}</span>;
        return m[1].split(";").map((lab, j) => {
          const mm = /^\s*(.*) p\.(\d+)\s*$/.exec(lab);
          const c = mm && citations.find((c) => shortName(c.doc) === mm[1] && c.page === Number(mm[2]));
          if (!c) return <span key={`${i}-${j}`} className="chip dead">{lab.trim()}</span>;
          const on = active && active.doc_id === c.doc_id && active.page === c.page;
          return (
            <button key={`${i}-${j}`} className={`chip ${on ? "on" : ""}`} onClick={() => onCite(c)}>
              {shortName(c.doc)} · p.{c.page}
            </button>
          );
        });
      })}
    </div>
  );
}

function AssistantMessage({ m, onCite, active }) {
  const f = m.final;
  if (m.error)
    return (
      <div className="msg assistant">
        <div className="error">{m.error}</div>
      </div>
    );
  if (!f)
    return (
      <div className="msg assistant pending">
        {m.text ? <div className="answer-text">{m.text}</div> : <Thinking />}
        <div className="meta muted">drafting · will be verified against the sources</div>
      </div>
    );
  const conf = f.confidence?.label;
  return (
    <div className={`msg assistant ${f.abstained ? "abstained" : ""}`}>
      {f.abstained ? (
        <div className="answer-text">
          <span className="abstain-icon" aria-hidden>
            ∅
          </span>
          {f.answer}
        </div>
      ) : (
        <AnswerText text={f.answer} citations={f.citations} onCite={onCite} active={active} />
      )}
      <div className="meta">
        {conf && <span className={`badge conf-${conf.toLowerCase()}`}>{conf} confidence</span>}
        {f.verified === true && <span className="badge ok">all sentences verified</span>}
        {f.verified === false && <span className="badge warn">unsupported sentences removed</span>}
        {f.unverified_numbers?.length > 0 && (
          <span className="badge warn" title="These numbers do not appear in the sources">
            check: {f.unverified_numbers.join(", ")}
          </span>
        )}
        {f.timings?.total_s != null && <span className="muted">{f.timings.total_s.toFixed(1)} s</span>}
        {f.resolved_question && f.resolved_question !== m.question && (
          <span className="muted" title="Follow-up rewritten using the conversation">
            read as: “{f.resolved_question}”
          </span>
        )}
      </div>
      {!f.abstained && f.citations.length > 0 && (
        <div className="sources">
          {f.citations.map((c) => (
            <button key={`${c.doc_id}-${c.page}`} className="source" onClick={() => onCite(c)}>
              <span className="src-head">
                {c.doc} · page {c.page}
              </span>
              <span className="src-ev">“{c.evidence.length > 160 ? c.evidence.slice(0, 160) + "…" : c.evidence}”</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Thinking() {
  return (
    <span className="thinking" aria-label="thinking">
      <i />
      <i />
      <i />
    </span>
  );
}

function PagePreview({ ws, c, onClose }) {
  const [text, setText] = useState("");
  const [imgOk, setImgOk] = useState(true);
  useEffect(() => {
    setImgOk(true);
    api.pageText(ws, c.doc_id, c.page).then((r) => setText(r.text)).catch(() => setText(""));
  }, [ws, c.doc_id, c.page]);
  const ev = c.evidence?.trim();
  const idx = ev ? text.indexOf(ev) : -1;
  return (
    <aside className="preview">
      <header>
        <div>
          <div className="pv-doc">{c.doc}</div>
          <div className="muted small">page {c.page}</div>
        </div>
        <button className="icon" onClick={onClose} title="Close">
          ×
        </button>
      </header>
      {ev && (
        <div className="pv-evidence">
          <div className="small muted">Cited passage</div>“{ev}”
        </div>
      )}
      {imgOk && <img className="pv-img" src={api.pageImageUrl(ws, c.doc_id, c.page)} alt={`${c.doc} page ${c.page}`} onError={() => setImgOk(false)} />}
      {!imgOk && text && (
        <pre className="pv-text">
          {idx >= 0 ? (
            <>
              {text.slice(0, idx)}
              <mark>{text.slice(idx, idx + ev.length)}</mark>
              {text.slice(idx + ev.length)}
            </>
          ) : (
            text
          )}
        </pre>
      )}
    </aside>
  );
}
