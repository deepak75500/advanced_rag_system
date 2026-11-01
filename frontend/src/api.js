// Thin client for the FastAPI backend (same origin in production, proxied by Vite in dev).

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  system: () => fetch("/system").then(json),
  createWorkspace: () => fetch("/workspaces", { method: "POST" }).then(json),
  documents: (ws) => fetch(`/workspaces/${ws}/documents`).then(json),
  upload(ws, file) {
    const body = new FormData();
    body.append("file", file);
    return fetch(`/workspaces/${ws}/documents`, { method: "POST", body }).then(json);
  },
  deleteDocument: (ws, id) => fetch(`/workspaces/${ws}/documents/${id}`, { method: "DELETE" }).then(json),
  pageText: (ws, id, page) => fetch(`/workspaces/${ws}/documents/${id}/pages/${page}`).then(json),
  pageImageUrl: (ws, id, page) => `/workspaces/${ws}/documents/${id}/pages/${page}/image?zoom=1.6`,

  // Server-sent events over a POST: "token" events carry the provisional text so far, "final" the verified answer.
  async askStream(ws, body, { onToken, signal }) {
    const res = await fetch(`/workspaces/${ws}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) await json(res);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const event = /^event: (.*)$/m.exec(raw)?.[1];
        const data = /^data: (.*)$/m.exec(raw)?.[1];
        if (!event || data === undefined) continue;
        const payload = JSON.parse(data);
        if (event === "token") onToken?.(payload);
        else if (event === "final") return payload;
        else if (event === "error") throw new Error(payload);
      }
    }
    throw new Error("The answer stream ended early.");
  },
};
