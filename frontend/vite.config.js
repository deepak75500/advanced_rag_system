import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes the UI into the Python package so FastAPI serves it offline at "/".
// `npm run dev` proxies API calls to a backend started with `python -m localpdf.serve`.
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../localpdf/web", emptyOutDir: true },
  server: { proxy: { "/workspaces": api, "/system": api } },
});
