import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

const lib = (p: string) => fileURLToPath(new URL(`../../packages/laya-web/src/${p}`, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      // Point at the library source so the playground hot-reloads without a build step.
      "@wexare/laya-web/worker": lib("worker.ts"),
      "@wexare/laya-web": lib("index.ts"),
    },
  },
  worker: { format: "es" },
  optimizeDeps: {
    // onnxruntime-web ships its own wasm loader; prebundling it breaks the asset paths.
    exclude: ["onnxruntime-web"],
  },
  server: {
    headers: {
      // SharedArrayBuffer (multi-threaded WASM) needs cross-origin isolation.
      // `credentialless` keeps the CORS fetches to huggingface.co working.
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});
