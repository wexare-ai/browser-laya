# @wexare/laya-web

Run [Laya](https://github.com/NandhaKishorM/laya), the open-source "System One" decision model,
entirely in the browser. You give it a **state** and typed **questions**, and it scores every
option in one forward pass. It returns probabilities, not generated text, so there is nothing to
parse. Inference runs on WebGPU, falling back to WebAssembly. There is no server and no API key,
and nothing leaves the page once the weights are cached.

```sh
npm i @wexare/laya-web
```

## Quick start

Run the model in a Web Worker so the page stays responsive. The worker file is one line:

```ts
// laya.worker.ts
import "@wexare/laya-web/worker";
```

```ts
import { LayaWorkerClient } from "@wexare/laya-web";

const laya = await LayaWorkerClient.load({
  worker: new Worker(new URL("./laya.worker.ts", import.meta.url), { type: "module" }),
  onProgress: (p) => console.log(p.phase, p.received, p.total),
});

const result = await laya.predict(
  { subject: "Duplicate charge", body: "We were billed twice for March." },
  {
    department: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: { billing: "invoices and refunds", technical: "bugs and outages" },
    },
    urgency: {
      type: "score",
      instructions: "How urgent is this?",
      criteria: ["not urgent", "soon", "critical"],
    },
    churn_risk: { type: "noul", instructions: "Might this customer leave?" },
  },
);

result.answers.department.choice;        // "billing"
result.answers.department.probabilities; // { billing: 0.9868, technical: 0.0132 }
result.answers.urgency.score;            // expected level, 0..2
result.answers.churn_risk.noul;          // P(true)
```

`Laya.load()` has the same API but runs on the main thread. That's fine for scripts and tests,
but the page will freeze during loading and inference.

## Question types

| Type | You give | You get |
| --- | --- | --- |
| `choice` | `criteria` as `{option: description}` or a list of names | the chosen option, a probability per option, confidence |
| `score` | `criteria` as an ordered list of levels | the expected level, the distribution, confidence |
| `noul` | nothing, or `criteria: {true, false}` | a calibrated `P(true)` |

All questions about one state are answered in a single forward pass. Request and response shapes
match the Python library's `agent.predict`, so a payload written for one runs on the other.

## Things to know

- **The first load downloads 846 MB** of weights from Hugging Face
  ([`sevenreasons/laya-onnx-fp16`](https://huggingface.co/sevenreasons/laya-onnx-fp16)). They are
  stored with the Cache API, so later loads take a few seconds. `clearCache()` removes them.
- **Backend:** WebGPU is used when available, otherwise WebAssembly (much slower). If WebGPU
  creates a session but cannot run the model, loading falls back to WebAssembly on its own, and
  `laya.info.fellBackFrom` says why. Pass `device: "wasm"` to force the CPU.
- **Multi-threaded WebAssembly** needs cross-origin isolation. Serve the page with
  `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`
  (`credentialless` keeps the Hugging Face downloads working). Without these headers, WebAssembly
  runs on a single thread. WebGPU is unaffected.
- **Vite dev server:** add `optimizeDeps: { exclude: ["onnxruntime-web"] }`. Pre-bundling
  onnxruntime-web breaks the paths it uses to load its `.wasm` files. Production builds work
  without this.
- **English only.** States are truncated at 512 tokens, and each question's options must fit
  in 192 tokens.
- Speed: about 125 ms for one question on WebGPU in Chrome on an Apple M4.

## Licences

MIT. `src/sequence.ts` is derived in part from [`receptron/laya`](https://github.com/receptron/laya)
(MIT), and `src/presets.ts` is ported from `NandhaKishorM/laya` (Apache 2.0). See
`LICENSE-THIRD-PARTY`. The Laya weights are Apache 2.0, published by
[Convai Innovations](https://huggingface.co/convaiinnovations). They are downloaded at run time
and not redistributed in this package.
