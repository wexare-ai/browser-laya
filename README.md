# System One in the browser

[![npm](https://img.shields.io/npm/v/@wexare/laya-web)](https://www.npmjs.com/package/@wexare/laya-web)
[![CI](https://github.com/wexare-ai/browser-laya/actions/workflows/publish.yml/badge.svg)](https://github.com/wexare-ai/browser-laya/actions/workflows/publish.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

[Laya](https://github.com/NandhaKishorM/laya) is a non-autoregressive "System 1" decision model:
you give it a **state** and typed **questions**, and it scores every option in a single forward
pass. It never generates text, so there is no output to parse.

This repository runs that model **entirely in a browser tab** — no server, no API key, no request
leaving the page after the weights are cached.

**[Try it in your browser →](https://wexare-ai.github.io/browser-laya/)** (a 900 MB download the
first time; WebGPU recommended)

```json
{
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
```

```
department → returns    returns .895   shipping .080   billing .025   confidence 0.643
```

## What is here

| Path | What it is |
| --- | --- |
| `packages/laya-web` | `@wexare/laya-web`, the library: tokenizer, sequence builder, ONNX Runtime Web session, decoding |
| `apps/playground` | A Vite page to paste a state and questions into and watch the distribution come back |
| `tools/gen_fixtures.py` | Generates golden fixtures by running the reference Python `laya` package |
| `tools/export_onnx.py` | Builds the default ONNX bundle from `convaiinnovations/laya` and checks it against PyTorch |

## Run the playground

The playground is hosted at **https://wexare-ai.github.io/browser-laya/**. GitHub Pages cannot send
the COOP/COEP headers that multi-threaded WebAssembly needs, so there WebGPU runs normally and the
WebAssembly fallback runs single-threaded. To run it locally, with those headers:

```sh
pnpm install
pnpm dev          # http://localhost:5173
```

Click **Load model**. The first load downloads 900 MB of weights and takes about 20 seconds on a
fast connection; after that the Cache API serves them and the model is ready in 2–3 seconds.

Add `?device=wasm` to force the CPU backend, or `?bundle=<id>` to pick a specific export. On phones
the playground picks the experimental 8-bit build (`laya-en-q8`, 633 MB, WebAssembly) on its own,
because loading the full 900 MB model needs more memory than phone browsers give a tab; it gives
the same answer as the full model on 99.3% of a 720-question test. `?bundle=laya-en-fp16`
overrides it.

## Use the library

Install it from npm:

```sh
npm install @wexare/laya-web
# or: pnpm add @wexare/laya-web
# or: yarn add @wexare/laya-web
```

It works with any bundler that supports ES modules and `new Worker(new URL(...))`. If you use the
Vite dev server, add `optimizeDeps: { exclude: ["onnxruntime-web"] }` to your Vite config; see
the [package README](packages/laya-web/README.md) for this and the other setup notes.

```ts
import { Laya } from "@wexare/laya-web";

const laya = await Laya.load({ onProgress: (p) => console.log(p) });

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
result.answers.urgency.score;            // 1.3685  (expected level, 0..2)
result.answers.churn_risk.noul;          // 0.0317  (P(true))
```

`Laya.load()` runs on the main thread. To keep the page responsive, drive it from a worker
instead:

```ts
import { LayaWorkerClient } from "@wexare/laya-web";

const laya = await LayaWorkerClient.load({
  worker: new Worker(new URL("./laya.worker.ts", import.meta.url), { type: "module" }),
});
```

where `laya.worker.ts` is one line: `import "@wexare/laya-web/worker";`

### Question types

| Type | You give | You get |
| --- | --- | --- |
| `choice` | `criteria` as `{option: description}` or a list of names | the chosen option, a probability per option, confidence |
| `score` | `criteria` as an ordered list of levels | the expected level, the distribution, confidence |
| `noul` | nothing, or `criteria: {true, false}` | a calibrated `P(true)` |

The request and response shapes match the Python library's `agent.predict`, so a payload written
for one runs on the other.

## Measured behaviour

Apple M4, Chrome, median of several runs. These were measured with the earlier fp16 export of the
same checkpoint; the current bundle (`wexare/laya-onnx`) has the same fp16 encoder plus an fp32
decision head, and its timings have not been re-measured yet.

| Run | Backend | Total | Per question |
| --- | --- | --- | --- |
| 1 question, 65 tokens | WebGPU | ~125 ms | ~125 ms |
| 5 questions, 492 tokens | WebGPU | ~1 060 ms* | ~210 ms* |
| 1 question, 65 tokens | WebAssembly | ~2 900 ms* | ~2 900 ms* |

\* From an earlier round of measurements, taken before the ~125 ms single-question figure. They
have not been re-run since, so they are likely pessimistic.

All five questions share a single forward pass, which is the same property the Python library
has. A batched request is one `predict()` call, not five.

A single question at ~125 ms is within about 3.5× of the ~35 ms the model does on a server GPU.
The remaining gap is the browser runtime, not the model: WebGPU has to run the graph with only
basic optimisation (see below), and every operator dispatch crosses the JavaScript boundary.

## Accuracy

The TypeScript port is checked against the reference Python implementation two ways, both in
`packages/laya-web/test`:

- **Tokenisation is exact.** `sequence.test.ts` asserts that the token ids and `[MASK]` marker
  positions match Python's `build_sequence` byte for byte across nine cases, including dictionary
  states, non-string criteria, falsy-but-real criteria (`0`, `false`), an option list that
  overflows the head budget, a truncated 600-token state, and non-ASCII text.
- **Answers agree.** `e2e.test.ts` runs the real ONNX bundle and compares every answer to the
  Python model's: same chosen option, same yes/no verdict, probabilities within 0.02.

```sh
pnpm test                                     # tokenisation parity, ~0.3s, offline after first run
pnpm --filter @wexare/laya-web test:e2e       # adds the end-to-end run against the real weights
```

The end-to-end run needs the 900 MB bundle. It downloads once to
`packages/laya-web/test/.model-cache/`, or set `LAYA_MODEL=/path/to/model.onnx` to point at a copy
you already have.

To regenerate the fixtures from a newer checkpoint:

```sh
uv venv -p 3.12 tools/.venv
uv pip install -p tools/.venv/bin/python laya
tools/.venv/bin/python tools/gen_fixtures.py
```

## Weights

The library loads ONNX exports of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya),
the 421M-parameter English checkpoint (ModernBERT-large encoder plus Laya's decision head). The
default bundle is our own export, built by `tools/export_onnx.py` and hosted at
[`wexare/laya-onnx`](https://huggingface.co/wexare/laya-onnx) together with a mirror of the
tokenizer and calibration config, so the library depends on one repository we control.

| Bundle id | Source | Size | Backends | Option limit |
| --- | --- | --- | --- | --- |
| `laya-en-fp16` | [`wexare/laya-onnx`](https://huggingface.co/wexare/laya-onnx) | 900 MB | WebGPU, WASM | none |
| `laya-en-int8-2opt` | [`Mattepiu/laya-onnx`](https://huggingface.co/Mattepiu/laya-onnx) | 581 MB | WASM only | **2** |

`laya-en-fp16` is the default on both backends. The int8 export is registered for completeness but
is not usable for most questions: its export froze the option axis at 2, so it can only answer
yes/no questions and two-way choices. The library refuses larger questions on it with a message
saying so rather than returning a wrong answer.

To rebuild the default bundle from the checkpoint:

```sh
uv pip install -p tools/.venv/bin/python laya onnx onnxscript onnxruntime
tools/.venv/bin/python tools/export_onnx.py      # writes tools/out/laya_fp16.onnx
```

The script exports with dynamic batch, sequence and option axes, runs the encoder in fp16 with its
LayerNorms and the decision head in fp32, and stops if any fixture's decision differs from
PyTorch. The current file differs from the PyTorch fp32 model by at most 0.00068 in probability,
passes the end-to-end parity suite on onnxruntime-web (WebAssembly), and has been checked by hand
on WebGPU in Chrome.

## Things worth knowing before you rely on this

- **First load is 900 MB.** It is cached afterwards, and the page asks for persistent storage, but
  a browser under storage pressure can still evict it.
- **WebGPU runs with basic graph optimisation only.** Skip Layer Normalization fusion is an
  extended-level optimisation, and its WebGPU kernel rejects ModernBERT's bias-free layer norms
  with `Beta must be 1D` on the first forward pass. The library therefore caps WebGPU at basic
  optimisation, which costs speed. WASM runs the fully fused graph.
- **A GPU can accept the session and still fail to run it.** The warm-up forward pass is part of
  loading for that reason: if WebGPU cannot run the model, the same weights are reopened on WASM
  and the page says the backend was downgraded.
- **Dynamic int8 is wrong on WebGPU**, silently, in onnxruntime-web 1.30
  ([microsoft/onnxruntime#32578](https://github.com/microsoft/onnxruntime/issues/32578)). That is
  why int8 bundles are marked WASM-only rather than merely discouraged.
- **English only.** The multilingual and typed-decisions checkpoints have no ONNX export yet.
- **States are truncated at 512 tokens**, and a question's options must fit 192 tokens.
- **Numbers inside a JSON state may tokenise differently** from Python for some non-integer
  floats, because JavaScript and Python format them differently.

## Adding your own export

`BUNDLES` in `packages/laya-web/src/bundles.ts` is the only place that knows about weights. A new
export needs the graph signature below and dynamic batch, sequence and option axes:

```
inputs   input_ids [B,L] int64, attention_mask [B,L] int64,
         marker_pos [B,K] int64, marker_mask [B,K] bool, qtype [B] int64
outputs  logits [B,K], and optionally the act head [B,2]
```

Exporting a smaller 4-bit bundle, or the multilingual checkpoint, is the obvious next step:
start from `tools/export_onnx.py`, then quantise with
`onnxruntime.quantization.matmul_nbits_quantizer`.

## Licences

The library and playground are MIT. `src/sequence.ts` is derived in part from
[`receptron/laya`](https://github.com/receptron/laya) (MIT). The Laya weights are Apache 2.0,
published by [Convai Innovations](https://huggingface.co/convaiinnovations).
