/**
 * The downloadable ONNX builds of Laya that this library knows how to run.
 *
 * All of them are ONNX exports of `convaiinnovations/laya` (the English checkpoint: ModernBERT-large
 * encoder + Laya's decision head). They share a graph signature:
 *
 *   inputs : input_ids [B,L] int64, attention_mask [B,L] int64,
 *            marker_pos [B,K] int64, marker_mask [B,K] bool, qtype [B] int64
 *   outputs: logits [B,K] (masked slots = -1e4), plus an act head
 *
 * They do NOT share their dynamic axes, and that is a correctness constraint rather than a
 * preference:
 *
 *  - `laya-en-fp16` is weXare's export (tools/export_onnx.py, hosted at wexare/laya-onnx). It is
 *    dynamic in batch, sequence and option count, and checked against the Python reference.
 *  - `laya-en-q8` is an experimental 8-bit build of the same export (MatMulNBits weights), 30%
 *    smaller, for phones where the fp16 model can exhaust memory. It changed 5 of 720 answers
 *    against the PyTorch model, so it is opt-in until it has proven itself on devices.
 *  - `laya-en-int8-2opt` froze the option axis at 2 during export, so it can only answer
 *    yes/no questions and two-way choices. It also uses MatMulInteger / DynamicQuantizeLinear,
 *    which silently return wrong numbers on onnxruntime-web's WebGPU backend as of 1.30
 *    (microsoft/onnxruntime#32578, fix PR #32579 still open), so it is WASM-only.
 */
import type { Bundle, Device } from "./types.js";

/**
 * weXare's mirror of the model, tokenizer and calibration config (Apache 2.0, from
 * convaiinnovations/laya). Everything loads from one repo we control, so a third-party repo
 * disappearing cannot break installs again, which is what happened to the first fp16 export.
 */
const WEXARE_REPO = "https://huggingface.co/wexare/laya-onnx/resolve/main";

const TOKENIZER = {
  tokenizerUrl: `${WEXARE_REPO}/tokenizer/tokenizer.json`,
  tokenizerConfigUrl: `${WEXARE_REPO}/tokenizer/tokenizer_config.json`,
  configUrl: `${WEXARE_REPO}/rl_agent_config.json`,
};

export const BUNDLES: Record<string, Bundle> = {
  "laya-en-fp16": {
    id: "laya-en-fp16",
    label: "Laya English · fp16",
    devices: ["webgpu", "wasm"],
    url: `${WEXARE_REPO}/laya_fp16.onnx`,
    bytes: 900_153_849,
    precision: "fp16 encoder; LayerNorms and decision head in fp32",
    source: "wexare/laya-onnx",
    maxOptions: null,
    ...TOKENIZER,
  },
  "laya-en-q8": {
    id: "laya-en-q8",
    label: "Laya English · 8-bit (experimental)",
    devices: ["webgpu", "wasm"],
    url: `${WEXARE_REPO}/laya_q8.onnx`,
    bytes: 632_942_133,
    precision:
      "8-bit weights; agrees with the full model on 99.3% of answers (720-question test)",
    source: "wexare/laya-onnx",
    maxOptions: null,
    ...TOKENIZER,
  },
  "laya-en-int8-2opt": {
    id: "laya-en-int8-2opt",
    label: "Laya English · int8 (2 options only)",
    devices: ["wasm"],
    url: "https://huggingface.co/Mattepiu/laya-onnx/resolve/main/int8/laya_int8.onnx",
    bytes: 581_105_897,
    precision: "dynamic int8 — WASM only, and its option axis is frozen at 2",
    source: "Mattepiu/laya-onnx",
    maxOptions: 2,
    ...TOKENIZER,
  },
};

/** The build to reach for on each execution provider when the caller does not choose one. */
export const DEFAULT_BUNDLE_FOR: Record<Device, string> = {
  webgpu: "laya-en-fp16",
  wasm: "laya-en-fp16",
};

export function resolveBundle(
  idOrBundle: string | Bundle | undefined,
  device: Device,
): Bundle {
  if (idOrBundle && typeof idOrBundle !== "string") return idOrBundle;
  const id = idOrBundle ?? DEFAULT_BUNDLE_FOR[device];
  const bundle = BUNDLES[id];
  if (!bundle) {
    throw new Error(
      `unknown bundle ${JSON.stringify(id)}; known: ${Object.keys(BUNDLES).join(", ")}`,
    );
  }
  return bundle;
}

/** ModernBERT special-token ids, asserted after the tokenizer loads. */
export const EXPECTED_SPECIAL_IDS = {
  cls: 50281,
  sep: 50282,
  pad: 50283,
  mask: 50284,
};
