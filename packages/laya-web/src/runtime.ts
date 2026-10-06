/**
 * ONNX Runtime Web session management: device probing, session creation, and the tensor
 * collation that mirrors `laya/common.py::collate_items`.
 */
import * as ort from "onnxruntime-web";
import type { BuiltSequence } from "./sequence.js";
import type { Device } from "./types.js";

export interface SessionIO {
  /** graph output name holding the option logits */
  logits: string;
  /** graph output name holding the act head, when the export exposes one */
  act: string | null;
}

/**
 * Minimal structural view of the WebGPU entry points we touch, so the library does not force
 * @webgpu/types on consumers.
 */
interface GPUAdapterLike {
  features: { has(name: string): boolean };
}
interface GPULike {
  requestAdapter(): Promise<GPUAdapterLike | null>;
}
const gpuOf = (): GPULike | undefined =>
  (globalThis.navigator as { gpu?: GPULike } | undefined)?.gpu;

/** True when this environment can actually create a WebGPU device. */
export async function webgpuAvailable(): Promise<boolean> {
  try {
    const gpu = gpuOf();
    if (!gpu) return false;
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

/** Whether the GPU reports native fp16 shader support (fp16 bundles run far better with it). */
export async function webgpuHasF16(): Promise<boolean> {
  try {
    const adapter = await gpuOf()?.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  } catch {
    return false;
  }
}

export async function pickDevice(preferred?: Device): Promise<Device> {
  if (preferred) return preferred;
  return (await webgpuAvailable()) ? "webgpu" : "wasm";
}

export function configureWasm(): void {
  const isolated =
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated ===
    true;
  const cores =
    (globalThis.navigator as { hardwareConcurrency?: number } | undefined)
      ?.hardwareConcurrency ?? 4;
  // Threads need SharedArrayBuffer, which needs COOP/COEP. Without isolation, stay single-threaded.
  ort.env.wasm.numThreads = isolated ? Math.max(1, Math.min(4, cores)) : 1;
  // We already run inside a worker; the extra proxy worker would just add a hop.
  ort.env.wasm.proxy = false;
}

export async function createSession(
  weights: Uint8Array,
  device: Device,
  extra?: ort.InferenceSession.SessionOptions,
): Promise<ort.InferenceSession> {
  if (device === "wasm") configureWasm();
  return ort.InferenceSession.create(weights, {
    executionProviders: [device],
    // Skip Layer Normalization fusion is an extended-level optimisation, and its WebGPU kernel
    // rejects ModernBERT's bias-free norms with "Beta must be 1D" on the first forward pass.
    // Basic is therefore the highest level WebGPU can take here; WASM runs the fused graph fine.
    graphOptimizationLevel: device === "webgpu" ? "basic" : "all",
    ...extra,
  });
}

/**
 * Resolve which graph outputs carry the logits and the act head.
 *
 * Community exports disagree on naming: receptron emits `logits` / `act_probs`,
 * weXare's export emits `logits` / `act_logits`, and the Mattepiu int8 build emits a
 * compiler-generated name for the act head. Match by name, then fall back to position.
 */
export function resolveIO(session: ort.InferenceSession): SessionIO {
  const names = session.outputNames;
  const logits =
    names.find(
      (n) =>
        n.toLowerCase().includes("logit") && !n.toLowerCase().startsWith("act"),
    ) ?? names[0];
  if (!logits) throw new Error("the ONNX graph exposes no outputs");
  const act = names.find((n) => n !== logits) ?? null;
  return { logits, act };
}

/** Whether the act output is already softmaxed (receptron's `act_probs`) or raw logits. */
export function actIsProbabilities(name: string | null): boolean {
  return name !== null && name.toLowerCase().includes("prob");
}

export interface CollatedBatch {
  feeds: Record<string, ort.Tensor>;
  /** padded width of the option dimension */
  K: number;
  inputTokens: number;
}

/** `laya/common.py::collate_items`: right-pad to the longest sequence and widest option set. */
export function collate(
  items: readonly BuiltSequence[],
  padId: number,
): CollatedBatch {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const K = Math.max(...items.map((it) => it.markers.length));

  const inputIds = new BigInt64Array(n * L).fill(BigInt(padId));
  const attention = new BigInt64Array(n * L);
  const markerPos = new BigInt64Array(n * K);
  const markerMask = new Uint8Array(n * K);
  const qtype = new BigInt64Array(n);
  let inputTokens = 0;

  items.forEach((it, i) => {
    it.ids.forEach((v, j) => {
      inputIds[i * L + j] = BigInt(v);
      attention[i * L + j] = 1n;
    });
    inputTokens += it.ids.length;
    it.markers.forEach((m, j) => {
      markerPos[i * K + j] = BigInt(m);
      markerMask[i * K + j] = 1;
    });
  });

  return {
    feeds: {
      input_ids: new ort.Tensor("int64", inputIds, [n, L]),
      attention_mask: new ort.Tensor("int64", attention, [n, L]),
      marker_pos: new ort.Tensor("int64", markerPos, [n, K]),
      marker_mask: new ort.Tensor("bool", markerMask, [n, K]),
      qtype: new ort.Tensor("int64", qtype, [n]),
    },
    K,
    inputTokens,
  };
}

/** Fill the qtype tensor in place (kept separate so `collate` stays about padding). */
export function setQTypes(
  batch: CollatedBatch,
  qtypes: readonly number[],
): void {
  const t = batch.feeds.qtype;
  if (!t) throw new Error("qtype tensor missing");
  const data = t.data as BigInt64Array;
  qtypes.forEach((q, i) => {
    data[i] = BigInt(q);
  });
}

/**
 * Read an output tensor as float32 regardless of how the export types its outputs.
 *
 * The fp16 bundle returns `float16` tensors. onnxruntime-web surfaces those as a native
 * `Float16Array` where the engine has one and as a `Uint16Array` of raw IEEE half bits where it
 * does not, and reading raw half bits as numbers would yield garbage, so decode explicitly.
 */
export function toFloat32(
  tensor: ort.Tensor | undefined,
  what: string,
): Float32Array {
  if (!tensor) throw new Error(`the ONNX graph did not return ${what}`);
  const d = tensor.data as ArrayLike<number> & { BYTES_PER_ELEMENT?: number };

  if (d instanceof Float32Array) return d;
  if (tensor.type === "float16") {
    // A native Float16Array already holds decoded values; a Uint16Array holds raw bits.
    const isRawBits = d instanceof Uint16Array;
    return isRawBits
      ? decodeFloat16(d)
      : Float32Array.from(d as ArrayLike<number>, Number);
  }
  if (d instanceof Float64Array || Array.isArray(d))
    return Float32Array.from(d as ArrayLike<number>, Number);
  throw new Error(`unexpected ${what} tensor type ${tensor.type}`);
}

/** IEEE 754 half-precision bits -> float32. */
export function decodeFloat16(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i] ?? 0;
    const sign = h & 0x8000 ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const fraction = h & 0x3ff;
    if (exponent === 0) {
      out[i] = sign * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1f) {
      out[i] = fraction === 0 ? sign * Infinity : NaN;
    } else {
      out[i] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }
  return out;
}
