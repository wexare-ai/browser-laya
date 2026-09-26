/**
 * Laya in the browser.
 *
 * `Laya.load()` resolves an ONNX bundle (Cache API, then network), builds the tokenizer and
 * opens an ONNX Runtime Web session. `predict()` answers every typed question about one state
 * in a single forward pass and decodes the logits exactly as `laya/agent.py::Agent.system_one`
 * does: per-cardinality temperature, softmax, argmax / expected level / P(true), and
 * entropy-based confidence.
 */
import type * as ortTypes from "onnxruntime-web";
import { resolveBundle } from "./bundles.js";
import {
  fetchBytes,
  fetchJson,
  isCached,
  requestPersistentStorage,
  type ProgressFn,
} from "./download.js";
import {
  buildSequence,
  confidenceFromProbs,
  QTYPES,
  renderOptions,
  round4,
  softmax,
  tempBucket,
  toInternal,
  type BuiltSequence,
  type InternalQ,
} from "./sequence.js";
import {
  actIsProbabilities,
  collate,
  createSession,
  pickDevice,
  resolveIO,
  setQTypes,
  toFloat32,
  webgpuHasF16,
  type SessionIO,
} from "./runtime.js";
import { loadTokenizer, type LoadedTokenizer } from "./tokenizer.js";
import type {
  ActionInfo,
  Answer,
  Bundle,
  Criterion,
  Device,
  LayaConfig,
  Question,
  SystemOneResult,
} from "./types.js";

export interface LayaOptions {
  /** bundle id from BUNDLES, or a Bundle object for a custom export */
  bundle?: string | Bundle;
  /** force an execution provider; by default WebGPU when available, else WASM */
  device?: Device;
  onProgress?: ProgressFn;
  /** extra onnxruntime session options */
  sessionOptions?: ortTypes.InferenceSession.SessionOptions;
  /** skip the warm-up forward pass (first real call will then be slower) */
  skipWarmup?: boolean;
}

/** Everything the UI wants to show about a loaded model. */
export interface LayaInfo {
  bundleId: string;
  bundleLabel: string;
  device: Device;
  precision: string;
  source: string;
  /** option-count cap imposed by the export, or null when the option axis is dynamic */
  maxOptions: number | null;
  bytes: number;
  wasCached: boolean;
  shaderF16: boolean;
  loadMs: number;
  maxLen: number;
  headMaxLen: number;
  /** set when a faster execution provider was tried first and could not run the model */
  fellBackFrom?: { device: Device; reason: string };
}

export class Laya {
  private constructor(
    private readonly session: ortTypes.InferenceSession,
    private readonly tok: LoadedTokenizer,
    private readonly io: SessionIO,
    readonly config: LayaConfig,
    readonly info: LayaInfo,
  ) {}

  /**
   * Load a bundle and prove it works.
   *
   * WebGPU is tried first when available, but a device can accept a session and still fail on
   * the first forward pass, so the warm-up run is part of loading rather than an optimisation.
   * If it fails and the caller did not pin a device, the same weights are reopened on WASM.
   */
  static async load(opts: LayaOptions = {}): Promise<Laya> {
    const started = performance.now();
    const device = await pickDevice(opts.device);

    try {
      return await Laya.open(opts, device, started);
    } catch (err) {
      const canFallBack = !opts.device && device === "webgpu";
      if (!canFallBack) throw err;
      const reason = err instanceof Error ? err.message : String(err);
      opts.onProgress?.({
        phase: "session",
        message: "WebGPU could not run this model, falling back to WebAssembly",
      });
      const laya = await Laya.open(opts, "wasm", started);
      laya.info.fellBackFrom = { device: "webgpu", reason };
      return laya;
    }
  }

  private static async open(
    opts: LayaOptions,
    device: Device,
    started: number,
  ): Promise<Laya> {
    const bundle = resolveBundle(opts.bundle, device);

    if (!bundle.devices.includes(device)) {
      throw new Error(
        `bundle ${bundle.id} only produces correct results on ${bundle.devices.join(" / ")}, ` +
          `but the session is on ${device}. Pass device: "${bundle.devices[0]}" or pick another bundle.`,
      );
    }

    await requestPersistentStorage();
    const wasCached = await isCached(bundle.url);

    const config = await fetchJson<LayaConfig>(
      bundle.configUrl,
      opts.onProgress,
      "config",
    );
    const tok = await loadTokenizer(
      bundle.tokenizerUrl,
      bundle.tokenizerConfigUrl,
      opts.onProgress,
    );

    const weights = await fetchBytes(bundle.url, {
      phase: "weights",
      expectedBytes: bundle.bytes,
      onProgress: opts.onProgress,
    });

    opts.onProgress?.({
      phase: "session",
      message: `creating ${device} session`,
    });
    const session = await createSession(weights, device, opts.sessionOptions);
    const io = resolveIO(session);

    const laya = new Laya(session, tok, io, config, {
      bundleId: bundle.id,
      bundleLabel: bundle.label,
      device,
      precision: bundle.precision,
      source: bundle.source,
      maxOptions: bundle.maxOptions,
      bytes: weights.byteLength,
      wasCached,
      shaderF16: device === "webgpu" ? await webgpuHasF16() : false,
      loadMs: 0,
      maxLen: config.max_len,
      headMaxLen: config.head_max_len,
    });

    if (!opts.skipWarmup) {
      opts.onProgress?.({ phase: "warmup", message: "warming up" });
      try {
        await laya.predict("warm up", {
          ok: { type: "noul", instructions: "Is this a warm-up?" },
        });
      } catch (err) {
        await laya.close().catch(() => undefined);
        throw err;
      }
    }

    laya.info.loadMs = Math.round(performance.now() - started);
    opts.onProgress?.({ phase: "ready" });
    return laya;
  }

  /** Answer every question about `state` in one forward pass. */
  async predict<Q extends Record<string, Question>>(
    state: unknown,
    questions: Q,
  ): Promise<SystemOneResult<Q>> {
    const qids = Object.keys(questions);
    if (qids.length === 0)
      throw new Error("predict: at least one question is required");

    const prepared = qids.map((qid) => {
      const raw = questions[qid] as Question;
      if (!raw || !(raw.type in QTYPES)) {
        throw new Error(
          `question ${JSON.stringify(qid)}: type must be one of choice, score, noul`,
        );
      }
      const q = toInternal(raw);
      const options = renderOptions(q);
      if (options.length < 2) {
        throw new Error(
          `question ${JSON.stringify(qid)}: needs at least 2 options, got ${options.length}`,
        );
      }
      const cap = this.info.maxOptions;
      if (cap !== null && options.length > cap) {
        throw new Error(
          `question ${JSON.stringify(qid)}: the ${this.info.bundleId} export is frozen at ${cap} options, ` +
            `but this question has ${options.length}. Load the laya-en-fp16 bundle instead.`,
        );
      }
      const seq: BuiltSequence = buildSequence(
        this.tok.encode,
        this.tok.ids,
        state,
        q,
        this.config.max_len,
        this.config.head_max_len,
      );
      if (seq.markers.length !== options.length) {
        throw new Error(
          `question ${JSON.stringify(qid)}: ${options.length} options do not fit in head_max_len=${this.config.head_max_len} tokens`,
        );
      }
      return { qid, q, seq, qtype: QTYPES[q.t] };
    });

    const batch = collate(
      prepared.map((p) => p.seq),
      this.tok.ids.pad,
    );
    setQTypes(
      batch,
      prepared.map((p) => p.qtype),
    );

    const t0 = performance.now();
    const out = await this.session.run(batch.feeds);
    const latency = performance.now() - t0;

    const logits = toFloat32(out[this.io.logits], "logits");
    const actRaw = this.io.act ? toFloat32(out[this.io.act], "act") : null;
    const actDims = this.io.act ? (out[this.io.act]?.dims ?? [0, 0]) : [0, 0];
    const nAct = Number(actDims[1] ?? 0);
    const actAlreadyProbs = actIsProbabilities(this.io.act);

    const answers: Record<string, Answer> = {};
    prepared.forEach(({ qid, q, seq, qtype }, r) => {
      const k = seq.markers.length;
      const bucket = tempBucket(qtype, k);
      const temp =
        this.config.temperature_by_options[bucket] ??
        this.config.temperature[qtype] ??
        1;
      const scale = Math.max(1e-3, temp);

      const z: number[] = [];
      for (let i = 0; i < k; i++)
        z.push((logits[r * batch.K + i] ?? 0) / scale);
      const p = softmax(z);

      answers[qid] = decode(q, p, k, action(actRaw, r, nAct, actAlreadyProbs));
    });

    return {
      model: "laya",
      device: this.info.device,
      answers: answers as SystemOneResult<Q>["answers"],
      usage: { input_tokens: batch.inputTokens, output_tokens: 0 },
      latency_ms: Math.round(latency * 10) / 10,
    };
  }

  async close(): Promise<void> {
    await this.session.release();
  }
}

function action(
  act: Float32Array | null,
  row: number,
  nAct: number,
  alreadyProbs: boolean,
): ActionInfo {
  if (!act || nAct < 1) return { act_probability: null };
  // An act head exported with a frozen batch of 1 only has row 0; fall back to it.
  const base = row * nAct + nAct <= act.length ? row * nAct : 0;
  const slice: number[] = [];
  for (let i = 0; i < nAct; i++) slice.push(act[base + i] ?? 0);
  const p = alreadyProbs ? slice : softmax(slice);
  return { act_probability: round4(p[0] ?? 0) };
}

function decode(q: InternalQ, p: number[], k: number, act: ActionInfo): Answer {
  const confidence = round4(confidenceFromProbs(p, k));

  if (q.t === "choice") {
    const keys = Object.keys((q.crit ?? {}) as Record<string, Criterion>);
    let best = 0;
    for (let i = 1; i < p.length; i++)
      if ((p[i] ?? 0) > (p[best] ?? 0)) best = i;
    return {
      type: "choice",
      choice: keys[best] ?? String(best),
      probabilities: Object.fromEntries(
        keys.map((key, i) => [key, round4(p[i] ?? 0)]),
      ),
      confidence,
      action: act,
    };
  }

  if (q.t === "score") {
    const levels = (q.crit ?? []) as Criterion[];
    return {
      type: "score",
      score: round4(p.reduce((s, v, i) => s + i * v, 0)),
      legend: Object.fromEntries(
        levels.map((c, i) => [
          String(i),
          typeof c === "string" ? c : String(c),
        ]),
      ),
      probabilities: Object.fromEntries(
        p.map((v, i) => [String(i), round4(v)]),
      ),
      confidence,
      action: act,
    };
  }

  const pTrue = p[1] ?? 0;
  return {
    type: "noul",
    noul: round4(pTrue),
    confidence: round4(Math.max(pTrue, 1 - pTrue)),
    action: act,
  };
}
