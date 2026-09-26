/**
 * Pure port of Laya's request rendering: `laya/common.py` (render_criterion, render_options,
 * build_sequence, confidence_from_probs, temp_bucket) plus `laya/agent.py::Agent._to_internal`.
 *
 * Deliberately free of any model or tokenizer dependency so it can be unit-tested against
 * golden fixtures produced by the Python implementation.
 *
 * Derived in part from receptron/laya (MIT) — see LICENSE-THIRD-PARTY.
 */
import type { Criterion, Question, QuestionType } from "./types.js";

export const QTYPES: Record<QuestionType, number> = {
  choice: 0,
  score: 1,
  noul: 2,
};
const QTYPE_NAMES: QuestionType[] = ["choice", "score", "noul"];

/** Maximum tokens kept per rendered option, before the [MASK] marker is prepended. */
const OPTION_TOKEN_CAP = 48;

export interface InternalQ {
  t: QuestionType;
  ins: string;
  crit:
    | Record<string, Criterion>
    | Criterion[]
    | { true?: Criterion; false?: Criterion }
    | null
    | undefined;
}

/** `Agent._to_internal`: a bare list of choice options becomes `{name: null}`. */
export function toInternal(q: Question): InternalQ {
  let crit: InternalQ["crit"] = q.criteria as InternalQ["crit"];
  if (q.type === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries((crit as string[]).map((c) => [c, null]));
  }
  const ins =
    typeof q.instructions === "string"
      ? q.instructions
      : pyJsonDumps(q.instructions);
  return { t: q.type, ins, crit };
}

/**
 * `common.render_criterion`: strings pass through, anything structured becomes compact JSON
 * with Python's `separators=(", ", ": ")`.
 */
export function renderCriterion(value: Criterion): string {
  if (typeof value === "string") return value;
  return pyJsonDumps(value);
}

/**
 * `common.render_options`: option texts in label-index order. `noul` is always [false, true],
 * so p[1] is P(true).
 *
 * Only `null` and `""` mean "no description"; `0` and `false` are legitimate criterion values.
 */
export function renderOptions(q: InternalQ): string[] {
  if (q.t === "choice") {
    const crit = (q.crit ?? {}) as Record<string, Criterion>;
    return Object.entries(crit).map(([k, v]) =>
      v === null || v === undefined || v === ""
        ? k
        : `${k}: ${renderCriterion(v)}`,
    );
  }
  if (q.t === "score") {
    const crit = (q.crit ?? []) as Criterion[];
    return crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  }
  const crit = (q.crit ?? {}) as { true?: Criterion; false?: Criterion };
  const f = crit.false;
  const t = crit.true;
  return [
    "false: " +
      (f === null || f === undefined || f === ""
        ? "no, the statement does not hold"
        : renderCriterion(f)),
    "true: " +
      (t === null || t === undefined || t === ""
        ? "yes, the statement holds"
        : renderCriterion(t)),
  ];
}

/**
 * Python's `json.dumps(obj, ensure_ascii=False)` with `", "` / `": "` separators and
 * insertion key order — the exact string the Python tokenizer sees.
 *
 * Known divergence: non-integer floats may format differently between Python's repr and
 * JavaScript's Number#toString for a few values.
 */
export function pyJsonDumps(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "string") return jsonString(v);
  if (typeof v === "number") {
    if (!Number.isFinite(v))
      return v > 0 ? "Infinity" : Number.isNaN(v) ? "NaN" : "-Infinity";
    return Number.isInteger(v) ? String(v) : String(v);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return "[" + v.map(pyJsonDumps).join(", ") + "]";
  return (
    "{" +
    Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${jsonString(k)}: ${pyJsonDumps(x)}`)
      .join(", ") +
    "}"
  );
}

/** JSON string literal with ensure_ascii=False semantics (non-ASCII stays literal). */
function jsonString(s: string): string {
  return JSON.stringify(s).replace(/\\u([0-9a-fA-F]{4})/g, (m, hex) => {
    const code = parseInt(hex, 16);
    // JSON.stringify only escapes control chars and surrogates beyond what Python keeps literal.
    return code < 0x20 ? m : String.fromCharCode(code);
  });
}

/** `common.serialize_state` */
export function serializeState(state: unknown): string {
  return typeof state === "string" ? state : pyJsonDumps(state);
}

/** `common.temp_bucket`: a 2-option noul and a 20-option choice need different scaling. */
export function tempBucket(qtype: number, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

/** `common.confidence_from_probs`: 1 - H(p)/log(k), clipped to [0, 1]. */
export function confidenceFromProbs(p: readonly number[], k: number): number {
  if (k < 2) return 1;
  let ent = 0;
  for (let i = 0; i < k; i++) {
    const x = p[i] ?? 0;
    ent -= x * Math.log(Math.min(Math.max(x, 1e-12), 1));
  }
  return Math.min(Math.max(1 - ent / Math.log(k), 0), 1);
}

export function softmax(z: readonly number[]): number[] {
  const zmax = Math.max(...z);
  const e = z.map((v) => Math.exp(v - zmax));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export interface SpecialIds {
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  /** literal mask token text, scrubbed from user text so it cannot inject a marker */
  maskTok: string;
}

/** Tokenizer surface the sequence builder needs: text -> ids, no special tokens added. */
export type Encode = (text: string) => number[];

export interface BuiltSequence {
  ids: number[];
  /** token index of each option's [MASK] marker */
  markers: number[];
}

/**
 * `common.build_sequence`:
 *   [CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 */
export function buildSequence(
  encode: Encode,
  ids: SpecialIds,
  state: unknown,
  q: InternalQ,
  maxLen = 512,
  headMaxLen = 192,
  optionOrder?: number[],
  truncateLeft = false,
): BuiltSequence {
  const scrub = (s: string) => s.split(ids.maskTok).join(" ");
  const opts = renderOptions(q);
  const order = optionOrder ?? opts.map((_, i) => i);

  let headIds = encode(`${q.t} question: ${scrub(q.ins)}`);
  let optIds = order.map((i) => [
    ids.mask,
    ...encode(" " + scrub(opts[i] ?? "")).slice(0, OPTION_TOKEN_CAP),
  ]);

  const total = (xs: number[][]) => xs.reduce((s, o) => s + o.length, 0);
  let optBudget = headMaxLen - total(optIds);
  if (optBudget < 16) {
    // too many / too long options: shrink every option evenly
    const per = Math.max(
      4,
      Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)),
    );
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - total(optIds);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));

  const seq: number[] = [ids.cls, ...headIds, ids.sep];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(seq.length);
    seq.push(...o);
  }
  seq.push(ids.sep);

  const room = Math.max(0, maxLen - seq.length - 1);
  const stateIds = encode(scrub(serializeState(state)));
  const st = truncateLeft
    ? stateIds.slice(stateIds.length - room)
    : stateIds.slice(0, room);
  seq.push(...st, ids.sep);

  return {
    ids: seq.slice(0, maxLen),
    markers: markers.filter((m) => m < maxLen),
  };
}

/** Round to 4 decimals, matching Python's `round(x, 4)` for the values Laya returns. */
export function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}
