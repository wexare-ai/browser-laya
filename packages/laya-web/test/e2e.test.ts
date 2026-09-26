/**
 * End-to-end check: run the real ONNX bundle through the library's own code path and compare
 * every answer against the reference Python implementation's output.
 *
 * Opt-in, because it needs the ~846 MB fp16 bundle on disk:
 *
 *   LAYA_E2E=1 LAYA_MODEL=/path/to/laya_fp16.onnx pnpm test
 *
 * With LAYA_MODEL unset the bundle is downloaded to test/.model-cache on first run.
 */
import { readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it, beforeAll } from "vitest";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web";
import {
  buildSequence,
  confidenceFromProbs,
  QTYPES,
  renderOptions,
  round4,
  softmax,
  tempBucket,
  toInternal,
  type SpecialIds,
} from "../src/sequence.js";
import { collate, resolveIO, setQTypes, toFloat32 } from "../src/runtime.js";
import { BUNDLES } from "../src/bundles.js";
import type { Question } from "../src/types.js";

const RUN = process.env.LAYA_E2E === "1";
const here = dirname(fileURLToPath(import.meta.url));
const CACHE = join(here, ".model-cache");

interface AnswerFixture {
  name: string;
  state: unknown;
  questions: Record<string, Question>;
  expected: {
    answers: Record<string, Record<string, unknown>>;
    usage: { input_tokens: number };
  };
}

const fixtures = JSON.parse(
  readFileSync(join(here, "fixtures/answers.json"), "utf8"),
) as {
  meta: {
    max_len: number;
    head_max_len: number;
    temperature: [number, number, number];
    temperature_by_options: Record<string, number>;
  };
  cases: AnswerFixture[];
};

let session: ort.InferenceSession;
let encode: (t: string) => number[];
let ids: SpecialIds;

async function cachedFile(url: string, name: string): Promise<string> {
  mkdirSync(CACHE, { recursive: true });
  const path = join(CACHE, name);
  if (existsSync(path)) return path;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
  writeFileSync(path, new Uint8Array(await res.arrayBuffer()));
  return path;
}

async function cachedJson(url: string, name: string): Promise<object> {
  mkdirSync(CACHE, { recursive: true });
  const path = join(CACHE, name);
  if (!existsSync(path)) writeFileSync(path, await (await fetch(url)).text());
  return JSON.parse(readFileSync(path, "utf8")) as object;
}

/** Run the library's own sequence + collation + decoding over the real session. */
async function predict(state: unknown, questions: Record<string, Question>) {
  const prepared = Object.entries(questions).map(([qid, raw]) => {
    const q = toInternal(raw);
    const seq = buildSequence(
      encode,
      ids,
      state,
      q,
      fixtures.meta.max_len,
      fixtures.meta.head_max_len,
    );
    return { qid, q, seq, qtype: QTYPES[q.t] };
  });
  const batch = collate(
    prepared.map((p) => p.seq),
    ids.pad,
  );
  setQTypes(
    batch,
    prepared.map((p) => p.qtype),
  );

  const io = resolveIO(session);
  const out = await session.run(batch.feeds);
  const logits = toFloat32(out[io.logits], "logits");

  const answers: Record<string, Record<string, unknown>> = {};
  prepared.forEach(({ qid, q, seq, qtype }, r) => {
    const k = seq.markers.length;
    const temp =
      fixtures.meta.temperature_by_options[tempBucket(qtype, k)] ??
      fixtures.meta.temperature[qtype] ??
      1;
    const z: number[] = [];
    for (let i = 0; i < k; i++)
      z.push((logits[r * batch.K + i] ?? 0) / Math.max(1e-3, temp));
    const p = softmax(z);

    if (q.t === "choice") {
      const keys = Object.keys((q.crit ?? {}) as Record<string, unknown>);
      let best = 0;
      for (let i = 1; i < p.length; i++)
        if ((p[i] as number) > (p[best] as number)) best = i;
      answers[qid] = {
        type: "choice",
        choice: keys[best],
        probabilities: Object.fromEntries(
          keys.map((key, i) => [key, round4(p[i] ?? 0)]),
        ),
        confidence: round4(confidenceFromProbs(p, k)),
      };
    } else if (q.t === "score") {
      answers[qid] = {
        type: "score",
        score: round4(p.reduce((s, v, i) => s + i * v, 0)),
        probabilities: Object.fromEntries(
          p.map((v, i) => [String(i), round4(v)]),
        ),
        confidence: round4(confidenceFromProbs(p, k)),
      };
    } else {
      const pTrue = p[1] ?? 0;
      answers[qid] = {
        type: "noul",
        noul: round4(pTrue),
        confidence: round4(Math.max(pTrue, 1 - pTrue)),
      };
    }
  });
  return { answers, usage: { input_tokens: batch.inputTokens } };
}

describe.skipIf(!RUN)("end-to-end against the Python reference answers", () => {
  beforeAll(async () => {
    const bundle = BUNDLES["laya-en-fp16"]!;
    const modelPath =
      process.env.LAYA_MODEL ??
      (await cachedFile(bundle.url, "laya_fp16.onnx"));
    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      cachedJson(bundle.tokenizerUrl, "tokenizer.json"),
      cachedJson(bundle.tokenizerConfigUrl, "tokenizer_config.json"),
    ]);
    const tok = new Tokenizer(tokenizerJson, tokenizerConfig);
    encode = (text: string) =>
      tok.encode(text, { add_special_tokens: false }).ids;
    ids = {
      cls: tok.token_to_id("[CLS]") as number,
      sep: tok.token_to_id("[SEP]") as number,
      mask: tok.token_to_id("[MASK]") as number,
      pad: tok.token_to_id("[PAD]") as number,
      maskTok: "[MASK]",
    };
    session = await ort.InferenceSession.create(readFileSync(modelPath), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }, 900_000);

  for (const testCase of fixtures.cases) {
    it(`${testCase.name}: matches the reference decision`, async () => {
      const got = await predict(testCase.state, testCase.questions);
      expect(got.usage.input_tokens).toBe(testCase.expected.usage.input_tokens);

      for (const [qid, want] of Object.entries(testCase.expected.answers)) {
        const mine = got.answers[qid]!;
        expect(mine.type).toBe(want.type);

        if (want.type === "choice") {
          // fp16 must not change which option wins
          expect(
            mine.choice,
            `${testCase.name}.${qid} chose a different option`,
          ).toBe(want.choice);
          const wantP = want.probabilities as Record<string, number>;
          for (const [k, v] of Object.entries(
            mine.probabilities as Record<string, number>,
          )) {
            expect(
              Math.abs(v - (wantP[k] ?? 0)),
              `${qid}.${k} probability drift`,
            ).toBeLessThan(0.02);
          }
        } else if (want.type === "score") {
          expect(
            Math.abs((mine.score as number) - (want.score as number)),
          ).toBeLessThan(0.05);
        } else {
          const wantNoul = want.noul as number;
          expect(Math.abs((mine.noul as number) - wantNoul)).toBeLessThan(0.02);
          // the yes/no verdict itself must agree
          expect((mine.noul as number) > 0.5).toBe(wantNoul > 0.5);
        }
      }
    }, 300_000);
  }
});
