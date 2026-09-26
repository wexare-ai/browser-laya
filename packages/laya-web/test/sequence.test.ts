/**
 * Parity tests: the TypeScript sequence builder must reproduce the reference Python
 * implementation's token ids and marker positions exactly, for every fixture case.
 *
 * Fixtures come from tools/gen_fixtures.py running the real `laya` package (0.3.3).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it, beforeAll } from "vitest";
import { Tokenizer } from "@huggingface/tokenizers";
import {
  buildSequence,
  confidenceFromProbs,
  pyJsonDumps,
  renderOptions,
  softmax,
  tempBucket,
  toInternal,
  type SpecialIds,
} from "../src/sequence.js";
import type { Question } from "../src/types.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = JSON.parse(
  readFileSync(join(here, "fixtures/sequences.json"), "utf8"),
) as {
  meta: {
    max_len: number;
    head_max_len: number;
    temperature_by_options: Record<string, number>;
  };
  cases: Array<{
    name: string;
    state: unknown;
    questions: Record<string, Question>;
    expected: Record<
      string,
      {
        rendered_options: string[];
        ids: number[];
        markers: number[];
        qtype: number;
      }
    >;
  }>;
};

/**
 * The tokenizer is fetched once from the Hub and cached on disk, so the suite stays offline
 * after the first run.
 */
const TOKENIZER_URL =
  "https://huggingface.co/convaiinnovations/laya/resolve/main/tokenizer/tokenizer.json";
const TOKENIZER_CONFIG_URL =
  "https://huggingface.co/convaiinnovations/laya/resolve/main/tokenizer/tokenizer_config.json";
const CACHE_DIR = join(here, ".tokenizer-cache");

async function cachedJson(url: string, name: string): Promise<object> {
  const { mkdirSync, existsSync, writeFileSync } = await import("node:fs");
  mkdirSync(CACHE_DIR, { recursive: true });
  const path = join(CACHE_DIR, name);
  if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8")) as object;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${name}: ${res.status}`);
  const text = await res.text();
  writeFileSync(path, text);
  return JSON.parse(text) as object;
}

let encode: (t: string) => number[];
let ids: SpecialIds;

beforeAll(async () => {
  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    cachedJson(TOKENIZER_URL, "tokenizer.json"),
    cachedJson(TOKENIZER_CONFIG_URL, "tokenizer_config.json"),
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
}, 60_000);

describe("special token ids", () => {
  it("match the ModernBERT checkpoint", () => {
    expect(ids).toMatchObject({
      cls: 50281,
      sep: 50282,
      pad: 50283,
      mask: 50284,
    });
  });
});

describe("build_sequence parity with the Python reference", () => {
  for (const testCase of fixtures.cases) {
    describe(testCase.name, () => {
      for (const [qid, expected] of Object.entries(testCase.expected)) {
        it(`${qid}: renders the same options`, () => {
          const q = toInternal(testCase.questions[qid] as Question);
          expect(renderOptions(q)).toEqual(expected.rendered_options);
        });

        it(`${qid}: produces identical token ids and markers`, () => {
          const q = toInternal(testCase.questions[qid] as Question);
          const built = buildSequence(
            encode,
            ids,
            testCase.state,
            q,
            fixtures.meta.max_len,
            fixtures.meta.head_max_len,
          );
          expect(built.markers).toEqual(expected.markers);
          expect(built.ids.length).toBe(expected.ids.length);
          expect(built.ids).toEqual(expected.ids);
        });
      }
    });
  }
});

describe("pyJsonDumps", () => {
  it("uses Python's separators and key order", () => {
    expect(pyJsonDumps({ b: 1, a: "x" })).toBe('{"b": 1, "a": "x"}');
    expect(pyJsonDumps([1, 2, "three"])).toBe('[1, 2, "three"]');
  });

  it("keeps non-ASCII literal, as ensure_ascii=False does", () => {
    expect(pyJsonDumps({ m: "Rückerstattung 支払い" })).toBe(
      '{"m": "Rückerstattung 支払い"}',
    );
  });

  it("renders Python's booleans and null", () => {
    expect(pyJsonDumps({ t: true, f: false, n: null })).toBe(
      '{"t": true, "f": false, "n": null}',
    );
  });
});

describe("tempBucket", () => {
  it("buckets by question type and option count", () => {
    expect(tempBucket(2, 2)).toBe("noul:2");
    expect(tempBucket(0, 3)).toBe("choice:3-5");
    expect(tempBucket(0, 10)).toBe("choice:6-10");
    expect(tempBucket(0, 14)).toBe("choice:11+");
    expect(tempBucket(1, 4)).toBe("score:3-5");
  });
});

describe("confidenceFromProbs", () => {
  it("is 1 for a one-hot distribution and 0 for a uniform one", () => {
    expect(confidenceFromProbs([1, 0, 0], 3)).toBeCloseTo(1, 6);
    expect(confidenceFromProbs([1 / 3, 1 / 3, 1 / 3], 3)).toBeCloseTo(0, 6);
  });
});

describe("softmax", () => {
  it("normalises and is shift-invariant", () => {
    const a = softmax([1, 2, 3]);
    const b = softmax([101, 102, 103]);
    expect(a.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i] as number, 10));
  });
});
