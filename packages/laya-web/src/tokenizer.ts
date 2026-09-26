/**
 * The checkpoint's ModernBERT tokenizer, loaded from the Hugging Face Hub with
 * @huggingface/tokenizers (pure JS, ~8 kB, works in browsers and workers).
 */
import { Tokenizer } from "@huggingface/tokenizers";
import { EXPECTED_SPECIAL_IDS } from "./bundles.js";
import { fetchJson, type ProgressFn } from "./download.js";
import type { Encode, SpecialIds } from "./sequence.js";

export interface LoadedTokenizer {
  encode: Encode;
  ids: SpecialIds;
  raw: Tokenizer;
}

export async function loadTokenizer(
  tokenizerUrl: string,
  tokenizerConfigUrl: string,
  onProgress?: ProgressFn,
): Promise<LoadedTokenizer> {
  const [tokenizerJson, tokenizerConfig] = await Promise.all([
    fetchJson<object>(tokenizerUrl, onProgress, "tokenizer"),
    fetchJson<object>(tokenizerConfigUrl, onProgress, "tokenizer"),
  ]);

  const raw = new Tokenizer(tokenizerJson, tokenizerConfig);

  const id = (token: string): number => {
    const v = raw.token_to_id(token);
    if (v === undefined || v === null)
      throw new Error(`special token ${token} missing from the tokenizer`);
    return v;
  };

  const ids: SpecialIds = {
    cls: id("[CLS]"),
    sep: id("[SEP]"),
    mask: id("[MASK]"),
    pad: id("[PAD]"),
    maskTok: "[MASK]",
  };

  // The sequence layout hard-codes nothing, but a checkpoint whose special ids moved would
  // silently produce garbage, so fail loudly instead.
  for (const [name, expected] of Object.entries(EXPECTED_SPECIAL_IDS)) {
    const actual = ids[name as keyof typeof EXPECTED_SPECIAL_IDS];
    if (actual !== expected) {
      throw new Error(
        `tokenizer mismatch: [${name.toUpperCase()}] is ${actual}, expected ${expected}`,
      );
    }
  }

  const encode: Encode = (text: string) =>
    raw.encode(text, { add_special_tokens: false }).ids;
  return { encode, ids, raw };
}
