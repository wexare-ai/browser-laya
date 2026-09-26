/**
 * Request / response shapes for Laya's `system_one` API.
 *
 * These mirror `laya/agent.py::Agent.system_one` in the reference Python implementation
 * (github.com/NandhaKishorM/laya, Apache 2.0) so that a payload written for the Python
 * library works unchanged here.
 */

export type QuestionType = "choice" | "score" | "noul";

/** A criterion value: a description string, or any JSON value (rendered as compact JSON). */
export type Criterion = string | null | number | boolean | object | unknown[];

export interface ChoiceQuestion {
  type: "choice";
  instructions: string | object;
  /** option -> short description (null/"" means "no description"), or a plain list of names */
  criteria: Record<string, Criterion> | string[];
}

export interface ScoreQuestion {
  type: "score";
  instructions: string | object;
  /** ordered levels, index 0 = lowest */
  criteria: Criterion[];
}

export interface NoulQuestion {
  type: "noul";
  instructions: string | object;
  criteria?: { true?: Criterion; false?: Criterion } | null;
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export interface ActionInfo {
  /** softmax(act_logits)[0]; null when the ONNX bundle does not expose the act head */
  act_probability: number | null;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  /** 1 - normalized entropy of the answer distribution */
  confidence: number;
  action: ActionInfo;
}

export interface ScoreAnswer {
  type: "score";
  /** expected level (0 .. levels-1) */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionInfo;
}

export interface NoulAnswer {
  type: "noul";
  /** P(true) */
  noul: number;
  confidence: number;
  action: ActionInfo;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer
  : Q extends ScoreQuestion
    ? ScoreAnswer
    : NoulAnswer;

export interface SystemOneResult<
  Q extends Record<string, Question> = Record<string, Question>,
> {
  model: string;
  /** the execution provider the forward pass actually ran on */
  device: Device;
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
  usage: { input_tokens: number; output_tokens: 0 };
  /** wall-clock time of the forward pass, milliseconds */
  latency_ms: number;
}

export type Device = "webgpu" | "wasm";

/** The calibration values from the checkpoint's `rl_agent_config.json`. */
export interface LayaConfig {
  max_len: number;
  head_max_len: number;
  temperature: [number, number, number];
  temperature_by_options: Record<string, number>;
}

/** One downloadable ONNX build of a Laya checkpoint. */
export interface Bundle {
  id: string;
  label: string;
  /** the execution providers this build produces correct results on */
  devices: Device[];
  url: string;
  bytes: number;
  /** precision note shown in the UI */
  precision: string;
  /** Hugging Face repo the tokenizer + rl_agent_config.json come from */
  tokenizerUrl: string;
  tokenizerConfigUrl: string;
  configUrl: string;
  /** provenance for the README / UI */
  source: string;
  /**
   * Hard cap on options per question imposed by the export's frozen axes, or null when the
   * option axis is dynamic.
   */
  maxOptions: number | null;
}

export interface LoadProgress {
  phase: "config" | "tokenizer" | "weights" | "session" | "warmup" | "ready";
  file?: string;
  received?: number;
  total?: number;
  /** true when the bytes came from the Cache API rather than the network */
  cached?: boolean;
  message?: string;
}
