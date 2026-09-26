/**
 * Laya playground: load the decision model into this tab, then ask it typed questions.
 */
import {
  BUNDLES,
  LayaWorkerClient,
  PRESETS,
  clearCache,
  formatBytes,
  pickDevice,
  presetById,
  type Device,
  type LoadProgress,
  type Question,
} from "@wexare/laya-web";
import { renderEmpty, renderProblem, renderResult } from "./render.js";

const STORAGE_KEY = "laya-playground-draft";
const params = new URLSearchParams(location.search);
const forcedDevice = (params.get("device") as Device | null) ?? undefined;
const forcedBundle = params.get("bundle") ?? undefined;

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <header class="rail">
    <h1 class="wordmark">Laya <span>in the browser</span></h1>
    <div class="rail-status" id="status" role="status" aria-live="polite">
      <span id="status-text">Checking this browser</span>
      <span class="gauge" id="gauge"><i></i></span>
    </div>
    <button id="load" disabled>Load model</button>
  </header>

  <section class="intro">
    <h2>A decision model that answers in probabilities, not prose.</h2>
    <p>
      Laya reads a state and a set of typed questions, then scores every option in one forward
      pass. Nothing is generated, so there is no output to parse and nothing to hallucinate.
    </p>
    <p>
      The 421M-parameter checkpoint downloads once into this tab and runs on your own hardware.
      No request leaves the page after that.
    </p>
  </section>

  <div class="bench">
    <form id="bench-form">
      <div class="field">
        <div class="field-head">
          <h3><label for="state">State</label></h3>
          <span class="note">Plain text, or JSON for a structured record</span>
        </div>
        <textarea id="state" rows="7" spellcheck="false"></textarea>
      </div>

      <div class="field">
        <div class="field-head">
          <h3><label for="questions">Questions</label></h3>
          <span class="note" id="questions-note">choice, score or noul</span>
        </div>
        <textarea id="questions" rows="16" spellcheck="false" aria-describedby="questions-note"></textarea>
      </div>

      <div class="controls">
        <button id="run" type="submit" disabled>Run</button>
        <label class="inline" for="preset">
          Example
          <select id="preset"></select>
        </label>
        <button id="reset-cache" type="button" class="quiet">Clear cached weights</button>
      </div>
    </form>

    <section class="readout" id="readout" aria-live="polite">
      <h2 class="visually-hidden">Answers</h2>
    </section>
  </div>

  <footer class="colophon">
    <p id="colophon-text"></p>
  </footer>
`;

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
const stateInput = $<HTMLTextAreaElement>("state");
const questionsInput = $<HTMLTextAreaElement>("questions");
const presetSelect = $<HTMLSelectElement>("preset");
const loadButton = $<HTMLButtonElement>("load");
const runButton = $<HTMLButtonElement>("run");
const clearButton = $<HTMLButtonElement>("reset-cache");
const statusText = $<HTMLSpanElement>("status-text");
const statusBar = $<HTMLDivElement>("status");
const gauge = $<HTMLSpanElement>("gauge");
const gaugeFill = gauge.querySelector("i") as HTMLElement;
const readout = $<HTMLElement>("readout");
const form = $<HTMLFormElement>("bench-form");
const colophon = $<HTMLParagraphElement>("colophon-text");

let client: LayaWorkerClient | null = null;
let busy = false;

/* ---------- presets and draft persistence ---------- */

for (const preset of PRESETS) {
  presetSelect.append(new Option(preset.label, preset.id));
}

function applyPreset(id: string): void {
  const preset = presetById(id);
  if (!preset) return;
  stateInput.value =
    typeof preset.state === "string"
      ? preset.state
      : JSON.stringify(preset.state, null, 2);
  questionsInput.value = JSON.stringify(preset.questions, null, 2);
  saveDraft();
  validate();
}

function saveDraft(): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: stateInput.value,
        questions: questionsInput.value,
      }),
    );
  } catch {
    /* private mode: drafts just do not persist */
  }
}

function restoreDraft(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const draft = JSON.parse(raw) as { state?: string; questions?: string };
    if (typeof draft.state !== "string" || typeof draft.questions !== "string")
      return false;
    stateInput.value = draft.state;
    questionsInput.value = draft.questions;
    return true;
  } catch {
    return false;
  }
}

/* ---------- input parsing ---------- */

/** A state is JSON when it parses as an object or array, otherwise it is plain text. */
function parseState(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return raw;
  try {
    return JSON.parse(trimmed);
  } catch {
    return raw;
  }
}

interface QuestionsParse {
  ok: boolean;
  questions?: Record<string, Question>;
  problem?: string;
}

function parseQuestions(raw: string): QuestionsParse {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    return {
      ok: false,
      problem: err instanceof Error ? err.message : "Invalid JSON",
    };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return {
      ok: false,
      problem: "Questions must be a JSON object keyed by question id.",
    };
  }
  const entries = Object.entries(parsed as Record<string, unknown>);
  if (entries.length === 0)
    return { ok: false, problem: "Add at least one question." };

  for (const [id, value] of entries) {
    if (typeof value !== "object" || value === null) {
      return { ok: false, problem: `${id} must be an object.` };
    }
    const q = value as Record<string, unknown>;
    if (q.type !== "choice" && q.type !== "score" && q.type !== "noul") {
      return {
        ok: false,
        problem: `${id} needs a type of choice, score or noul.`,
      };
    }
    if (
      typeof q.instructions !== "string" &&
      typeof q.instructions !== "object"
    ) {
      return { ok: false, problem: `${id} needs instructions.` };
    }
    if (q.type === "choice") {
      const c = q.criteria;
      const count = Array.isArray(c)
        ? c.length
        : c && typeof c === "object"
          ? Object.keys(c).length
          : 0;
      if (count < 2)
        return {
          ok: false,
          problem: `${id} needs at least two options in criteria.`,
        };
    }
    if (
      q.type === "score" &&
      (!Array.isArray(q.criteria) || q.criteria.length < 2)
    ) {
      return {
        ok: false,
        problem: `${id} needs criteria as a list of at least two levels.`,
      };
    }
  }
  return { ok: true, questions: parsed as Record<string, Question> };
}

function validate(): boolean {
  const parsed = parseQuestions(questionsInput.value);
  questionsInput.setAttribute("aria-invalid", String(!parsed.ok));
  runButton.disabled = busy || client === null || !parsed.ok;
  return parsed.ok;
}

/* ---------- status rail ---------- */

function setStatus(
  text: string,
  chips: string[] = [],
  progress?: number | "indeterminate",
): void {
  statusText.textContent = text;
  statusBar.querySelectorAll(".chip").forEach((c) => c.remove());
  for (const chip of chips) {
    const span = document.createElement("span");
    span.className = "chip";
    if (chip === "webgpu" || chip === "wasm") span.dataset.device = chip;
    span.textContent = chip;
    statusBar.insertBefore(span, gauge);
  }
  if (progress === undefined) {
    gauge.style.display = "none";
    gauge.dataset.indeterminate = "false";
  } else {
    gauge.style.display = "";
    gauge.dataset.indeterminate = String(progress === "indeterminate");
    gaugeFill.style.width =
      progress === "indeterminate" ? "" : `${Math.round(progress * 100)}%`;
  }
}

function onProgress(p: LoadProgress): void {
  if (p.phase === "weights" && p.total) {
    const share = (p.received ?? 0) / p.total;
    const label = p.cached
      ? `Reading cached weights (${formatBytes(p.total)})`
      : `Downloading weights ${formatBytes(p.received ?? 0)} of ${formatBytes(p.total)}`;
    setStatus(label, [], share);
    return;
  }
  const phases: Record<LoadProgress["phase"], string> = {
    config: "Fetching calibration",
    tokenizer: "Fetching tokenizer",
    weights: "Fetching weights",
    session: "Starting the inference session",
    warmup: "Warming up",
    ready: "Ready",
  };
  setStatus(
    phases[p.phase],
    [],
    p.phase === "ready" ? undefined : "indeterminate",
  );
}

/* ---------- actions ---------- */

async function load(): Promise<void> {
  loadButton.disabled = true;
  busy = true;
  validate();
  try {
    const worker = new Worker(new URL("./laya.worker.ts", import.meta.url), {
      type: "module",
    });
    client = await LayaWorkerClient.load({
      worker,
      device: forcedDevice,
      bundle: forcedBundle,
      onProgress,
    });
    const {
      device,
      bundleLabel,
      bytes,
      wasCached,
      loadMs,
      shaderF16,
      fellBackFrom,
    } = client.info;
    setStatus(
      wasCached
        ? `Ready from cache in ${(loadMs / 1000).toFixed(1)}s`
        : `Ready in ${(loadMs / 1000).toFixed(1)}s`,
      [
        device,
        bundleLabel,
        formatBytes(bytes),
        ...(device === "webgpu" && !shaderF16 ? ["no shader-f16"] : []),
      ],
    );
    loadButton.hidden = true;
    if (fellBackFrom) {
      renderProblem(
        readout,
        `Running on ${device} instead of ${fellBackFrom.device}`,
        `${fellBackFrom.reason} The answers are identical, but each run takes longer. Run the questions to see for yourself.`,
      );
    } else {
      renderEmpty(
        readout,
        "Model loaded. Run the questions to see the distribution it returns.",
      );
    }
  } catch (err) {
    loadButton.disabled = false;
    setStatus("Could not load the model");
    renderProblem(
      readout,
      "The model did not load",
      err instanceof Error ? err.message : String(err),
    );
  } finally {
    busy = false;
    validate();
  }
}

async function run(): Promise<void> {
  if (!client || busy) return;
  const parsed = parseQuestions(questionsInput.value);
  if (!parsed.ok || !parsed.questions) {
    renderProblem(
      readout,
      "Check the questions",
      parsed.problem ?? "Invalid questions.",
    );
    return;
  }
  busy = true;
  validate();
  runButton.textContent = "Running";
  try {
    const result = await client.predict(
      parseState(stateInput.value),
      parsed.questions,
    );
    renderResult(readout, result);
  } catch (err) {
    renderProblem(
      readout,
      "The model could not answer",
      err instanceof Error ? err.message : String(err),
    );
  } finally {
    busy = false;
    runButton.textContent = "Run";
    validate();
  }
}

/* ---------- wiring ---------- */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void run();
});
loadButton.addEventListener("click", () => void load());
presetSelect.addEventListener("change", () => applyPreset(presetSelect.value));
stateInput.addEventListener("input", () => {
  saveDraft();
});
questionsInput.addEventListener("input", () => {
  saveDraft();
  validate();
});
clearButton.addEventListener("click", async () => {
  await clearCache();
  clearButton.textContent = "Cached weights cleared";
  setTimeout(() => (clearButton.textContent = "Clear cached weights"), 2400);
});

// Cmd/Ctrl+Enter runs from either editor.
for (const input of [stateInput, questionsInput]) {
  input.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void run();
    }
  });
}

async function boot(): Promise<void> {
  if (!restoreDraft()) applyPreset(PRESETS[0]!.id);
  validate();

  const device = await pickDevice(forcedDevice);
  const bundle =
    BUNDLES[
      forcedBundle ?? (device === "webgpu" ? "laya-en-fp16" : "laya-en-fp16")
    ]!;
  setStatus(
    `${formatBytes(bundle.bytes)} to download once, then it is cached`,
    [device],
  );
  loadButton.disabled = false;
  renderEmpty(
    readout,
    device === "webgpu"
      ? "Load the model to start. It runs on your GPU through WebGPU."
      : "Load the model to start. This browser has no WebGPU, so it will run on the CPU through WebAssembly, which is slower.",
  );
  colophon.innerHTML =
    `Weights: <a href="https://huggingface.co/${bundle.source}">${bundle.source}</a>, an ONNX export of ` +
    `<a href="https://huggingface.co/convaiinnovations/laya">convaiinnovations/laya</a> (Apache 2.0) by ` +
    `<a href="https://github.com/NandhaKishorM/laya">Convai Innovations</a>. ` +
    `Inference runs through ONNX Runtime Web. Append <code>?device=wasm</code> to compare backends.`;
}

void boot();
