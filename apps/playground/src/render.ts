/**
 * Rendering the readout. A Laya answer is a distribution, so the primary mark is one strip of
 * probability mass per question: segment widths are the probabilities, and the segment the
 * model picked is the only thing drawn in the signal colour.
 */
import type { Answer, SystemOneResult } from "@wexare/laya-web";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const prob = (v: number) => v.toFixed(3).replace(/^0\./, ".");

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c);
  return node;
}

/** The distribution strip plus its legend. */
function distribution(
  entries: [string, number][],
  topIndex: number,
): DocumentFragment {
  const frag = document.createDocumentFragment();

  const strip = el("div", {
    class: "mass",
    role: "img",
    "aria-label": describe(entries),
  });
  entries.forEach(([label, p], i) => {
    const seg = el("div", {
      class: "mass-seg",
      "data-top": String(i === topIndex),
      title: `${label} ${pct(p)}`,
    });
    // flex-grow carries the probability, so the segments always sum to the full width
    seg.style.flexGrow = String(Math.max(p, 0.0005));
    seg.style.flexBasis = "0";
    strip.append(seg);
  });
  frag.append(strip);

  const legend = el("dl", { class: "legend" });
  entries.forEach(([label, p], i) => {
    const row = el("div");
    row.append(
      el("dt", { "data-top": String(i === topIndex), title: label }, [label]),
      el("dd", {}, [prob(p)]),
    );
    legend.append(row);
  });
  frag.append(legend);
  return frag;
}

function describe(entries: [string, number][]): string {
  return entries.map(([l, p]) => `${l} ${pct(p)}`).join(", ");
}

/** A thin gauge for a 0..1 reading such as confidence. */
function meter(
  label: string,
  value: number,
  display = value.toFixed(3),
): HTMLElement {
  const track = el("div", { class: "meter-track" });
  const fill = el("i");
  fill.style.width = pct(Math.min(Math.max(value, 0), 1));
  track.append(fill);
  return el("div", { class: "meter" }, [label, track, el("b", {}, [display])]);
}

function answerBlock(id: string, answer: Answer): HTMLElement {
  const block = el("article", { class: "answer" });
  const head = el("div", { class: "answer-head" }, [
    el("h3", { class: "answer-id" }, [id]),
  ]);

  if (answer.type === "choice") {
    const entries = Object.entries(answer.probabilities);
    const topIndex = entries.findIndex(([k]) => k === answer.choice);
    head.append(el("div", { class: "answer-verdict" }, [answer.choice]));
    block.append(
      head,
      distribution(entries, topIndex),
      meter("confidence", answer.confidence),
    );
    return block;
  }

  if (answer.type === "score") {
    const entries = Object.entries(answer.probabilities).map(
      ([i, p]) => [`${i} · ${answer.legend[i] ?? ""}`, p] as [string, number],
    );
    let topIndex = 0;
    entries.forEach(([, p], i) => {
      if (p > (entries[topIndex]?.[1] ?? 0)) topIndex = i;
    });
    const top = Math.max(entries.length - 1, 1);
    head.append(
      el("div", { class: "answer-verdict" }, [
        `${answer.score.toFixed(2)} / ${top}`,
      ]),
    );
    block.append(
      head,
      distribution(entries, topIndex),
      meter("expected level", answer.score / top, answer.score.toFixed(3)),
      meter("confidence", answer.confidence),
    );
    return block;
  }

  const entries: [string, number][] = [
    ["false", 1 - answer.noul],
    ["true", answer.noul],
  ];
  head.append(
    el("div", { class: "answer-verdict" }, [
      answer.noul >= 0.5 ? "true" : "false",
    ]),
  );
  // A noul's confidence is max(p, 1 - p), so a second gauge would restate P(true).
  block.append(
    head,
    distribution(entries, answer.noul >= 0.5 ? 1 : 0),
    meter("P(true)", answer.noul, prob(answer.noul)),
  );
  return block;
}

export function renderResult(host: HTMLElement, result: SystemOneResult): void {
  host.replaceChildren();
  for (const [id, answer] of Object.entries(result.answers)) {
    host.append(answerBlock(id, answer as Answer));
  }

  const nQuestions = Object.keys(result.answers).length;
  const facts = [
    `${result.latency_ms} ms`,
    `${nQuestions} question${nQuestions === 1 ? "" : "s"} in one forward pass`,
    `${result.usage.input_tokens} input tokens`,
    "0 tokens generated",
    result.device,
  ];
  host.append(
    el(
      "div",
      { class: "footline" },
      facts.map((f) => el("span", {}, [f])),
    ),
  );

  const raw = el("details", { class: "raw" });
  raw.append(
    el("summary", {}, ["Response JSON"]),
    el("pre", {}, [JSON.stringify(result, null, 2)]),
  );
  host.append(raw);
}

export function renderProblem(
  host: HTMLElement,
  title: string,
  detail: string,
): void {
  host.replaceChildren(
    el("div", { class: "problem" }, [
      el("h4", {}, [title]),
      el("div", {}, [detail]),
    ]),
  );
}

export function renderEmpty(host: HTMLElement, message: string): void {
  host.replaceChildren(el("p", { class: "readout-empty" }, [message]));
}
