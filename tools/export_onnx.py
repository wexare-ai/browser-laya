"""Export convaiinnovations/laya to the ONNX bundle @wexare/laya-web loads.

Run:
    uv pip install -p tools/.venv/bin/python laya onnx onnxscript onnxruntime
    tools/.venv/bin/python tools/export_onnx.py [--out tools/out]

Steps, each checked before the next:
  1. load the Apache-2.0 checkpoint with the reference `laya` package (fp32, CPU)
  2. export fp32 with dynamic batch, sequence and option axes; compare with PyTorch
  3. export mixed precision: the ModernBERT encoder in fp16 with its LayerNorms in fp32, the
     small decision head in fp32; compare its decisions with PyTorch fp32 using the e2e test's
     tolerances (same winner, probabilities within 0.02)

Mixed precision is set up in PyTorch before export rather than by rewriting the ONNX graph
afterwards: the graph-level fp16 converters either took 30+ minutes or emitted duplicate value
names on this model.

Graph signature (what packages/laya-web/src/bundles.ts expects):
  inputs   input_ids [B,L] int64, attention_mask [B,L] int64,
           marker_pos [B,K] int64, marker_mask [B,K] bool, qtype [B] int64
  outputs  logits [B,K] float32, act_logits [B,A] float32
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "packages", "laya-web", "test", "fixtures", "answers.json")
INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUTS = ["logits", "act_logits"]
OPSET = 18


class Exportable(torch.nn.Module):
    """The decision model with a fixed positional signature and no training-only arguments."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)


class FP32LayerNorm(torch.nn.Module):
    """A LayerNorm that computes in fp32 inside an fp16 network: variance overflows fp16."""

    def __init__(self, ln):
        super().__init__()
        self.ln = ln.float()

    def forward(self, x):
        return self.ln(x.float()).to(x.dtype)


def to_mixed_precision(model):
    """fp16 encoder (about 95% of the weights) with fp32 LayerNorms; the decision head stays fp32.

    The head consumes the encoder's fp16 output; PyTorch promotes `h + type_emb(...)` to fp32, so
    everything downstream of the encoder runs in fp32 without extra casts.
    """
    m = copy.deepcopy(model)
    enc = m.model.encoder.half()
    for parent in list(enc.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.LayerNorm):
                setattr(parent, name, FP32LayerNorm(child))
    return m


def batches(agent):
    """One collated batch per fixture case (all of a case's questions together), as the library sends them."""
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, collate_items

    cfg = agent.cfg
    for case in json.load(open(FIXTURES))["cases"]:
        items = []
        for qdef in case["questions"].values():
            q = Agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, case["state"], q, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        yield case, {
            "input_ids": b["input_ids"].long(),
            "attention_mask": b["attention_mask"].long(),
            "marker_pos": b["marker_pos"].long(),
            "marker_mask": b["marker_mask"].bool(),
            "qtype": b["qtype"].long(),
        }


def decide(logits, marker_mask, qtypes, cfg):
    """Per-question probabilities, decoded exactly as the library does (temperature, softmax)."""
    from laya.common import temp_bucket

    out = []
    for r in range(logits.shape[0]):
        k = int(marker_mask[r].sum())
        qt = int(qtypes[r])
        t = cfg.get("temperature_by_options", {}).get(temp_bucket(qt, k)) or cfg["temperature"][qt]
        z = logits[r, :k].astype(np.float64) / max(1e-3, t)
        p = np.exp(z - z.max())
        out.append(p / p.sum())
    return out


def export(model, example, path, external_data):
    # torch.export tracks shapes symbolically; the older TorchScript tracer froze a sequence-length
    # reshape inside ModernBERT's attention, so its graph only ran at the traced length.
    from torch.export import Dim

    dyn = {k: {0: Dim.DYNAMIC, 1: Dim.DYNAMIC} for k in INPUTS if k != "qtype"}
    dyn["qtype"] = {0: Dim.DYNAMIC}
    with torch.no_grad():
        program = torch.onnx.export(
            model,
            tuple(example[k] for k in INPUTS),
            input_names=INPUTS,
            output_names=OUTPUTS,
            opset_version=OPSET,
            dynamo=True,
            dynamic_shapes=dyn,
            optimize=True,
        )
    program.save(path, external_data=external_data)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--skip-fp32", action="store_true", help="only build and check the fp16 bundle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import onnx
    import onnxruntime as ort
    from laya.agent import Agent

    # The fused TransformerEncoderLayer fast path is one opaque kernel; export needs the plain ops.
    torch.backends.mha.set_fastpath_enabled(False)

    print("loading", args.model, flush=True)
    agent = Agent(args.model, device="cpu")
    model = Exportable(agent.model.float().eval())
    cfg = agent.cfg
    cases = list(batches(agent))
    # a multi-question case gives the tracer B>1 and K>2, so nothing is specialised to 1 or 2
    example = max(cases, key=lambda c: (c[1]["input_ids"].shape[0], c[1]["marker_pos"].shape[1]))[1]

    ref = []
    with torch.no_grad():
        for _, feeds in cases:
            lg, _ = model(*[feeds[k] for k in INPUTS])
            ref.append(lg.numpy())

    def check(path, label):
        onnx.checker.check_model(path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        worst, ok = 0.0, True
        for (case, feeds), rlg in zip(cases, ref):
            lg, act = sess.run(OUTPUTS, {k: v.numpy() for k, v in feeds.items()})
            assert lg.dtype == np.float32 and act.dtype == np.float32, (lg.dtype, act.dtype)
            mm, qt = feeds["marker_mask"].numpy(), feeds["qtype"].numpy()
            for a, b in zip(decide(lg, mm, qt, cfg), decide(rlg, mm, qt, cfg)):
                drift = float(np.abs(a - b).max())
                worst = max(worst, drift)
                if int(a.argmax()) != int(b.argmax()) or drift >= 0.02:
                    ok = False
                    print(f"  {label} MISMATCH in {case['name']}: drift {drift:.4f}", flush=True)
        print(f"{label}: worst probability drift vs PyTorch fp32 {worst:.5f}, decisions {'all match' if ok else 'DIFFER'}", flush=True)
        return ok

    if not args.skip_fp32:
        fp32_path = os.path.join(args.out, "laya_fp32.onnx")
        print("exporting fp32", flush=True)
        export(model, example, fp32_path, external_data=True)  # 1.7 GB of weights
        if not check(fp32_path, "fp32"):
            sys.exit("fp32 export does not match PyTorch")

    fp16_path = os.path.join(args.out, "laya_fp16.onnx")
    print("exporting fp16 (mixed precision)", flush=True)
    export(to_mixed_precision(model), example, fp16_path, external_data=False)  # one file for the browser
    if not check(fp16_path, "fp16"):
        sys.exit("fp16 bundle changes decisions")

    meta = {"file": "laya_fp16.onnx", "bytes": os.path.getsize(fp16_path), "sha256": sha256(fp16_path),
            "opset": OPSET, "source": args.model, "torch": torch.__version__, "onnx": onnx.__version__,
            "onnxruntime_checked": ort.__version__}
    json.dump(meta, open(os.path.join(args.out, "export.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
