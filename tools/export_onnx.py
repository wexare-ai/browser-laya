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
  4. with --q4: quantize the fp32 export's MatMul weights to 4 bits (ONNX Runtime MatMulNBits,
     block 32) for phones and the WebAssembly path, and run the same comparison

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
    ap.add_argument("--skip-fp32", action="store_true", help="reuse out/laya_fp32.onnx instead of re-exporting it")
    ap.add_argument("--skip-fp16", action="store_true", help="do not rebuild the fp16 bundle")
    ap.add_argument("--q4", action="store_true", help="also build the 4-bit bundle from the fp32 export")
    ap.add_argument("--q4-block", type=int, default=32)
    ap.add_argument("--q4-exclude-head", action="store_true", help="keep the decision head's MatMuls in fp32")
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
        worst, ok, flips, n = 0.0, True, 0, 0
        for (case, feeds), rlg in zip(cases, ref):
            lg, act = sess.run(OUTPUTS, {k: v.numpy() for k, v in feeds.items()})
            assert lg.dtype == np.float32 and act.dtype == np.float32, (lg.dtype, act.dtype)
            mm, qt = feeds["marker_mask"].numpy(), feeds["qtype"].numpy()
            for a, b in zip(decide(lg, mm, qt, cfg), decide(rlg, mm, qt, cfg)):
                drift = float(np.abs(a - b).max())
                worst = max(worst, drift)
                n += 1
                flipped = int(a.argmax()) != int(b.argmax())
                flips += flipped
                if flipped or drift >= 0.02:
                    ok = False
                    print(f"  {label} {'WINNER CHANGED' if flipped else 'drift'} in {case['name']}: {drift:.4f}", flush=True)
        print(f"{label}: worst drift {worst:.5f}; winner changed in {flips}/{n} questions; {'within' if ok else 'OUTSIDE'} e2e tolerance", flush=True)
        return ok

    fp32_path = os.path.join(args.out, "laya_fp32.onnx")
    if not args.skip_fp32:
        print("exporting fp32", flush=True)
        export(model, example, fp32_path, external_data=True)  # 1.7 GB of weights
        if not check(fp32_path, "fp32"):
            sys.exit("fp32 export does not match PyTorch")

    if not args.skip_fp16:
        fp16_path = os.path.join(args.out, "laya_fp16.onnx")
        print("exporting fp16 (mixed precision)", flush=True)
        export(to_mixed_precision(model), example, fp16_path, external_data=False)  # one file for the browser
        if not check(fp16_path, "fp16"):
            sys.exit("fp16 bundle changes decisions")
        write_meta(args, "laya_fp16.onnx", {"precision": "fp16 encoder; fp32 LayerNorms and decision head"})

    if args.q4:
        q4_path = os.path.join(args.out, "laya_q4.onnx")
        print("quantizing to 4-bit (block %d%s)" % (args.q4_block, ", head kept fp32" if args.q4_exclude_head else ""), flush=True)
        quantize_q4(fp32_path, q4_path, args.q4_block, args.q4_exclude_head)
        ok = check(q4_path, "q4")
        write_meta(args, "laya_q4.onnx", {"precision": "4-bit MatMulNBits, block %d" % args.q4_block,
                                           "head_fp32": args.q4_exclude_head, "parity_ok": ok})
        if not ok:
            print("q4 is outside the e2e tolerance; not fit to ship as is", flush=True)


def quantize_q4(src, dst, block, exclude_head):
    """4-bit weights for every MatMul (optionally not the decision head's), via ONNX Runtime's
    MatMulNBits, which both the WebAssembly and WebGPU backends of onnxruntime-web implement."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    m = onnx.load(src, load_external_data=True)
    exclude = []
    if exclude_head:
        # The exporter renames weights to opaque ids, but nodes are in execution order: the
        # encoder's weight MatMuls come first (4 per ModernBERT layer: Wqkv, Wo, Wi, MLP Wo), and
        # every weight MatMul after them belongs to the decision head, scorer and act head.
        from transformers import AutoConfig

        layers = AutoConfig.from_pretrained("answerdotai/ModernBERT-large").num_hidden_layers
        init = {t.name: t for t in m.graph.initializer}
        weighted = [n for n in m.graph.node if n.op_type == "MatMul" and any(i in init for i in n.input)]
        head = weighted[4 * layers:]
        exclude = [n.name for n in head]
        shapes = [tuple(init[next(i for i in n.input if i in init)].dims) for n in head]
        print("  keeping %d head MatMuls in fp32: %s" % (len(exclude), shapes), flush=True)
    algo = None
    if os.environ.get("Q4_ALGO") == "hqq":
        from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig
        algo = HQQWeightOnlyQuantConfig(block_size=block, bits=4)
    q = MatMulNBitsQuantizer(m, bits=4, block_size=block, is_symmetric=algo is None,
                             nodes_to_exclude=exclude, algo_config=algo)
    q.process()
    onnx.save(q.model.model, dst)


def write_meta(args, name, extra):
    import onnx
    import onnxruntime as ort

    path = os.path.join(args.out, name)
    meta = {"file": name, "bytes": os.path.getsize(path), "sha256": sha256(path), "opset": OPSET,
            "source": args.model, "torch": torch.__version__, "onnx": onnx.__version__,
            "onnxruntime_checked": ort.__version__, **extra}
    json.dump(meta, open(os.path.join(args.out, name.replace(".onnx", ".json")), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
