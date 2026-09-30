"""Fuse the exported laya graph for ONNX Runtime's CUDA provider.

Why this exists
---------------
The plain export (tools/export_onnx.py) is ~4300 nodes: RoPE is spelled out as
Slice/Neg/Concat/Mul, every LayerNorm + residual is separate, and GELU is an Erf
chain. On the CUDA provider each of those is its own kernel launch, so latency is
dominated by launch overhead rather than math.

ORT's transformer optimizer fuses them (SkipLayerNormalization, Gelu,
RotaryEmbedding) and halves the node count. Two of its assumptions do not hold for
ModernBERT, and both are repaired here, otherwise the fused graph is wrong or
fails to run:

1. One RoPE cache for every layer. ModernBERT alternates global layers
   (theta = 160000, every Nth layer, starting at 0) and local layers
   (theta = 10000). The optimizer writes a single cos/sin cache and points all
   56 RotaryEmbedding nodes at it, which silently changes the answers. The caches
   are rebuilt here per theta, from the inverse frequencies read out of the
   original graph, and each node is pointed at the one its layer uses.
2. position_ids of batch 1. The export feeds RoPE a [1, seq] position tensor;
   RotaryEmbedding requires [batch, seq], so any batch > 1 fails. It is expanded
   to input_ids' shape.

Attention fusion is left off: ORT does not match ModernBERT's attention pattern,
and forcing it is where fusion goes wrong.

`--verify` (on by default) runs the original and fused graphs on CPU over random
batched inputs, with padding, and fails if any output differs by more than the
tolerance or any argmax changes.

Usage
-----
    python tools/optimize_onnx.py --src onnx/laya_ctx8192.onnx --out onnx/laya_ctx8192.opt.onnx

Requires: onnx, onnxruntime (>= 1.28), sympy, numpy.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper

LAYER_RE = re.compile(r"/encoder/layers\.(\d+)/")

# Ops on the hidden-state path; the RoPE tables never pass through them.
HIDDEN_PATH_OPS = {"MatMul", "Gemm", "LayerNormalization", "Softmax"}


def _constants(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    out = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    return out


def rope_frequencies(src: str, head_dim: int) -> dict[int, np.ndarray]:
    """Read each distinct RoPE inverse-frequency vector out of the original graph.

    Keyed by the theta it corresponds to (rounded), so the caller can match a
    layer to its table without trusting the node names.
    """
    g = onnx.load(src).graph
    consts = _constants(g)
    producer = {o: n for n in g.node for o in n.output}
    half = head_dim // 2

    def find(name: str, depth: int, seen: set[str]) -> np.ndarray | None:
        if depth > 12 or name in seen:
            return None
        seen.add(name)
        a = consts.get(name)
        if a is not None and a.dtype == np.float32 and a.size == half:
            return a.reshape(-1).astype(np.float64)
        n = producer.get(name)
        if n is None:
            return None
        for i in n.input:
            r = find(i, depth + 1, seen)
            if r is not None:
                return r
        return None

    out: dict[int, np.ndarray] = {}
    for n in g.node:
        if n.op_type == "Cos":
            inv = find(n.input[0], 0, set())
            if inv is None:
                raise SystemExit(f"cannot find RoPE inverse frequencies feeding {n.name}")
            theta = int(round((1.0 / inv[1]) ** (half)))
            ref = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
            if np.abs(inv - ref).max() > 1e-6:
                raise SystemExit(f"{n.name}: frequencies do not match theta={theta}")
            out[theta] = inv
    if not out:
        raise SystemExit("no RoPE (Cos) nodes in the source graph")
    return out


def layer_thetas(src: str, freqs: dict[int, np.ndarray]) -> dict[int, int]:
    """Which RoPE table each encoder layer reads, traced in the original graph.

    The walk goes backward from a layer's attention Mul nodes and stops at the
    hidden-state path (MatMul / LayerNormalization / Softmax), so it can never
    cross into an earlier layer. It must not stop at other layers' node names:
    the exporter de-duplicates the cos/sin Unsqueeze, so later layers read
    tensors produced by nodes named after the first layer that used them.
    """
    g = onnx.load(src, load_external_data=False).graph
    producer = {o: n for n in g.node for o in n.output}
    cos_theta: dict[str, int] = {}

    # Tag each Cos node with its theta by re-reading the frequencies it uses.
    consts = _constants(g)
    half = next(iter(freqs.values())).size

    def inv_of(name, depth=0, seen=None):
        seen = seen if seen is not None else set()
        if depth > 12 or name in seen:
            return None
        seen.add(name)
        a = consts.get(name)
        if a is not None and a.dtype == np.float32 and a.size == half:
            return a.reshape(-1).astype(np.float64)
        n = producer.get(name)
        return None if n is None else next(
            (r for i in n.input if (r := inv_of(i, depth + 1, seen)) is not None), None)

    for n in g.node:
        if n.op_type == "Cos":
            inv = inv_of(n.input[0])
            cos_theta[n.name] = min(freqs, key=lambda t: np.abs(freqs[t] - inv).max())

    out: dict[int, int] = {}
    for n in g.node:
        m = LAYER_RE.search(n.name)
        if not m or "/attn/" not in n.name or n.op_type != "Mul":
            continue
        layer = int(m.group(1))
        stack, seen = list(n.input), set()
        while stack:
            p = producer.get(stack.pop())
            if p is None or p.name in seen:
                continue
            seen.add(p.name)
            if p.op_type in HIDDEN_PATH_OPS:
                continue  # hidden states, not the rotary tables
            if p.op_type == "Cos":
                prev = out.setdefault(layer, cos_theta[p.name])
                if prev != cos_theta[p.name]:
                    raise SystemExit(f"layer {layer} reads two RoPE tables")
                continue
            stack.extend(p.input)
    return out


def fuse(src: str, num_heads: int, hidden: int):
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    opts = FusionOptions("bert")
    opts.enable_attention = False  # ModernBERT attention does not match; see module doc
    model = optimize_model(src, model_type="bert", num_heads=num_heads, hidden_size=hidden,
                           optimization_options=opts, opt_level=0, use_gpu=True)
    stats = {k: v for k, v in model.get_fused_operator_statistics().items() if v}
    return model, stats


def repair_rope(graph: onnx.GraphProto, freqs: dict[int, np.ndarray],
                thetas: dict[int, int], max_pos: int) -> collections.Counter:
    """Give every RotaryEmbedding node the cache of its own layer's theta."""
    ropes = [n for n in graph.node if n.op_type == "RotaryEmbedding"]
    if not ropes:
        raise SystemExit("the optimizer fused no RotaryEmbedding nodes; nothing to repair")
    old = {n.input[i] for n in ropes for i in (2, 3)}
    keep = [i for i in graph.initializer if i.name not in old]
    del graph.initializer[:]
    graph.initializer.extend(keep)

    pos = np.arange(max_pos, dtype=np.float64)[:, None]
    for theta, inv in freqs.items():
        f = pos * inv[None, :]
        graph.initializer.extend([
            numpy_helper.from_array(np.cos(f).astype(np.float32), f"laya_cos_{theta}"),
            numpy_helper.from_array(np.sin(f).astype(np.float32), f"laya_sin_{theta}"),
        ])

    used = collections.Counter()
    for n in ropes:
        m = LAYER_RE.search(n.input[0]) or LAYER_RE.search(n.name)
        if not m:
            raise SystemExit(f"cannot tell which layer {n.name} belongs to")
        layer = int(m.group(1))
        if layer not in thetas:
            raise SystemExit(f"layer {layer} has no traced RoPE table")
        theta = thetas[layer]
        n.input[2], n.input[3] = f"laya_cos_{theta}", f"laya_sin_{theta}"
        used[theta] += 1
    return used


def expand_positions(graph: onnx.GraphProto) -> str:
    """Broadcast the [1, seq] position tensor to [batch, seq] for RotaryEmbedding."""
    ropes = [n for n in graph.node if n.op_type == "RotaryEmbedding"]
    sources = {n.input[1] for n in ropes}
    if len(sources) != 1:
        raise SystemExit(f"expected one position_ids source, found {sorted(sources)}")
    src = sources.pop()
    out = "laya_position_ids"
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["laya_input_ids_shape"], name="laya_input_ids_shape"),
        helper.make_node("Expand", [src, "laya_input_ids_shape"], [out], name="laya_position_expand"),
    ]
    at = next(i for i, n in enumerate(graph.node) if src in n.output) + 1
    for k, n in enumerate(nodes):
        graph.node.insert(at + k, n)
    for n in ropes:
        n.input[1] = out
    return src


def to_fp16(model) -> None:
    """Store weights and run the body in fp16, keeping fp32/int64/bool IO.

    Long inputs are compute-bound, and fp32 is what made them slow: 2 x 512
    tokens cost ~45 ms in fp32 even on TensorRT. The Python original runs bf16
    autocast; fp16 is the CUDA-provider equivalent. IO stays as it was, so the
    Go side binds the same buffers.
    """
    model.convert_float_to_float16(keep_io_types=True, use_symbolic_shape_infer=True)


def verify_probs(src: str, dst: str, tol: float) -> None:
    """Compare a reduced-precision graph by its decisions, not its raw logits.

    fp16 legitimately moves logits in the low bits, so the check is on what the
    service reports: the softmax over each row's valid markers and the act
    probability. Both graphs run on the CPU provider.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    rng = np.random.default_rng(1)
    cases = []
    for batch, seq, k in ((1, 128, 4), (2, 300, 4), (2, 512, 6)):
        ids = rng.integers(5, 50000, size=(batch, seq), dtype=np.int64)
        mask = np.ones((batch, seq), dtype=np.int64)
        pos = np.sort(rng.choice(np.arange(1, seq), size=(batch, k), replace=False), axis=1).astype(np.int64)
        cases.append({"input_ids": ids, "attention_mask": mask, "marker_pos": pos,
                      "marker_mask": np.ones((batch, k), dtype=bool),
                      "qtype": np.arange(batch, dtype=np.int64) % 3})

    def softmax(x):
        x = x - x.max(-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(-1, keepdims=True)

    def run(path):
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return [dict(zip([o.name for o in s.get_outputs()], s.run(None, f))) for f in cases]

    ref, got = run(src), run(dst)
    worst = 0.0
    for ci, (a, b) in enumerate(zip(ref, got)):
        for name in ("logits", "act_logits"):
            pa, pb = softmax(a[name].astype(np.float64)), softmax(b[name].astype(np.float64))
            d = float(np.abs(pa - pb).max())
            worst = max(worst, d)
            print(f"   case {ci} {name:10s} max|dp|={d:.2e}")
    if worst > tol:
        raise SystemExit(f"verify: fp16 probabilities differ by {worst:.2e} (> {tol:.1e})")
    print(f"verify: OK (fp16, max |dp| = {worst:.2e})")


def verify(src: str, dst: str, rtol: float) -> None:
    """Run both graphs on CPU over batched, padded inputs and compare outputs."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    rng = np.random.default_rng(0)

    def feeds(batch: int, seq: int, k: int) -> dict[str, np.ndarray]:
        ids = rng.integers(5, 50000, size=(batch, seq), dtype=np.int64)
        mask = np.ones((batch, seq), dtype=np.int64)
        mask[-1, seq // 2:] = 0  # one padded row
        pos = np.sort(rng.choice(np.arange(1, seq // 2), size=(batch, k)), axis=1).astype(np.int64)
        mm = np.ones((batch, k), dtype=bool)
        mm[0, -1] = False
        return {"input_ids": ids, "attention_mask": mask, "marker_pos": pos,
                "marker_mask": mm, "qtype": np.arange(batch, dtype=np.int64) % 3}

    cases = [feeds(1, 128, 4), feeds(2, 300, 4), feeds(4, 512, 6)]

    def run(path: str):
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        names = [o.name for o in s.get_outputs()]
        return names, [s.run(None, f) for f in cases]

    names, ref = run(src)
    _, got = run(dst)
    worst = 0.0
    for ci, (a, b) in enumerate(zip(ref, got)):
        for name, x, y in zip(names, a, b):
            fx, fy = np.isfinite(x), np.isfinite(y)
            if not np.array_equal(fx, fy):
                raise SystemExit(f"verify: case {ci} {name}: masked positions differ")
            d = float(np.abs(x[fx] - y[fy]).max()) if fx.any() else 0.0
            # Relative to the output's own scale: act_logits run in the
            # thousands, where fp32 reordering alone moves them by ~1e-2.
            scale = max(1.0, float(np.abs(x[fx]).max())) if fx.any() else 1.0
            worst = max(worst, d / scale)
            ax = np.argmax(np.where(fx, x, -np.inf), axis=-1)
            ay = np.argmax(np.where(fy, y, -np.inf), axis=-1)
            print(f"   case {ci} {name:10s} shape={x.shape} max|d|={d:.2e} argmax-equal={np.array_equal(ax, ay)}")
            if not np.array_equal(ax, ay):
                raise SystemExit(f"verify: case {ci} {name}: argmax changed")
    if worst > rtol:
        raise SystemExit(f"verify: max relative |d| = {worst:.2e} exceeds --rtol {rtol:.1e}")
    print(f"verify: OK (max relative |d| = {worst:.2e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="exported laya .onnx")
    ap.add_argument("--out", required=True, help="where to write the fused graph")
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--max-pos", type=int, default=8192, help="RoPE cache length (max sequence)")
    ap.add_argument("--rtol", type=float, default=1e-4,
                    help="largest allowed output difference, relative to each output's magnitude")
    ap.add_argument("--fp16", action="store_true",
                    help="also convert the body to fp16 (IO stays fp32); ~2x faster on long inputs")
    ap.add_argument("--prob-tol", type=float, default=2e-2,
                    help="with --fp16: largest allowed probability difference")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    t = time.time()
    freqs = rope_frequencies(args.src, args.head_dim)
    thetas = layer_thetas(args.src, freqs)
    print(f"rope: thetas {sorted(freqs)}; layers per theta "
          f"{dict(collections.Counter(thetas.values()))}")

    model, stats = fuse(args.src, args.num_heads, args.hidden)
    graph = model.model.graph
    print(f"fused: {stats}; nodes {len(graph.node)}")

    used = repair_rope(graph, freqs, thetas, args.max_pos)
    print(f"rope: RotaryEmbedding nodes per theta {dict(used)}")
    src_pos = expand_positions(graph)
    print(f"positions: {src_pos} expanded to [batch, seq]")

    if args.fp16:
        to_fp16(model)
        print("precision: fp16 body, fp32 IO")

    onnx.save(model.model, args.out)
    print(f"wrote {args.out} in {time.time() - t:.0f}s")

    if not args.no_verify:
        if args.fp16:
            verify_probs(args.src, args.out, args.prob_tol)
        else:
            verify(args.src, args.out, args.rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
