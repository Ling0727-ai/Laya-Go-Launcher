"""Export the laya decision model to ONNX with a *dynamic* sequence length.

Why this exists
---------------
`torch.onnx.export` on the model as-is produces a graph that is only correct at
the length it was traced at, which makes a TensorRT engine built from it accept
exactly one sequence length. The symptom is:

    IShuffleLayer /layers.0/self_attn/Reshape_4: reshaping failed
    input shape:{128,1,1024}, requested shape:{512,16,64}

The `512` in the requested shape is the trace length, baked into the graph as a
constant.

Two things cause it, and both have to be handled:

1. `nn.TransformerEncoder` (the head container) lowers to
   `aten::_transformer_encoder_layer_fwd`, which folds the sequence length in.
   Iterating `head.layers` explicitly avoids the container.
2. `nn.TransformerEncoderLayer` itself still folds it in its own fused attention
   path. So the attention has to be written out here, using the layer's own
   weights, with `scaled_dot_product_attention` on tensors whose shape comes from
   the input.

The result is the same math — `verify` checks the exported graph against the
unmodified model at both the traced length and a different one, so a mistake in
this reimplementation shows up immediately.

Usage
-----
    python tools/export_onnx.py --out laya_dyn.onnx --seq-len 512 --markers 16

Then build an engine with a dynamic profile:

    trtexec --onnx=laya_dyn.onnx --saveEngine=laya.engine --fp16 \
      --minShapes=input_ids:1x64,attention_mask:1x64,marker_pos:1x2,marker_mask:1x2,qtype:1 \
      --optShapes=input_ids:1x256,attention_mask:1x256,marker_pos:1x8,marker_mask:1x8,qtype:1 \
      --maxShapes=input_ids:1x512,attention_mask:1x512,marker_pos:1x16,marker_mask:1x16,qtype:1

Inputs: input_ids, attention_mask (int64 [batch, seq]); marker_pos (int64
[batch, markers]); marker_mask (bool [batch, markers]); qtype (int64 [batch]).
Outputs: logits (float32 [batch, markers]), act_logits (float32 [batch, 2]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _attention(mha, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Multi-head self-attention over `mha`'s own weights, shape-generic.

    Written out rather than calling `mha(...)` so no fused kernel can bake the
    sequence length into the graph. Equivalent to `nn.MultiheadAttention` in eval
    mode with `batch_first=True` and no bias/add-zero-attn.
    """
    batch, seq, dim = x.shape
    heads = mha.num_heads
    head_dim = dim // heads

    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.reshape(batch, seq, heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, seq, heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, seq, heads, head_dim).transpose(1, 2)

    attn_mask = None
    if pad_mask is not None:
        # [batch, 1, 1, seq] so it broadcasts over heads and query positions.
        attn_mask = torch.zeros(batch, 1, 1, seq, dtype=x.dtype, device=x.device)
        attn_mask = attn_mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).reshape(batch, seq, dim)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


def _encoder_layer(layer, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """One `nn.TransformerEncoderLayer` with norm_first=True, written out."""
    h = layer.norm1(x)
    h = _attention(layer.self_attn, h, pad_mask)
    h = layer.dropout1(h)
    x = x + h

    h = layer.norm2(x)
    h = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
    h = layer.dropout2(h)
    return x + h


class ExportWrapper(torch.nn.Module):
    """The decision model with a shape-generic head.

    The encoder is used as-is (ModernBERT exports fine). The head — the part that
    folded the trace length into a reshape — is reimplemented above.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.inner

        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]

        if m.head is not None:
            pad_mask = ~attention_mask.bool()
            for layer in m.head.layers:
                h = _encoder_layer(layer, h, pad_mask)

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        gathered = torch.gather(h, 1, idx)
        logits = m.scorer(gathered).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)

        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)

        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = m.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def make_inputs(seq: int, markers: int, cls_id: int, batch: int = 1):
    input_ids = torch.randint(100, 20000, (batch, seq), dtype=torch.long)
    input_ids[:, 0] = cls_id
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones((batch, seq), dtype=torch.long),
        "marker_pos": torch.arange(1, markers + 1, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch, 1),
        "marker_mask": torch.ones((batch, markers), dtype=torch.bool),
        "qtype": torch.zeros(batch, dtype=torch.long),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--out", default="laya_dyn.onnx")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="length to trace at; the graph is dynamic, this only "
                             "fixes what the tracer sees")
    parser.add_argument("--markers", type=int, default=16,
                        help="option markers to trace with")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import laya

    print(f"loading {args.model}"
          + (f" (subfolder={args.subfolder})" if args.subfolder else ""))
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    model = agent.model.eval()
    cls_id = agent.tok.cls_token_id

    # Keeps the encoder off any fused path too.
    torch.backends.mha.set_fastpath_enabled(False)

    wrapper = ExportWrapper(model)
    traced = make_inputs(args.seq_len, args.markers, cls_id)
    print(f"tracing at seq={args.seq_len} markers={args.markers}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(traced.values()),
            args.out,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "markers"},
                "marker_mask": {0: "batch", 1: "markers"},
                "qtype": {0: "batch"},
                "logits": {0: "batch", 1: "markers"},
                "act_logits": {0: "batch"},
            },
            opset_version=args.opset,
            dynamo=False,
        )

    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024 / 1024:.0f} MB)")
    verify(args.out, model, cls_id, args)
    return 0


def verify(path: str, model: torch.nn.Module, cls_id: int, args) -> None:
    """Check the exported graph against the *unmodified* model.

    Compared at the traced length and at a different one, because matching only at
    the traced length is exactly the bug this script exists to avoid.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # The exporter can leave the module in a different state than it found it
    # (hooks, training flags). Restore eval mode and the fastpath setting before
    # using the model as the reference, or the comparison measures the export
    # side effect rather than the exported graph.
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)

    cases = [
        ("traced length", args.seq_len, args.markers),
        ("different length", max(16, args.seq_len // 4), max(2, args.markers // 2)),
    ]

    failures = 0
    for label, seq, markers in cases:
        # One input set, used for both sides. Generating it twice would compare
        # two different random documents and report a large spurious difference.
        feed = make_inputs(seq, markers, cls_id)
        with torch.no_grad():
            ref_logits, ref_act = model(
                feed["input_ids"], feed["attention_mask"], feed["marker_pos"],
                feed["marker_mask"], feed["qtype"],
            )
        got_logits, got_act = sess.run(None, {k: v.numpy() for k, v in feed.items()})

        d_logits = float(np.abs(ref_logits.numpy() - got_logits).max())
        d_act = float(np.abs(ref_act.numpy() - got_act).max())
        ok = d_logits < 1e-3
        print(f"[{'OK' if ok else 'FAIL'}] {label} (seq={seq}, markers={markers}): "
              f"max|Δlogits|={d_logits:.3e} max|Δact|={d_act:.3e}")
        if not ok:
            failures += 1
            print(f"      torch : {ref_logits.numpy().ravel()[:6]}")
            print(f"      onnx  : {got_logits.ravel()[:6]}")

    if failures:
        print("the exported graph disagrees with the model; see the notes at the top "
              "of this file", file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps({
        "onnx": path,
        "suggested_profile": {
            "minShapes": "input_ids:1x64,attention_mask:1x64,marker_pos:1x2,marker_mask:1x2,qtype:1",
            "optShapes": "input_ids:1x256,attention_mask:1x256,marker_pos:1x8,marker_mask:1x8,qtype:1",
            "maxShapes": f"input_ids:1x{args.seq_len},attention_mask:1x{args.seq_len},"
                         f"marker_pos:1x{args.markers},marker_mask:1x{args.markers},qtype:1",
        },
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
