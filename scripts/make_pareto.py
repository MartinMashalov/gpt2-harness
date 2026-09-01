"""Part 7: one quality/size frontier with every configuration on it.

This script deliberately re-evaluates every configuration itself rather than
reading perplexities out of the other result files. Those scripts each use their
own evaluation slice, and points measured on different data are not on the same
Pareto front -- putting them on one chart would be the kind of quiet
apples-to-oranges comparison this repository exists to avoid. Here every point is
scored by the same function on the same fixed batches.

Writes ``results/pareto.json``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_dataset, get_gpt2, get_tokenizer, write_json
from transformer_internals.benchmark import model_size_bytes
from transformer_internals.pruning import (
    _keep_mask,
    head_importance,
    neuron_importance,
    params_removed_by_head_pruning,
    params_removed_by_neuron_pruning,
    pruned,
)
from transformer_internals.quantization import quantize_model

QUANT_SCHEMES = [
    ("int8/chan", 8, "per_channel", False),
    ("int8/tensor", 8, "per_tensor", False),
    ("int4/chan", 4, "per_channel", False),
    ("int4/tensor", 4, "per_tensor", False),
]
PRUNE_LEVELS = [0.1, 0.2, 0.3, 0.5]


@torch.no_grad()
def score(model, batches, device) -> float:
    """Mean next-token loss on a fixed batch list. The one eval used by everything."""
    total, n = 0.0, 0
    for x, y in batches:
        logits = model(x.to(device))["logits"].float().cpu()
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "pareto.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    tok, _ = get_tokenizer(args.local_only)
    model, _ = get_gpt2(device=device, local_only=args.local_only)
    dataset = get_dataset(tok, block_size=128, max_chars=1_000_000, local_only=args.local_only)
    batches = dataset.sequential_batches(
        "val", batch_size=args.batch_size, limit=args.eval_batches
    )
    n_tokens = sum(y.numel() for _, y in batches)
    fp32_bytes = model_size_bytes(model)
    total_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())

    points = []
    base_loss = score(model, batches, device)
    points.append({
        "label": "GPT-2 124M fp32",
        "family": "baseline",
        "size_mb": fp32_bytes / 1e6,
        "loss": base_loss,
        "ppl": math.exp(base_loss),
        "detail": "the verified reference point",
    })
    print(f"fp32 baseline: loss {base_loss:.4f}  ppl {math.exp(base_loss):.3f}  "
          f"{fp32_bytes / 1e6:.1f} MB  ({n_tokens:,} eval tokens)")

    for label, bits, gran, inc in QUANT_SCHEMES:
        qm, stats = quantize_model(model, bits=bits, granularity=gran, include_embedding=inc)
        loss = score(qm, batches, device)
        points.append({
            "label": label,
            "family": "quantization",
            "size_mb": stats["packed_bytes"] / 1e6,
            "loss": loss,
            "ppl": math.exp(loss),
            "detail": f"{bits}-bit, {gran} scales",
        })
        print(f"  {label:<14} loss {loss:.4f}  ppl {math.exp(loss):8.3f}  "
              f"{stats['packed_bytes'] / 1e6:6.1f} MB")
        del qm

    print("computing pruning importance ...")
    h_imp = head_importance(model, dataset, device=device)
    n_imp = neuron_importance(model, dataset, device=device)
    for kind, imp, remover in (
        ("heads", h_imp, params_removed_by_head_pruning),
        ("neurons", n_imp, params_removed_by_neuron_pruning),
    ):
        for sparsity in PRUNE_LEVELS:
            mask = _keep_mask(imp, sparsity)
            kw = {"head_mask": mask} if kind == "heads" else {"neuron_mask": mask}
            with pruned(model, **kw):
                loss = score(model, batches, device)
            removed = remover(model, mask)
            size_mb = (total_params - removed) * 4 / 1e6
            points.append({
                "label": f"{kind[:4]} -{sparsity:.0%}",
                "family": "pruning",
                "size_mb": size_mb,
                "loss": loss,
                "ppl": math.exp(min(loss, 20.0)),
                "detail": f"{kind} pruned at {sparsity:.0%} ({removed:,} params removed)",
                "label_it": sparsity in (0.2, 0.5),
            })
            print(f"  {kind:<8} {sparsity:>4.0%}  loss {loss:.4f}  "
                  f"ppl {math.exp(min(loss, 20.0)):9.3f}  {size_mb:6.1f} MB")

    # The Pareto-optimal subset: no other point is both smaller and better.
    for p in points:
        p["pareto_optimal"] = not any(
            q["size_mb"] <= p["size_mb"] and q["ppl"] < p["ppl"] for q in points
        )
    front = [p["label"] for p in points if p["pareto_optimal"]]
    print(f"\nPareto front: {', '.join(front)}")

    write_json(args.out, {
        "points": points,
        "pareto_front": front,
        "meta": {
            "device": str(device),
            "eval_tokens": n_tokens,
            "eval_batches": len(batches),
            "fp32_bytes": fp32_bytes,
            "n_params": total_params,
            "caption": (
                f"every point scored by one function on the same {n_tokens:,} held-out "
                f"tokens · quantized sizes are measured packed files, baseline and "
                f"pruned sizes are parameter counts x 4 bytes"
            ),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
