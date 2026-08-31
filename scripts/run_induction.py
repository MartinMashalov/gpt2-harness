"""Part 3: find induction heads in GPT-2 small.

Writes ``results/induction.json``: per-head prefix-matching, previous-token,
copying and causal-ablation scores, plus the in-context-learning curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_gpt2, write_json
from transformer_internals.induction import in_context_learning_curve, score_heads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ablation-batch-size", type=int, default=4)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "induction.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    model, _ = get_gpt2(device=device, local_only=args.local_only)
    print(f"device {device} | GPT-2 {model.config.n_layer}L x {model.config.n_head}H")

    scores = score_heads(
        model,
        seq_len=args.seq_len,
        batch_size=args.ablation_batch_size if not args.no_ablation else args.batch_size,
        seed=0,
        device=device,
        with_ablation=not args.no_ablation,
    )
    curve = in_context_learning_curve(
        model, seq_len=args.seq_len, batch_size=args.batch_size, device=device
    )

    print(f"\nchance prefix-matching = {scores.chance_level:.4f}")
    print(f"\n{'head':<8} {'prefix':>8} {'prev-tok':>9} {'copying':>8} {'ablation Δ':>11}")
    print("-" * 48)
    for r in scores.top_heads(8):
        abl = f"{r['ablation']:+.4f}" if r["ablation"] is not None else "n/a"
        print(
            f"{r['name']:<8} {r['prefix_matching']:>8.3f} {r['previous_token']:>9.3f} "
            f"{r['copying']:>8.3f} {abl:>11}"
        )
    print("\ntop previous-token heads (the other half of the circuit):")
    for r in scores.top_heads(4, key="previous_token"):
        print(f"  {r['name']:<8} prev-token {r['previous_token']:.3f}")
    if scores.ablation is not None:
        print("\nmost damaging to induction when ablated:")
        for r in scores.top_heads(6, key="ablation"):
            print(f"  {r['name']:<8} Δ second-copy loss {r['ablation']:+.4f} "
                  f"(prefix {r['prefix_matching']:.3f})")

    print(
        f"\nin-context learning: first copy {curve['first_copy_loss']:.3f} nats -> "
        f"second copy {curve['second_copy_loss']:.3f} nats "
        f"(bump {curve['induction_bump_nats']:.3f})"
    )

    payload = scores.to_dict()
    payload["in_context_learning"] = curve
    threshold = 0.3
    pm = torch.tensor(payload["prefix_matching"])
    payload["named_induction_heads"] = [
        {
            "name": f"L{int(i // pm.shape[1])}H{int(i % pm.shape[1])}",
            "layer": int(i // pm.shape[1]),
            "head": int(i % pm.shape[1]),
            "prefix_matching": float(pm.flatten()[i]),
        }
        for i in torch.argsort(pm.flatten(), descending=True)
        if float(pm.flatten()[i]) >= threshold
    ]
    payload["threshold"] = threshold
    write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
