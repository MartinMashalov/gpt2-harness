"""Part 7: structured pruning of attention heads and MLP neurons.

Ranks heads and neurons by gradient-based importance, prunes at several
sparsity levels, and tracks both held-out loss and the induction behaviour.
Writes ``results/pruning.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    RESULTS,
    device_from_arg,
    get_dataset,
    get_gpt2,
    get_tokenizer,
    read_json,
    write_json,
)
from transformer_internals.induction import make_repeated_sequence, second_copy_loss
from transformer_internals.pruning import head_importance, neuron_importance, prune_sweep

SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS / "pruning.json"))
    ap.add_argument("--induction", default=str(RESULTS / "induction.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    tok, _ = get_tokenizer(args.local_only)
    model, _ = get_gpt2(device=device, local_only=args.local_only)
    dataset = get_dataset(tok, block_size=128, max_chars=1_000_000, local_only=args.local_only)

    seqs = make_repeated_sequence(4, 60, model.config.vocab_size, 50256,
                                  generator=torch.Generator().manual_seed(0)).to(device)

    def probe(m) -> float:
        return second_copy_loss(m, seqs, 60)

    print("computing head importance ...")
    h_imp = head_importance(model, dataset, device=device)
    print("computing neuron importance ...")
    n_imp = neuron_importance(model, dataset, device=device)

    sweeps = {}
    for kind, imp in (("heads", h_imp), ("neurons", n_imp)):
        print(f"sweeping {kind} ...")
        rows, _ = prune_sweep(
            model, dataset, SPARSITIES, kind=kind, device=device, importance=imp,
            induction_probe=probe if kind == "heads" else None,
        )
        sweeps[kind] = [r.to_dict() for r in rows]
        print(f"\n{kind}: {'sparsity':>9} {'params -%':>10} {'loss':>8} {'ppl':>9} {'induction':>10}")
        for r in rows:
            ind = (f"{r.induction_second_copy_loss:.3f}"
                   if r.induction_second_copy_loss is not None else "--")
            print(f"{'':<7} {r.sparsity:>9.0%} {100 * r.params_removed / r.params_total:>10.1f} "
                  f"{r.val_loss:>8.4f} {r.val_ppl:>9.2f} {ind:>10}")

    # Where do the named induction heads rank on the importance criterion, and
    # does gradient importance agree with the direct-ablation measurement?
    tie_in = {}
    ind_path = Path(args.induction)
    if ind_path.exists():
        ind = read_json(ind_path)
        flat = h_imp.flatten()
        order = torch.argsort(flat, descending=True)
        rank_of = {int(v): i for i, v in enumerate(order.tolist())}
        n_head = model.config.n_head
        tie_in["induction_head_ranks"] = [
            {
                "name": h["name"],
                "prefix_matching": h["prefix_matching"],
                "importance": float(flat[h["layer"] * n_head + h["head"]]),
                "importance_rank": rank_of[h["layer"] * n_head + h["head"]],
                "of_total_heads": int(flat.numel()),
            }
            for h in ind["named_induction_heads"]
        ]
        if ind.get("ablation"):
            abl = torch.tensor(ind["ablation"]).flatten()
            # Spearman: rank-correlate the cheap criterion against direct ablation.
            ra = torch.argsort(torch.argsort(abl)).float()
            rb = torch.argsort(torch.argsort(flat)).float()
            ra, rb = ra - ra.mean(), rb - rb.mean()
            tie_in["spearman_importance_vs_ablation"] = float(
                (ra @ rb) / (ra.norm() * rb.norm())
            )
        print("\ninduction heads on the pruning criterion:")
        for r in tie_in["induction_head_ranks"]:
            print(f"  {r['name']:<8} importance rank {r['importance_rank']:>3} "
                  f"of {r['of_total_heads']}")
        if "spearman_importance_vs_ablation" in tie_in:
            print(f"  Spearman(gradient importance, direct ablation) = "
                  f"{tie_in['spearman_importance_vs_ablation']:+.3f}")

    write_json(args.out, {
        "sweeps": sweeps,
        "head_importance": h_imp.tolist(),
        "induction_tie_in": tie_in,
        "meta": {
            "device": str(device),
            "sparsities": SPARSITIES,
            "criterion": "|dL/d(mask)| accumulated over calibration batches, L2-normalised per layer",
            "n_params": model.num_parameters(),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
