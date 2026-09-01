"""Part 5: prove the implementation computes the same function as GPT-2.

Writes ``results/verification.json`` and prints the layer-by-layer table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_dataset, get_gpt2, get_tokenizer, write_json
from transformer_internals.verify import run_verification


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", help="cpu (default) keeps fp32 tolerances honest")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS / "verification.json"))
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--no-perplexity", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    print(f"device: {device}")

    tok, _ = get_tokenizer(args.local_only)
    model, ckpt = get_gpt2(device=device, local_only=args.local_only)
    print(f"loaded {model.num_parameters():,} parameters from {ckpt}")

    dataset = None
    if not args.no_perplexity:
        dataset = get_dataset(tok, block_size=128, max_chars=1_000_000, local_only=args.local_only)

    report = run_verification(
        model, tok, dataset=dataset, max_new_tokens=args.max_new_tokens, device=device
    )

    print(f"\n{'activation':<18} {'max abs':>11} {'max rel':>11} {'mean abs':>11} {'|ref|max':>10}")
    print("-" * 66)
    for row in report.layers:
        print(
            f"{row.name:<18} {row.max_abs:>11.3e} {row.max_rel:>11.3e} "
            f"{row.mean_abs:>11.3e} {row.ref_scale:>10.2f}"
        )
    worst = report.worst_layer
    print("-" * 66)
    print(f"worst activation : {worst.name} at {worst.max_abs:.3e}")
    print(f"final logits     : {report.logits_max_abs:.3e} (tolerance {report.logit_tolerance:g})")
    for g in report.generation:
        status = "EXACT" if g["match"] else f"DIVERGED at token {g['first_divergence']}"
        print(f"greedy {g['n_new_tokens']:>4} tok : {status:<24} {g['prompt'][:38]!r}")
    if report.perplexity:
        p = report.perplexity
        print(
            f"perplexity       : ours {p['ours_ppl']:.4f}  reference {p['reference_ppl']:.4f}  "
            f"(|Δ| {p['abs_ppl_diff']:.2e} over {p['n_tokens']:,} tokens)"
        )

    write_json(args.out, report.to_dict())
    ok = report.logits_max_abs < report.logit_tolerance and report.all_generations_match
    print("\nVERIFICATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
