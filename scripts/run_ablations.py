"""Part 3: train the ablation grid and write ``results/ablations.json``.

Nine configurations x three seeds, all under an identical budget and an identical
data order. Reports mean +- standard deviation of final validation loss.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_compact_dataset, get_tokenizer, write_json
from transformer_internals.ablations import (
    BASELINE_MODEL,
    BASELINE_TRAIN,
    build_arms,
    run_arm,
    summarize,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None, help="override the step budget")
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arm keys")
    ap.add_argument("--out", default=str(RESULTS / "ablations.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    tok, _ = get_tokenizer(args.local_only)
    base_train = BASELINE_TRAIN
    if args.steps is not None:
        from transformer_internals.config import TrainConfig

        base_train = TrainConfig(**{**BASELINE_TRAIN.to_dict(), "steps": args.steps})

    dataset, coverage = get_compact_dataset(
        tok,
        block_size=base_train.block_size,
        vocab_size=BASELINE_MODEL.vocab_size,
        local_only=args.local_only,
    )
    arms = [a for a in build_arms() if args.arms is None or a.key in args.arms]

    print(f"device {device} | {len(arms)} arms x {len(args.seeds)} seeds "
          f"| {base_train.steps} steps | {len(dataset.train):,} train tokens "
          f"| vocab {BASELINE_MODEL.vocab_size} covering {coverage:.2%} of occurrences")

    results: dict[str, list] = {}
    t_all = time.perf_counter()
    for arm in arms:
        results[arm.key] = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            r = run_arm(arm, dataset, seed, BASELINE_MODEL, base_train, device=device)
            results[arm.key].append(r)
            print(
                f"  {arm.key:<18} seed {seed}  val {r.final_val_loss:.4f}  "
                f"{time.perf_counter() - t0:5.1f}s"
                + ("  DIVERGED" if r.diverged else ""),
                flush=True,
            )

    summary = summarize(results, arms)
    summary["meta"] = {
        "device": str(device),
        "seeds": args.seeds,
        "total_wall_clock_s": time.perf_counter() - t_all,
        "base_model_config": BASELINE_MODEL.to_dict(),
        "base_train_config": base_train.to_dict(),
        "vocab_coverage": coverage,
        "train_tokens": len(dataset.train),
        "val_tokens": len(dataset.val),
    }
    summary["raw"] = {k: [r.to_dict() for r in v] for k, v in results.items()}

    print(f"\n{'arm':<26} {'val loss':>16} {'Δ vs base':>11} {'verdict':>19} {'s/run':>7}")
    print("-" * 84)
    for row in summary["rows"]:
        print(
            f"{row['label']:<26} {row['val_loss_mean']:>8.4f} ± {row['val_loss_std']:<5.4f} "
            f"{row['delta_vs_baseline']:>+11.4f} {row['verdict']:>19} "
            f"{row['wall_clock_s_mean']:>7.1f}"
        )
    write_json(args.out, summary)
    print(f"total {summary['meta']['total_wall_clock_s'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
