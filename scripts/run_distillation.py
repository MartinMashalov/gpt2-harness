"""Part 4: distil the verified GPT-2 teacher into a small student.

Trains the same student twice under an identical budget -- once from scratch,
once against the teacher's distribution -- and reports the gap.
Writes ``results/distillation.json``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, device_from_arg, get_dataset, get_gpt2, get_tokenizer, write_json
from transformer_internals.config import TrainConfig
from transformer_internals.distill import STUDENT_CONFIG, train_student


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # CPU by default, deliberately. The student must share the teacher's full
    # 50257-token vocabulary for the KL to mean anything, and that large
    # cross-entropy reduction is exactly where the MPS backward stops being
    # bit-deterministic: an earlier MPS run of this experiment produced a
    # from-scratch control of 4.99 +- 1.47 where the reproducible CPU control is
    # tight. A control arm that noisy cannot support any claim about the
    # treatment, so this experiment trades speed for reproducibility.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0.1, 0.5, 0.9],
        help="distillation weights to sweep; 0.0 (the control) is always included",
    )
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--out", default=str(RESULTS / "distillation.json"))
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    tok, _ = get_tokenizer(args.local_only)
    teacher, _ = get_gpt2(device=device, local_only=args.local_only)
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, block_size=args.block_size,
        lr=3e-4, warmup_steps=max(5, args.steps // 10),
        eval_interval=max(10, args.steps // 6), eval_batches=8,
    )
    dataset = get_dataset(tok, block_size=cfg.block_size, local_only=args.local_only)
    print(f"device {device} | teacher {teacher.num_parameters():,} params "
          f"| student {args.steps} steps x {len(args.seeds)} seeds")

    arms_spec = [("from scratch", 0.0)] + [
        (f"distilled α={a:g}", a) for a in args.alphas
    ]
    arms = []
    for label, alpha in arms_spec:
        runs = []
        for seed in args.seeds:
            c = TrainConfig(**{**cfg.to_dict(), "seed": seed})
            _, r = train_student(
                STUDENT_CONFIG, c, dataset, teacher=teacher if alpha > 0 else None,
                alpha=alpha, temperature=args.temperature, device=device, label=label,
            )
            runs.append(r)
            print(f"  {label:<34} seed {seed}  val {r.final_val_loss:.4f}  "
                  f"{r.wall_clock_s:5.1f}s", flush=True)
        losses = [r.final_val_loss for r in runs]
        # Mean the per-step histories across seeds so the figure shows a curve,
        # not one arbitrary seed.
        steps = [h["step"] for h in runs[0].history]
        mean_history = [
            {"step": s,
             "val_loss": statistics.fmean([r.history[i]["val_loss"] for r in runs])}
            for i, s in enumerate(steps)
        ]
        arms.append({
            "label": label,
            "alpha": alpha,
            "temperature": args.temperature,
            "val_loss_mean": statistics.fmean(losses),
            "val_loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "val_losses": losses,
            "wall_clock_s_mean": statistics.fmean([r.wall_clock_s for r in runs]),
            "mean_history": mean_history,
            "n_params": runs[0].n_params,
            "runs": [r.to_dict() for r in runs],
        })

    scratch = arms[0]
    # The best distilled arm is the one that should be compared against the
    # control -- reporting only a single arbitrary alpha would make the verdict
    # a function of that choice rather than of distillation.
    distilled = min(arms[1:], key=lambda a: a["val_loss_mean"])
    delta = distilled["val_loss_mean"] - scratch["val_loss_mean"]
    pooled = (scratch["val_loss_std"] ** 2 + distilled["val_loss_std"] ** 2) ** 0.5
    print(f"\n{'arm':<36} {'val loss':>16} {'s/run':>8}")
    print("-" * 62)
    for a in arms:
        print(f"{a['label']:<36} {a['val_loss_mean']:>8.4f} ± {a['val_loss_std']:<5.4f} "
              f"{a['wall_clock_s_mean']:>8.1f}")
    print(f"\nbest distilled arm: {distilled['label']}")
    print(f"distillation Δ = {delta:+.4f} nats (pooled sd {pooled:.4f}) at "
          f"{distilled['wall_clock_s_mean'] / scratch['wall_clock_s_mean']:.1f}x the wall-clock")

    write_json(args.out, {
        "arms": arms,
        "best_distilled": distilled["label"],
        "delta": delta,
        "pooled_std": pooled,
        "distinguishable": bool(abs(delta) > pooled),
        "cost_ratio": distilled["wall_clock_s_mean"] / scratch["wall_clock_s_mean"],
        "meta": {
            "device": str(device),
            "seeds": args.seeds,
            "student_config": STUDENT_CONFIG.to_dict(),
            "train_config": cfg.to_dict(),
            "teacher_params": teacher.num_parameters(),
            "alphas": args.alphas,
            "caption": (
                f"same 4-layer student, same {args.steps} steps and batches, "
                f"{len(args.seeds)} seeds, alpha swept over {args.alphas}"
            ),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
