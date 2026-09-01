"""The ablation grid: what is each GPT-2 design decision actually worth?

Every arm changes exactly one field of :class:`~transformer_internals.config.GPTConfig`
against a shared baseline and is trained by the same
:func:`~transformer_internals.train.train` under an identical budget, an identical data
order and three identical seeds. What is reported is the mean and sample standard
deviation of the final validation loss across those seeds, plus wall-clock.

**The seed spread is the point.** A single-seed ablation table is close to
worthless: at this scale the seed-to-seed spread on a *fixed* configuration is
often larger than the effect of the design decision being tested. Reporting
mean +- std makes it possible to say "this made no measurable difference", which
is a real finding and the one tutorials never report. :func:`summarize` computes
an explicit "is it distinguishable from baseline" verdict on exactly that basis:
a gap smaller than the pooled spread is called *indistinguishable*, and is
written down as such rather than being narrated as a win.

Head-count arms are constructed at **fixed parameter count**: attention's
parameters are ``4 * n_embd^2`` regardless of how ``n_embd`` is partitioned into
heads, so 16x16, 8x32, 4x64 and 2x128 are the same model size and the same FLOPs.
The only thing that changes is how much of the residual stream each head can see
at once -- and therefore how many distinct things the layer can attend to.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.data import TokenDataset
from transformer_internals.train import TrainResult, train

__all__ = ["BASELINE_MODEL", "BASELINE_TRAIN", "Arm", "build_arms", "run_arm", "summarize"]

#: The shared baseline: a 6-layer, 256-wide GPT-2 with every published design
#: decision switched on. Small enough that 27 runs fit in a coffee break, deep
#: enough that pre-LN vs post-LN is a real question rather than a formality.
#:
#: ``vocab_size`` is 4096, not 50257: see
#: :func:`transformer_internals.data.compact_vocabulary`. With the full GPT-2
#: vocabulary the output projection is three quarters of this model and dominates
#: the compute, so the budget would be spent on an embedding table that no
#: ablation touches.
BASELINE_MODEL = GPTConfig(
    vocab_size=4096,
    n_positions=128,
    n_layer=6,
    n_head=8,
    n_embd=256,
    dropout=0.0,
)

#: Learning rate 3e-4, not 6e-4. At 6e-4 this model sits close enough to an
#: instability that the tiny non-determinism of the MPS backward -- ~3e-3 on bias
#: gradients, which are atomic reductions -- is amplified into completely
#: different trajectories: two runs at the same seed finished at 3.89 and 5.95.
#: At 3e-4 the grid is bit-exact reproducible across repeated runs, which is
#: verified rather than assumed (``tests/test_determinism.py``).
BASELINE_TRAIN = TrainConfig(
    steps=250,
    batch_size=24,
    block_size=96,
    lr=3e-4,
    warmup_steps=50,
    eval_interval=50,
    eval_batches=10,
)


@dataclass
class Arm:
    """One cell of the grid.

    Attributes:
        key: Machine-readable identifier, used as the results filename.
        label: Human label for tables and figures.
        group: Which question this arm belongs to, so the report can group
            baseline-and-variant together.
        overrides: :class:`GPTConfig` fields to change from the baseline.
        is_baseline: Whether this arm *is* the shared baseline. Head-count arms
            have their own within-group baseline, hence a flag rather than a
            name check.
        note: What the arm is testing, carried through into the results JSON.
    """

    key: str
    label: str
    group: str
    overrides: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False
    note: str = ""

    def config(self, base: GPTConfig = BASELINE_MODEL) -> GPTConfig:
        """Materialise this arm's model config."""
        return GPTConfig(**{**base.to_dict(), **self.overrides})


def build_arms() -> list[Arm]:
    """Enumerate the grid.

    Returns:
        Every arm, baseline first. Six questions, nine configurations.
    """
    return [
        Arm(
            key="baseline",
            label="GPT-2 defaults",
            group="baseline",
            is_baseline=True,
            note="pre-LN, tied weights, learned positions, 8x32 heads, scaled residual init, GELU-tanh",
        ),
        Arm(
            key="post_ln",
            label="post-LN",
            group="norm_position",
            overrides={"norm_position": "post"},
            note="LayerNorm on the residual stream after the add, as in the 2017 Transformer",
        ),
        Arm(
            key="untied",
            label="untied embeddings",
            group="tie_weights",
            overrides={"tie_weights": False},
            note="separate input embedding and output head (+1.05M parameters at this vocab size)",
        ),
        Arm(
            key="sinusoidal",
            label="sinusoidal positions",
            group="pos_embedding",
            overrides={"pos_embedding": "sinusoidal"},
            note="fixed sin/cos positions instead of learned ones",
        ),
        Arm(
            key="heads_16x16",
            label="16 heads x 16 dim",
            group="n_head",
            overrides={"n_head": 16},
            note="same parameters, narrower heads",
        ),
        Arm(
            key="heads_4x64",
            label="4 heads x 64 dim",
            group="n_head",
            overrides={"n_head": 4},
            note="same parameters, wider heads",
        ),
        Arm(
            key="heads_2x128",
            label="2 heads x 128 dim",
            group="n_head",
            overrides={"n_head": 2},
            note="same parameters, very wide heads",
        ),
        Arm(
            key="no_resid_init",
            label="no residual-scaled init",
            group="residual_scaled_init",
            overrides={"residual_scaled_init": False},
            note="residual projections initialised at 0.02 instead of 0.02/sqrt(2L)",
        ),
        Arm(
            key="relu",
            label="ReLU",
            group="activation",
            overrides={"activation": "relu"},
            note="ReLU in the MLP instead of GPT-2's tanh-approximated GELU",
        ),
    ]


def run_arm(
    arm: Arm,
    dataset: TokenDataset,
    seed: int,
    base_model: GPTConfig = BASELINE_MODEL,
    base_train: TrainConfig = BASELINE_TRAIN,
    device: str | torch.device | None = None,
    progress: bool = False,
) -> TrainResult:
    """Train one arm at one seed.

    Args:
        arm: The arm.
        dataset: Shared dataset.
        seed: Seed for init and batch order.
        base_model: Baseline architecture.
        base_train: Baseline optimisation.
        device: Device.
        progress: Print progress.

    Returns:
        The training result, with the arm's identity recorded in ``meta``.
    """
    mcfg = arm.config(base_model)
    tcfg = TrainConfig(**{**base_train.to_dict(), "seed": seed})
    _, result = train(mcfg, tcfg, dataset, device=device, progress=progress)
    result.meta["arm"] = arm.key
    result.meta["label"] = arm.label
    result.meta["group"] = arm.group
    result.meta["note"] = arm.note
    result.meta["seed"] = seed
    return result


def summarize(results: dict[str, list[TrainResult]], arms: list[Arm]) -> dict[str, Any]:
    """Aggregate per-arm results into the table the README prints.

    Args:
        results: ``arm_key -> list of per-seed results``.
        arms: The arm definitions, for labels and grouping.

    Returns:
        A dict with one row per arm, each carrying mean/std of final validation
        loss, mean wall-clock, the delta against the baseline, and a verdict.

    The verdict rule, stated once so it cannot drift: an arm is called
    *distinguishable* only if ``|delta|`` exceeds the pooled standard deviation of
    the two arms being compared. That is a deliberately conservative bar -- with
    three seeds a formal test has almost no power, so rather than dress the
    comparison up as significance testing, the table reports the effect against
    the noise and lets the reader see both numbers.
    """
    by_key = {a.key: a for a in arms}
    rows: list[dict[str, Any]] = []

    def stats(key: str) -> tuple[float, float, float, list[float]]:
        losses = [r.final_val_loss for r in results[key]]
        walls = [r.wall_clock_s for r in results[key]]
        std = statistics.stdev(losses) if len(losses) > 1 else 0.0
        return statistics.fmean(losses), std, statistics.fmean(walls), losses

    # The baseline may be absent when a subset of arms is run (``--arms``). In
    # that case every delta is measured against the first arm present and the
    # payload says so, rather than crashing on a partial sweep.
    reference_key = "baseline" if results.get("baseline") else next(iter(results))
    base_mean, base_std, base_wall, _ = stats(reference_key)

    for arm in arms:
        if arm.key not in results or not results[arm.key]:
            continue
        mean, std, wall, losses = stats(arm.key)
        delta = mean - base_mean
        pooled = (std**2 + base_std**2) ** 0.5
        if arm.key == reference_key:
            verdict = "reference"
        elif abs(delta) <= pooled:
            verdict = "indistinguishable"
        elif delta > 0:
            verdict = "worse"
        else:
            verdict = "better"
        rows.append(
            {
                "arm": arm.key,
                "label": arm.label,
                "group": arm.group,
                "note": arm.note,
                "n_seeds": len(losses),
                "n_params": results[arm.key][0].n_params,
                "val_loss_mean": mean,
                "val_loss_std": std,
                "val_losses": losses,
                "delta_vs_baseline": delta,
                "pooled_std": pooled,
                "verdict": verdict,
                "wall_clock_s_mean": wall,
                "wall_clock_ratio": wall / base_wall if base_wall else float("nan"),
                "diverged_seeds": sum(1 for r in results[arm.key] if r.diverged),
                "max_grad_norm": max(
                    (max(r.grad_norms) if r.grad_norms else 0.0) for r in results[arm.key]
                ),
                "mean_grad_norm": statistics.fmean(
                    [g for r in results[arm.key] for g in r.grad_norms] or [0.0]
                ),
            }
        )

    return {
        "reference_arm": reference_key,
        "baseline": {"val_loss_mean": base_mean, "val_loss_std": base_std, "wall_clock_s": base_wall},
        "verdict_rule": (
            "distinguishable only if |delta| > sqrt(std_arm^2 + std_baseline^2); "
            "otherwise reported as indistinguishable"
        ),
        "rows": rows,
        "arm_count": len(by_key),
    }


def save_results(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a results payload as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
