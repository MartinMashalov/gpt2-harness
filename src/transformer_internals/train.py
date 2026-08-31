"""Training loop: AdamW, cosine schedule with warmup, clipping, grad accumulation.

Nothing here is exotic; the value is that every arm of the ablation grid goes
through *this* function, so a difference between two arms cannot be a difference
in the optimiser. The seed controls parameter init, dropout and the batch
sampler, and the batch sampler has its own generator seeded identically across
arms -- so two arms that differ only in, say, weight tying see the same tokens in
the same order at every step.

The learning-rate schedule is written out rather than pulled from
``torch.optim.lr_scheduler`` because the exact shape matters and is worth being
able to read: linear warmup to the peak, then cosine decay to a floor of
``min_lr_ratio * lr``.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT

__all__ = ["TrainResult", "estimate_loss", "lr_at_step", "pick_device", "set_seed", "train"]


def pick_device(prefer: str | None = None) -> torch.device:
    """Choose the best available device.

    Args:
        prefer: Force a device string. ``None`` auto-detects.

    Returns:
        MPS on Apple Silicon, else CUDA, else CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    import random

    random.seed(seed)
    # Seeds NumPy's legacy global RNG on purpose: this is about making any
    # third-party code that reaches for it reproducible too, not about how
    # this package draws its own randomness (it uses explicit Generators).
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    """Learning rate for a given step: linear warmup, then cosine decay.

    Warmup exists because Adam's second-moment estimate is meaningless for the
    first few steps -- ``v`` is initialised at zero and the bias correction makes
    the effective step size roughly ``lr`` regardless of gradient scale. On a
    fresh transformer that first full-size step is enough to push LayerNorm
    statistics somewhere it takes hundreds of steps to climb back from.

    The cosine floor (10% of peak, following GPT-3) rather than zero: the last
    few percent of a schedule that decays fully to zero does no work.

    Args:
        step: Zero-based optimiser step.
        cfg: Training config.

    Returns:
        The learning rate.
    """
    if step < cfg.warmup_steps:
        # +1 so step 0 gets a non-zero rate; a first step of exactly zero wastes
        # a step and, worse, makes the schedule depend on whether you count from
        # 0 or 1.
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


@dataclass
class TrainResult:
    """What one training run produced.

    Attributes:
        history: Per-eval records of ``step``, ``train_loss``, ``val_loss``, ``lr``.
        final_val_loss: Validation loss after the last step.
        best_val_loss: Best validation loss seen at any eval.
        steps: Steps actually run.
        wall_clock_s: Total training seconds, excluding the final evaluation.
        tokens_seen: Tokens consumed.
        n_params: Parameter count.
        diverged: True if the loss ever became non-finite. Recorded rather than
            raised, because "this configuration diverges" is a result.
        grad_norms: Pre-clip global gradient norm at every step, which is what
            makes the pre-LN vs post-LN stability claim measurable instead of
            anecdotal.
    """

    history: list[dict[str, float]] = field(default_factory=list)
    final_val_loss: float = float("nan")
    best_val_loss: float = float("nan")
    steps: int = 0
    wall_clock_s: float = 0.0
    tokens_seen: int = 0
    n_params: int = 0
    diverged: bool = False
    grad_norms: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "steps": self.steps,
            "wall_clock_s": self.wall_clock_s,
            "tokens_seen": self.tokens_seen,
            "n_params": self.n_params,
            "diverged": self.diverged,
            "grad_norms": self.grad_norms,
            "meta": self.meta,
        }


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset: TokenDataset,
    cfg: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Average loss over a fixed number of batches from each split.

    The generator is passed in and *reused*, so the eval batches are the same
    sequence of batches for every arm of an ablation. Fresh random eval batches
    would add variance that has nothing to do with the thing being measured.

    Args:
        model: The model.
        dataset: Data source.
        cfg: Provides ``eval_batches`` and ``batch_size``.
        device: Device.
        generator: RNG for batch selection.

    Returns:
        ``{"train": ..., "val": ...}``.
    """
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_batches)
        for i in range(cfg.eval_batches):
            x, y = dataset.get_batch(split, cfg.batch_size, device=device, generator=generator)
            losses[i] = model(x, targets=y)["loss"].item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


def train(
    model_config: GPTConfig,
    train_config: TrainConfig,
    dataset: TokenDataset,
    device: torch.device | str | None = None,
    progress: bool = False,
) -> tuple[GPT, TrainResult]:
    """Train a model and return it with its result record.

    Args:
        model_config: Architecture.
        train_config: Optimisation.
        dataset: Data.
        device: Device; auto-detected if ``None``.
        progress: Print a line every ``log_every`` steps.

    Returns:
        ``(model, result)``.
    """
    cfg, mcfg = train_config, model_config
    device = pick_device(device) if not isinstance(device, torch.device) else device

    set_seed(cfg.seed)
    model = GPT(mcfg).to(device)
    model.train()
    opt = model.configure_optimizers(cfg.weight_decay, cfg.lr, cfg.betas, cfg.eps)

    # Two independent, identically-seeded generators: one drives the training
    # stream, one drives evaluation. Sharing a single generator would make the
    # training batches depend on how often evaluation ran.
    train_gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen_seed = cfg.seed + 10_000

    # autocast is only enabled where it is actually a win and actually stable.
    # On MPS, bf16 autocast in torch 2.2 silently changes numerics in LayerNorm;
    # the ablation grid is small enough that fp32 costs little, and a comparison
    # run under different numerics than the baseline is not a comparison.
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    result = TrainResult(n_params=model.num_parameters())
    result.meta = {
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "model_config": mcfg.to_dict(),
        "train_config": cfg.to_dict(),
    }

    start = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at_step(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = dataset.get_batch("train", cfg.batch_size, device=device, generator=train_gen)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = model(x, targets=y)["loss"]
            else:
                loss = model(x, targets=y)["loss"]
            # Divide by grad_accum so the accumulated gradient is the *mean* over
            # the effective batch, not the sum -- otherwise changing grad_accum
            # silently changes the effective learning rate.
            scaler.scale(loss / cfg.grad_accum).backward()

        if use_amp:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        result.grad_norms.append(float(grad_norm))
        scaler.step(opt)
        scaler.update()

        result.tokens_seen += cfg.batch_size * cfg.grad_accum * cfg.block_size
        result.steps = step + 1

        if not math.isfinite(float(loss.item())):
            result.diverged = True
            break

        is_last = step == cfg.steps - 1
        if (step + 1) % cfg.eval_interval == 0 or is_last:
            gen = torch.Generator().manual_seed(eval_gen_seed)
            losses = estimate_loss(model, dataset, cfg, device, gen)
            result.history.append(
                {
                    "step": step + 1,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                }
            )
            if progress and ((step + 1) % cfg.log_every == 0 or is_last):
                print(
                    f"  step {step + 1:5d}/{cfg.steps}  train {losses['train']:.4f}  "
                    f"val {losses['val']:.4f}  lr {lr:.2e}",
                    flush=True,
                )

    result.wall_clock_s = time.perf_counter() - start

    if result.history:
        result.final_val_loss = result.history[-1]["val_loss"]
        result.best_val_loss = min(h["val_loss"] for h in result.history)
    return model, result
