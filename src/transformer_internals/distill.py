"""Knowledge distillation from the verified GPT-2 teacher into a small student.

The experiment is a controlled one, and the control is the point: the *same*
student architecture, the *same* optimiser, the *same* number of steps and the
*same* batches, trained two ways -- once on the hard next-token labels alone, and
once on a mixture of those labels and the teacher's full output distribution.
Anything else measured here would be confounded.

**Why a distribution is worth more than a label.** A one-hot target tells the
student one bit of structure per token: which word came next. The teacher's
distribution tells it that after "The capital of France is" the mass sits on
" Paris" but also, a little, on " located" and " the" -- what Hinton called the
dark knowledge. That is a far denser training signal per token, which is exactly
what a small model starved of data needs.

**Temperature, and the T^2.** Both logits are divided by ``T`` before the KL. That
flattens both distributions and exposes the teacher's relative ranking among
*low*-probability tokens, which is where most of the extra information is. But
softening also shrinks the gradients by roughly ``1/T^2``, so the KL term is
multiplied back by ``T^2`` to keep its contribution scale-free -- otherwise
changing ``T`` silently changes the effective learning rate on that term, and the
ablation over ``T`` measures the wrong thing. This factor is in the original
Hinton et al. paper and is the detail most reimplementations drop.

The teacher runs in ``no_grad`` on every step. That is the dominant cost: the
teacher is 124M parameters against the student's few million, so distillation
here is roughly 3-5x the wall-clock of training from scratch. That cost is
reported alongside the quality, because "distillation helps" is not useful
without "and it cost this much".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT
from transformer_internals.train import lr_at_step, pick_device, set_seed

__all__ = ["STUDENT_CONFIG", "DistillResult", "distillation_loss", "train_student"]

#: The student: 4 layers, 256 wide, 8 heads. Small enough to train in under a
#: minute, large enough that the gap between the two training signals is
#: measurable rather than swamped by the model's own capacity limit.
STUDENT_CONFIG = GPTConfig(
    vocab_size=50257,
    n_positions=128,
    n_layer=4,
    n_head=8,
    n_embd=256,
    dropout=0.0,
)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine hard-label cross-entropy with soft-target KL divergence.

    Args:
        student_logits: ``(B, T, V)``.
        teacher_logits: ``(B, T, V)``, detached.
        targets: ``(B, T)`` true next tokens.
        temperature: Softening temperature ``T``.
        alpha: Weight on the KL term; ``1 - alpha`` on the hard labels.
            ``alpha = 0`` reduces exactly to ordinary training, which is how the
            control arm is run -- same code path, so the comparison cannot be
            confounded by an implementation difference.

    Returns:
        ``(total, ce, kl)``. ``ce`` is reported separately because it is the
        quantity comparable across arms; the total is not, since it includes a
        term the control arm does not have.
    """
    v = student_logits.size(-1)
    ce = F.cross_entropy(student_logits.reshape(-1, v), targets.reshape(-1))

    if alpha == 0.0:
        zero = torch.zeros((), device=student_logits.device)
        return ce, ce, zero

    s = F.log_softmax(student_logits.reshape(-1, v) / temperature, dim=-1)
    t = F.softmax(teacher_logits.reshape(-1, v) / temperature, dim=-1)
    # batchmean: the KL is summed over the vocabulary and averaged over tokens,
    # which is the definition. torch's default 'mean' would divide by the
    # vocabulary size too and make the term 50257x too small.
    kl = F.kl_div(s, t, reduction="batchmean") * (temperature**2)
    return (1 - alpha) * ce + alpha * kl, ce, kl


@dataclass
class DistillResult:
    """Outcome of one student training run."""

    label: str
    alpha: float
    temperature: float
    seed: int
    final_val_loss: float
    final_val_ppl: float
    history: list[dict[str, float]] = field(default_factory=list)
    wall_clock_s: float = 0.0
    n_params: int = 0
    teacher_params: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "seed": self.seed,
            "final_val_loss": self.final_val_loss,
            "final_val_ppl": self.final_val_ppl,
            "history": self.history,
            "wall_clock_s": self.wall_clock_s,
            "n_params": self.n_params,
            "teacher_params": self.teacher_params,
            "meta": self.meta,
        }


def train_student(
    student_config: GPTConfig,
    train_config: TrainConfig,
    dataset: TokenDataset,
    teacher: GPT | None = None,
    alpha: float = 0.0,
    temperature: float = 2.0,
    device: torch.device | str | None = None,
    label: str = "student",
    progress: bool = False,
) -> tuple[GPT, DistillResult]:
    """Train a student, optionally against a teacher.

    Args:
        student_config: Student architecture.
        train_config: Optimisation; shared with the control arm.
        dataset: Data.
        teacher: The teacher model. Required when ``alpha > 0``.
        alpha: Weight on the distillation term. ``0`` is the from-scratch control.
        temperature: Softening temperature.
        device: Device.
        label: Name for the result record.
        progress: Print progress.

    Returns:
        ``(student, result)``.
    """
    if alpha > 0 and teacher is None:
        raise ValueError("distillation requires a teacher")

    cfg = train_config
    device = pick_device(device) if not isinstance(device, torch.device) else device

    set_seed(cfg.seed)
    student = GPT(student_config).to(device)
    student.train()
    opt = student.configure_optimizers(cfg.weight_decay, cfg.lr, cfg.betas, cfg.eps)

    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Same seed as the control arm, so both see identical batches at every step.
    train_gen = torch.Generator().manual_seed(cfg.seed)
    result = DistillResult(
        label=label,
        alpha=alpha,
        temperature=temperature,
        seed=cfg.seed,
        final_val_loss=float("nan"),
        final_val_ppl=float("nan"),
        n_params=student.num_parameters(),
        teacher_params=teacher.num_parameters() if teacher is not None else 0,
    )

    start = time.perf_counter()
    for step in range(cfg.steps):
        lr = lr_at_step(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        x, y = dataset.get_batch("train", cfg.batch_size, device=device, generator=train_gen)
        s_logits = student(x)["logits"]
        if alpha > 0 and teacher is not None:
            with torch.no_grad():
                t_logits = teacher(x)["logits"]
        else:
            t_logits = s_logits.detach()
        loss, ce, kl = distillation_loss(s_logits, t_logits, y, temperature, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()

        is_last = step == cfg.steps - 1
        if (step + 1) % cfg.eval_interval == 0 or is_last:
            val = _val_loss(student, dataset, cfg, device)
            result.history.append(
                {
                    "step": step + 1,
                    "val_loss": val,
                    "train_ce": float(ce.item()),
                    "kl": float(kl.item()),
                    "lr": lr,
                }
            )
            if progress:
                print(
                    f"  [{label}] step {step + 1:4d}/{cfg.steps} val {val:.4f} "
                    f"ce {ce.item():.4f} kl {kl.item():.4f}",
                    flush=True,
                )

    result.wall_clock_s = time.perf_counter() - start
    if result.history:
        result.final_val_loss = result.history[-1]["val_loss"]
        result.final_val_ppl = math.exp(min(result.final_val_loss, 20.0))
    result.meta = {
        "student_config": student_config.to_dict(),
        "train_config": cfg.to_dict(),
        "device": str(device),
    }
    return student, result


@torch.no_grad()
def _val_loss(model: GPT, dataset: TokenDataset, cfg: TrainConfig, device: torch.device) -> float:
    """Validation loss on a fixed, arm-independent set of batches.

    The generator is re-seeded from a constant every time, so every arm and every
    eval point is scored on exactly the same windows.
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_batch("val", cfg.batch_size, device=device, generator=gen)
        total += model(x, targets=y)["loss"].item()
    model.train(was_training)
    return total / cfg.eval_batches
