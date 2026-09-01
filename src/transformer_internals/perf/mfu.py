"""Model-FLOPs utilisation: count the arithmetic a training step owes, divide by wall clock.

MFU is the fraction of the machine's achievable arithmetic rate that a training
step actually uses::

    MFU = model FLOPs per step / step seconds / peak FLOP/s

The numerator is deliberately *model* FLOPs, not hardware FLOPs: it counts the
arithmetic the model definition requires, and charges nothing for recomputation,
for padding, or for a kernel that does the same work twice. A run that turns on
activation checkpointing does more hardware FLOPs and its MFU goes *down*, which
is the point. Hardware-FLOPs utilisation (HFU) is the other convention and is
flattering for the same reason. This module reports MFU, as defined in the PaLM
paper (Chowdhery et al., 2022, appendix B).

Two ways to count the numerator, both implemented:

**6ND.** Every parameter takes part in one multiply and one add per token in the
forward pass, so forward is ``2N`` FLOPs per token. The backward pass computes
two gradients per parameter (one with respect to the input, one with respect to
the weight), each about as expensive as the forward, so backward is ``4N``.
Total ``6N`` per token, ``6ND`` for D tokens. It is a good rule of thumb and it
is wrong in one specific way: it has no sequence-length term, because it charges
nothing for ``QK^T`` and ``attention @ V``, which have no parameters.

**Exact per-layer.** Same 6N term, plus the attention quadratic term
``12 * n_layer * n_embd * seq_len`` FLOPs per token of training, which is three
times the forward cost ``2 * (2 * T * C)`` per layer per token. This is the term
that makes long-context training expensive, and it is the difference between an
MFU number that holds at 512 tokens and one that quietly drifts at 8192.

:func:`flops_per_token_exact` returns the breakdown so the two can be compared;
:func:`measure_step_mfu` uses the exact count, and reports the ratio to 6ND so
the reader can see how much the choice was worth at that shape.

Non-matmul work (LayerNorm, GELU, softmax, residual adds, the optimiser update)
is excluded from the numerator by convention, because peak FLOP/s in the
denominator is a matmul number. Those ops cost real time, and that time shows up
as *lower* MFU. Finding out how much of it they account for is what
:mod:`~transformer_internals.perf.roofline` and
:mod:`~transformer_internals.perf.profiling` are for.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT

__all__ = [
    "PUBLISHED_ACCELERATORS",
    "MFUReport",
    "flops_6nd",
    "flops_per_token_exact",
    "measure_step_mfu",
    "mfu_on_published_gpu",
]


#: Published dense peaks for accelerators this repository does **not** have.
#: Every number here is copied from a vendor datasheet, is a *dense* figure with
#: structured sparsity off, and is used only as a denominator in arithmetic that
#: is labelled ``modelled`` wherever it is reported. None of it is measured.
PUBLISHED_ACCELERATORS: dict[str, dict[str, Any]] = {
    "NVIDIA A100 80GB SXM": {
        "peak_flops_per_s": 312e12,
        "precision": "bf16 tensor core, dense",
        "memory_bytes_per_s": 2039e9,
        "memory": "HBM2e",
        "source": "NVIDIA A100 Tensor Core GPU datasheet, rev. 2021 (312 TFLOPS BFLOAT16 dense, 2039 GB/s)",
    },
    "NVIDIA H100 SXM5": {
        "peak_flops_per_s": 989.4e12,
        "precision": "bf16 tensor core, dense",
        "memory_bytes_per_s": 3350e9,
        "memory": "HBM3",
        "source": "NVIDIA H100 Tensor Core GPU datasheet, rev. 2023 (1979 TFLOPS BFLOAT16 with sparsity, i.e. 989.4 dense; 3.35 TB/s)",
    },
    "NVIDIA H200 SXM": {
        "peak_flops_per_s": 989.4e12,
        "precision": "bf16 tensor core, dense",
        "memory_bytes_per_s": 4800e9,
        "memory": "HBM3e",
        "source": "NVIDIA H200 Tensor Core GPU datasheet, rev. 2024 (same SM peak as H100 SXM; 4.8 TB/s)",
    },
}


def flops_6nd(n_params: int, n_tokens: int) -> float:
    """The 6ND estimate of training FLOPs.

    Args:
        n_params: Parameter count N. Use the non-embedding count if you want the
            convention that Kaplan et al. (2020) used; use the total if you want
            the one the PaLM paper used. :func:`flops_per_token_exact` reports
            both so the choice is explicit rather than inherited.
        n_tokens: Tokens processed, D.

    Returns:
        ``6 * n_params * n_tokens``.
    """
    return 6.0 * n_params * n_tokens


def flops_per_token_exact(
    cfg: GPTConfig,
    seq_len: int,
    causal_aware: bool = False,
) -> dict[str, float]:
    """Per-token training FLOPs, counted operator by operator.

    Everything is per token of the batch, per training step (forward plus
    backward, i.e. three times the forward cost of each matmul).

    Args:
        cfg: Model architecture.
        seq_len: Context length T. Only the attention term depends on it.
        causal_aware: If True, halve the ``QK^T`` and ``attn @ V`` terms, which
            is what a masked-tile kernel such as FlashAttention actually does:
            it never computes the upper triangle. The dense einsum path in
            :mod:`transformer_internals.model` *does* compute it and then masks
            it, so the default is False and matches this implementation. The
            distinction is worth 1.5% of the total at T=512 for GPT-2 small and
            considerably more at long context.

    Returns:
        A breakdown dict. ``total`` is the sum of the three FLOP terms;
        ``ratio_to_6nd_total`` compares it to ``6N`` with N the total parameter
        count.
    """
    c = float(cfg.n_embd)
    t = float(seq_len)
    kv_c = float(cfg.kv_heads * cfg.head_dim)

    # Per layer, per token, forward: qkv, output projection, both MLP matmuls.
    per_layer_proj_fwd = 2.0 * c * (c + 2.0 * kv_c) + 2.0 * c * c + 2.0 * (4.0 * c * c) * 2.0
    # Per layer, per token, forward: QK^T and attn @ V, both 2*T*C.
    quad = 2.0 * (2.0 * t * c)
    if causal_aware:
        quad /= 2.0

    n_layer = float(cfg.n_layer)
    head_fwd = 2.0 * c * float(cfg.vocab_size)

    fwd_params = n_layer * per_layer_proj_fwd + head_fwd
    fwd_attn = n_layer * quad
    total_fwd = fwd_params + fwd_attn

    n_total = _param_count(cfg)
    n_nonembed = _param_count(cfg, non_embedding=True)
    return {
        "seq_len": t,
        "forward_projection_flops_per_token": fwd_params,
        "forward_attention_quadratic_flops_per_token": fwd_attn,
        "forward_total_flops_per_token": total_fwd,
        "training_flops_per_token": 3.0 * total_fwd,
        "training_attention_quadratic_flops_per_token": 3.0 * fwd_attn,
        "attention_quadratic_fraction": fwd_attn / total_fwd,
        "n_params_total": float(n_total),
        "n_params_non_embedding": float(n_nonembed),
        "flops_6n_total_per_token": 6.0 * n_total,
        "flops_6n_non_embedding_per_token": 6.0 * n_nonembed,
        "ratio_to_6nd_total": (3.0 * total_fwd) / (6.0 * n_total),
        "causal_aware": float(causal_aware),
    }


def _param_count(cfg: GPTConfig, non_embedding: bool = False) -> int:
    """Parameter count from the config alone, without building the model.

    Kept analytic so the FLOP arithmetic can be checked at any shape without
    allocating 124M parameters. ``tests/test_perf.py`` asserts it agrees with
    ``GPT(cfg).num_parameters()``.
    """
    c, v, p, ln = cfg.n_embd, cfg.vocab_size, cfg.n_positions, cfg.n_layer
    kv_c = cfg.kv_heads * cfg.head_dim
    per_layer = 0
    per_layer += c * (c + 2 * kv_c) + (c + 2 * kv_c if cfg.attn_bias else 0)  # qkv
    per_layer += c * c + (c if cfg.attn_bias else 0)  # attn out
    per_layer += c * 4 * c + (4 * c if cfg.mlp_bias else 0)  # mlp up
    per_layer += 4 * c * c + (c if cfg.mlp_bias else 0)  # mlp down
    per_layer += 4 * c  # two LayerNorms, gain and bias
    total = ln * per_layer + 2 * c  # + final LayerNorm
    total += v * c  # token embedding
    if cfg.pos_embedding == "learned":
        total += p * c
    if not cfg.tie_weights:
        total += v * c
    if non_embedding:
        total -= v * c
        if cfg.pos_embedding == "learned":
            total -= p * c
    return total


@dataclass
class MFUReport:
    """A measured MFU number and everything needed to check it.

    Attributes:
        device: Where the steps ran.
        batch / seq: Shape of one step.
        steps_timed: How many optimiser steps were timed.
        step_s: Fastest step wall clock over the timed steps. The minimum, not
            the mean: contention only ever adds time, so the fastest step is the
            closest estimate of the cost with nothing in the way, and it stays
            comparable across runs measured on a machine that is also doing
            other work.
        tokens_per_s: Throughput from that step.
        model_flops_per_step: Numerator, exact count.
        achieved_flops_per_s: ``model_flops_per_step / median_step_s``.
        peak_flops_per_s: Denominator, measured on this machine.
        mfu: The ratio. Measured end to end.
        flops_breakdown: Output of :func:`flops_per_token_exact`.
        modelled_gpu_mfu: The same achieved rate divided by published peaks.
            Arithmetic on a datasheet, not a measurement, and named so.
    """

    device: str
    batch: int
    seq: int
    steps_timed: int
    step_s: float
    tokens_per_s: float
    model_flops_per_step: float
    achieved_flops_per_s: float
    peak_flops_per_s: float
    mfu: float
    step_times_s: list[float] = field(default_factory=list)
    flops_breakdown: dict[str, float] = field(default_factory=dict)
    modelled_gpu_mfu: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mfu_on_published_gpu(achieved_flops_per_s: float) -> list[dict[str, Any]]:
    """What this achieved rate would score against published accelerator peaks.

    This is division, not measurement. It answers "if a run sustained this model
    FLOP/s on an A100, what MFU would that be", which is the only honest thing
    that can be said about hardware that is not in the machine. It does **not**
    predict what rate an A100 would sustain on this workload.

    Args:
        achieved_flops_per_s: A measured model-FLOP/s rate.

    Returns:
        One row per accelerator, each carrying its datasheet citation.
    """
    rows: list[dict[str, Any]] = []
    for name, spec in PUBLISHED_ACCELERATORS.items():
        rows.append(
            {
                "accelerator": name,
                "status": "modelled from published spec, not measured",
                "peak_flops_per_s": spec["peak_flops_per_s"],
                "precision": spec["precision"],
                "mfu_if_this_rate_were_sustained": achieved_flops_per_s
                / spec["peak_flops_per_s"],
                "ridge_point_flops_per_byte": spec["peak_flops_per_s"]
                / spec["memory_bytes_per_s"],
                "source": spec["source"],
            }
        )
    return rows


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_step_mfu(
    peak_flops_per_s: float,
    cfg: GPTConfig | None = None,
    batch: int = 4,
    seq: int = 256,
    steps: int = 6,
    warmup: int = 2,
    device: torch.device | str = "cpu",
    model: GPT | None = None,
    lr: float = 1e-4,
) -> MFUReport:
    """Time real training steps and report their MFU.

    A step is forward, backward and one AdamW update on synthetic token ids. The
    data is synthetic on purpose: this measures the machine's arithmetic rate,
    and a real dataloader would put its own stall time inside the number. The
    dataloader is measured separately, by
    :mod:`~transformer_internals.perf.diagnose`.

    Args:
        peak_flops_per_s: Denominator. Pass a *measured* peak.
        cfg: Model config. Defaults to GPT-2 124M with the given seq as context.
        batch / seq: Step shape.
        steps: Timed steps.
        warmup: Untimed steps first, which matter: the first step allocates,
            and on MPS it also compiles kernels.
        device: Where to run.
        model: Reuse an existing model instead of building one.
        lr: Optimiser learning rate. Irrelevant to timing, present so the update
            is a real update.

    Returns:
        An :class:`MFUReport`.
    """
    device = torch.device(device)
    cfg = cfg or GPTConfig(n_positions=max(seq, 64))
    model = model if model is not None else GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    x = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = model(x, targets=y)["loss"]
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one_step()
    _synchronize(device)

    samples: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        one_step()
        _synchronize(device)
        samples.append(time.perf_counter() - t0)

    breakdown = flops_per_token_exact(cfg, seq)
    tokens = batch * seq
    flops_per_step = breakdown["training_flops_per_token"] * tokens
    best = min(samples)
    achieved = flops_per_step / best

    return MFUReport(
        device=str(device),
        batch=batch,
        seq=seq,
        steps_timed=steps,
        step_s=best,
        tokens_per_s=tokens / best,
        model_flops_per_step=flops_per_step,
        achieved_flops_per_s=achieved,
        peak_flops_per_s=peak_flops_per_s,
        mfu=achieved / peak_flops_per_s,
        step_times_s=samples,
        flops_breakdown=breakdown,
        modelled_gpu_mfu=mfu_on_published_gpu(achieved),
        meta={
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "model_config": cfg.to_dict(),
            "statistic": "minimum over the timed steps",
        },
    )
