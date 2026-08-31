"""Structured pruning of attention heads and MLP neurons.

Structured, not unstructured, on purpose. Zeroing scattered individual weights
gives a sparsity number and no speedup at all on any hardware that does dense
matmuls -- you still multiply by the zeros. Removing an entire attention head or
an entire MLP neuron removes whole rows and columns, so the matrices genuinely
get smaller and the FLOPs genuinely go away. Structured sparsity is the kind that
converts into serving cost.

**The importance criterion, stated up front.** Michel, Levy & Neubig (2019)
observe that if you attach a multiplicative mask ``xi_h`` to head ``h``'s output
and hold it at 1, then ``|dL/dxi_h|`` is a first-order estimate of how much the
loss changes if that head is removed -- it is the magnitude of the linear term in
the Taylor expansion of the loss around "head present". It costs one backward
pass for all heads at once, against one forward pass *per head* for direct
ablation, and it agrees with direct ablation closely enough to be the standard
criterion.

This repository is in the unusual position of being able to check that claim
rather than cite it: :mod:`transformer_internals.induction` already measures
*direct* ablation damage for every head, so the results compare the two rankings
head to head and report the correlation.

Scores are normalised **within each layer** before being compared across layers.
Raw gradient magnitudes differ systematically by depth -- later layers sit closer
to the loss and have larger gradients -- so a global ranking on raw scores prunes
layer 0 almost exclusively, for reasons that have nothing to do with importance.
This is exactly the normalisation Michel et al. apply, and skipping it is the
most common way this method is implemented wrongly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT

__all__ = [
    "PruneResult",
    "head_importance",
    "neuron_importance",
    "params_removed_by_head_pruning",
    "params_removed_by_neuron_pruning",
    "prune_sweep",
    "pruned",
]


def _calibration_batches(
    dataset: TokenDataset, n_batches: int, batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic calibration data, shared by every criterion and sweep."""
    return dataset.sequential_batches("train", batch_size=batch_size, limit=n_batches)


def head_importance(
    model: GPT,
    dataset: TokenDataset,
    n_batches: int = 4,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    normalize_per_layer: bool = True,
) -> torch.Tensor:
    """Gradient-based importance for every attention head.

    Args:
        model: The model.
        dataset: Calibration data.
        n_batches: Calibration batches.
        batch_size: Windows per batch.
        device: Device.
        normalize_per_layer: Divide each layer's scores by that layer's L2 norm,
            so scores are comparable across depth. See the module docstring.

    Returns:
        ``(n_layer, n_head)`` importance, higher meaning more important.
    """
    model = model.to(device).eval()
    cfg = model.config
    masks = []
    for block in model.h:
        m = torch.ones(cfg.n_head, device=device, requires_grad=True)
        block.attn.head_mask = m
        masks.append(m)

    scores = torch.zeros(cfg.n_layer, cfg.n_head)
    try:
        for x, y in _calibration_batches(dataset, n_batches, batch_size):
            model.zero_grad(set_to_none=True)
            for m in masks:
                if m.grad is not None:
                    m.grad = None
            loss = model(x.to(device), targets=y.to(device))["loss"]
            loss.backward()
            for layer, m in enumerate(masks):
                if m.grad is not None:
                    # Accumulate |dL/dxi| across batches: the expectation of the
                    # absolute value, not the absolute value of the expectation,
                    # since a head can help on one batch and hurt on another and
                    # signed cancellation would call it unimportant.
                    scores[layer] += m.grad.detach().abs().cpu()
    finally:
        for block in model.h:
            block.attn.head_mask = None
        model.zero_grad(set_to_none=True)

    if normalize_per_layer:
        norms = scores.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scores = scores / norms
    return scores


def neuron_importance(
    model: GPT,
    dataset: TokenDataset,
    n_batches: int = 4,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    normalize_per_layer: bool = True,
) -> torch.Tensor:
    """Gradient-based importance for every MLP neuron.

    Identical construction to :func:`head_importance`, on the ``4 * n_embd``
    hidden units of each block's MLP.

    Returns:
        ``(n_layer, 4 * n_embd)`` importance.
    """
    model = model.to(device).eval()
    cfg = model.config
    hidden = 4 * cfg.n_embd
    masks = []
    for block in model.h:
        m = torch.ones(hidden, device=device, requires_grad=True)
        block.mlp.neuron_mask = m
        masks.append(m)

    scores = torch.zeros(cfg.n_layer, hidden)
    try:
        for x, y in _calibration_batches(dataset, n_batches, batch_size):
            model.zero_grad(set_to_none=True)
            for m in masks:
                if m.grad is not None:
                    m.grad = None
            loss = model(x.to(device), targets=y.to(device))["loss"]
            loss.backward()
            for layer, m in enumerate(masks):
                if m.grad is not None:
                    scores[layer] += m.grad.detach().abs().cpu()
    finally:
        for block in model.h:
            block.mlp.neuron_mask = None
        model.zero_grad(set_to_none=True)

    if normalize_per_layer:
        norms = scores.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scores = scores / norms
    return scores


@contextmanager
def pruned(
    model: GPT,
    head_mask: torch.Tensor | None = None,
    neuron_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Temporarily apply structured masks.

    Args:
        model: The model.
        head_mask: ``(n_layer, n_head)`` of 0/1, or ``None``.
        neuron_mask: ``(n_layer, 4 * n_embd)`` of 0/1, or ``None``.

    Yields:
        Nothing; masks are restored on exit.
    """
    prev_h = [b.attn.head_mask for b in model.h]
    prev_n = [b.mlp.neuron_mask for b in model.h]
    try:
        for i, block in enumerate(model.h):
            if head_mask is not None:
                block.attn.head_mask = head_mask[i]
            if neuron_mask is not None:
                block.mlp.neuron_mask = neuron_mask[i]
        yield
    finally:
        for i, block in enumerate(model.h):
            block.attn.head_mask = prev_h[i]
            block.mlp.neuron_mask = prev_n[i]


def _keep_mask(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Build a 0/1 mask keeping the highest-scoring ``1 - sparsity`` fraction.

    The ranking is global across layers (after per-layer normalisation), so the
    method is free to prune a whole layer's worth of heads if that layer really
    is redundant -- which, for attention heads, it sometimes is.
    """
    flat = scores.flatten()
    n_prune = round(sparsity * flat.numel())
    mask = torch.ones_like(flat)
    if n_prune > 0:
        victims = torch.argsort(flat)[:n_prune]
        mask[victims] = 0.0
    return mask.view_as(scores)


def params_removed_by_head_pruning(model: GPT, mask: torch.Tensor) -> int:
    """Parameters a real implementation would delete for a given head mask.

    Removing one head deletes its slice of the q, k and v projections and the
    matching slice of the output projection: ``4 * head_dim * n_embd`` weights,
    plus ``3 * head_dim`` bias entries.

    Args:
        model: The model.
        mask: ``(n_layer, n_head)`` of 0/1.

    Returns:
        Parameter count removed.
    """
    cfg = model.config
    per_head = 4 * cfg.head_dim * cfg.n_embd + (3 * cfg.head_dim if cfg.attn_bias else 0)
    return int((mask == 0).sum().item()) * per_head


def params_removed_by_neuron_pruning(model: GPT, mask: torch.Tensor) -> int:
    """Parameters deleted for a given MLP-neuron mask.

    One neuron is a row of ``c_fc`` (``n_embd`` weights + 1 bias) and a column of
    ``c_proj`` (``n_embd`` weights).
    """
    cfg = model.config
    per_neuron = 2 * cfg.n_embd + (1 if cfg.mlp_bias else 0)
    return int((mask == 0).sum().item()) * per_neuron


@dataclass
class PruneResult:
    """One point on the pruning Pareto curve."""

    kind: str
    sparsity: float
    val_loss: float
    val_ppl: float
    params_removed: int
    params_total: int
    induction_second_copy_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sparsity": self.sparsity,
            "val_loss": self.val_loss,
            "val_ppl": self.val_ppl,
            "params_removed": self.params_removed,
            "params_total": self.params_total,
            "fraction_removed": self.params_removed / self.params_total,
            "induction_second_copy_loss": self.induction_second_copy_loss,
        }


@torch.no_grad()
def _masked_val_loss(
    model: GPT,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: str | torch.device,
) -> float:
    total, n = 0.0, 0
    for x, y in batches:
        logits = model(x.to(device))["logits"].float().cpu()
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
    return total / n


def prune_sweep(
    model: GPT,
    dataset: TokenDataset,
    sparsities: list[float],
    kind: str = "heads",
    n_eval_batches: int = 6,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    importance: torch.Tensor | None = None,
    induction_probe: Any = None,
) -> tuple[list[PruneResult], torch.Tensor]:
    """Prune at several sparsity levels and measure quality at each.

    Args:
        model: The model.
        dataset: Data for calibration and evaluation.
        sparsities: Fractions to prune, e.g. ``[0.0, 0.1, 0.25, 0.5]``.
        kind: ``"heads"`` or ``"neurons"``.
        n_eval_batches: Held-out batches for the loss.
        batch_size: Windows per batch.
        device: Device.
        importance: Precomputed importance, to avoid recomputing it per sweep.
        induction_probe: Optional callable ``(model) -> float`` evaluated under
            each mask. Used to answer "what happens when you prune a head that
            matters" by tracking the induction behaviour, not just the loss.

    Returns:
        ``(results, importance)``.
    """
    model = model.to(device).eval()
    if importance is None:
        importance = (
            head_importance(model, dataset, device=device)
            if kind == "heads"
            else neuron_importance(model, dataset, device=device)
        )
    eval_batches = dataset.sequential_batches("val", batch_size=batch_size, limit=n_eval_batches)
    total_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())

    results: list[PruneResult] = []
    for sparsity in sparsities:
        mask = _keep_mask(importance, sparsity)
        kw = {"head_mask": mask} if kind == "heads" else {"neuron_mask": mask}
        with pruned(model, **kw):
            loss = _masked_val_loss(model, eval_batches, device)
            probe = float(induction_probe(model)) if induction_probe is not None else None
        removed = (
            params_removed_by_head_pruning(model, mask)
            if kind == "heads"
            else params_removed_by_neuron_pruning(model, mask)
        )
        results.append(
            PruneResult(
                kind=kind,
                sparsity=sparsity,
                val_loss=loss,
                val_ppl=math.exp(min(loss, 20.0)),
                params_removed=removed,
                params_total=total_params,
                induction_second_copy_loss=probe,
            )
        )
    return results, importance
