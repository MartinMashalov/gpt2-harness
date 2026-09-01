"""Sharded data parallel: ZeRO-1, ZeRO-2 and ZeRO-3 (FSDP), written out.

Data parallel replicates everything. For a model with ``N`` parameters trained
in fp32 with Adam, every rank holds

* ``4N`` bytes of parameters,
* ``4N`` bytes of gradients,
* ``8N`` bytes of optimizer state (two Adam moments),

so ``16N`` bytes per rank, ``p`` times over, and ``p - 1`` copies of it are
redundant. ZeRO (Rajbhandari et al., "ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models", SC20) removes the redundancy in three
stages:

===========  ==========================================  ===================
stage        what is sharded                             bytes per rank
===========  ==========================================  ===================
DDP          nothing                                     ``16N``
ZeRO-1       optimizer state                             ``8N + 8N/p``
ZeRO-2       optimizer state, gradients                  ``4N + 12N/p``
ZeRO-3       optimizer state, gradients, parameters       ``16N/p``
===========  ==========================================  ===================

Those are the published formulas. The numbers this file *measures* are in
``results/parallel_comms.json``: the bytes of live tensor storage each rank
actually holds, counted per distinct storage so weight tying is charged once.

The communication trade is exact and worth stating, because it is the reason
ZeRO-3 is not free. DDP moves one all-reduce of ``N`` per step. ZeRO-2 replaces
it with one reduce-scatter of ``N`` plus one all-gather of ``N`` -- the same ring
volume as the all-reduce, since an all-reduce *is* a reduce-scatter followed by
an all-gather. ZeRO-3 adds an all-gather of ``N`` in the forward pass and
another in the backward, so it moves 1.5x to 2x the DDP volume in exchange for
``p``-fold parameter memory.

The ZeRO-3 implementation here gathers a unit's parameters immediately before
the unit runs, frees them as soon as it is done, and gathers them again in the
backward pass. Freeing is the whole point -- an implementation that keeps the
gathered parameters alive for the backward pass has ZeRO-3's communication cost
and DDP's memory. Because the parameters are gone by the time the backward pass
arrives, the unit's forward is recomputed there, which is exactly FSDP combined
with activation checkpointing.

Wire dtypes, and the fp32 master shard
--------------------------------------
Both optimizers take two independent dtype axes, matching FSDP's
``MixedPrecision(param_dtype=..., reduce_dtype=...)``:

* ``param_dtype`` is what the *parameter* all-gather carries.
* ``reduce_dtype`` is what the *gradient* reduction carries.

Neither of them is what the optimizer updates. Rank ``r`` holds a persistent
**fp32 master shard** of its slice of the flat parameter vector, and every AdamW
update is applied to that. This is the piece that makes bf16 collectives safe to
use at all: a bf16 value carries 8 significand bits, an Adam update is commonly
1e-4 of the weight it is added to, and adding those inside bf16 rounds the
update away. Keeping the master in fp32 confines the bf16 rounding to the wire,
where it is a one-off error per step rather than an error that compounds into
the optimizer state.

``results/parallel_comms.json`` measures what choosing bf16 on the wire actually
costs, by running the same multi-step trajectory comparison at both dtypes.

Gradient clipping is not a local operation
------------------------------------------
A global gradient-norm clip needs the norm over *every* parameter, and rank
``r`` holds only its slice of the gradient. So a correct clip needs the ranks to
agree on one number before any of them updates, which is an extra collective:
each rank all-reduces its local sum of squares, takes the square root of the
total, and scales its own shard by the same coefficient
``torch.nn.utils.clip_grad_norm_`` would have used.

The extra collective is tiny -- one fp32 scalar, 4 bytes -- and it is therefore
pure latency, which is the expensive kind of collective. It is counted like
every other collective in this repository, and ZeRO-1 does not pay it: stage 1
all-reduces the *full* gradient anyway, so every rank can compute the global
norm locally. That difference between the stages is measured, not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call

from transformer_internals.parallel import comms
from transformer_internals.parallel.common import (
    identical_batch,
    identical_model,
    parallel_config,
    state_bytes,
)
from transformer_internals.parallel.data_parallel import average_gradients, local_shard
from transformer_internals.precision import reduce_dtype_of

__all__ = [
    "ShardedAdamW",
    "Zero3Model",
    "adamw_update_",
    "clip_coefficient",
    "decay_mask",
    "global_grad_norm",
    "zero3_equivalence_worker",
    "zero_equivalence_worker",
]


# --------------------------------------------------------------------------- #
# flat-buffer helpers
# --------------------------------------------------------------------------- #


def _padded_numel(total: int, world_size: int) -> int:
    """Round ``total`` up to a multiple of ``world_size``.

    Sharding a flat buffer needs every rank's slice to be the same length --
    ``all_gather_into_tensor`` and ``reduce_scatter_tensor`` both require it --
    so the flat buffer is padded. The padding carries zero gradient, so it
    receives a zero Adam update and never affects a real parameter.
    """
    return int(math.ceil(total / world_size) * world_size)


def _flatten(tensors: Sequence[torch.Tensor], numel: int) -> torch.Tensor:
    """Concatenate into one padded flat buffer, on the tensors' own device."""
    flat = torch.zeros(numel, dtype=tensors[0].dtype, device=tensors[0].device)
    offset = 0
    for t in tensors:
        n = t.numel()
        flat[offset : offset + n] = t.reshape(-1)
        offset += n
    return flat


def decay_mask(params: Sequence[nn.Parameter]) -> torch.Tensor:
    """A flat 0/1 mask marking which elements get weight decay.

    Matmul weights (``dim >= 2``) decay; biases and LayerNorm gains do not.
    Decaying a LayerNorm gain toward zero is decaying the network toward the
    identity, which is not regularisation. Keeping the rule as a flat mask is
    what lets a *sharded* optimizer reproduce a two-parameter-group AdamW
    exactly: the shard boundary can fall in the middle of a tensor, so the rule
    has to be expressible per element.
    """
    return torch.cat(
        [
            torch.full((p.numel(),), float(p.dim() >= 2), device=p.device)
            for p in params
        ]
    )


def adamw_update_(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: torch.Tensor | float,
) -> None:
    """One decoupled AdamW update, in place, elementwise.

    Written out rather than delegated so that the sharded optimizer is provably
    the same arithmetic as ``torch.optim.AdamW`` -- the update order here
    (decoupled decay first, then the bias-corrected moment ratio) is torch's
    ``_single_tensor_adamw``, and matching it is what makes the multi-step
    trajectory comparison meaningful rather than approximate.
    """
    if isinstance(weight_decay, torch.Tensor) or weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)


# --------------------------------------------------------------------------- #
# gradient clipping under sharding
# --------------------------------------------------------------------------- #


def global_grad_norm(
    shard_sumsq: torch.Tensor,
    replicated_sumsq: torch.Tensor | float = 0.0,
    *,
    reduce: bool,
) -> torch.Tensor:
    """The global gradient L2 norm, from a rank that holds only a slice of it.

    Args:
        shard_sumsq: Sum of squares of the gradient elements this rank *owns*,
            which no other rank holds. Summed across ranks when ``reduce``.
        replicated_sumsq: Sum of squares of gradients that are already identical
            on every rank, such as ZeRO-3's root parameters after their
            all-reduce. Added *after* the collective, deliberately: putting it
            inside would count it ``p`` times and shrink the norm by a factor of
            ``sqrt(p)`` on the replicated part, which is a bug that grows with
            the world size and therefore passes on one GPU.
        reduce: Whether the shard sums need a collective. False for ZeRO-1,
            where every rank already holds the whole averaged gradient and can
            compute the norm without talking to anyone.

    Returns:
        A scalar tensor: the global norm.
    """
    total = shard_sumsq.clone()
    if reduce:
        comms.all_reduce(total)
    return (total + replicated_sumsq).sqrt()


def clip_coefficient(total_norm: torch.Tensor, max_norm: float) -> torch.Tensor:
    """The scale ``torch.nn.utils.clip_grad_norm_`` applies, to the letter.

    ``max_norm / (total_norm + 1e-6)``, clamped at 1 and then applied
    unconditionally. The clamp and the epsilon both matter for reproducing
    torch's result exactly: an ``if norm > max_norm`` written by hand differs
    from torch by the epsilon, which is small but not zero, and a trajectory
    comparison over several steps will find it.
    """
    return torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)


# --------------------------------------------------------------------------- #
# ZeRO-1 / ZeRO-2
# --------------------------------------------------------------------------- #


class ShardedAdamW:
    """AdamW whose state -- and, from stage 2, whose gradients -- are sharded.

    The parameters stay replicated (that is what distinguishes stages 1 and 2
    from stage 3), but rank ``r`` owns only slice ``r`` of the flattened
    parameter vector: it holds the two Adam moments for that slice and nobody
    else's, updates only that slice, and then all-gathers the updated slices so
    that every rank ends the step with identical parameters again.

    Stage 1 all-reduces the full gradient and then throws away the parts it does
    not own. Stage 2 reduce-scatters instead, so a rank never materialises a
    gradient it will not use -- same ring volume, ``p``-fold less gradient
    memory.

    Args:
        params: The replicated parameters, in a fixed order that must be the
            same on every rank.
        rank / world_size: Position in the group.
        stage: 1 or 2.
        lr, betas, eps, weight_decay: AdamW hyper-parameters.
        decay_matmul_only: Apply weight decay only to ``dim >= 2`` tensors.
        grad_clip: Global gradient-norm clip, or ``None`` for no clipping. Under
            sharding this is not a local operation; see the module docstring.
        reduce_dtype: Dtype the gradient reduction carries. ``None`` is fp32.
        param_dtype: Dtype the parameter all-gather carries. ``None`` is fp32.
            The master shard this optimizer updates is fp32 either way.
    """

    def __init__(
        self,
        params: Sequence[nn.Parameter],
        rank: int,
        world_size: int,
        stage: int = 2,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        decay_matmul_only: bool = True,
        grad_clip: float | None = None,
        reduce_dtype: str | torch.dtype | None = None,
        param_dtype: str | torch.dtype | None = None,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError(f"ShardedAdamW implements stages 1 and 2, got {stage}")
        self.params = list(params)
        self.rank = rank
        self.world_size = world_size
        self.stage = stage
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.grad_clip = grad_clip
        self.reduce_dtype = reduce_dtype_of(reduce_dtype)
        self.param_dtype = reduce_dtype_of(param_dtype)
        #: Norm of the *global* gradient at the last step, before clipping.
        #: Recorded so the sharded clip can be compared against a single-process
        #: one on the number itself and not only on the parameters it produced.
        self.last_grad_norm: float = float("nan")

        self.numel = sum(p.numel() for p in self.params)
        self.padded = _padded_numel(self.numel, world_size)
        self.shard_size = self.padded // world_size
        self.shard_start = rank * self.shard_size

        device = self.params[0].device
        full_decay = (
            decay_mask(self.params) * weight_decay
            if decay_matmul_only
            else torch.full((self.numel,), weight_decay, device=device)
        )
        padded_decay = torch.zeros(self.padded, device=device)
        padded_decay[: self.numel] = full_decay
        #: Per-element weight decay for this rank's shard only.
        self.decay = padded_decay[self.shard_start : self.shard_start + self.shard_size].clone()

        # The sharded state. This is the memory ZeRO exists to save: 2 * N/p
        # elements instead of 2 * N.
        self.exp_avg = torch.zeros(self.shard_size, device=device)
        self.exp_avg_sq = torch.zeros(self.shard_size, device=device)
        #: Stage 2 keeps the reduced gradient shard here and frees ``p.grad``.
        self.grad_shard = torch.zeros(self.shard_size, device=device)
        #: The fp32 master copy of this rank's slice of the flat parameter
        #: vector, updated in place every step. Persistent, rather than
        #: re-derived from the replicated parameters each step, which is the
        #: whole point: with ``param_dtype`` bf16 the replicated copies have been
        #: through a bf16 all-gather and re-deriving from them would let that
        #: rounding compound into the optimizer trajectory.
        self.master_shard = (
            _flatten([p.detach() for p in self.params], self.padded)[
                self.shard_start : self.shard_start + self.shard_size
            ]
            .clone()
            .float()
        )

    # -- memory accounting ------------------------------------------------

    def state_bytes(self) -> int:
        """Bytes of Adam moments this rank holds -- the ``8N/p`` of the ZeRO table.

        The fp32 master shard is *not* counted here. It is a parameter copy, not
        Adam state, and folding it into the optimizer figure would make the
        sharded number look worse than the ZeRO formulas for a reason that has
        nothing to do with ZeRO. It is reported on its own in
        :meth:`resident_bytes`.
        """
        return state_bytes([self.exp_avg, self.exp_avg_sq])

    def resident_bytes(self) -> dict[str, int]:
        """Parameter / gradient / optimizer bytes resident on this rank.

        ``decay_mask`` is listed on its own rather than folded into
        ``optimizer``. It is real memory this implementation holds -- a flat
        per-element weight-decay coefficient for the shard -- but it is an
        artifact of expressing parameter groups on a flat buffer, not part of
        the Adam state that the ZeRO memory formulas talk about. Reporting it
        inside ``optimizer`` would make the sharded number look worse than
        ZeRO's for a reason that has nothing to do with ZeRO.
        """
        param_bytes = state_bytes(self.params)
        if self.stage >= 2:
            grad_bytes = state_bytes([self.grad_shard])
        else:
            grad_bytes = state_bytes([p.grad for p in self.params if p.grad is not None])
        opt_bytes = self.state_bytes()
        decay_bytes = state_bytes([self.decay])
        master_bytes = state_bytes([self.master_shard])
        return {
            "params": param_bytes,
            "grads": grad_bytes,
            "optimizer": opt_bytes,
            "master_shard": master_bytes,
            "decay_mask": decay_bytes,
            "total": param_bytes + grad_bytes + opt_bytes + master_bytes + decay_bytes,
        }

    # -- the step ---------------------------------------------------------

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Reduce the gradients, clip, update this rank's shard, re-gather."""
        self.step_count += 1
        grads = [
            p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params
        ]
        flat_grad = _flatten(grads, self.padded)

        # Whether the gradient reduction needs a collective in a narrower dtype
        # is a wire decision; the accumulation stays fp32 on both stages.
        if self.stage == 1:
            # Every rank ends up with the whole averaged gradient and then uses
            # 1/p of it. The wasted bytes are exactly what stage 2 removes.
            if self.reduce_dtype != flat_grad.dtype:
                wire = flat_grad.to(self.reduce_dtype)
                comms.all_reduce(wire)
                flat_grad = wire.to(torch.float32)
            else:
                comms.all_reduce(flat_grad)
            flat_grad.div_(self.world_size)
            self.grad_shard.copy_(
                flat_grad[self.shard_start : self.shard_start + self.shard_size]
            )
            # Stage 1 holds the entire averaged gradient, so the global norm is
            # already computable here with no further communication.
            local_sumsq = (
                flat_grad.pow(2).sum() if self.grad_clip is not None else None
            )
            needs_norm_collective = False
        else:
            if self.reduce_dtype != flat_grad.dtype:
                wire_in = flat_grad.to(self.reduce_dtype)
                wire_out = torch.empty(
                    self.shard_size, dtype=self.reduce_dtype, device=flat_grad.device
                )
                comms.reduce_scatter_into(wire_out, wire_in)
                self.grad_shard.copy_(wire_out.to(torch.float32))
                del wire_in, wire_out
            else:
                comms.reduce_scatter_into(self.grad_shard, flat_grad)
            self.grad_shard.div_(self.world_size)
            # The full gradient is no longer needed anywhere on this rank.
            self.zero_grad()
            # This rank owns a disjoint slice, so the sums have to be added up
            # across ranks before anybody can clip. One extra collective, four
            # bytes, all latency.
            local_sumsq = (
                self.grad_shard.pow(2).sum() if self.grad_clip is not None else None
            )
            needs_norm_collective = True
        del flat_grad

        # No clip, no collective. The norm is not computed at all when it would
        # not be used, so turning clipping on is visible in the byte count.
        if self.grad_clip is not None and local_sumsq is not None:
            total_norm = global_grad_norm(local_sumsq, reduce=needs_norm_collective)
            self.last_grad_norm = float(total_norm)
            self.grad_shard.mul_(clip_coefficient(total_norm, self.grad_clip))

        adamw_update_(
            self.master_shard,
            self.grad_shard,
            self.exp_avg,
            self.exp_avg_sq,
            step=self.step_count,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.decay,
        )

        # Every rank needs every parameter for the next forward pass, so the
        # updated shards go back out. This all-gather is the second half of what
        # DDP's all-reduce was doing all along. Under param_dtype=bf16 it is the
        # collective that rounds; the master shard above is untouched by it.
        gathered = torch.empty(
            self.padded, dtype=self.param_dtype, device=self.master_shard.device
        )
        comms.all_gather_into(gathered, self.master_shard.to(self.param_dtype))

        offset = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                p.copy_(gathered[offset : offset + n].view_as(p))
                offset += n


# --------------------------------------------------------------------------- #
# ZeRO-3
# --------------------------------------------------------------------------- #


class _GatherRunFree(torch.autograd.Function):
    """Run one sharded unit: all-gather its parameters, use them, free them.

    Forward: all-gather the flat parameter shard into the full flat parameter,
    run the unit under ``no_grad``, drop the gathered buffer. The only things
    that survive are the unit's *input* and its output, so peak parameter memory
    is one unit's worth, not the model's.

    Backward: all-gather again, recompute the unit's forward with grad enabled,
    backpropagate through it, then reduce-scatter the parameter gradient so each
    rank keeps only the shard it owns. The division by the world size turns the
    sum over ranks into the mean over the global batch, which is what makes this
    match a single-process run on the full batch.

    ``param_dtype`` and ``reduce_dtype`` say what the two collectives carry.
    Unlike :class:`ShardedAdamW`, where the parameters stay replicated and bf16
    only ever touches the wire, here the gathered buffer *is* what the unit
    computes with, so ``param_dtype`` sets the compute dtype of the block as
    well. That is what FSDP's ``MixedPrecision(param_dtype=...)`` does, and it is
    why the two implementations are worth measuring separately.

    The shard itself, and therefore the optimizer's view of the parameters, stays
    fp32 on both paths.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        shard: torch.Tensor,
        module: nn.Module,
        meta: tuple[tuple[str, torch.Size, int], ...],
        world_size: int,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.module = module
        ctx.meta = meta
        ctx.world_size = world_size
        ctx.param_dtype = param_dtype
        ctx.reduce_dtype = reduce_dtype
        ctx.input_dtype = x.dtype
        ctx.save_for_backward(x, shard)
        with torch.no_grad():
            full = torch.empty(
                shard.numel() * world_size, dtype=param_dtype, device=shard.device
            )
            comms.all_gather_into(full, shard.to(param_dtype))
            out = _run_unit(module, full, meta, x.to(param_dtype))
            del full
        # The residual stream leaves the unit in the dtype it entered in, so the
        # replicated root of the model and the loss stay in fp32 whatever the
        # blocks computed in.
        return out.to(ctx.input_dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        x, shard = ctx.saved_tensors
        world_size = ctx.world_size
        param_dtype = ctx.param_dtype
        full = torch.empty(
            shard.numel() * world_size, dtype=param_dtype, device=shard.device
        )
        comms.all_gather_into(full, shard.to(param_dtype))
        full.requires_grad_(True)
        x_in = x.detach().to(param_dtype).requires_grad_(True)
        with torch.enable_grad():
            out = _run_unit(ctx.module, full, ctx.meta, x_in)
        grad_x, grad_full = torch.autograd.grad(
            out, [x_in, full], grad_outputs=grad_out.to(param_dtype)
        )
        wire = grad_full.contiguous().to(ctx.reduce_dtype)
        wire_shard = torch.empty(shard.numel(), dtype=ctx.reduce_dtype, device=shard.device)
        comms.reduce_scatter_into(wire_shard, wire)
        grad_shard = wire_shard.to(shard.dtype)
        grad_shard.div_(world_size)
        del full, grad_full, wire, wire_shard
        return grad_x.to(ctx.input_dtype), grad_shard, None, None, None, None, None


def _run_unit(
    module: nn.Module,
    full: torch.Tensor,
    meta: tuple[tuple[str, torch.Size, int], ...],
    x: torch.Tensor,
) -> torch.Tensor:
    """Call ``module`` with parameters taken as views into the flat buffer."""
    params: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, numel in meta:
        params[name] = full[offset : offset + numel].view(shape)
        offset += numel
    out = functional_call(module, params, (x,))
    return out[0] if isinstance(out, tuple) else out


class Zero3Model(nn.Module):
    """A GPT whose transformer blocks are parameter-sharded across ranks.

    The blocks are the units, which is the same choice FSDP recipes make with
    ``transformer_auto_wrap_policy``: one flat parameter per block, gathered for
    that block and freed after it. The root parameters -- embeddings, final
    LayerNorm, tied head -- stay replicated and are kept in sync by an ordinary
    gradient all-reduce, again matching what FSDP's root unit does.

    After construction the blocks' own ``nn.Parameter`` storages are released,
    so :func:`~transformer_internals.parallel.common.state_bytes` on this model
    reports the sharded footprint and not a hidden full copy.

    Args:
        model: A constructed :class:`~transformer_internals.model.GPT`. It is
            consumed, not copied.
        rank / world_size: Position in the group.
        param_dtype: Dtype the per-unit parameter all-gather carries, and
            therefore the dtype the unit computes in. ``None`` is fp32.
        reduce_dtype: Dtype the per-unit gradient reduce-scatter carries.
            ``None`` is fp32. The shards stay fp32 on both settings.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        param_dtype: str | torch.dtype | None = None,
        reduce_dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.param_dtype = reduce_dtype_of(param_dtype)
        self.reduce_dtype = reduce_dtype_of(reduce_dtype)

        self.unit_meta: list[tuple[tuple[str, torch.Size, int], ...]] = []
        self.shards = nn.ParameterList()
        for block in model.h:
            named = list(block.named_parameters())
            meta = tuple((n, p.shape, p.numel()) for n, p in named)
            numel = sum(m[2] for m in meta)
            padded = _padded_numel(numel, world_size)
            flat = _flatten([p.detach() for _, p in named], padded)
            shard_size = padded // world_size
            shard = flat[rank * shard_size : (rank + 1) * shard_size].clone()
            self.unit_meta.append(meta)
            self.shards.append(nn.Parameter(shard))
            # Release the replicated copies. This is the memory saving; without
            # it the shard would merely be a fourth copy of the same weights.
            for _, p in named:
                p.data = torch.empty(0, device=p.device)
                p.requires_grad_(False)

        #: Parameters outside the blocks, replicated across ranks.
        self.root_params = [
            p
            for n, p in model.named_parameters()
            if not n.startswith("h.") and p.requires_grad and p.numel() > 0
        ]

    def unit_decay_mask(self, unit: int, weight_decay: float) -> torch.Tensor:
        """Per-element weight decay for this rank's shard of one unit."""
        meta = self.unit_meta[unit]
        device = self.shards[unit].device
        full = torch.cat(
            [
                torch.full((numel,), float(len(shape) >= 2), device=device)
                for _, shape, numel in meta
            ]
        )
        shard_size = self.shards[unit].numel()
        padded = torch.zeros(shard_size * self.world_size, device=device)
        padded[: full.numel()] = full
        return padded[self.rank * shard_size : (self.rank + 1) * shard_size] * weight_decay

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Embed, run every sharded block, then the replicated head.

        Deliberately a re-implementation of :meth:`GPT.forward` rather than a
        call into it: the blocks no longer own their parameters, so the only way
        through them is the gather/run/free path.
        """
        model = self.model
        _b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = model.drop(model.wte(idx) + model.position_embeddings(pos))

        for unit, block in enumerate(model.h):
            x = _GatherRunFree.apply(
                x,
                self.shards[unit],
                block,
                self.unit_meta[unit],
                self.world_size,
                self.param_dtype,
                self.reduce_dtype,
            )

        x = model.ln_f(x)
        logits = model.lm_head(x)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            out["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return out

    def resident_bytes(
        self,
        optimizer_state: Sequence[torch.Tensor] = (),
        constants: Sequence[torch.Tensor] = (),
    ) -> dict[str, int]:
        """Bytes of parameters, gradients and optimizer state on this rank."""
        params = list(self.shards) + self.root_params
        grads = [p.grad for p in params if p.grad is not None]
        out = {
            "params": state_bytes(params),
            "grads": state_bytes(grads),
            "optimizer": state_bytes(optimizer_state),
            "decay_mask": state_bytes(constants),
        }
        out["total"] = sum(out.values())
        return out


class Zero3AdamW:
    """AdamW for :class:`Zero3Model`: shard-local for units, replicated for root.

    Unit shards need no gradient communication at all -- ``_GatherRunFree``
    already reduce-scattered them, so rank ``r``'s shard gradient is final.
    The replicated root parameters still need the ordinary DDP all-reduce.

    Clipping is the one place the two kinds of parameter have to be handled
    differently. The unit shards are disjoint across ranks, so their squared
    norms must be summed by a collective; the root gradients have already been
    all-reduced, so they are identical on every rank and adding them before that
    collective would count them ``p`` times. They are therefore added after it.

    Args:
        model: The sharded model.
        lr, betas, eps, weight_decay: AdamW hyper-parameters.
        grad_clip: Global gradient-norm clip, or ``None``.
        reduce_dtype: Dtype the *root* gradient all-reduce carries. The unit
            gradients were already reduced inside the backward pass, in the
            dtype the model was constructed with.
    """

    def __init__(
        self,
        model: Zero3Model,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_clip: float | None = None,
        reduce_dtype: str | torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.reduce_dtype = reduce_dtype_of(reduce_dtype)
        self.last_grad_norm: float = float("nan")
        self.step_count = 0
        self.state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.decay: dict[int, torch.Tensor] = {}
        for i, shard in enumerate(model.shards):
            self.state[id(shard)] = (torch.zeros_like(shard), torch.zeros_like(shard))
            self.decay[id(shard)] = model.unit_decay_mask(i, weight_decay)
        for p in model.root_params:
            self.state[id(p)] = (torch.zeros_like(p), torch.zeros_like(p))

    def zero_grad(self) -> None:
        for p in list(self.model.shards) + self.model.root_params:
            p.grad = None

    def optimizer_tensors(self) -> list[torch.Tensor]:
        """The Adam moments. The decay masks are reported separately."""
        return [t for pair in self.state.values() for t in pair]

    def constant_tensors(self) -> list[torch.Tensor]:
        """Per-element weight-decay coefficients; see ShardedAdamW.resident_bytes."""
        return list(self.decay.values())

    def step(self) -> None:
        self.step_count += 1
        # Root parameters are replicated, so their gradients are per-rank
        # partials and need the DDP all-reduce.
        average_gradients(
            self.model.root_params,
            self.model.world_size,
            bucket_bytes=0,
            reduce_dtype=self.reduce_dtype,
        )

        coefficient = None
        if self.grad_clip is not None:
            device = self.model.shards[0].device
            shard_sumsq = torch.zeros((), device=device)
            for shard in self.model.shards:
                if shard.grad is not None:
                    shard_sumsq = shard_sumsq + shard.grad.pow(2).sum()
            root_sumsq = sum(
                (
                    p.grad.pow(2).sum()
                    for p in self.model.root_params
                    if p.grad is not None
                ),
                torch.zeros((), device=device),
            )
            total_norm = global_grad_norm(shard_sumsq, root_sumsq, reduce=True)
            self.last_grad_norm = float(total_norm)
            coefficient = clip_coefficient(total_norm, self.grad_clip)

        with torch.no_grad():
            if coefficient is not None:
                for p in list(self.model.shards) + self.model.root_params:
                    if p.grad is not None:
                        p.grad.mul_(coefficient)
            for shard in self.model.shards:
                if shard.grad is None:
                    continue
                m, v = self.state[id(shard)]
                adamw_update_(
                    shard.data,
                    shard.grad,
                    m,
                    v,
                    step=self.step_count,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    weight_decay=self.decay[id(shard)],
                )
            for p in self.model.root_params:
                if p.grad is None:
                    continue
                m, v = self.state[id(p)]
                adamw_update_(
                    p.data,
                    p.grad,
                    m,
                    v,
                    step=self.step_count,
                    lr=self.lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                    weight_decay=self.weight_decay if p.dim() >= 2 else 0.0,
                )

    def gathered_block_params(self, unit: int) -> torch.Tensor:
        """All-gather one unit's flat parameter, for comparison against a reference."""
        shard = self.model.shards[unit]
        full = torch.empty(
            shard.numel() * self.model.world_size, dtype=shard.dtype, device=shard.device
        )
        comms.all_gather_into(full, shard.detach())
        return full


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #


def _reference_adamw(model: nn.Module, lr: float, betas, eps: float, weight_decay: float):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


def zero_equivalence_worker(
    rank: int,
    world_size: int,
    stage: int = 2,
    steps: int = 4,
    batch: int = 8,
    seq: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    grad_clip: float | None = None,
    reduce_dtype: str | None = None,
    param_dtype: str | None = None,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with :class:`ShardedAdamW` and against single-process AdamW.

    The reference is ``torch.optim.AdamW`` on the full batch in this same
    process. The assertion that matters is on the *parameters after the step*,
    for several steps: a sharded optimizer that got the moments wrong still
    produces a plausible first step and diverges by the third.

    Args:
        rank / world_size: Position in the group.
        stage: 1 or 2.
        steps: Optimiser steps to compare.
        batch / seq: Global batch shape.
        lr / weight_decay: AdamW hyper-parameters, shared with the reference.
        grad_clip: When set, both the sharded and the reference path clip to
            this global norm, and the two norms are compared as well as the
            parameters.
        reduce_dtype / param_dtype: What the two collectives carry. ``None`` is
            fp32 on both, which is the exact-equivalence configuration.
        config_kwargs: Overrides for the tested model.

    Returns:
        Per-step max absolute parameter error, the gradient norms both paths
        computed, and the measured resident bytes for the sharded and
        replicated paths.
    """
    config = parallel_config(**(config_kwargs or {}))
    inputs, targets = identical_batch(config, batch, seq)
    my_inputs = local_shard(inputs, rank, world_size)
    my_targets = local_shard(targets, rank, world_size)

    model = identical_model(config, seed=0)
    ref = identical_model(config, seed=0)
    betas = (0.9, 0.95)
    opt = ShardedAdamW(
        list(model.parameters()),
        rank=rank,
        world_size=world_size,
        stage=stage,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        reduce_dtype=reduce_dtype,
        param_dtype=param_dtype,
    )
    ref_opt = _reference_adamw(ref, lr, betas, 1e-8, weight_decay)

    # A replicated baseline built the same way, purely so the memory comparison
    # is measured against something real rather than against the formula.
    baseline = identical_model(config, seed=0)
    baseline_opt = _reference_adamw(baseline, lr, betas, 1e-8, weight_decay)

    errors: list[float] = []
    norms: list[dict[str, float]] = []
    for _ in range(steps):
        opt.zero_grad()
        model(my_inputs, targets=my_targets)["loss"].backward()
        opt.step()
        comms.get_counter().steps += 1

        ref_opt.zero_grad(set_to_none=True)
        ref(inputs, targets=targets)["loss"].backward()
        # The reference clips with torch's own utility, so the comparison is
        # against the thing everyone actually calls rather than against a second
        # hand-written norm.
        if grad_clip is not None:
            ref_norm = float(torch.nn.utils.clip_grad_norm_(ref.parameters(), grad_clip))
            norms.append({"sharded": opt.last_grad_norm, "reference": ref_norm})
        ref_opt.step()

        # The replicated baseline exists only so the memory comparison is
        # against something measured. Its all-reduces are counted in their own
        # scope so they do not appear in ZeRO's per-step communication volume.
        with comms.counter_scope(world_size):
            baseline_opt.zero_grad(set_to_none=True)
            baseline(local_shard(inputs, rank, world_size), targets=my_targets)[
                "loss"
            ].backward()
            average_gradients(baseline.parameters(), world_size, bucket_bytes=0)
            baseline_opt.step()

        errors.append(
            max(
                float((a - b).abs().max())
                for a, b in zip(model.parameters(), ref.parameters(), strict=True)
            )
        )

    baseline_state = [
        t
        for st in baseline_opt.state.values()
        for t in (st.get("exp_avg"), st.get("exp_avg_sq"))
        if t is not None
    ]
    baseline_bytes = {
        "params": state_bytes(baseline),
        "grads": state_bytes([p.grad for p in baseline.parameters() if p.grad is not None]),
        "optimizer": state_bytes(baseline_state),
        "decay_mask": 0,
    }
    baseline_bytes["total"] = sum(baseline_bytes.values())

    return {
        "rank": rank,
        "stage": stage,
        "grad_clip": grad_clip,
        "reduce_dtype": reduce_dtype or "fp32",
        "param_dtype": param_dtype or "fp32",
        "max_param_error": max(errors),
        "param_errors": errors,
        "grad_norms": norms,
        "max_grad_norm_error": (
            max(abs(n["sharded"] - n["reference"]) for n in norms) if norms else None
        ),
        "n_params": sum(p.numel() for p in model.parameters()),
        # The scale the errors above should be read against. The largest
        # parameter in this model is a LayerNorm gain at 1.0, and bf16's grid
        # spacing there is 2^-8, which is what caps the param_dtype column.
        "reference_param_scale": max(float(p.detach().abs().max()) for p in ref.parameters()),
        "shard_numel": opt.shard_size,
        "sharded_bytes": opt.resident_bytes(),
        "replicated_bytes": baseline_bytes,
    }


def zero3_equivalence_worker(
    rank: int,
    world_size: int,
    steps: int = 3,
    batch: int = 8,
    seq: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    grad_clip: float | None = None,
    reduce_dtype: str | None = None,
    param_dtype: str | None = None,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a :class:`Zero3Model` and compare with single-process AdamW.

    Comparing parameters is not straightforward here, because after the step
    this rank holds only a shard of each block. The comparison therefore
    all-gathers each unit's flat parameter and compares it against the same
    block's parameters flattened from the reference model -- so the check covers
    the parts of the model this rank does *not* own, which is where a wrong
    shard boundary would hide.
    """
    config = parallel_config(**(config_kwargs or {}))
    inputs, targets = identical_batch(config, batch, seq)
    my_inputs = local_shard(inputs, rank, world_size)
    my_targets = local_shard(targets, rank, world_size)

    betas = (0.9, 0.95)
    model = Zero3Model(
        identical_model(config, seed=0),
        rank=rank,
        world_size=world_size,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    opt = Zero3AdamW(
        model,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        reduce_dtype=reduce_dtype,
    )
    ref = identical_model(config, seed=0)
    ref_opt = _reference_adamw(ref, lr, betas, 1e-8, weight_decay)

    errors: list[float] = []
    losses: list[float] = []
    norms: list[dict[str, float]] = []
    for _ in range(steps):
        opt.zero_grad()
        out = model(my_inputs, targets=my_targets)
        out["loss"].backward()
        opt.step()
        comms.get_counter().steps += 1
        losses.append(float(out["loss"]))

        ref_opt.zero_grad(set_to_none=True)
        ref(inputs, targets=targets)["loss"].backward()
        if grad_clip is not None:
            norms.append(
                {
                    "sharded": opt.last_grad_norm,
                    "reference": float(
                        torch.nn.utils.clip_grad_norm_(ref.parameters(), grad_clip)
                    ),
                }
            )
        ref_opt.step()

        worst = 0.0
        # The verification all-gathers are counted in their own scope: they are
        # part of the test, not part of the step, and folding them into the
        # reported per-step volume would overstate ZeRO-3's traffic by 50%.
        with comms.counter_scope(world_size):
            gathered = [opt.gathered_block_params(unit) for unit in range(len(model.shards))]
        for unit, block in enumerate(ref.h):
            got = gathered[unit]
            want = torch.cat([p.detach().reshape(-1) for p in block.parameters()])
            worst = max(worst, float((got[: want.numel()] - want).abs().max()))
        for got_p, want_p in zip(
            model.root_params,
            [p for n, p in ref.named_parameters() if not n.startswith("h.")],
            strict=True,
        ):
            worst = max(worst, float((got_p.detach() - want_p.detach()).abs().max()))
        errors.append(worst)

    resident = model.resident_bytes(opt.optimizer_tensors(), opt.constant_tensors())
    return {
        "rank": rank,
        "grad_clip": grad_clip,
        "reduce_dtype": reduce_dtype or "fp32",
        "param_dtype": param_dtype or "fp32",
        "max_param_error": max(errors),
        "param_errors": errors,
        "grad_norms": norms,
        "max_grad_norm_error": (
            max(abs(n["sharded"] - n["reference"]) for n in norms) if norms else None
        ),
        "losses": losses,
        "n_params": sum(p.numel() for p in ref.parameters()),
        "reference_param_scale": max(float(p.detach().abs().max()) for p in ref.parameters()),
        "n_block_params": sum(p.numel() for p in ref.h.parameters()),
        "sharded_bytes": resident,
        "n_units": len(model.shards),
        "shard_numel": [int(s.numel()) for s in model.shards],
    }
