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

__all__ = [
    "ShardedAdamW",
    "Zero3Model",
    "adamw_update_",
    "decay_mask",
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
    """Concatenate into one padded flat fp32 buffer."""
    flat = torch.zeros(numel, dtype=tensors[0].dtype)
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
        [torch.full((p.numel(),), float(p.dim() >= 2)) for p in params]
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

        self.numel = sum(p.numel() for p in self.params)
        self.padded = _padded_numel(self.numel, world_size)
        self.shard_size = self.padded // world_size
        self.shard_start = rank * self.shard_size

        full_decay = (
            decay_mask(self.params) * weight_decay
            if decay_matmul_only
            else torch.full((self.numel,), weight_decay)
        )
        padded_decay = torch.zeros(self.padded)
        padded_decay[: self.numel] = full_decay
        #: Per-element weight decay for this rank's shard only.
        self.decay = padded_decay[self.shard_start : self.shard_start + self.shard_size].clone()

        # The sharded state. This is the memory ZeRO exists to save: 2 * N/p
        # elements instead of 2 * N.
        self.exp_avg = torch.zeros(self.shard_size)
        self.exp_avg_sq = torch.zeros(self.shard_size)
        #: Stage 2 keeps the reduced gradient shard here and frees ``p.grad``.
        self.grad_shard = torch.zeros(self.shard_size)

    # -- memory accounting ------------------------------------------------

    def state_bytes(self) -> int:
        """Bytes of Adam moments this rank holds -- the ``8N/p`` of the ZeRO table."""
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
        return {
            "params": param_bytes,
            "grads": grad_bytes,
            "optimizer": opt_bytes,
            "decay_mask": decay_bytes,
            "total": param_bytes + grad_bytes + opt_bytes + decay_bytes,
        }

    # -- the step ---------------------------------------------------------

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Reduce the gradients, update this rank's shard, re-gather parameters."""
        self.step_count += 1
        grads = [
            p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params
        ]
        flat_grad = _flatten(grads, self.padded)

        if self.stage == 1:
            # Every rank ends up with the whole averaged gradient and then uses
            # 1/p of it. The wasted bytes are exactly what stage 2 removes.
            comms.all_reduce(flat_grad)
            flat_grad.div_(self.world_size)
            self.grad_shard.copy_(
                flat_grad[self.shard_start : self.shard_start + self.shard_size]
            )
        else:
            comms.reduce_scatter_into(self.grad_shard, flat_grad)
            self.grad_shard.div_(self.world_size)
            # The full gradient is no longer needed anywhere on this rank.
            self.zero_grad()
        del flat_grad

        flat_param = _flatten([p.detach() for p in self.params], self.padded)
        shard = flat_param[self.shard_start : self.shard_start + self.shard_size].clone()

        adamw_update_(
            shard,
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
        # DDP's all-reduce was doing all along.
        gathered = torch.empty(self.padded)
        comms.all_gather_into(gathered, shard)

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
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        shard: torch.Tensor,
        module: nn.Module,
        meta: tuple[tuple[str, torch.Size, int], ...],
        world_size: int,
    ) -> torch.Tensor:
        ctx.module = module
        ctx.meta = meta
        ctx.world_size = world_size
        ctx.save_for_backward(x, shard)
        with torch.no_grad():
            full = torch.empty(shard.numel() * world_size, dtype=shard.dtype)
            comms.all_gather_into(full, shard)
            out = _run_unit(module, full, meta, x)
            del full
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        x, shard = ctx.saved_tensors
        world_size = ctx.world_size
        full = torch.empty(shard.numel() * world_size, dtype=shard.dtype)
        comms.all_gather_into(full, shard)
        full.requires_grad_(True)
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _run_unit(ctx.module, full, ctx.meta, x_in)
        grad_x, grad_full = torch.autograd.grad(out, [x_in, full], grad_outputs=grad_out)
        grad_shard = torch.empty_like(shard)
        comms.reduce_scatter_into(grad_shard, grad_full.contiguous())
        grad_shard.div_(world_size)
        del full, grad_full
        return grad_x, grad_shard, None, None, None


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
    """

    def __init__(self, model: nn.Module, rank: int, world_size: int) -> None:
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size

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
                p.data = torch.empty(0)
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
        full = torch.cat(
            [torch.full((numel,), float(len(shape) >= 2)) for _, shape, numel in meta]
        )
        shard_size = self.shards[unit].numel()
        padded = torch.zeros(shard_size * self.world_size)
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
                x, self.shards[unit], block, self.unit_meta[unit], self.world_size
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
    """

    def __init__(
        self,
        model: Zero3Model,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
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
        average_gradients(self.model.root_params, self.model.world_size, bucket_bytes=0)

        with torch.no_grad():
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
        full = torch.empty(shard.numel() * self.model.world_size, dtype=shard.dtype)
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
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with :class:`ShardedAdamW` and against single-process AdamW.

    The reference is ``torch.optim.AdamW`` on the full batch in this same
    process. The assertion that matters is on the *parameters after the step*,
    for several steps: a sharded optimizer that got the moments wrong still
    produces a plausible first step and diverges by the third.

    Returns:
        Per-step max absolute parameter error, and the measured resident bytes
        for the sharded and replicated paths.
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
    )
    ref_opt = _reference_adamw(ref, lr, betas, 1e-8, weight_decay)

    # A replicated baseline built the same way, purely so the memory comparison
    # is measured against something real rather than against the formula.
    baseline = identical_model(config, seed=0)
    baseline_opt = _reference_adamw(baseline, lr, betas, 1e-8, weight_decay)

    errors: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        model(my_inputs, targets=my_targets)["loss"].backward()
        opt.step()
        comms.get_counter().steps += 1

        ref_opt.zero_grad(set_to_none=True)
        ref(inputs, targets=targets)["loss"].backward()
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
        "max_param_error": max(errors),
        "param_errors": errors,
        "n_params": sum(p.numel() for p in model.parameters()),
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
    model = Zero3Model(identical_model(config, seed=0), rank=rank, world_size=world_size)
    opt = Zero3AdamW(model, lr=lr, betas=betas, weight_decay=weight_decay)
    ref = identical_model(config, seed=0)
    ref_opt = _reference_adamw(ref, lr, betas, 1e-8, weight_decay)

    errors: list[float] = []
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        out = model(my_inputs, targets=my_targets)
        out["loss"].backward()
        opt.step()
        comms.get_counter().steps += 1
        losses.append(float(out["loss"]))

        ref_opt.zero_grad(set_to_none=True)
        ref(inputs, targets=targets)["loss"].backward()
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
        "max_param_error": max(errors),
        "param_errors": errors,
        "losses": losses,
        "n_params": sum(p.numel() for p in ref.parameters()),
        "n_block_params": sum(p.numel() for p in ref.h.parameters()),
        "sharded_bytes": resident,
        "n_units": len(model.shards),
        "shard_numel": [int(s.numel()) for s in model.shards],
    }
