"""Data parallelism: replicate the model, split the batch, average the gradients.

The whole of DDP is one identity. If the loss is a mean over tokens and every
rank holds the same number of tokens, then

    mean_over_all_tokens(grad) == (1/p) * sum_over_ranks(grad_on_that_rank)

so an all-reduce with SUM followed by a division by ``p`` reproduces the
single-process gradient on the full batch *exactly*, not approximately. That is
what :func:`ddp_equivalence_worker` asserts, and it is the only thing that makes
a data-parallel run interpretable: the batch size changed, nothing else did.

The part that is engineering rather than algebra is bucketing. One all-reduce
per parameter on a 124M model is 148 collectives per step, each one paying the
latency of the interconnect on a buffer as small as a 768-element LayerNorm
bias. Real DDP coalesces gradients into flat buckets (25 MB by default) and
issues one collective per bucket, in reverse parameter order, because that is
roughly the order in which the backward pass produces them. Bucketing does not
change a single number in the result; it changes the number of collectives, and
that is visible in ``results/parallel_comms.json``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn

from transformer_internals.parallel import comms
from transformer_internals.parallel.common import identical_batch, identical_model, parallel_config
from transformer_internals.perf.activation_memory import ActivationMeter
from transformer_internals.precision import reduce_dtype_of

__all__ = [
    "DEFAULT_BUCKET_BYTES",
    "average_gradients",
    "build_buckets",
    "ddp_equivalence_worker",
    "local_shard",
]

#: torch.nn.parallel.DistributedDataParallel's default, in bytes.
DEFAULT_BUCKET_BYTES = 25 * 1024 * 1024


def local_shard(tensor: torch.Tensor, rank: int, world_size: int, dim: int = 0) -> torch.Tensor:
    """This rank's contiguous slice of ``tensor`` along ``dim``.

    Raises:
        ValueError: If the axis does not divide evenly. Uneven data sharding is
            a real problem (it makes the per-rank token counts differ, so a
            plain gradient average is no longer the full-batch gradient), and
            silently padding it here would hide that.
    """
    n = tensor.shape[dim]
    if n % world_size:
        raise ValueError(f"axis {dim} of length {n} does not split across {world_size} ranks")
    per = n // world_size
    return tensor.narrow(dim, rank * per, per).contiguous()


def build_buckets(
    params: Sequence[nn.Parameter], bucket_bytes: int = DEFAULT_BUCKET_BYTES
) -> list[list[nn.Parameter]]:
    """Group parameters into flat all-reduce buckets, in reverse order.

    Reverse order is not cosmetic: gradients are produced by the backward pass
    from the output of the network towards the input, so bucketing in reverse
    parameter order is what lets bucket ``0`` be ready (and, on real hardware,
    its all-reduce launched) while the rest of the backward pass is still
    running.

    Args:
        params: Parameters in forward-declaration order.
        bucket_bytes: Cap on a bucket's payload. A parameter larger than the cap
            gets a bucket of its own.

    Returns:
        A list of buckets, each a list of parameters.
    """
    buckets: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    size = 0
    for p in reversed(list(params)):
        nbytes = p.numel() * p.element_size()
        if current and size + nbytes > bucket_bytes:
            buckets.append(current)
            current, size = [], 0
        current.append(p)
        size += nbytes
    if current:
        buckets.append(current)
    return buckets


def average_gradients(
    params: Iterable[nn.Parameter],
    world_size: int,
    bucket_bytes: int = DEFAULT_BUCKET_BYTES,
    reduce_dtype: torch.dtype | str | None = None,
) -> int:
    """All-reduce every gradient and divide by the world size, in place.

    Parameters whose gradient is ``None`` are skipped -- a parameter that took
    no part in the loss has no gradient to average, and materialising a zero for
    it would move bytes for nothing.

    Args:
        params: The replicated parameters.
        world_size: Ranks participating.
        bucket_bytes: Bucket cap; ``0`` means one collective per parameter.
        reduce_dtype: Dtype the collective carries. ``None`` (default) keeps the
            gradient's own dtype, which is fp32 everywhere in this repository.
            Passing ``"bf16"`` casts the bucket down before the all-reduce and
            back up after it, which halves the bytes on the wire and rounds
            every rank's contribution to 8 significand bits before they are
            summed. This is the same knob as FSDP's
            ``MixedPrecision(reduce_dtype=...)``; the cost of choosing bf16 is
            measured in ``results/parallel_comms.json`` rather than assumed.

    Returns:
        The number of collectives issued.
    """
    dtype = reduce_dtype_of(reduce_dtype) if reduce_dtype is not None else None
    with_grad = [p for p in params if p.grad is not None]
    if not with_grad:
        return 0
    buckets = (
        [[p] for p in reversed(with_grad)]
        if bucket_bytes <= 0
        else build_buckets(with_grad, bucket_bytes)
    )
    for bucket in buckets:
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        # The division by the world size happens in the accumulation dtype,
        # after the cast back. Dividing inside the reduced dtype would round
        # twice for no saving on the wire.
        if dtype is not None and flat.dtype != dtype:
            wire = flat.to(dtype)
            comms.all_reduce(wire)
            flat = wire.to(flat.dtype)
        else:
            comms.all_reduce(flat)
        flat.div_(world_size)
        offset = 0
        for p in bucket:
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n
    return len(buckets)


def _grad_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in model.parameters()
        ]
    )


def ddp_equivalence_worker(
    rank: int,
    world_size: int,
    batch: int = 8,
    seq: int = 16,
    steps: int = 3,
    bucket_bytes: int = DEFAULT_BUCKET_BYTES,
    reduce_dtype: str | None = None,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run DDP for ``steps`` steps and compare against the single-process run.

    Both runs happen inside this process. The reference is a full-batch forward
    and backward with no collectives at all; the DDP run sees only this rank's
    slice of the batch and reconstructs the same gradient by all-reduce. The
    comparison is done on the flattened gradient of every parameter, and on the
    parameters themselves after an SGD step, so a sharding bug cannot hide in a
    parameter the loss happens to be insensitive to.

    Args:
        rank / world_size: Position in the group.
        batch / seq: Global batch shape; this rank sees ``batch / world_size``.
        steps: Optimiser steps to compare.
        bucket_bytes: All-reduce bucket cap; ``0`` is one collective per tensor.
        reduce_dtype: Dtype the gradient all-reduce carries. ``None`` is fp32.
        config_kwargs: Overrides for the tested model.

    Returns:
        Per-step max absolute gradient error, the loss on both paths, and the
        bucket count.
    """
    config = parallel_config(**(config_kwargs or {}))
    inputs, targets = identical_batch(config, batch, seq)

    ddp_model = identical_model(config, seed=0)
    ref_model = identical_model(config, seed=0)
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.1)

    my_inputs = local_shard(inputs, rank, world_size)
    my_targets = local_shard(targets, rank, world_size)

    grad_errors: list[float] = []
    param_errors: list[float] = []
    losses: list[float] = []
    ref_losses: list[float] = []
    buckets = 0
    local_activations: dict[str, Any] = {}
    reference_activations: dict[str, Any] = {}

    for step in range(steps):
        ddp_opt.zero_grad(set_to_none=True)
        # Measured on the first step only, because the stash is the same every
        # step and the meter is not free. It changes nothing: the pack hook
        # returns its tensor unaltered, so the graph and the collectives are
        # identical whether it is installed or not.
        if step == 0:
            meter = ActivationMeter(
                exclude=[*ddp_model.parameters(), *ddp_model.buffers()]
            )
            with meter:
                out = ddp_model(my_inputs, targets=my_targets)
            local_activations = meter.snapshot()
        else:
            out = ddp_model(my_inputs, targets=my_targets)
        out["loss"].backward()
        buckets = average_gradients(
            ddp_model.parameters(), world_size, bucket_bytes, reduce_dtype=reduce_dtype
        )
        comms.get_counter().steps += 1

        ref_opt.zero_grad(set_to_none=True)
        if step == 0:
            ref_meter = ActivationMeter(
                exclude=[*ref_model.parameters(), *ref_model.buffers()]
            )
            with ref_meter:
                ref_out = ref_model(inputs, targets=targets)
            reference_activations = ref_meter.snapshot()
        else:
            ref_out = ref_model(inputs, targets=targets)
        ref_out["loss"].backward()

        grad_errors.append(float((_grad_vector(ddp_model) - _grad_vector(ref_model)).abs().max()))
        ddp_opt.step()
        ref_opt.step()
        param_errors.append(
            max(
                float((a - b).abs().max())
                for a, b in zip(ddp_model.parameters(), ref_model.parameters(), strict=True)
            )
        )
        losses.append(float(out["loss"]))
        ref_losses.append(float(ref_out["loss"]))

    return {
        "rank": rank,
        "reduce_dtype": reduce_dtype or "fp32",
        "max_grad_error": max(grad_errors),
        "max_param_error": max(param_errors),
        "grad_errors": grad_errors,
        "local_losses": losses,
        "reference_losses": ref_losses,
        "buckets": buckets,
        "n_params": sum(p.numel() for p in ddp_model.parameters()),
        "grad_bytes": sum(p.numel() * p.element_size() for p in ddp_model.parameters()),
        # Activations are the one memory term data parallelism shrinks, and it
        # shrinks it for a reason that has nothing to do with data parallelism:
        # this rank sees 1/p of the batch. Nothing about the strategy shards an
        # activation.
        "activation_bytes": local_activations.get("activation_bytes", 0),
        "reference_activation_bytes": reference_activations.get("activation_bytes", 0),
        "activation_detail": local_activations,
    }
