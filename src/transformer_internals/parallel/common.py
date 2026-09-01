"""Shared pieces: deterministic model/batch construction, device, memory accounting.

Every equivalence proof in this package has the same shape. Build the *same*
model and the *same* data in every process from a seed, run the sharded thing,
run the single-process reference, subtract. That only works if construction is
bit-identical across processes, which is what the helpers here guarantee -- no
weights are broadcast to hide a divergence, because a broadcast would also hide
a bug in the sharding.

**Device.** The same proofs run on gloo/CPU and on NCCL/CUDA, and the difference
must not be a second copy of the code. It is a single process-wide setting:
:func:`set_device` is called once per rank by the launcher in
:mod:`transformer_internals.parallel.comms`, and every allocation in this
package either takes its device from a tensor already in hand or asks
:func:`current_device`. On CPU it is ``cpu`` and nothing changes.

Construction stays on the CPU and is moved afterwards, deliberately. Building a
model directly on a CUDA device would draw from the CUDA RNG, and then "the same
seed gives the same weights" would depend on the device rather than on the seed.
Building on CPU and calling ``.to(device)`` keeps the sharded run and the
single-process reference bit-identical on any hardware.
"""

from __future__ import annotations

import resource
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT, Block

__all__ = [
    "current_device",
    "device_scope",
    "identical_batch",
    "identical_block",
    "identical_model",
    "max_rss_bytes",
    "parallel_config",
    "set_device",
    "state_bytes",
]


#: The device this rank places its tensors on. ``cpu`` unless a launcher has
#: said otherwise, which is what keeps the gloo path exactly as it was.
_DEVICE = torch.device("cpu")


def current_device() -> torch.device:
    """The device this rank is running on."""
    return _DEVICE


def set_device(device: torch.device | str | None) -> torch.device:
    """Set the process-wide device for this rank, and return it.

    Called once per rank, by the launcher, before any model is built. Passing
    ``None`` leaves the current setting alone so callers can forward an optional
    argument without a conditional.
    """
    global _DEVICE
    if device is not None:
        _DEVICE = torch.device(device)
    return _DEVICE


@contextmanager
def device_scope(device: torch.device | str | None) -> Iterator[torch.device]:
    """Run a block with the rank device temporarily set."""
    global _DEVICE
    previous = _DEVICE
    try:
        yield set_device(device)
    finally:
        _DEVICE = previous


def parallel_config(**overrides: Any) -> GPTConfig:
    """A small but structurally complete GPT-2: >1 layer, >1 head, no dropout.

    Dropout is 0 because every test in this package is an exact-equivalence
    test, and dropout would make the sharded and single-process runs draw
    different masks. That is a property of the test, not of the implementation.
    """
    base = {
        "vocab_size": 128,
        "n_positions": 64,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 32,
        "dropout": 0.0,
    }
    base.update(overrides)
    return GPTConfig(**base)


def identical_model(config: GPTConfig, seed: int = 0) -> GPT:
    """A GPT built from ``seed``, on this rank's device. Bit-identical everywhere.

    Built on the CPU and moved, so the weights depend on the seed and not on
    which device drew them.
    """
    torch.manual_seed(seed)
    return GPT(config).train().to(current_device())


def identical_block(config: GPTConfig, seed: int = 0, layer_idx: int = 0) -> Block:
    """A single transformer block built from ``seed``, on this rank's device."""
    torch.manual_seed(seed)
    return Block(config, layer_idx=layer_idx).to(current_device())


def identical_batch(
    config: GPTConfig, batch: int, seq: int, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``(inputs, targets)`` token batch on this rank's device.

    Drawn on the CPU from an explicit generator and moved, for the same reason
    :func:`identical_model` builds on the CPU: the tokens must depend on the
    seed, not on the device.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, config.vocab_size, (batch, seq + 1), generator=g)
    device = current_device()
    return (
        idx[:, :-1].contiguous().to(device),
        idx[:, 1:].contiguous().to(device),
    )


def state_bytes(tensors: Any) -> int:
    """Bytes of distinct storage held by an iterable of tensors or a module.

    Counted by storage, not by tensor, so views and the tied ``lm_head``/``wte``
    weight are charged once rather than twice -- which is the difference between
    a memory number that means something and one that double-counts weight
    tying by 38% of the model.
    """
    if isinstance(tensors, nn.Module):
        tensors = list(tensors.parameters())
    seen: dict[int, int] = {}
    for t in tensors:
        if t is None:
            continue
        storage = t.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
    return sum(seen.values())


def max_rss_bytes() -> int:
    """Peak resident set size of this process, from the OS.

    ``ru_maxrss`` is in bytes on macOS and kilobytes on Linux. This is a coarse
    process-level number that includes the interpreter and the BLAS pools, so it
    is reported alongside the exact per-tensor accounting rather than instead of
    it.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return int(raw) if sys.platform == "darwin" else int(raw) * 1024
