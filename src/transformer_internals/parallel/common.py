"""Shared pieces: deterministic model/batch construction and memory accounting.

Every equivalence proof in this package has the same shape. Build the *same*
model and the *same* data in every process from a seed, run the sharded thing,
run the single-process reference, subtract. That only works if construction is
bit-identical across processes, which is what the helpers here guarantee -- no
weights are broadcast to hide a divergence, because a broadcast would also hide
a bug in the sharding.
"""

from __future__ import annotations

import resource
from typing import Any

import torch
import torch.nn as nn

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT, Block

__all__ = [
    "identical_batch",
    "identical_block",
    "identical_model",
    "max_rss_bytes",
    "parallel_config",
    "state_bytes",
]


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
    """A GPT built from ``seed``. Bit-identical in every process."""
    torch.manual_seed(seed)
    return GPT(config).train()


def identical_block(config: GPTConfig, seed: int = 0, layer_idx: int = 0) -> Block:
    """A single transformer block built from ``seed``."""
    torch.manual_seed(seed)
    return Block(config, layer_idx=layer_idx)


def identical_batch(
    config: GPTConfig, batch: int, seq: int, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``(inputs, targets)`` token batch. Bit-identical in every process."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, config.vocab_size, (batch, seq + 1), generator=g)
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()


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
