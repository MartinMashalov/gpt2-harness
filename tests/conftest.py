"""Shared fixtures.

Nothing here touches the network. The weight-dependent tests are marked
``weights`` and skip cleanly when the GPT-2 checkpoint is not available, so the
whole suite runs on an offline CPU runner.
"""

from __future__ import annotations

import os

import pytest
import torch

from transformer_internals.config import GPTConfig


def machine_is_oversubscribed() -> bool:
    """True when there are more runnable processes than cores.

    Used to gate wall-clock assertions, and only wall-clock assertions. A
    one-minute load average above the core count means every process is waiting
    for a core, so a comparison between two measured times is a comparison of
    scheduler luck: on this laptop at a load average of 176 on ten cores, a
    step that normally takes 2 ms took 90, and the per-iteration spread on a
    25 ms collective step was 160 ms.

    Every structural assertion runs regardless. What is gated is the handful of
    assertions whose subject is a duration, and a gated one prints the load
    average so a skipped comparison is visible in the log rather than silent.
    An idle CI runner never trips it.

    Returns False where the load average is unavailable, so the assertions are
    on by default rather than off.
    """
    if not hasattr(os, "getloadavg"):
        return False
    try:
        one_minute = os.getloadavg()[0]
    except OSError:
        return False
    cores = os.cpu_count() or 1
    if one_minute > cores:
        print(
            f"\n[timing] not asserting on wall clock: load average "
            f"{one_minute:.1f} on {cores} cores"
        )
        return True
    return False


@pytest.fixture(scope="session")
def tiny_config() -> GPTConfig:
    """A small but structurally complete model: >1 layer, >1 head, even widths."""
    return GPTConfig(
        vocab_size=97,
        n_positions=32,
        n_layer=3,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config: GPTConfig):
    from transformer_internals.model import GPT

    torch.manual_seed(0)
    return GPT(tiny_config).eval()


@pytest.fixture
def tiny_batch(tiny_config: GPTConfig) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, tiny_config.vocab_size, (2, 11))


@pytest.fixture(scope="session")
def gpt2_available() -> bool:
    """Whether the published GPT-2 checkpoint can be loaded without the network."""
    try:
        from transformer_internals.weights import resolve_checkpoint_dir

        resolve_checkpoint_dir(local_files_only=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def gpt2_pair(gpt2_available: bool):
    """Our GPT-2 and the HuggingFace reference, both fp32 on CPU."""
    if not gpt2_available:
        pytest.skip("GPT-2 checkpoint not available offline")
    transformers = pytest.importorskip("transformers")
    from transformer_internals.tokenizer import BPETokenizer
    from transformer_internals.weights import load_pretrained_gpt2

    model, ckpt = load_pretrained_gpt2(local_files_only=True)
    ref = transformers.GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float32
    ).eval()
    tok = BPETokenizer.from_pretrained(ckpt)
    return model, ref, tok
