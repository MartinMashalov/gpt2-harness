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

    Used to gate timing assertions, and only timing assertions. A one-minute
    load average above the core count means every process is waiting for a
    core, so a comparison between two measured times is a comparison of
    scheduler luck: on this laptop at a load average of 176 on ten cores, a
    step that normally takes 2 ms took 90, and the per-iteration spread on a
    25 ms collective step was 160 ms.

    Not every assertion about a duration is gated, and an earlier commit
    message (6b426d9, "one explicit gate for every wall-clock assertion in the
    suite") overstated this. The rule the suite actually follows is two tiers:

    * A loose bound with measured margin against contention, which runs
      everywhere. Six of these: the prefetch speedup and the collective fit in
      ``test_cluster.py``, and matmul's share of self time, the injected fetch
      stall, the stall fraction of a healthy loader and the diagnosis cost
      fraction in ``test_perf.py``. Every one was rerun on this laptop at load
      averages between 168 and 239 on ten cores while writing this, and every
      one passed with room: prefetch 1.89x to 2.18x against a bound of 1.3,
      the collective fit R^2 0.973 to 0.991 against 0.9.
    * A tight bound that contention can push you across, gated on this
      function. Five of these, at four sites: matmul's share against the
      runner-up and the diagnosis cost fraction above 0.8 in ``test_perf.py``,
      the paired collective-probe median in ``test_perf.py``, and the two
      bounds on the measured pipeline bubble in ``test_parallel.py``.

    A third kind used to exist and no longer does: a comparison of two measured
    durations with no margin at all. The async-checkpoint test asserted that an
    overlapped save blocked for less time than a synchronous one, which is red
    on any loaded machine and took a CI leg down. It now asserts that control
    returns before the write completes, observed rather than timed.

    A gated assertion prints the load average when it declines to measure, so a
    skipped comparison is visible in the log rather than silent. An idle CI
    runner never trips it.

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


def no_headroom_for(world_size: int) -> bool:
    """True when a timed multi-process run has no spare core to be timed on.

    ``machine_is_oversubscribed`` reads the load average, which is a one-minute
    average and therefore says nothing about the load this test is about to
    create itself. A two-core CI runner sitting idle passes that check, then
    spawns ``world_size`` workers and times a collective between them with zero
    cores left over. The two rates being compared are then a comparison of
    scheduler luck, which is what took a CI leg down: all-gather measured
    1.77 GB/s against all-reduce's 2.07 on a two-core runner at world size 2,
    reversing an ordering that holds with room to spare on any machine that has
    a spare core.

    A comparison of two measured rates needs at least one core that is not
    already committed to a worker.
    """
    cores = os.cpu_count() or 1
    if cores - world_size < 1:
        print(
            f"\n[timing] not asserting on wall clock: {cores} cores and "
            f"world size {world_size} leaves no core to absorb the measurement"
        )
        return True
    return False
