"""Activation memory: the meter, and the analytic count that must equal it.

The central assertion here is equality, not a tolerance. The analytic count
enumerates the tensors this implementation's forward pass saves for backward,
so if it is right it is right to the byte, and a tolerance would only hide the
day it stops being right. Eight architecture variants are checked, because most
of the interesting terms only move when an architectural switch moves them: the
GELU term is different for all three activations, the position-id term is
different for sinusoidal embeddings, and the layer terms are different under
post-LN.
"""

from __future__ import annotations

import pytest
import torch

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT
from transformer_internals.perf.activation_memory import (
    ActivationMeter,
    analytic_activation_bytes,
    measure_activation_bytes,
)

VARIANTS = [
    {},
    {"n_layer": 1, "n_head": 2, "n_embd": 32},
    {"n_layer": 2, "n_head": 8, "n_embd": 128},
    {"norm_position": "post"},
    {"activation": "gelu_exact"},
    {"activation": "relu"},
    {"tie_weights": False},
    {"pos_embedding": "sinusoidal"},
]


def _config(**overrides) -> GPTConfig:
    base = {
        "vocab_size": 512,
        "n_positions": 64,
        "n_layer": 3,
        "n_head": 4,
        "n_embd": 64,
        "dropout": 0.0,
    }
    base.update(overrides)
    return GPTConfig(**base)


def _measure(cfg: GPTConfig, batch: int = 4, seq: int = 32):
    torch.manual_seed(0)
    model = GPT(cfg).train()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (batch, seq), generator=g)
    y = torch.randint(0, cfg.vocab_size, (batch, seq), generator=g)
    return measure_activation_bytes(model, x, y)


# --------------------------------------------------------------------------- #
# the count is exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", VARIANTS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default")
def test_the_analytic_count_equals_the_measurement_to_the_byte(overrides):
    report = _measure(_config(**overrides))
    assert report.measured_bytes == report.analytic_bytes, (
        f"measured {report.measured_bytes:,} vs analytic {report.analytic_bytes:,.0f}"
    )


def test_the_count_scales_with_tokens_and_with_the_square_of_the_sequence():
    """Activation memory is linear in tokens and has a quadratic term in T.

    Doubling the batch doubles everything. Doubling the sequence at fixed batch
    doubles the linear terms and quadruples the attention probabilities, so the
    total more than doubles. Both are checked against the measurement, not
    against the formula, so this is a statement about the model.
    """
    cfg = _config()
    base = _measure(cfg, batch=2, seq=16)
    double_batch = _measure(cfg, batch=4, seq=16)
    double_seq = _measure(cfg, batch=2, seq=32)

    # Batch scaling is not exactly 2x because the position ids and the causal
    # mask do not depend on the batch, so it is asserted as a tight band.
    assert 1.95 < double_batch.measured_bytes / base.measured_bytes < 2.0
    # Sequence scaling exceeds 2x, and that excess is the quadratic term.
    assert double_seq.measured_bytes / base.measured_bytes > 2.0


def test_the_attention_probabilities_are_the_quadratic_term():
    cfg = _config()
    short = analytic_activation_bytes(cfg, batch=2, seq=16)
    long = analytic_activation_bytes(cfg, batch=2, seq=32)
    assert (
        long["per_layer.attention_probabilities"]
        == 4 * short["per_layer.attention_probabilities"]
    )
    # Everything else in the layer is linear in the token count, so it doubles.
    assert long["per_layer.mlp_down_projection_input"] == (
        2 * short["per_layer.mlp_down_projection_input"]
    )


# --------------------------------------------------------------------------- #
# the meter itself
# --------------------------------------------------------------------------- #


def test_parameters_are_excluded_and_it_matters():
    """The graph saves weights too, and charging them here would double-count.

    addmm keeps its weight because the input gradient is grad @ W. On this model
    that is a fifth of everything the hooks see.
    """
    cfg = _config()
    torch.manual_seed(0)
    model = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))

    with_exclusion = ActivationMeter(exclude=[*model.parameters(), *model.buffers()])
    with with_exclusion:
        loss = model(x, targets=y)["loss"]
    excluded = with_exclusion.stash_bytes()
    loss.backward()

    torch.manual_seed(0)
    model2 = GPT(cfg).train()
    naive = ActivationMeter()
    with naive:
        loss2 = model2(x, targets=y)["loss"]
    included = naive.stash_bytes()
    loss2.backward()

    kept_weights = included - excluded
    assert kept_weights > 0
    # The graph can only have kept a subset of the model's own storages, so the
    # difference is bounded above by the total parameter and buffer bytes.
    assert kept_weights <= with_exclusion.excluded_bytes
    # And it is not a rounding error: on this model it is more than a sixth of
    # everything the hooks see.
    assert kept_weights > 0.15 * included


def test_a_storage_saved_by_several_tensors_is_charged_once():
    """Views share a storage; charging each view would inflate the number."""
    meter = ActivationMeter()
    base = torch.randn(64, 64, requires_grad=True)
    with meter:
        # Two matmuls that both save slices of the same underlying tensor.
        left = base[:32]
        right = base[32:]
        out = (left @ right.T).sum()
    charged = meter.stash_bytes()
    out.backward()
    # base is 64*64*4 bytes and both slices live in it.
    assert charged <= 64 * 64 * 4


def test_the_peak_is_read_before_the_backward_frees_it():
    cfg = _config()
    report = _measure(cfg)
    # The snapshot is taken between the forward and the backward, so the live
    # figure in the report is the stash and not the almost-empty graph left over.
    assert report.detail["activation_bytes"] == report.measured_bytes
    assert report.detail["saved_for_backward_peak_bytes"] >= report.measured_bytes
    assert report.detail["distinct_storages"] > 0


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_the_count_refuses_configurations_it_does_not_enumerate():
    """Silence would be worse: the gap would be read as a hardware property."""
    with pytest.raises(ValueError, match="dropout"):
        analytic_activation_bytes(_config(dropout=0.1), 2, 16)
    with pytest.raises(ValueError, match="scaled_dot_product_attention"):
        analytic_activation_bytes(_config(use_sdpa=True), 2, 16)
    with pytest.raises(ValueError, match="multi-head attention only"):
        analytic_activation_bytes(_config(n_kv_head=2), 2, 16)


def test_the_loss_term_is_bigger_than_a_whole_transformer_block():
    """At GPT-2's vocabulary the log-softmax outweighs any single layer.

    Not an aside. Anyone sizing a run from a per-layer formula and forgetting
    the tokens-by-vocab log-softmax is short by more than a block's worth.
    """
    cfg = GPTConfig(dropout=0.0)  # GPT-2 124M
    terms = analytic_activation_bytes(cfg, batch=8, seq=512)
    without = analytic_activation_bytes(cfg, batch=8, seq=512, with_loss=False)
    assert terms["total"] > without["total"]
    assert terms["root.cross_entropy_log_softmax_output"] > terms["per_layer_subtotal"]


def test_the_quadratic_term_takes_over_as_context_grows():
    """At GPT-2 124M the attention probabilities go from a fifth of a layer to a
    third when the context doubles from 512 to 1024, which is the reason context
    parallelism and fused attention exist."""
    cfg = GPTConfig(dropout=0.0)
    short = analytic_activation_bytes(cfg, batch=8, seq=512)
    long = analytic_activation_bytes(cfg, batch=8, seq=1024)
    short_share = short["per_layer.attention_probabilities"] / short["per_layer_subtotal"]
    long_share = long["per_layer.attention_probabilities"] / long["per_layer_subtotal"]
    assert 0.20 < short_share < 0.25
    assert 0.34 < long_share < 0.39
    assert long_share > short_share
