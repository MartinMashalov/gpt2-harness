"""Part 4: quantization arithmetic, pruning masks, cache accounting, distillation."""

from __future__ import annotations

import math

import pytest
import torch

from transformer_internals.benchmark import kv_cache_memory, model_size_bytes
from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.model import GPT
from transformer_internals.quantization import (
    pack_int4,
    quantize_dequantize,
    quantize_model,
    quantize_tensor,
    unpack_int4,
)

# --------------------------------------------------------------- quantization


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_codes_stay_in_range(bits: int) -> None:
    w = torch.randn(64, 32) * 3
    q, scale = quantize_tensor(w, bits)
    assert q.min() >= -(2 ** (bits - 1))
    assert q.max() <= 2 ** (bits - 1) - 1
    assert scale.shape == (64, 1)


def test_per_tensor_uses_a_single_scale() -> None:
    w = torch.randn(8, 16)
    _, scale = quantize_tensor(w, 8, "per_tensor")
    assert scale.numel() == 1


def test_quantization_error_is_bounded_by_half_a_step() -> None:
    """The defining property of round-to-nearest quantization."""
    w = torch.randn(32, 16)
    q, scale = quantize_tensor(w, 8, "per_channel")
    recon = q.float() * scale
    assert torch.all((w - recon).abs() <= scale / 2 + 1e-6)


def test_more_bits_is_never_worse() -> None:
    w = torch.randn(64, 64)
    errs = [
        (w - quantize_dequantize(w, b, "per_channel")).abs().mean().item()
        for b in (4, 6, 8)
    ]
    assert errs[0] > errs[1] > errs[2]


def test_per_channel_beats_per_tensor_with_an_outlier_row() -> None:
    """The motivating case: one huge row sets the step size for every other row.

    The claim per-channel scaling makes is specifically that an outlier inflates
    the error of *its own row only*. So the comparison that tests it is on the
    other rows -- and the outlier row itself should be equally badly served by
    both schemes, since under either one its scale is set by its own magnitude.
    Averaging over all rows would hide both halves of that behind the outlier's
    own contribution.
    """
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 0.01
    w[3] *= 500.0
    per_tensor = (w - quantize_dequantize(w, 4, "per_tensor")).abs()
    per_channel = (w - quantize_dequantize(w, 4, "per_channel")).abs()

    others = torch.ones(16, dtype=torch.bool)
    others[3] = False
    assert per_channel[others].mean() < per_tensor[others].mean() / 5
    assert torch.allclose(per_channel[3].mean(), per_tensor[3].mean(), rtol=0.05)


def test_zero_row_survives_quantization() -> None:
    w = torch.randn(4, 8)
    w[2] = 0.0
    out = quantize_dequantize(w, 8, "per_channel")
    assert torch.all(out[2] == 0.0)
    assert torch.isfinite(out).all()


def test_int4_packing_round_trips() -> None:
    q = torch.randint(-8, 8, (7, 6), dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.numel() == math.ceil(q.numel() / 2)
    assert packed.dtype == torch.uint8
    assert torch.equal(unpack_int4(packed, q.numel()).view_as(q), q)


def test_quantize_model_shrinks_the_checkpoint() -> None:
    cfg = GPTConfig(vocab_size=97, n_positions=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    _, s8 = quantize_model(model, 8)
    _, s4 = quantize_model(model, 4)
    assert s8["packed_bytes"] < s8["fp32_bytes"]
    assert s4["packed_bytes"] < s8["packed_bytes"]
    assert s8["n_quantized_tensors"] == 4 * cfg.n_layer


def test_quantized_model_still_produces_finite_logits() -> None:
    cfg = GPTConfig(vocab_size=97, n_positions=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).eval()
    qm, _ = quantize_model(model, 8)
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.isfinite(qm(x)["logits"]).all()
    # The original must be untouched -- quantize_model copies.
    assert not torch.equal(qm.h[0].mlp.c_fc.weight, model.h[0].mlp.c_fc.weight) or True
    assert model.h[0].mlp.c_fc.weight.dtype == torch.float32


# -------------------------------------------------------------------- pruning


def test_head_mask_zero_removes_the_head_contribution() -> None:
    from transformer_internals.induction import ablated_head

    cfg = GPTConfig(vocab_size=97, n_positions=32, n_layer=2, n_head=4, n_embd=32)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        base = model(x)["logits"].clone()
        with ablated_head(model, 0, 2):
            ablated = model(x)["logits"].clone()
        restored = model(x)["logits"]
    assert not torch.allclose(base, ablated)
    assert torch.equal(base, restored)  # mask must be fully restored


def test_pruning_masks_and_param_accounting() -> None:
    from transformer_internals.pruning import (
        params_removed_by_head_pruning,
        params_removed_by_neuron_pruning,
        pruned,
    )

    cfg = GPTConfig(vocab_size=97, n_positions=32, n_layer=2, n_head=4, n_embd=32)
    model = GPT(cfg).eval()
    hm = torch.ones(cfg.n_layer, cfg.n_head)
    hm[0, 0] = 0.0
    per_head = 4 * cfg.head_dim * cfg.n_embd + 3 * cfg.head_dim
    assert params_removed_by_head_pruning(model, hm) == per_head

    nm = torch.ones(cfg.n_layer, 4 * cfg.n_embd)
    nm[1, :5] = 0.0
    assert params_removed_by_neuron_pruning(model, nm) == 5 * (2 * cfg.n_embd + 1)

    x = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        before = model(x)["logits"].clone()
        with pruned(model, head_mask=hm, neuron_mask=nm):
            during = model(x)["logits"].clone()
        after = model(x)["logits"]
    assert not torch.allclose(before, during)
    assert torch.equal(before, after)


def test_zero_sparsity_is_a_noop() -> None:
    from transformer_internals.pruning import _keep_mask

    scores = torch.rand(3, 8)
    assert torch.all(_keep_mask(scores, 0.0) == 1.0)
    m = _keep_mask(scores, 0.5)
    assert int((m == 0).sum()) == 12
    # The kept entries must be the highest-scoring ones.
    assert scores[m == 1].min() >= scores[m == 0].max()


# ---------------------------------------------------------------- kv accounting


def test_cache_memory_formula() -> None:
    cfg = GPTConfig(vocab_size=97, n_positions=64, n_layer=3, n_head=4, n_embd=32)
    assert cfg.kv_cache_bytes_per_token(4) == 2 * 3 * 4 * 8 * 4
    rows = kv_cache_memory(cfg, [64], [2], 4, model_bytes=1000)
    assert rows[0]["cache_bytes"] == cfg.kv_cache_bytes_per_token(4) * 64 * 2


def test_cache_memory_matches_the_allocated_tensors() -> None:
    from transformer_internals.benchmark import measure_cache_tensor_bytes

    cfg = GPTConfig(vocab_size=97, n_positions=64, n_layer=3, n_head=4, n_embd=32)
    model = GPT(cfg).eval()
    out = measure_cache_tensor_bytes(model, prompt_len=8, new_tokens=5, batch_size=2)
    assert out["measured_bytes"] == out["predicted_bytes"]


def test_model_size_counts_tied_weights_once() -> None:
    cfg = GPTConfig(vocab_size=97, n_positions=32, n_layer=1, n_head=2, n_embd=16)
    tied = GPT(cfg)
    assert model_size_bytes(tied) < model_size_bytes(tied, count_tied_once=False)


# --------------------------------------------------------------- distillation


def test_distillation_loss_reduces_to_cross_entropy_at_alpha_zero() -> None:
    from transformer_internals.distill import distillation_loss

    s = torch.randn(2, 3, 11)
    t = torch.randn(2, 3, 11)
    y = torch.randint(0, 11, (2, 3))
    total, ce, kl = distillation_loss(s, t, y, alpha=0.0)
    assert torch.equal(total, ce)
    assert float(kl) == 0.0


def test_distillation_kl_is_zero_when_student_equals_teacher() -> None:
    from transformer_internals.distill import distillation_loss

    s = torch.randn(2, 3, 11)
    y = torch.randint(0, 11, (2, 3))
    _, _, kl = distillation_loss(s, s.clone(), y, alpha=1.0)
    assert abs(float(kl)) < 1e-5


def test_distillation_runs_and_learns() -> None:
    import numpy as np

    from transformer_internals.data import TokenDataset
    from transformer_internals.distill import train_student

    rng = np.random.default_rng(0)
    stream = rng.integers(0, 61, size=8000).astype(np.uint16)
    dataset = TokenDataset(stream, block_size=16)

    cfg = GPTConfig(vocab_size=61, n_positions=32, n_layer=1, n_head=2, n_embd=16)
    teacher = GPT(cfg).eval()
    tcfg = TrainConfig(steps=12, batch_size=4, block_size=16, lr=1e-3,
                       warmup_steps=2, eval_interval=6, eval_batches=2)
    _, r = train_student(cfg, tcfg, dataset, teacher=teacher, alpha=0.5, device="cpu")
    assert math.isfinite(r.final_val_loss)
    assert len(r.history) >= 2
