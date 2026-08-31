"""Model invariants: shapes, dtypes, init, weight tying, config validation."""

from __future__ import annotations

import math

import pytest
import torch

from transformer_internals.config import GPTConfig, gpt2_config
from transformer_internals.model import GPT, sinusoidal_position_embeddings


def test_forward_shapes_and_dtypes(tiny_model: GPT, tiny_batch: torch.Tensor) -> None:
    out = tiny_model(tiny_batch, targets=tiny_batch)
    B, T = tiny_batch.shape
    assert out["logits"].shape == (B, T, tiny_model.config.vocab_size)
    assert out["logits"].dtype == torch.float32
    assert out["loss"].shape == ()
    assert torch.isfinite(out["loss"])


def test_loss_at_init_is_near_uniform(tiny_config: GPTConfig) -> None:
    """A correctly initialised LM starts at ln(vocab_size).

    This is the cheapest possible smoke test for a catastrophic init or a
    mis-shaped output layer, and it fails loudly if the head is not roughly
    symmetric across the vocabulary.
    """
    torch.manual_seed(0)
    model = GPT(tiny_config).eval()
    x = torch.randint(0, tiny_config.vocab_size, (8, 16))
    loss = model(x, targets=x)["loss"].item()
    assert abs(loss - math.log(tiny_config.vocab_size)) < 0.5


def test_weight_tying_shares_storage(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config)
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    untied = GPT(GPTConfig(**{**tiny_config.to_dict(), "tie_weights": False}))
    assert untied.lm_head.weight.data_ptr() != untied.wte.weight.data_ptr()
    assert untied.num_parameters() > model.num_parameters()


def test_residual_scaled_init_shrinks_the_writing_projections(tiny_config: GPTConfig) -> None:
    torch.manual_seed(0)
    scaled = GPT(tiny_config)
    torch.manual_seed(0)
    plain = GPT(GPTConfig(**{**tiny_config.to_dict(), "residual_scaled_init": False}))
    expected = 1.0 / math.sqrt(2 * tiny_config.n_layer)
    ratio = scaled.h[0].attn.c_proj.weight.std() / plain.h[0].attn.c_proj.weight.std()
    assert abs(float(ratio) - expected) < 0.15
    # Only the writing projections are touched.
    assert torch.allclose(scaled.h[0].attn.c_attn.weight, plain.h[0].attn.c_attn.weight)


def test_gpt2_parameter_count() -> None:
    """The published 124M figure, reproduced from our own module shapes."""
    model = GPT(gpt2_config("gpt2", dropout=0.0))
    assert model.num_parameters() == 124_439_808


@pytest.mark.parametrize("pos", ["learned", "sinusoidal"])
def test_position_embedding_variants_run(tiny_config: GPTConfig, pos: str) -> None:
    cfg = GPTConfig(**{**tiny_config.to_dict(), "pos_embedding": pos})
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    assert model(x)["logits"].shape == (2, 9, cfg.vocab_size)
    assert (model.wpe is None) == (pos == "sinusoidal")


def test_sinusoidal_embeddings_are_bounded_and_shaped() -> None:
    pe = sinusoidal_position_embeddings(16, 8)
    assert pe.shape == (16, 8)
    assert pe.abs().max() <= 1.0 + 1e-6
    assert not torch.allclose(pe[0], pe[1])


@pytest.mark.parametrize("norm", ["pre", "post"])
def test_norm_position_variants_run(tiny_config: GPTConfig, norm: str) -> None:
    cfg = GPTConfig(**{**tiny_config.to_dict(), "norm_position": norm})
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    assert torch.isfinite(model(x, targets=x)["loss"])


def test_config_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        GPTConfig(n_embd=10, n_head=3)
    with pytest.raises(ValueError):
        GPTConfig(n_embd=32, n_head=4, n_kv_head=3)
    with pytest.raises(ValueError):
        GPTConfig(norm_position="middle")  # type: ignore[arg-type]


def test_context_limit_is_enforced(tiny_config: GPTConfig) -> None:
    model = GPT(tiny_config).eval()
    too_long = torch.zeros(1, tiny_config.n_positions + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="n_positions"):
        model(too_long)


def test_optimizer_excludes_norms_and_biases_from_decay(tiny_model: GPT) -> None:
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_taps_cover_every_sublayer(tiny_model: GPT, tiny_batch: torch.Tensor) -> None:
    taps = tiny_model(tiny_batch, collect_taps=True)["taps"]
    for i in range(tiny_model.config.n_layer):
        for sub in ("ln_1", "attn", "resid_mid", "ln_2", "mlp", "resid_out"):
            assert f"h.{i}.{sub}" in taps
    assert {"embed", "ln_f", "logits"} <= set(taps)
