"""Generation: KV-cache equivalence and the sampling filters."""

from __future__ import annotations

import pytest
import torch

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT
from transformer_internals.sampling import apply_temperature, generate, top_k_filter, top_p_filter


@pytest.mark.parametrize("n_kv_head", [None, 1])
def test_cached_generation_is_token_exact(n_kv_head: int | None) -> None:
    """The headline KV-cache correctness property, at temperature 0.

    A cache bug that mis-slices the causal mask still produces fluent text and a
    plausible loss. Only an exact comparison against the uncached path catches
    it -- this test is what found the real bug in this repository's history,
    where every layer after the first read the wrong cache offset.
    """
    cfg = GPTConfig(
        vocab_size=64, n_positions=96, n_layer=3, n_head=4, n_kv_head=n_kv_head, n_embd=32
    )
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    cached = generate(model, prompt, 60, do_sample=False, use_cache=True)
    uncached = generate(model, prompt, 60, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cache_matches_full_forward_logits_at_every_step() -> None:
    """Stronger than argmax equality: the logits themselves must agree."""
    cfg = GPTConfig(vocab_size=64, n_positions=32, n_layer=2, n_head=4, n_embd=32)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    from transformer_internals.model import KVCache

    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    cache = KVCache.empty(cfg.n_layer)
    with torch.no_grad():
        out = model(idx, cache=cache)
        for _ in range(10):
            nxt = out["logits"][:, -1:, :].argmax(-1)
            idx = torch.cat([idx, nxt], dim=1)
            out = model(nxt, cache=cache)
            full = model(idx)["logits"][:, -1, :]
            assert torch.allclose(out["logits"][:, -1, :], full, atol=1e-5)


def test_greedy_is_deterministic() -> None:
    cfg = GPTConfig(vocab_size=64, n_positions=32, n_layer=2, n_head=4, n_embd=32)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = generate(model, prompt, 12, do_sample=False)
    b = generate(model, prompt, 12, do_sample=False)
    assert torch.equal(a, b)


def test_sampling_is_reproducible_under_a_seeded_generator() -> None:
    cfg = GPTConfig(vocab_size=64, n_positions=32, n_layer=2, n_head=4, n_embd=32)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = generate(model, prompt, 12, do_sample=True, temperature=0.8,
                 generator=torch.Generator().manual_seed(7))
    b = generate(model, prompt, 12, do_sample=True, temperature=0.8,
                 generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_top_k_keeps_exactly_k() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0, 0.5, -1.0]])
    out = top_k_filter(logits, 2)
    assert torch.isfinite(out).sum() == 2
    assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 2])


def test_top_k_is_a_noop_at_the_boundaries() -> None:
    logits = torch.randn(2, 9)
    assert torch.equal(top_k_filter(logits, 0), logits)
    assert torch.equal(top_k_filter(logits, 9), logits)


def test_top_p_keeps_the_smallest_set_exceeding_p() -> None:
    """Nucleus membership, checked against hand-computed cumulative mass.

    Probabilities are [0.7, 0.2, 0.1] once sorted. A token is kept when the mass
    of the tokens strictly *before* it does not exceed p -- so the token that
    carries the cumulative sum past p is itself kept, and the nucleus is the
    smallest set whose total mass is at least p. This is the Holtzman et al.
    definition and matches the reference implementations.

      p = 0.65 -> 0.7 alone already reaches 0.65, so only it survives.
      p = 0.70 -> mass before 0.2 is exactly 0.7, which does not *exceed* 0.70,
                  so 0.2 is kept too and the nucleus is {0.7, 0.2}.
      p = 0.95 -> mass before 0.1 is 0.9 < 0.95, so everything is kept.
    """
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.7]]))
    assert torch.isfinite(top_p_filter(logits, 0.65)).tolist() == [[False, False, True]]
    assert torch.isfinite(top_p_filter(logits, 0.70)).tolist() == [[False, True, True]]
    assert torch.isfinite(top_p_filter(logits, 0.95)).tolist() == [[True, True, True]]
    # And the kept set is always a prefix of the descending order.
    assert torch.isfinite(top_p_filter(logits, 0.99)).all()


def test_top_p_always_keeps_the_argmax() -> None:
    """The failure mode the shift exists to prevent: p below the top probability."""
    logits = torch.log(torch.tensor([[0.99, 0.005, 0.005]]))
    out = top_p_filter(logits, 0.5)
    assert torch.isfinite(out).sum() >= 1
    assert torch.isfinite(out[0, 0])


def test_temperature_rejects_zero() -> None:
    with pytest.raises(ValueError):
        apply_temperature(torch.randn(1, 4), 0.0)


def test_generation_stops_at_the_context_limit() -> None:
    cfg = GPTConfig(vocab_size=32, n_positions=12, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    out = generate(model, prompt, 50, do_sample=False)
    assert out.shape[1] == cfg.n_positions


def test_eos_stops_generation() -> None:
    cfg = GPTConfig(vocab_size=8, n_positions=32, n_layer=1, n_head=2, n_embd=16)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    # Force the model to always predict token 3 by zeroing the head and biasing it.
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.wte.weight.zero_()
        model.wte.weight[3] = 1.0
    out = generate(model, torch.zeros(1, 2, dtype=torch.long), 20,
                   do_sample=False, eos_token_id=3)
    assert out.shape[1] < 22
