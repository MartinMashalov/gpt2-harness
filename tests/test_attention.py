"""Attention: the causal mask, the reference loop, and the fused/GQA variants."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT, CausalSelfAttention, KVCache


def naive_causal_attention(
    x: torch.Tensor, attn: CausalSelfAttention
) -> torch.Tensor:
    """A deliberately slow, obviously-correct reference written with Python loops.

    No batching tricks, no einsum, no broadcasting cleverness: for every batch
    element, every head and every query position, sum over the allowed key
    positions explicitly. If the vectorised implementation disagrees with this,
    the vectorised one is wrong.
    """
    B, T, C = x.shape
    nh, hd = attn.n_head, attn.head_dim
    qkv = attn.c_attn(x)
    q, k, v = qkv.split([attn.n_embd, attn.kv_dim, attn.kv_dim], dim=2)
    q = q.view(B, T, nh, hd)
    k = k.view(B, T, attn.n_kv_head, hd)
    v = v.view(B, T, attn.n_kv_head, hd)

    out = torch.zeros(B, T, nh, hd, dtype=x.dtype)
    groups = nh // attn.n_kv_head
    for b in range(B):
        for h in range(nh):
            kvh = h // groups
            for i in range(T):
                scores = []
                for j in range(i + 1):  # causal: j <= i only
                    scores.append(torch.dot(q[b, i, h], k[b, j, kvh]) / math.sqrt(hd))
                w = torch.softmax(torch.stack(scores), dim=0)
                for j in range(i + 1):
                    out[b, i, h] += w[j] * v[b, j, kvh]
    return attn.c_proj(out.reshape(B, T, C))


def test_attention_matches_naive_reference(tiny_config: GPTConfig) -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config, layer_idx=0).eval()
    x = torch.randn(2, 7, tiny_config.n_embd)
    with torch.no_grad():
        fast, _ = attn(x)
        slow = naive_causal_attention(x, attn)
    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()


def test_no_position_attends_to_the_future(tiny_config: GPTConfig) -> None:
    """The property the mask exists for, asserted directly on the probabilities."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config, layer_idx=0).eval()
    T = 9
    x = torch.randn(2, T, tiny_config.n_embd)
    with torch.no_grad():
        _, w = attn(x, need_weights=True)
    assert w is not None
    # Strictly-upper-triangular entries must be exactly zero, not merely small.
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(w[:, :, future] == 0.0)
    # And every row must still be a probability distribution.
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)


def test_future_tokens_cannot_change_earlier_logits(tiny_config: GPTConfig) -> None:
    """A causal model's prediction at position i must not depend on tokens > i.

    This is the behavioural version of the mask test: it would catch a leak
    anywhere in the stack, not just inside the attention module.
    """
    torch.manual_seed(0)
    model = GPT(tiny_config).eval()
    a = torch.randint(0, tiny_config.vocab_size, (1, 12))
    b = a.clone()
    b[0, 7:] = torch.randint(0, tiny_config.vocab_size, (5,))
    with torch.no_grad():
        la = model(a)["logits"]
        lb = model(b)["logits"]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-6)
    assert not torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-4)


def test_sdpa_arm_matches_reference_path(tiny_config: GPTConfig) -> None:
    """The optimisation arm must compute the same function as the reference."""
    torch.manual_seed(0)
    ref_cfg = tiny_config
    fast_cfg = GPTConfig(**{**tiny_config.to_dict(), "use_sdpa": True})
    ref = GPT(ref_cfg).eval()
    fast = GPT(fast_cfg).eval()
    fast.load_state_dict(ref.state_dict())
    x = torch.randint(0, tiny_config.vocab_size, (2, 10))
    with torch.no_grad():
        a = ref(x)["logits"]
        b = fast(x)["logits"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_shapes_and_cache_size(n_kv_head: int) -> None:
    """GQA must shrink the cache by exactly n_head / n_kv_head and still run."""
    cfg = GPTConfig(
        vocab_size=53, n_positions=16, n_layer=2, n_head=4, n_kv_head=n_kv_head, n_embd=32
    )
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    cache = KVCache.empty(cfg.n_layer)
    with torch.no_grad():
        out = model(x, cache=cache)
    assert out["logits"].shape == (2, 6, cfg.vocab_size)
    assert cache.keys[0].shape == (2, n_kv_head, 6, cfg.head_dim)

    mha = GPTConfig(**{**cfg.to_dict(), "n_kv_head": None})
    assert mha.kv_cache_bytes_per_token() / cfg.kv_cache_bytes_per_token() == (
        cfg.n_head / n_kv_head
    )


def test_masked_softmax_rows_are_normalised_at_position_zero(tiny_config: GPTConfig) -> None:
    """Position 0 attends only to itself, so its attention must be exactly 1."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config, layer_idx=0).eval()
    with torch.no_grad():
        _, w = attn(torch.randn(1, 5, tiny_config.n_embd), need_weights=True)
    assert torch.allclose(w[0, :, 0, 0], torch.ones(tiny_config.n_head), atol=1e-6)
    assert torch.all(w[0, :, 0, 1:] == 0.0)


def test_gelu_is_the_tanh_approximation() -> None:
    """GPT-2 shipped the tanh approximation; it is not the exact erf GELU."""
    from transformer_internals.model import gelu_tanh

    x = torch.linspace(-4, 4, 400)
    exact = F.gelu(x)
    ours = gelu_tanh(x)
    assert torch.allclose(ours, F.gelu(x, approximate="tanh"), atol=1e-6)
    # And it is measurably different from the exact form -- if this ever stops
    # being true, the distinction the model docstring makes has evaporated.
    assert (ours - exact).abs().max() > 1e-4
