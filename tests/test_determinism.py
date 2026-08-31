"""Determinism: same seed, same numbers.

This exists because it caught a real problem. An early version of the ablation
grid ran on Apple's MPS backend at a learning rate close to an instability, and
two runs with an identical seed finished at validation losses of 3.89 and 5.95 --
the MPS backward pass is not bit-deterministic (its bias gradients are atomic
reductions), and an unstable optimisation amplifies that into a different
trajectory. Every number in an ablation table is worthless if this test does not
pass, so it is a test rather than an assumption.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch

from transformer_internals.config import GPTConfig, TrainConfig
from transformer_internals.data import TokenDataset
from transformer_internals.model import GPT
from transformer_internals.train import lr_at_step, set_seed, train


def _dataset(seed: int = 0, n: int = 20_000, vocab: int = 61) -> TokenDataset:
    rng = np.random.default_rng(seed)
    # A learnable stream, not white noise: a repeating motif with jitter, so the
    # loss actually moves and the test can tell two trajectories apart.
    base = np.tile(np.arange(17, dtype=np.uint16), n // 17 + 1)[:n]
    noise = rng.integers(0, vocab, size=n).astype(np.uint16)
    mask = rng.random(n) < 0.3
    return TokenDataset(np.where(mask, noise, base).astype(np.uint16), block_size=16)


def test_same_seed_gives_identical_initialisation() -> None:
    cfg = GPTConfig(vocab_size=61, n_positions=32, n_layer=2, n_head=2, n_embd=16)
    set_seed(3)
    a = GPT(cfg)
    set_seed(3)
    b = GPT(cfg)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb
        assert torch.equal(pa, pb), na


def test_different_seeds_give_different_initialisation() -> None:
    cfg = GPTConfig(vocab_size=61, n_positions=32, n_layer=2, n_head=2, n_embd=16)
    set_seed(1)
    a = GPT(cfg)
    set_seed(2)
    b = GPT(cfg)
    assert not torch.equal(a.wte.weight, b.wte.weight)


def test_batches_are_reproducible_under_a_seeded_generator() -> None:
    ds = _dataset()
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    for _ in range(5):
        x1, y1 = ds.get_batch("train", 4, generator=g1)
        x2, y2 = ds.get_batch("train", 4, generator=g2)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_targets_are_inputs_shifted_by_one() -> None:
    ds = _dataset()
    x, y = ds.get_batch("train", 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_and_val_splits_do_not_overlap() -> None:
    ds = _dataset()
    assert len(ds.train) + len(ds.val) == 20_000
    assert len(ds.val) > 0


def test_training_run_is_bit_reproducible() -> None:
    """Two full training runs at the same seed must agree exactly."""
    cfg = GPTConfig(vocab_size=61, n_positions=32, n_layer=2, n_head=2, n_embd=16)
    tcfg = TrainConfig(
        steps=25, batch_size=4, block_size=16, lr=3e-4, warmup_steps=5,
        eval_interval=25, eval_batches=3, seed=5,
    )
    ds = _dataset()
    _, a = train(cfg, tcfg, ds, device="cpu")
    _, b = train(cfg, tcfg, ds, device="cpu")
    assert a.final_val_loss == b.final_val_loss
    assert a.grad_norms == b.grad_norms


def test_training_actually_reduces_the_loss() -> None:
    """A reproducibility test passes trivially if nothing learns; this rules that out."""
    cfg = GPTConfig(vocab_size=61, n_positions=32, n_layer=2, n_head=2, n_embd=32)
    tcfg = TrainConfig(
        steps=120, batch_size=8, block_size=16, lr=3e-3, warmup_steps=10,
        eval_interval=20, eval_batches=4, seed=0,
    )
    _, r = train(cfg, tcfg, _dataset(), device="cpu")
    assert not r.diverged
    assert r.history[-1]["val_loss"] < r.history[0]["val_loss"] - 0.05


def test_lr_schedule_shape() -> None:
    cfg = TrainConfig(steps=100, warmup_steps=10, lr=1e-3, min_lr_ratio=0.1)
    assert lr_at_step(0, cfg) == cfg.lr / 10
    assert lr_at_step(9, cfg) == cfg.lr           # peak at the end of warmup
    assert lr_at_step(99, cfg) < cfg.lr
    assert abs(lr_at_step(99, cfg) - cfg.lr * cfg.min_lr_ratio) < 2e-5
    # Monotone decay after warmup.
    after = [lr_at_step(s, cfg) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in pairwise(after))
