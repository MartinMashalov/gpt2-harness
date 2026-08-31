"""Inference-efficiency measurement: KV cache latency, throughput and memory.

Everyone asserts that a KV cache turns O(T^2) decoding into O(T). This module
measures it, on both arms, at several context lengths, and reports the crossover
where the cache starts paying for itself -- which is not at T = 1, because the
cache costs a concatenation and a memory write per step, and at short context
that is a real fraction of the work.

The second half is the number that actually decides serving capacity: **how big
the cache gets**. For GPT-2 124M in fp32 the cache is 2 * 12 layers * 12 heads *
64 dims * 4 bytes = 73728 bytes *per token, per sequence*. At 1024 tokens that is
72 MB for a single sequence against 475 MB of fp32 weights; at batch 8 the cache
is larger than the model. This is why grouped-query attention exists, and the
reduction it buys is exact and reported here rather than estimated.

**Timing methodology.** Every measurement discards warmup iterations (MPS and
CUDA both compile kernels lazily, and the first call is not representative),
synchronises the device before stopping the clock, and reports the *median* of
several repeats rather than the mean -- a background process on a laptop
produces occasional 3x outliers, and a mean is not robust to them.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT, KVCache
from transformer_internals.sampling import generate

__all__ = [
    "BenchRow",
    "attention_variant_memory",
    "benchmark_generation",
    "kv_cache_memory",
    "model_size_bytes",
    "synchronize",
    "time_call",
]


def synchronize(device: torch.device) -> None:
    """Block until queued device work has finished.

    Both MPS and CUDA dispatch asynchronously. Without this, a timing loop
    measures how fast Python can enqueue kernels, which on a fast host is
    essentially free and produces impressively wrong numbers.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_call(fn: Any, device: torch.device, repeats: int = 5, warmup: int = 2) -> dict[str, float]:
    """Time a callable, robustly.

    Args:
        fn: Zero-argument callable to time.
        device: Device to synchronise on.
        repeats: Timed repetitions.
        warmup: Untimed repetitions first.

    Returns:
        ``median_s``, ``min_s``, ``mean_s``, ``std_s``.
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - t0)

    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "mean_s": statistics.fmean(samples),
        "std_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


@dataclass
class BenchRow:
    """One (prompt length, new tokens, cache on/off) measurement."""

    use_cache: bool
    prompt_len: int
    new_tokens: int
    batch_size: int
    median_s: float
    min_s: float
    std_s: float
    ms_per_token: float
    tokens_per_s: float
    device: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_cache": self.use_cache,
            "prompt_len": self.prompt_len,
            "new_tokens": self.new_tokens,
            "batch_size": self.batch_size,
            "median_s": self.median_s,
            "min_s": self.min_s,
            "std_s": self.std_s,
            "ms_per_token": self.ms_per_token,
            "tokens_per_s": self.tokens_per_s,
            "device": self.device,
            **self.extra,
        }


@torch.no_grad()
def benchmark_generation(
    model: GPT,
    prompt_lens: list[int],
    new_tokens: int = 64,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
    repeats: int = 3,
    warmup: int = 1,
    both_arms: bool = True,
) -> list[BenchRow]:
    """Time greedy generation with and without the KV cache.

    The uncached arm re-encodes the entire prefix at every step, so its cost at
    step ``t`` is proportional to ``(prompt + t)`` tokens of full forward work;
    summed over the generation that is quadratic in the total length. The cached
    arm encodes the prompt once and then does one token of work per step, which
    is linear. Sweeping ``prompt_lens`` makes the two curves separate visibly.

    Args:
        model: The model.
        prompt_lens: Prompt lengths to sweep.
        new_tokens: Tokens generated per measurement.
        batch_size: Sequences per measurement.
        device: Device.
        repeats: Timed repeats per configuration.
        warmup: Warmup repeats.
        both_arms: Measure the uncached arm too. It is much slower at long
            context, so it can be switched off for quick runs.

    Returns:
        One :class:`BenchRow` per configuration.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()
    rows: list[BenchRow] = []

    arms = [True, False] if both_arms else [True]
    for prompt_len in prompt_lens:
        if prompt_len + new_tokens > model.config.n_positions:
            continue
        # A fixed random prompt: content does not affect timing, but reusing the
        # same tensor across arms removes allocation noise from the comparison.
        torch.manual_seed(0)
        prompt = torch.randint(
            0, model.config.vocab_size, (batch_size, prompt_len), device=device
        )
        for use_cache in arms:
            stats = time_call(
                lambda p=prompt, uc=use_cache: generate(
                    model, p, new_tokens, do_sample=False, use_cache=uc
                ),
                device,
                repeats=repeats,
                warmup=warmup,
            )
            total_tokens = new_tokens * batch_size
            rows.append(
                BenchRow(
                    use_cache=use_cache,
                    prompt_len=prompt_len,
                    new_tokens=new_tokens,
                    batch_size=batch_size,
                    median_s=stats["median_s"],
                    min_s=stats["min_s"],
                    std_s=stats["std_s"],
                    ms_per_token=1000.0 * stats["median_s"] / new_tokens,
                    tokens_per_s=total_tokens / stats["median_s"],
                    device=str(device),
                )
            )
            gc.collect()
    return rows


def model_size_bytes(model: GPT, count_tied_once: bool = True) -> int:
    """Total bytes of the model's parameters.

    Args:
        model: The model.
        count_tied_once: Count a tied embedding/head tensor once, which is what
            it actually costs in memory. Counting it twice would flatter every
            cache-vs-weights ratio.

    Returns:
        Bytes.
    """
    # ``model.parameters()`` already de-duplicates shared tensors, so iterating
    # it can never produce the double-counted number. ``remove_duplicate=False``
    # is what actually yields the tied tensor twice.
    params = model.named_parameters(remove_duplicate=count_tied_once)
    return sum(p.numel() * p.element_size() for _, p in params)


def kv_cache_memory(
    config: GPTConfig,
    seq_lens: list[int],
    batch_sizes: list[int],
    dtype_bytes: int = 4,
    model_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Cache size as a function of context length and batch, against model size.

    The formula is exact, not an estimate:
    ``2 * n_layer * kv_heads * head_dim * dtype_bytes`` bytes per token per
    sequence -- a factor 2 for keys and values, and one entry per layer because
    every layer keeps its own.

    Args:
        config: Model config.
        seq_lens: Context lengths.
        batch_sizes: Batch sizes.
        dtype_bytes: 4 for fp32, 2 for fp16/bf16.
        model_bytes: Parameter bytes, for the ratio column. The ratio is the
            point of the table: once it exceeds 1, the accelerator is holding
            more cache than model, and the serving bottleneck has moved.

    Returns:
        One record per (seq_len, batch_size).
    """
    per_token = config.kv_cache_bytes_per_token(dtype_bytes)
    out: list[dict[str, Any]] = []
    for bs in batch_sizes:
        for t in seq_lens:
            total = per_token * t * bs
            out.append(
                {
                    "seq_len": t,
                    "batch_size": bs,
                    "dtype_bytes": dtype_bytes,
                    "bytes_per_token_per_seq": per_token,
                    "cache_bytes": total,
                    "cache_mb": total / 1e6,
                    "model_bytes": model_bytes,
                    "cache_fraction_of_model": (total / model_bytes) if model_bytes else None,
                }
            )
    return out


def attention_variant_memory(
    base: GPTConfig,
    kv_head_options: list[int | None],
    seq_len: int = 1024,
    batch_size: int = 8,
    dtype_bytes: int = 4,
    model_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Exact KV-cache savings of MQA and GQA against full multi-head attention.

    The saving is exactly ``n_head / n_kv_head``, because the cache holds one key
    and one value vector per *kv* head. What it costs in quality is a separate,
    empirical question, answered by the training arms in the results -- the two
    must be reported together or the tradeoff is invisible.

    Args:
        base: Baseline config (MHA).
        kv_head_options: ``None`` for MHA, or a kv-head count.
        seq_len: Context length for the comparison.
        batch_size: Batch size for the comparison.
        dtype_bytes: Bytes per element.
        model_bytes: Parameter bytes for the ratio column.

    Returns:
        One record per option.
    """
    rows: list[dict[str, Any]] = []
    mha_per_token = GPTConfig(**{**base.to_dict(), "n_kv_head": None}).kv_cache_bytes_per_token(
        dtype_bytes
    )
    for opt in kv_head_options:
        cfg = GPTConfig(**{**base.to_dict(), "n_kv_head": opt})
        per_token = cfg.kv_cache_bytes_per_token(dtype_bytes)
        total = per_token * seq_len * batch_size
        kv = cfg.kv_heads
        rows.append(
            {
                "n_kv_head": kv,
                "variant": (
                    "MHA" if kv == cfg.n_head else ("MQA" if kv == 1 else f"GQA-{kv}")
                ),
                "bytes_per_token_per_seq": per_token,
                "cache_mb": total / 1e6,
                "reduction_vs_mha": mha_per_token / per_token,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "cache_fraction_of_model": (total / model_bytes) if model_bytes else None,
            }
        )
    return rows


@torch.no_grad()
def measure_cache_tensor_bytes(
    model: GPT, prompt_len: int, new_tokens: int, batch_size: int = 1, device: str = "cpu"
) -> dict[str, Any]:
    """Empirically measure the cache's real memory footprint.

    The analytic formula above should be exactly right; this checks it against
    the actual allocated tensors, because a formula that has never been compared
    to reality is a guess with good posture.

    Args:
        model: The model.
        prompt_len: Prompt length.
        new_tokens: Tokens to generate.
        batch_size: Batch size.
        device: Device.

    Returns:
        Measured and predicted bytes, and their ratio.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    prompt = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=dev)
    cache = KVCache.empty(model.config.n_layer)
    idx = prompt
    out = model(idx, cache=cache)
    for _ in range(new_tokens):
        nxt = out["logits"][:, -1:, :].argmax(-1)
        idx = torch.cat([idx, nxt], dim=1)
        out = model(nxt, cache=cache)

    measured = sum(
        t.numel() * t.element_size()
        for lst in (cache.keys, cache.values)
        for t in lst
        if t is not None
    )
    total_len = prompt_len + new_tokens
    predicted = (
        model.config.kv_cache_bytes_per_token(
            next(model.parameters()).element_size()
        )
        * total_len
        * batch_size
    )
    return {
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "batch_size": batch_size,
        "total_len": total_len,
        "measured_bytes": measured,
        "predicted_bytes": predicted,
        "ratio": measured / predicted if predicted else float("nan"),
    }
