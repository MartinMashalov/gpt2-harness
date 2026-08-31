"""Autoregressive generation: KV-cached decoding, and the four sampling rules.

The sampling functions are separated from the decode loop on purpose. A logit
filter is a pure function ``(logits) -> logits`` and can be unit-tested against
hand-computed expectations; folding them into the loop is how off-by-one bugs in
top-p survive review.

On the KV cache: it changes the arithmetic but must not change the *result*.
``tests/test_generation.py`` asserts that cached and uncached greedy generation
produce identical token sequences over hundreds of steps, which is the only
convincing way to know the cache is right -- a cache bug that mis-slices the
causal mask still produces fluent English.
"""

from __future__ import annotations

import torch

from transformer_internals.model import GPT, KVCache

__all__ = [
    "apply_temperature",
    "generate",
    "top_k_filter",
    "top_p_filter",
]


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by ``temperature``.

    Temperature acts on the *logits*, before the softmax, so it rescales
    log-probabilities and therefore sharpens or flattens the distribution
    multiplicatively in probability space. ``T -> 0`` is the argmax; ``T = 1`` is
    the model's own distribution; ``T > 1`` flattens toward uniform.

    Args:
        logits: ``(..., vocab)``.
        temperature: Must be positive. ``0`` is not accepted here -- greedy is a
            separate, exactly-defined code path in :func:`generate`, and dividing
            by a tiny epsilon to fake it produces infinities in fp16.

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use do_sample=False for greedy decoding")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` highest logits per row, set the rest to ``-inf``.

    Args:
        logits: ``(B, vocab)``.
        k: Number to keep. ``k <= 0`` or ``k >= vocab`` is a no-op.

    Returns:
        Filtered logits. Renormalisation is left to the softmax, which handles it
        for free.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    # The k-th largest value per row is the threshold; strictly-smaller logits go.
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose mass exceeds ``p``.

    The subtlety is the shift. After sorting descending and taking the cumulative
    sum, ``cumsum[i] > p`` marks the first token that *completes* the nucleus --
    and that token must be kept, not dropped, or a distribution with one token at
    0.99 probability and ``p = 0.9`` would remove every candidate. Hence the
    right-shift of the removal mask, and the explicit "always keep index 0".

    Args:
        logits: ``(B, vocab)``.
        p: Cumulative probability threshold in ``(0, 1]``. ``p >= 1`` is a no-op.

    Returns:
        Filtered logits with removed entries set to ``-inf``.
    """
    if p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    remove_sorted = cumulative - probs > p
    remove_sorted[..., 0] = False  # never drop the argmax

    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a continuation.

    Args:
        model: The model. Set to ``eval()`` internally for the duration.
        idx: ``(B, T)`` prompt token ids. All rows must be the same length --
            left-padding with a proper attention mask is out of scope, and
            pretending otherwise would silently produce wrong results.
        max_new_tokens: How many tokens to append.
        do_sample: ``False`` gives exact greedy decoding (argmax), which is what
            the verification suite compares against HuggingFace token by token.
        temperature: Only used when ``do_sample``.
        top_k: Keep only the top-k logits. ``0`` disables.
        top_p: Nucleus threshold. ``1.0`` disables. Applied *after* top-k when
            both are set, matching the usual convention.
        eos_token_id: Stop early once every row has emitted this token.
        use_cache: Use the KV cache. Off is the O(T^2) reference path, kept
            because it is what the cache is tested against.
        generator: Optional RNG for reproducible sampling.

    Returns:
        ``(B, T + n)`` ids, prompt included. ``n <= max_new_tokens``: generation
        stops early on EOS, and also when the context limit is reached, since
        learned position embeddings simply do not exist beyond ``n_positions``.
    """
    was_training = model.training
    model.eval()
    try:
        cache = KVCache.empty(model.config.n_layer) if use_cache else None
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for step in range(max_new_tokens):
            if idx.size(1) >= model.config.n_positions:
                break

            if cache is not None:
                # First step encodes the whole prompt and fills the cache; every
                # later step feeds exactly one token and reads the rest from it.
                step_input = idx if step == 0 else idx[:, -1:]
            else:
                step_input = idx[:, -model.config.n_positions :]

            logits = model(step_input, cache=cache)["logits"][:, -1, :]

            if do_sample:
                logits = apply_temperature(logits, temperature)
                logits = top_k_filter(logits, top_k)
                logits = top_p_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                # A finished row keeps emitting EOS so the tensor stays
                # rectangular and its content stays meaningful.
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token.squeeze(1) == eos_token_id

            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return idx
    finally:
        model.train(was_training)
