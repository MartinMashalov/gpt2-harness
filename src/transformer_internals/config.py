"""Model and training configuration.

Two dataclasses, both frozen-by-convention: ``GPTConfig`` describes the network
and ``TrainConfig`` describes the optimisation. They are deliberately separate --
an ablation changes one architectural field and nothing else, and keeping the
optimiser settings in a different object makes it impossible to accidentally
confound "did pre-LN help?" with "did pre-LN get a different learning rate?".

Every architectural switch that the ablation grid toggles lives here as an
explicit field with a documented default that reproduces real GPT-2. The default
``GPTConfig()`` *is* GPT-2 124M; if you construct one with no arguments and load
the OpenAI weights into it, the verification suite passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = ["GPT2_SIZES", "GPTConfig", "TrainConfig", "gpt2_config"]


@dataclass
class GPTConfig:
    """Architecture of a GPT-2 style decoder-only transformer.

    The defaults are GPT-2 124M ("small") exactly as OpenAI released it. The
    non-default options exist so that Part 6 of this repository can ablate one
    design decision at a time; they are never used on the verification path.

    Attributes:
        vocab_size: Size of the byte-level BPE vocabulary. GPT-2 uses 50257 =
            256 byte tokens + 50000 merges + 1 ``<|endoftext|>``.
        n_positions: Maximum context length. Learned position embeddings mean
            this is a hard limit, not a soft one -- see ``pos_embedding``.
        n_layer: Number of transformer blocks.
        n_head: Number of query heads. Must divide ``n_embd``.
        n_kv_head: Number of key/value heads. ``None`` means ``n_head``, i.e.
            ordinary multi-head attention, which is what GPT-2 is. Setting it to
            1 gives multi-query attention (MQA); setting it to a divisor of
            ``n_head`` gives grouped-query attention (GQA). This changes only the
            k/v projections, so it shrinks the KV cache by exactly
            ``n_head / n_kv_head`` -- the reason every modern serving model uses
            it. It is incompatible with the published GPT-2 checkpoint, so the
            verification path always leaves it at ``None``.
        n_embd: Residual stream width.
        dropout: Applied to attention probabilities, to the output of each
            residual projection, and to the embedding sum. GPT-2 was trained
            with 0.1; inference and our short ablations use 0.0 so that runs are
            deterministic and comparable.
        layer_norm_epsilon: Matches the OpenAI checkpoint. A different epsilon
            shifts logits by ~1e-5, which is enough to break token-exact
            generation matching over hundreds of tokens.
        norm_position: ``"pre"`` puts LayerNorm on the residual *branch* (GPT-2,
            and every modern transformer); ``"post"`` puts it on the residual
            *stream* after the add (the original 2017 Transformer). This is the
            single most consequential switch in the file -- see the ablations.
        pos_embedding: ``"learned"`` (GPT-2) or ``"sinusoidal"`` (Vaswani et al.).
        tie_weights: Share the token embedding matrix with the output head. GPT-2
            does this; it removes ``vocab_size * n_embd`` = 38.6M parameters from
            the 124M model, i.e. 31% of it.
        activation: ``"gelu_tanh"`` is the tanh approximation OpenAI actually
            shipped -- *not* the exact erf GELU. Using the exact one changes the
            logits at the 1e-3 level, which is visible in the verification table.
        residual_scaled_init: Scale the init of the two projections that write
            into the residual stream by ``1/sqrt(2 * n_layer)``, so that the
            variance of the stream stays O(1) as depth grows.
        attn_bias / mlp_bias: GPT-2's Conv1D layers all carry biases.
        use_sdpa: Use PyTorch's fused ``scaled_dot_product_attention`` instead of
            the explicit einsum path. The explicit path is the reference and is
            what the tests check; this flag exists only for the speed arm, and
            ``tests/test_attention.py`` asserts the two agree numerically.
    """

    vocab_size: int = 50257
    n_positions: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None
    n_embd: int = 768
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5

    # --- ablation switches (defaults reproduce GPT-2) ---
    norm_position: Literal["pre", "post"] = "pre"
    pos_embedding: Literal["learned", "sinusoidal"] = "learned"
    tie_weights: bool = True
    activation: Literal["gelu_tanh", "gelu_exact", "relu"] = "gelu_tanh"
    residual_scaled_init: bool = True
    attn_bias: bool = True
    mlp_bias: bool = True
    use_sdpa: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}"
            )
        if self.n_kv_head is not None and (
            self.n_kv_head < 1 or self.n_head % self.n_kv_head != 0
        ):
            raise ValueError(
                f"n_head={self.n_head} must be divisible by n_kv_head={self.n_kv_head}"
            )
        if self.norm_position not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {self.norm_position!r}")
        if self.pos_embedding not in ("learned", "sinusoidal"):
            raise ValueError(f"unknown pos_embedding {self.pos_embedding!r}")
        if self.activation not in ("gelu_tanh", "gelu_exact", "relu"):
            raise ValueError(f"unknown activation {self.activation!r}")

    @property
    def head_dim(self) -> int:
        """Width of a single attention head."""
        return self.n_embd // self.n_head

    @property
    def kv_heads(self) -> int:
        """Number of key/value heads actually used (``n_head`` for plain MHA)."""
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        """KV-cache bytes for one token of one sequence, across all layers.

        ``2 (k and v) * n_layer * kv_heads * head_dim * dtype_bytes``. This is the
        number that decides how many concurrent sequences a serving box can hold,
        and it is linear in context length -- which is why long-context serving is
        a memory problem before it is a compute problem.
        """
        return 2 * self.n_layer * self.kv_heads * self.head_dim * dtype_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The four released GPT-2 sizes. Only ``gpt2`` (124M) is verified here, but the
#: loader is written against the general shape so the others load unchanged.
GPT2_SIZES: dict[str, dict[str, int]] = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}


def gpt2_config(size: str = "gpt2", **overrides: Any) -> GPTConfig:
    """Build the exact config of a released GPT-2 checkpoint.

    Args:
        size: One of the keys of :data:`GPT2_SIZES`.
        **overrides: Any :class:`GPTConfig` field, applied after the size preset.
            Used mainly to set ``dropout=0.0`` for deterministic verification.

    Returns:
        A :class:`GPTConfig` whose parameter shapes match the published weights.
    """
    if size not in GPT2_SIZES:
        raise ValueError(f"unknown GPT-2 size {size!r}; expected one of {sorted(GPT2_SIZES)}")
    return GPTConfig(**{**GPT2_SIZES[size], **overrides})


@dataclass
class TrainConfig:
    """Optimisation settings for the training loop.

    The defaults are the small-scale settings used by the ablation grid, not
    GPT-2's own (OpenAI trained with batch 512 x 1024 tokens for 300k steps).
    They are chosen so a single arm finishes in well under a minute on an M-series
    laptop while still showing a clean, monotone loss curve.

    Attributes:
        steps: Optimiser steps (not micro-batches).
        batch_size: Sequences per micro-batch.
        grad_accum: Micro-batches per optimiser step. Effective batch is the
            product; this exists so the same token budget can be hit on a machine
            that cannot hold the full batch.
        block_size: Sequence length in tokens.
        lr: Peak learning rate, reached at the end of warmup.
        min_lr_ratio: Cosine decays to ``lr * min_lr_ratio``, not to zero -- GPT-3
            used 10%, and decaying fully to zero wastes the tail of the schedule.
        warmup_steps: Linear warmup. Without it the first few updates on a
            freshly-initialised transformer are large enough to move LayerNorm
            statistics into a regime it takes hundreds of steps to recover from.
        weight_decay: Applied to matmul weights only, never to biases or
            LayerNorm gains -- decaying a LayerNorm gain toward zero is decaying
            the network toward the identity, which is not regularisation.
        grad_clip: Global L2 norm clip.
        betas / eps: AdamW moments.
        seed: Seeds Python, NumPy and torch, and controls the batch sampler.
        eval_interval / eval_batches: Held-out loss cadence and averaging.
        amp: Run the forward pass under autocast. The parameters and the
            optimiser state stay in fp32 whatever this is set to; see
            :mod:`transformer_internals.precision` for why.
        amp_dtype: ``"bf16"`` or ``"fp16"``. bf16 is the default and needs no
            loss scaling; fp16 gets a ``GradScaler`` and is only reachable on
            CUDA. Refused rather than silently downgraded where the device
            cannot do it.

    There is deliberately no ``reduce_dtype`` here. The dtype a gradient
    reduction carries is a real and separate knob, but :func:`train` is a
    single-process loop with no reduction in it, so a field here would be a
    setting that does nothing. It lives where the reductions are:
    :func:`~transformer_internals.parallel.data_parallel.average_gradients`,
    :class:`~transformer_internals.parallel.zero.ShardedAdamW` and
    :class:`~transformer_internals.parallel.zero.Zero3Model` all take it, and
    ``results/parallel_comms.json`` measures what choosing bf16 costs.
    """

    steps: int = 600
    batch_size: int = 16
    grad_accum: int = 1
    block_size: int = 128
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 60
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    seed: int = 0
    eval_interval: int = 50
    eval_batches: int = 20
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    log_every: int = 50
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
