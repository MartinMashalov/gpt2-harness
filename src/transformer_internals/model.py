"""GPT-2 in pure PyTorch: no ``transformers``, no ``nn.MultiheadAttention``.

Everything on the reference path is written out explicitly -- the q/k/v
projection, the head reshape, the scaled dot product, the causal mask, the
softmax and the output projection are all visible lines of code, because the
whole point of the repository is that you can read them and check them against
the paper. ``F.scaled_dot_product_attention`` is available behind
``GPTConfig.use_sdpa`` as a speed arm only, and ``tests/test_attention.py``
asserts that arm agrees with the reference to 1e-6.

Design notes that are easy to get subtly wrong and are therefore called out at
the point they happen:

* **Pre-LN vs post-LN** changes what the residual stream *is*. Under pre-LN the
  stream is never normalised, so it is a clean additive accumulator that every
  block reads and writes -- which is what makes the "residual stream" framing of
  interpretability work at all, and what makes deep transformers trainable
  without a warmup you tune by hand.
* **The causal mask must be built from the absolute positions**, not from the
  shape of the score matrix. With a KV cache the query block is short and the key
  block is long, and a mask built from ``(T, T)`` silently masks the wrong
  things. See :meth:`CausalSelfAttention.forward`.
* **GPT-2 shipped the tanh approximation of GELU.** Using the exact erf form
  moves the logits by ~1e-3, which is far above the 1e-4 tolerance this repo
  verifies to, and is enough to change greedy generations.
* **The residual-scaling init** (``0.02 / sqrt(2 * n_layer)``) applies to the two
  projections that *write into* the stream (attention output and MLP output),
  and to nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_internals.config import GPTConfig

__all__ = [
    "GPT",
    "MLP",
    "Block",
    "CausalSelfAttention",
    "KVCache",
    "gelu_tanh",
    "sinusoidal_position_embeddings",
]

NEG_INF = float("-inf")


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU, tanh approximation -- the one GPT-2 was actually trained with.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    Hendrycks & Gimpel give both this and the exact ``x * Phi(x)``; the OpenAI
    TensorFlow code used this one, so a faithful reimplementation must too. The
    two differ by up to ~1e-3 in the activation, which propagates to roughly the
    same order in the logits -- an order of magnitude above the tolerance the
    verification suite asserts, so this is not a stylistic choice.

    Args:
        x: Any shape.

    Returns:
        Elementwise GELU, same shape and dtype.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def sinusoidal_position_embeddings(n_positions: int, n_embd: int) -> torch.Tensor:
    """Fixed sinusoidal position embeddings from Vaswani et al. (2017).

    Even channels get ``sin(pos / 10000^(2i/d))``, odd channels the matching
    cosine. The appeal is that the embedding of position ``p + k`` is a fixed
    linear function of the embedding of ``p``, so relative offsets are in
    principle linearly decodable, and nothing has to be learned or stored.

    Used only by the ablation arm; GPT-2 itself learns its position embeddings.

    Args:
        n_positions: Maximum sequence length.
        n_embd: Embedding width. Odd widths are supported (the final cosine
            channel is dropped).

    Returns:
        A ``(n_positions, n_embd)`` float tensor.
    """
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd)
    )
    pe = torch.zeros(n_positions, n_embd, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : n_embd // 2]
    return pe


@dataclass
class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Stores the keys and values for every position generated so far, so that step
    ``t`` costs O(t) attention work instead of O(t^2) re-encoding of the whole
    prefix. This is *only* valid because the mask is causal: position ``t``'s key
    and value do not depend on anything after ``t``, so they never need
    recomputing.

    Attributes:
        keys / values: Lists of length ``n_layer``, each ``(B, n_head, T, hd)``,
            or ``None`` before the first forward pass.
    """

    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]

    @classmethod
    def empty(cls, n_layer: int) -> KVCache:
        return cls(keys=[None] * n_layer, values=[None] * n_layer)

    @property
    def seq_len(self) -> int:
        """Positions cached for layer 0, i.e. the model-level offset.

        Read this only *before* any layer has been updated in the current step --
        in practice, at the top of :meth:`GPT.forward`, where it gives the number
        of positions already processed and therefore the absolute position of the
        incoming tokens.
        """
        return self.length(0)

    def length(self, layer: int) -> int:
        """Positions currently cached for one layer (0 if empty).

        Attention must use *this*, not :attr:`seq_len`. Layers are updated in
        sequence within a single forward pass, so by the time layer 1 runs, layer
        0's entry already includes the current step and ``seq_len`` reports a
        past length that is too large by ``T``. That builds the causal mask at the
        wrong offset for every layer but the first -- a bug that still produces
        fluent text and a falling loss, and is invisible without a
        cached-vs-uncached equality test.
        """
        k = self.keys[layer]
        return 0 if k is None else int(k.shape[-2])

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's k/v for one layer and return the full history.

        Args:
            layer: Layer index.
            k: ``(B, n_head, T_new, head_dim)``.
            v: Same shape as ``k``.

        Returns:
            The concatenated ``(k, v)`` over all positions seen so far.
        """
        prev_k, prev_v = self.keys[layer], self.values[layer]
        if prev_k is not None and prev_v is not None:
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)
        self.keys[layer] = k
        self.values[layer] = v
        return k, v


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out longhand.

    The single fused ``c_attn`` projection producing q, k and v at once is not a
    stylistic choice -- it is what the GPT-2 checkpoint stores, so keeping it
    fused is what lets the published weights load without reshaping games. It is
    also faster: one ``(B*T, C) x (C, 3C)`` matmul beats three ``(C, C)`` ones.

    Args:
        config: Model config.
        layer_idx: Index of the containing block; used only for diagnostics.
    """

    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.kv_heads
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        # Queries always get n_head heads; keys and values get n_kv_head. For
        # plain MHA these coincide and c_attn is the familiar (C, 3C) matrix that
        # the GPT-2 checkpoint stores.
        self.kv_dim = self.n_kv_head * self.head_dim

        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.kv_dim, bias=config.attn_bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # A lower-triangular boolean mask, registered as a non-persistent buffer:
        # it is derived from the config, so writing it into every checkpoint
        # would waste 1 MB and, worse, would make a checkpoint saved at one
        # context length fail to load at another.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool)),
            persistent=False,
        )

        #: Optional ``(n_head,)`` multiplier on each head's output, applied after
        #: attention and before the output projection. ``None`` means no masking.
        #: Zeroing an entry is a clean *causal* ablation of that head: the head
        #: still runs, but its contribution to the residual stream is removed, so
        #: the change in loss is attributable to that head alone. This is how the
        #: induction analysis moves from "this head attends to the right place" to
        #: "removing this head damages the behaviour".
        self.head_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run attention.

        Args:
            x: ``(B, T, C)`` residual-branch input (already normalised under
                pre-LN).
            cache: If given, this layer's k/v are appended to it and attention
                runs over the full history. ``T`` is then the number of *new*
                positions, typically 1.
            need_weights: Also return the ``(B, n_head, T, T_total)`` attention
                probabilities. Off by default because materialising them defeats
                any fused kernel; the induction-head analysis turns it on.

        Returns:
            ``(output, weights_or_None)`` where output is ``(B, T, C)``.
        """
        B, T, C = x.shape
        # This layer's own cached length -- see KVCache.length for why using the
        # model-level seq_len here is wrong for every layer after the first.
        past_len = cache.length(self.layer_idx) if cache is not None else 0

        # --- q/k/v in one matmul, then split ------------------------------
        qkv = self.c_attn(x)  # (B, T, C + 2 * kv_dim)
        q, k, v = qkv.split([self.n_embd, self.kv_dim, self.kv_dim], dim=2)

        # --- reshape into heads -------------------------------------------
        # (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd). The transpose is what
        # puts the head axis next to the batch axis so the matmuls below are
        # batched over (B, nh) and each head genuinely attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # The cache stores the UNEXPANDED k/v -- n_kv_head of them, not n_head.
        # That is the entire memory win of GQA/MQA, and expanding before caching
        # would throw it away while still paying the quality cost.
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        T_total = k.shape[-2]

        if self.n_kv_head != self.n_head:
            # Broadcast each kv head across its group of query heads.
            # repeat_interleave (not repeat) so head g of the queries pairs with
            # kv head g // group_size, matching the grouping GQA defines.
            groups = self.n_head // self.n_kv_head
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # --- the mask, built from ABSOLUTE positions ----------------------
        # Query i in this block is at absolute position ``past_len + i``; key j is
        # at absolute position ``j``. Slicing the precomputed triangle by those
        # absolute ranges is what makes the cached and uncached paths agree. A
        # mask built as ``tril(ones(T, T_total))`` would be wrong the moment
        # ``past_len > 0``.
        mask = self.causal_mask[past_len : past_len + T, :T_total]

        if self.config.use_sdpa and not need_weights:
            # Speed arm. Same maths, fused kernel; kept behind a flag so the
            # reference path stays the thing under test.
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask.view(1, 1, T, T_total),
                dropout_p=self.config.dropout if self.training else 0.0,
            )
            attn = None
        else:
            # scaled dot product: (B, nh, T, hd) @ (B, nh, hd, T_total)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Mask *before* the softmax, with -inf, so masked positions get
            # exactly zero probability. Masking after the softmax would leave the
            # normaliser contaminated by the future.
            att = att.masked_fill(~mask.view(1, 1, T, T_total), NEG_INF)
            att = F.softmax(att, dim=-1)
            attn = att if need_weights else None
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hd)

        # --- merge heads and project --------------------------------------
        # ``contiguous`` is required: the transpose leaves a non-contiguous view
        # and ``view`` cannot reinterpret it.
        if self.head_mask is not None:
            # (B, nh, T, hd) * (1, nh, 1, 1)
            y = y * self.head_mask.to(y.device, y.dtype).view(1, -1, 1, 1)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, attn


class MLP(nn.Module):
    """The position-wise feed-forward block: 4x expansion, activation, project.

    The 4x is GPT-2's; it is where roughly two thirds of the parameters live
    (``8 * n_embd^2`` per block against attention's ``4 * n_embd^2``).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.mlp_bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)

        #: Optional ``(4 * n_embd,)`` multiplier on the hidden activations. The
        #: counterpart of ``CausalSelfAttention.head_mask``: zeroing an entry
        #: removes one MLP neuron, and leaving the mask differentiable turns it
        #: into the importance variable used for gradient-based pruning.
        self.neuron_mask: torch.Tensor | None = None

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation."""
        if self.config.activation == "gelu_tanh":
            return gelu_tanh(x)
        if self.config.activation == "gelu_exact":
            return F.gelu(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        if self.neuron_mask is not None:
            h = h * self.neuron_mask.to(h.device, h.dtype)
        return self.dropout(self.c_proj(h))


class Block(nn.Module):
    """One transformer block: attention sub-layer then MLP sub-layer.

    Under pre-LN (GPT-2, and the default) each sub-layer is
    ``x = x + f(LN(x))``: the normalisation sits on the branch, and the residual
    stream itself is never touched. Under post-LN (the 2017 original) it is
    ``x = LN(x + f(x))``: the stream is renormalised after every add.

    That difference is why pre-LN trains without a carefully tuned warmup and
    post-LN does not. Under post-LN the gradient reaching layer ``l`` from the
    loss is multiplied by a LayerNorm Jacobian at every one of the ``L - l``
    blocks above it; under pre-LN there is an unbroken identity path from the
    loss to every block. The ablation in this repository measures what that is
    worth at 6 layers -- where post-LN is still trainable, so the comparison is
    a fair one rather than a rigged one.

    Args:
        config: Model config.
        layer_idx: Index of this block.
    """

    def __init__(self, config: GPTConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        need_weights: bool = False,
        taps: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the block.

        Args:
            x: ``(B, T, C)`` residual stream.
            cache: Optional KV cache.
            need_weights: Return attention probabilities.
            taps: If given, intermediate activations are written into this dict.
                This is how the layer-by-layer verification table is produced --
                the alternative, forward hooks, would not see the pre-LN
                sub-results that are the interesting ones.
            prefix: Key prefix for ``taps``.

        Returns:
            ``(x, attn_weights_or_None)``.
        """
        pre = self.config.norm_position == "pre"

        if pre:
            h = self.ln_1(x)
            if taps is not None:
                taps[f"{prefix}ln_1"] = h
            a, attn = self.attn(h, cache=cache, need_weights=need_weights)
            if taps is not None:
                taps[f"{prefix}attn"] = a
            x = x + a
            if taps is not None:
                taps[f"{prefix}resid_mid"] = x
            h2 = self.ln_2(x)
            m = self.mlp(h2)
            if taps is not None:
                taps[f"{prefix}ln_2"] = h2
                taps[f"{prefix}mlp"] = m
            x = x + m
        else:
            a, attn = self.attn(x, cache=cache, need_weights=need_weights)
            if taps is not None:
                taps[f"{prefix}attn"] = a
            x = self.ln_1(x + a)
            if taps is not None:
                taps[f"{prefix}resid_mid"] = x
            m = self.mlp(x)
            if taps is not None:
                taps[f"{prefix}mlp"] = m
            x = self.ln_2(x + m)

        if taps is not None:
            taps[f"{prefix}resid_out"] = x
        return x, attn


class GPT(nn.Module):
    """A GPT-2 style decoder-only language model.

    Module names mirror the OpenAI/HuggingFace checkpoint (``wte``, ``wpe``,
    ``h.<i>.ln_1``, ...) so that :mod:`transformer_internals.weights` is a near-identity
    mapping and any discrepancy is a real difference rather than a naming
    accident.

    Args:
        config: Model config.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_embedding == "learned":
            self.wpe: nn.Embedding | None = nn.Embedding(config.n_positions, config.n_embd)
        else:
            self.wpe = None
            self.register_buffer(
                "sinusoidal_pe",
                sinusoidal_position_embeddings(config.n_positions, config.n_embd),
                persistent=False,
            )
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # Tying is a genuine constraint, not a copy: assigning the Parameter
            # object makes the two modules share storage, so a gradient from the
            # output head lands on the input embedding as well.
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        if config.residual_scaled_init:
            self._apply_residual_scaling()

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2's initialisation: normal(0, 0.02), zero biases, unit LayerNorm."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self) -> None:
        """Scale the residual-writing projections by ``1/sqrt(2 * n_layer)``.

        The residual stream is a sum of ``2 * n_layer`` sub-layer outputs plus the
        embedding. If each contribution has variance ``s^2`` and they are roughly
        independent at init, the stream's variance grows linearly in depth --
        so a 48-layer model starts with a stream ~10x larger than a 1-layer one,
        and the first LayerNorm has to undo it. Scaling each *writing* projection
        by ``1/sqrt(2 * n_layer)`` cancels that growth exactly, keeping the
        stream O(1) at initialisation regardless of depth.

        Only ``attn.c_proj`` and ``mlp.c_proj`` write into the stream, so only
        they are scaled. This is the ``NANOGPT_SCALE_INIT`` trick, and it is
        what OpenAI's ``model.py`` does with its ``n_layer`` division.
        """
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.h:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=std)

    # --------------------------------------------------------------- forward

    def position_embeddings(self, pos: torch.Tensor) -> torch.Tensor:
        """Look up position embeddings for absolute positions ``pos``."""
        if self.wpe is not None:
            return self.wpe(pos)
        return self.sinusoidal_pe[pos]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        need_weights: bool = False,
        collect_taps: bool = False,
    ) -> dict[str, Any]:
        """Run the model.

        Args:
            idx: ``(B, T)`` int64 token ids.
            targets: ``(B, T)`` next-token targets. When given, the returned dict
                carries ``loss``; ``-100`` is ignored, matching torch's default.
            cache: Optional KV cache. When present, ``idx`` holds only the *new*
                tokens and positions are offset by the cache length.
            need_weights: Collect per-layer attention probabilities.
            collect_taps: Collect intermediate activations for verification.

        Returns:
            A dict with ``logits`` ``(B, T, vocab)`` and, depending on the flags,
            ``loss``, ``attentions`` (list of ``(B, nh, T, T_total)``) and
            ``taps`` (dict of named activations).
        """
        _batch, T = idx.shape
        past_len = cache.seq_len if cache is not None else 0
        if past_len + T > self.config.n_positions:
            raise ValueError(
                f"sequence of length {past_len + T} exceeds n_positions="
                f"{self.config.n_positions}; learned position embeddings cannot "
                f"extrapolate"
            )

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.position_embeddings(pos)
        x = self.drop(tok_emb + pos_emb)

        taps: dict[str, torch.Tensor] | None = {} if collect_taps else None
        if taps is not None:
            taps["embed"] = x

        attentions: list[torch.Tensor] = []
        for i, block in enumerate(self.h):
            x, attn = block(
                x,
                cache=cache,
                need_weights=need_weights,
                taps=taps,
                prefix=f"h.{i}.",
            )
            if attn is not None:
                attentions.append(attn)

        x = self.ln_f(x)
        if taps is not None:
            taps["ln_f"] = x
        logits = self.lm_head(x)

        out: dict[str, Any] = {"logits": logits}
        if taps is not None:
            taps["logits"] = logits
            out["taps"] = taps
        if need_weights:
            out["attentions"] = attentions
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return out

    # ----------------------------------------------------------------- utils

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count parameters.

        Args:
            non_embedding: Exclude the position embeddings. Following the GPT-2
                and nanoGPT convention the *token* embedding is still counted,
                because under weight tying it is also the output head and so does
                real work in the forward pass.

        Returns:
            Parameter count.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.wpe is not None:
            n -= self.wpe.weight.numel()
        return n

    def configure_optimizers(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
    ) -> torch.optim.AdamW:
        """Build AdamW with decay applied only to matmul weights.

        Every parameter of rank >= 2 (embeddings and linear weights) is decayed;
        biases and LayerNorm gains are not. Decaying a LayerNorm gain toward zero
        is decaying the whole layer toward outputting its own bias, which is not
        regularisation -- it is damage. Decaying a bias just shifts it, which
        the next layer's own bias can undo for free.

        Args:
            weight_decay: Decay for the decayed group.
            lr: Learning rate.
            betas: AdamW betas.
            eps: AdamW epsilon.

        Returns:
            The optimiser.
        """
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
