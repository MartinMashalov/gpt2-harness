"""Finding induction heads in GPT-2 small.

An **induction head** implements the rule *"[A][B] ... [A] -> [B]"*: having seen
the bigram ``A B`` earlier in the context, when ``A`` appears again it predicts
``B``. Olsson et al. (2022) identify it as the mechanism behind a large part of
in-context learning, and it is a two-head circuit: a **previous-token head** in an
early layer writes token ``i - 1``'s identity into position ``i``'s residual
stream, and an induction head in a later layer uses that to search for earlier
positions whose *predecessor* matches the current token, then copies what
followed.

The reason this is a good test of whether you understand attention -- rather than
just its shape -- is that it is falsifiable. The prediction is completely
specific: on a sequence of random tokens repeated twice, an induction head at
position ``i`` in the second copy must attend to position ``i - (T - 1)``, where
``T`` is the repeat period. Random tokens are essential: they cannot be predicted
from unigram or bigram statistics, so *any* better-than-chance prediction in the
second half has to come from copying the first half.

Three scores are computed per head, because "attends to the right place",
"writes the right thing" and "actually matters" are three different claims:

* **Prefix-matching score** -- where the head looks. Mean attention from each
  second-copy position to the token that followed the previous occurrence of the
  current token. Chance is ~``1/context``, i.e. under 1%.
* **Copying score** -- what the head writes. A head can attend perfectly and
  still write something unrelated. This passes token embeddings through the
  head's OV circuit ``W_U W_O W_V W_E`` and asks how often the token's own
  identity comes out on top. In GPT-2 small it identifies a real population of
  copying heads (L11H3 at 0.639, L11H10 at 0.605; 17% of heads above 0.1 against
  a median of 0.001) -- but **essentially none of them are the prefix-matching
  heads**, which all score below 0.01. That is reported rather than hidden.
  Folding in the LayerNorm gains does not change it (tested). The construction
  only sees the *direct* path to the unembedding, while an induction head writes
  into a residual stream that later layers read and transform, so its copying is
  not legible in the direct path alone.
* **Ablation score** -- whether the head is used. Zero the head's contribution to
  the residual stream and measure how much the second-copy loss rises. This is
  causal rather than correlational, and it is the score that decides the
  question: a head that attends perfectly but whose removal costs nothing was not
  doing the job.

A **previous-token score** is also reported, since it identifies the other half of
the circuit and makes the layer ordering visible: previous-token heads must sit
below induction heads for the circuit to compose.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from transformer_internals.model import GPT

__all__ = [
    "InductionScores",
    "ablated_head",
    "ablation_scores",
    "copying_scores",
    "in_context_learning_curve",
    "make_repeated_sequence",
    "prefix_matching_scores",
    "score_heads",
    "second_copy_loss",
]


@dataclass
class InductionScores:
    """Per-head scores, all shaped ``(n_layer, n_head)``.

    Attributes:
        prefix_matching: Attention mass on the induction target position.
        previous_token: Attention mass on position ``i - 1``.
        copying: Fraction of sampled tokens the head's OV circuit maps to itself.
        copying_ln_folded: The same score with the LayerNorm gains folded in --
            computed so that the negative result can be shown not to be an
            artefact of leaving them out.
        ablation: Increase in second-copy loss (nats) when the head is zeroed.
            The causal measure: it says the head is *used*, not merely present.
        chance_level: Attention a uniform head would place on a single position,
            averaged over the queried positions. The honest baseline for the
            attention-based scores.
        meta: Run parameters.
    """

    prefix_matching: torch.Tensor
    previous_token: torch.Tensor
    copying: torch.Tensor
    copying_ln_folded: torch.Tensor | None = None
    chance_level: float = 0.0
    ablation: torch.Tensor | None = None
    baseline_second_copy_loss: float = float("nan")
    meta: dict[str, Any] = field(default_factory=dict)

    def top_heads(self, k: int = 10, key: str = "prefix_matching") -> list[dict[str, Any]]:
        """Rank heads by one score.

        Args:
            k: How many to return.
            key: ``prefix_matching``, ``previous_token``, ``copying`` or
                ``ablation``.

        Returns:
            Records of ``layer``, ``head``, and all three scores, best first.
        """
        scores = getattr(self, key)
        flat = scores.flatten()
        order = torch.argsort(flat, descending=True)[:k]
        n_head = scores.shape[1]
        return [
            {
                "layer": int(i // n_head),
                "head": int(i % n_head),
                "name": f"L{int(i // n_head)}H{int(i % n_head)}",
                "prefix_matching": float(self.prefix_matching.flatten()[i]),
                "previous_token": float(self.previous_token.flatten()[i]),
                "copying": float(self.copying.flatten()[i]),
                "ablation": (
                    float(self.ablation.flatten()[i]) if self.ablation is not None else None
                ),
            }
            for i in order
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_matching": self.prefix_matching.tolist(),
            "previous_token": self.previous_token.tolist(),
            "copying": self.copying.tolist(),
            "copying_ln_folded": (
                self.copying_ln_folded.tolist() if self.copying_ln_folded is not None else None
            ),
            "chance_level": self.chance_level,
            "ablation": self.ablation.tolist() if self.ablation is not None else None,
            "baseline_second_copy_loss": self.baseline_second_copy_loss,
            "meta": self.meta,
        }


def make_repeated_sequence(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    bos_token_id: int,
    generator: torch.Generator | None = None,
    low: int = 1000,
    high: int = 20000,
) -> torch.Tensor:
    """Build ``[BOS] X X`` where ``X`` is a random token sequence.

    Args:
        batch_size: Number of sequences.
        seq_len: Length ``T`` of the repeated block.
        vocab_size: Unused except for validation; sampling is from ``[low, high)``.
        bos_token_id: Prepended token.
        generator: RNG.
        low / high: Token id range to sample from. Restricted to the middle of the
            vocabulary on purpose: ids below ~1000 are single bytes and common
            sub-words with strong unigram priors, and the very high ids are rare
            junk. Sampling from a band of ordinary, roughly equally-frequent
            tokens keeps the sequence genuinely unpredictable, so a high score
            cannot come from the model simply knowing that ``" the"`` is common.

    Returns:
        ``(batch_size, 2 * seq_len + 1)`` int64 ids.
    """
    if high > vocab_size:
        raise ValueError(f"high={high} exceeds vocab_size={vocab_size}")
    block = torch.randint(low, high, (batch_size, seq_len), generator=generator)
    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
    return torch.cat([bos, block, block], dim=1)


def _attention_offset_score(
    attentions: list[torch.Tensor], query_positions: torch.Tensor, offset: int
) -> torch.Tensor:
    """Mean attention from given queries to ``query - offset``.

    Args:
        attentions: Per-layer ``(B, n_head, T, T)`` probability tensors.
        query_positions: 1-D long tensor of query indices to average over.
        offset: Positive lag; key index is ``query - offset``.

    Returns:
        ``(n_layer, n_head)`` mean attention.
    """
    keys = query_positions - offset
    if int(keys.min()) < 0:
        raise ValueError("offset puts a key before position 0")
    rows = []
    for att in attentions:
        # (B, nh, len(queries)) -- gather the single (query, key) entry per query.
        picked = att[:, :, query_positions, keys]
        rows.append(picked.mean(dim=(0, 2)))
    return torch.stack(rows)


def prefix_matching_scores(
    model: GPT,
    seq_len: int = 60,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Score every head on the repeated-random-tokens probe.

    Args:
        model: The model.
        seq_len: Repeat period ``T``.
        batch_size: Sequences to average over.
        seed: RNG seed for the random tokens.
        device: Device.

    Returns:
        ``(prefix_matching, previous_token, chance_level, sequences)``, the first
        two shaped ``(n_layer, n_head)``.
    """
    gen = torch.Generator().manual_seed(seed)
    seqs = make_repeated_sequence(
        batch_size,
        seq_len,
        model.config.vocab_size,
        bos_token_id=50256 if model.config.vocab_size > 50256 else 0,
        generator=gen,
    ).to(device)

    with torch.no_grad():
        out = model(seqs, need_weights=True)
    attentions = out["attentions"]

    # Second-copy positions: token at 1 + T + j equals the token at 1 + j, and the
    # induction target -- the token that followed it -- sits at 1 + j + 1. The lag
    # between them is therefore exactly T - 1, independent of j. j stops at T - 2
    # so the target stays inside the first copy.
    j = torch.arange(0, seq_len - 1)
    queries = 1 + seq_len + j
    prefix = _attention_offset_score(attentions, queries, offset=seq_len - 1)
    prev = _attention_offset_score(attentions, queries, offset=1)

    # A uniform causal head at position i spreads its mass over i + 1 keys, so the
    # mass it puts on any one key is 1/(i+1). Averaging that over the queried
    # positions is the correct chance line for these scores.
    chance = float((1.0 / (queries.float() + 1.0)).mean())
    return prefix.cpu(), prev.cpu(), chance, seqs.cpu()


def copying_scores(
    model: GPT,
    n_tokens: int = 2000,
    seed: int = 0,
    top_k: int = 1,
    device: str | torch.device = "cpu",
    fold_ln: bool = False,
) -> torch.Tensor:
    """Score every head's OV circuit on whether it copies token identity.

    For head ``h`` the OV circuit is the linear map
    ``W_U @ W_O[h] @ W_V[h] @ W_E``: embed a token, project it into the head's
    value subspace, project back out to the residual stream, unembed. If the
    head's job is to copy, that composition should map a token to a logit vector
    whose largest entry is the token itself.

    Args:
        model: The model.
        n_tokens: How many random vocabulary items to test.
        seed: RNG seed for the token sample.
        top_k: Count a token as copied if its own logit is in the top ``k``.
        device: Device.
        fold_ln: Fold the LayerNorm gains into the circuit -- the block's
            ``ln_1`` gain into the embedding it reads, and ``ln_f``'s gain into
            the unembedding it writes through. This is the standard refinement,
            and the point of computing it is to check that the conclusion does
            not depend on omitting it. The centring and per-token scaling of
            LayerNorm are still not modelled; only the learned gains are, which
            is what "folding" conventionally means here.

    Returns:
        ``(n_layer, n_head)`` fraction of tokens copied.
    """
    gen = torch.Generator().manual_seed(seed)
    cfg = model.config
    tokens = torch.randint(0, cfg.vocab_size, (n_tokens,), generator=gen)
    W_E = model.wte.weight.detach()[tokens.to(model.wte.weight.device)].to(device)  # (N, C)
    W_U = model.wte.weight.detach().to(device).t()  # (C, vocab) -- tied
    if fold_ln:
        W_U = W_U * model.ln_f.weight.detach().to(device).unsqueeze(1)

    scores = torch.zeros(cfg.n_layer, cfg.n_head)
    hd = cfg.head_dim
    for layer, block in enumerate(model.h):
        attn = block.attn
        # c_attn packs [q | k | v] along the output dimension; nn.Linear stores
        # (out, in), so the value block is rows [2C : 3C].
        W_v_all = attn.c_attn.weight.detach()[2 * cfg.n_embd :, :].to(device)  # (C, C)
        b_v_all = (
            attn.c_attn.bias.detach()[2 * cfg.n_embd :].to(device)
            if attn.c_attn.bias is not None
            else torch.zeros(cfg.n_embd, device=device)
        )
        W_o_all = attn.c_proj.weight.detach().to(device)  # (C_out, C_in)
        # The head reads the stream after this block's ln_1, so its gain belongs
        # on the embedding side of the circuit.
        E = W_E * block.ln_1.weight.detach().to(device) if fold_ln else W_E
        for head in range(cfg.n_head):
            sl = slice(head * hd, (head + 1) * hd)
            v = E @ W_v_all[sl, :].t() + b_v_all[sl]  # (N, hd)
            resid = v @ W_o_all[:, sl].t()  # (N, C)
            logits = resid @ W_U  # (N, vocab)
            if top_k == 1:
                hit = logits.argmax(dim=-1).cpu() == tokens
            else:
                topk = logits.topk(top_k, dim=-1).indices.cpu()
                hit = (topk == tokens.unsqueeze(1)).any(dim=1)
            scores[layer, head] = hit.float().mean()
    return scores


@contextmanager
def ablated_head(model: GPT, layer: int, head: int) -> Iterator[None]:
    """Temporarily zero one attention head's contribution.

    The head still computes its attention pattern; only its *output* into the
    residual stream is removed. That is the right ablation for the question
    "is this head used?" -- it changes nothing about the rest of the forward
    pass, so any change in loss is attributable to this head.

    Args:
        model: The model.
        layer: Layer index.
        head: Head index within the layer.

    Yields:
        Nothing; the mask is restored on exit, including on exception.
    """
    attn = model.h[layer].attn
    previous = attn.head_mask
    mask = torch.ones(model.config.n_head)
    mask[head] = 0.0
    attn.head_mask = mask
    try:
        yield
    finally:
        attn.head_mask = previous


@torch.no_grad()
def second_copy_loss(model: GPT, seqs: torch.Tensor, seq_len: int) -> float:
    """Mean next-token loss over the second copy of a repeated sequence.

    Args:
        model: The model.
        seqs: ``(B, 2T + 1)`` repeated sequences.
        seq_len: The repeat period ``T``.

    Returns:
        Mean loss in nats over positions that predict inside the second copy.
    """
    logits = model(seqs)["logits"]
    x, y = logits[:, :-1], seqs[:, 1:]
    loss = F.cross_entropy(
        x.reshape(-1, x.size(-1)), y.reshape(-1), reduction="none"
    ).view(y.shape)
    return loss[:, seq_len : 2 * seq_len - 1].mean().item()


def ablation_scores(
    model: GPT,
    seq_len: int = 60,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, float]:
    """Zero each head in turn and measure the damage to induction behaviour.

    Runs ``n_layer * n_head`` forward passes (144 for GPT-2 small) on one fixed
    batch, so the comparison across heads is exact -- every head is scored on the
    same sequences against the same baseline.

    Args:
        model: The model.
        seq_len: Repeat period.
        batch_size: Sequences.
        seed: RNG seed for the random tokens.
        device: Device.

    Returns:
        ``(delta_loss, baseline_loss)`` where ``delta_loss`` is ``(n_layer,
        n_head)`` of ``ablated - baseline`` second-copy loss in nats. Positive
        means removing the head made induction worse.
    """
    gen = torch.Generator().manual_seed(seed)
    seqs = make_repeated_sequence(
        batch_size,
        seq_len,
        model.config.vocab_size,
        bos_token_id=50256 if model.config.vocab_size > 50256 else 0,
        generator=gen,
    ).to(device)

    baseline = second_copy_loss(model, seqs, seq_len)
    deltas = torch.zeros(model.config.n_layer, model.config.n_head)
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            with ablated_head(model, layer, head):
                deltas[layer, head] = second_copy_loss(model, seqs, seq_len) - baseline
    return deltas, baseline


def in_context_learning_curve(
    model: GPT,
    seq_len: int = 60,
    batch_size: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Behavioural evidence that the circuit is actually used.

    Head scores say a mechanism is present; this says it *does something*. On
    ``[BOS] X X`` the first copy is unpredictable by construction, so the loss
    there is essentially ``log(vocab_size_of_the_sampling_band)``. If the model
    copies, the loss on the second copy collapses. The gap between them is the
    induction bump, measured in nats.

    Args:
        model: The model.
        seq_len: Repeat period.
        batch_size: Sequences.
        seed: RNG seed.
        device: Device.

    Returns:
        Per-position mean loss, the two half-averages, and their difference.
    """
    gen = torch.Generator().manual_seed(seed)
    seqs = make_repeated_sequence(
        batch_size,
        seq_len,
        model.config.vocab_size,
        bos_token_id=50256 if model.config.vocab_size > 50256 else 0,
        generator=gen,
    ).to(device)
    with torch.no_grad():
        logits = model(seqs)["logits"]
    x, y = logits[:, :-1], seqs[:, 1:]
    loss = F.cross_entropy(
        x.reshape(-1, x.size(-1)), y.reshape(-1), reduction="none"
    ).view(y.shape)
    per_pos = loss.mean(0).cpu()

    # Positions 0..T-2 predict inside the first copy; T..2T-2 inside the second.
    first = per_pos[: seq_len - 1].mean().item()
    second = per_pos[seq_len : 2 * seq_len - 1].mean().item()
    return {
        "per_position_loss": per_pos.tolist(),
        "first_copy_loss": first,
        "second_copy_loss": second,
        "induction_bump_nats": first - second,
        "uniform_band_loss": math.log(19000),
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


def score_heads(
    model: GPT,
    seq_len: int = 60,
    batch_size: int = 8,
    n_copy_tokens: int = 2000,
    seed: int = 0,
    device: str | torch.device = "cpu",
    with_ablation: bool = True,
) -> InductionScores:
    """Run the full head analysis.

    Args:
        model: The model.
        seq_len: Repeat period for the attention probe.
        batch_size: Sequences for the attention probe.
        n_copy_tokens: Vocabulary sample size for the copying score.
        seed: RNG seed.
        device: Device.
        with_ablation: Also run the causal head-ablation sweep. This is the
            expensive part (one forward pass per head) but it is the score that
            actually settles the question, so it is on by default.

    Returns:
        The populated :class:`InductionScores`.
    """
    prefix, prev, chance, _ = prefix_matching_scores(
        model, seq_len=seq_len, batch_size=batch_size, seed=seed, device=device
    )
    copying = copying_scores(model, n_tokens=n_copy_tokens, seed=seed, device=device)
    copying_folded = copying_scores(
        model, n_tokens=n_copy_tokens, seed=seed, device=device, fold_ln=True
    )
    ablation: torch.Tensor | None = None
    baseline = float("nan")
    if with_ablation:
        ablation, baseline = ablation_scores(
            model, seq_len=seq_len, batch_size=batch_size, seed=seed, device=device
        )
    return InductionScores(
        prefix_matching=prefix,
        previous_token=prev,
        copying=copying,
        copying_ln_folded=copying_folded,
        chance_level=chance,
        ablation=ablation,
        baseline_second_copy_loss=baseline,
        meta={
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_copy_tokens": n_copy_tokens,
            "seed": seed,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
        },
    )
