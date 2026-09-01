"""Context parallelism: shard the sequence, and make attention cross the shards.

Every other kind of parallelism leaves the sequence axis alone. Context
parallelism cuts it: rank ``r`` holds positions ``[r*T/p, (r+1)*T/p)`` and
nothing else. That is the only way to train at a context length whose
activations do not fit on one device, because activation memory is linear in
``T`` and the attention score matrix is quadratic in it.

Everything except attention is trivially shardable this way -- LayerNorm, the
MLP and the residual add are position-wise, so a rank can do its own positions
with no communication at all. Attention is the exception, and it is the whole
problem: query ``i`` needs keys and values at every position ``<= i``, and under
a causal mask most of those live on other ranks.

Two ways to get them, both implemented here:

**All-gather KV.** Every rank all-gathers K and V, then attends locally. Simple,
and correct in one collective, but every rank materialises the full
``(B, H, T, d)`` K and V, so KV memory is not sharded at all -- only the queries,
the scores and the activations are. Communication is ``2 * B * H * T * d``
elements per rank per attention.

**Ring attention** (Liu, Zaharia & Abbeel, arXiv:2310.01889). Rotate the KV
blocks around a ring of ``p`` ranks, ``p`` steps, accumulating the softmax
online with the running-maximum rescaling from FlashAttention. No rank ever
holds more than one remote KV block, so KV memory *is* sharded, and the
point-to-point traffic per step is ``2 * B * H * (T/p) * d``. Under a causal
mask, roughly half the blocks a rank receives are entirely in its future and are
skipped without any arithmetic -- which is where causal ring attention gets its
load imbalance, since rank 0 skips ``p-1`` blocks and rank ``p-1`` skips none.

The equivalence claim is the same in both cases: the output at this rank's
positions must equal the corresponding rows of a single-process attention over
the whole sequence.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from transformer_internals.parallel import comms
from transformer_internals.parallel.common import (
    current_device,
    identical_block,
    parallel_config,
)

__all__ = [
    "all_gather_kv_attention",
    "gather_sequence",
    "ring_attention",
    "sequence_parallel_worker",
]


class _GatherSequence(torch.autograd.Function):
    """All-gather along a sequence axis; reduce-scatter the gradient back.

    The backward is a reduce-scatter and not a slice. Every rank uses the whole
    gathered tensor, so every rank produces a gradient for every position; this
    rank's positions must collect the contributions from all of them. Slicing
    instead of reducing would keep only this rank's own contribution and lose
    the rest -- a bug that leaves the forward pass perfect and silently drops
    ``(p-1)/p`` of the key/value gradient.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:  # type: ignore[override]
        ctx.dim = dim
        ctx.world_size = world_size
        moved = x.movedim(dim, 0).contiguous()
        out = torch.empty(
            (moved.shape[0] * world_size, *moved.shape[1:]),
            dtype=moved.dtype,
            device=moved.device,
        )
        comms.all_gather_into(out, moved)
        return out.movedim(0, dim).contiguous()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # type: ignore[override]
        moved = grad.movedim(ctx.dim, 0).contiguous()
        shard = torch.empty(
            (moved.shape[0] // ctx.world_size, *moved.shape[1:]),
            dtype=moved.dtype,
            device=moved.device,
        )
        comms.reduce_scatter_into(shard, moved)
        return shard.movedim(0, ctx.dim).contiguous(), None, None


def gather_sequence(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """Differentiable all-gather of a sequence-sharded tensor."""
    return _GatherSequence.apply(x, dim, world_size)


def all_gather_kv_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    rank: int,
    world_size: int,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """Attention over a sharded sequence, by gathering K and V.

    Args:
        q_local / k_local / v_local: ``(B, H, T/p, d)`` for this rank's positions.
        rank / world_size: Position in the group.
        causal_mask: A ``(T, T)`` lower-triangular bool mask over *absolute*
            positions.

    Returns:
        ``(B, H, T/p, d)`` -- this rank's rows of the full attention output.
    """
    t_local = q_local.shape[-2]
    head_dim = q_local.shape[-1]
    k_full = gather_sequence(k_local, 2, world_size)
    v_full = gather_sequence(v_local, 2, world_size)
    t_total = k_full.shape[-2]

    att = (q_local @ k_full.transpose(-2, -1)) / math.sqrt(head_dim)
    # The mask has to be built from absolute positions: this rank's query i is
    # at position rank*t_local + i, and slicing a (t_local, t_total) triangle
    # instead would let rank 1 attend to its own future.
    mask = causal_mask[rank * t_local : (rank + 1) * t_local, :t_total]
    att = att.masked_fill(~mask.view(1, 1, t_local, t_total), float("-inf"))
    return F.softmax(att, dim=-1) @ v_full


def ring_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Attention over a sharded sequence, by rotating KV blocks around a ring.

    The softmax is accumulated online: a running per-row maximum ``m``, a running
    denominator ``l`` and a running numerator, rescaled by ``exp(m_old - m_new)``
    whenever a new block raises the maximum. That rescaling is the only reason
    this is numerically safe -- attending to blocks one at a time and summing the
    unnormalised exponentials would overflow on the first block with large
    scores.

    Causal masking is applied at block granularity where it can be: a block
    entirely above the diagonal contributes nothing and is skipped, a block
    entirely below is dense, and only the diagonal block needs a triangular
    mask.

    Forward only. The backward pass of ring attention needs a second ring to
    carry ``dK``/``dV`` and is not implemented here; the all-gather path is the
    differentiable one.

    Returns:
        ``(B, H, T/p, d)`` -- this rank's rows of the full attention output.
    """
    b, h, t_local, head_dim = q_local.shape
    device = q_local.device
    scale = 1.0 / math.sqrt(head_dim)
    running_max = torch.full((b, h, t_local, 1), float("-inf"), device=device)
    running_sum = torch.zeros((b, h, t_local, 1), device=device)
    acc = torch.zeros_like(q_local)

    k_cur, v_cur = k_local.contiguous(), v_local.contiguous()
    block = rank
    for step in range(world_size):
        if block <= rank:
            scores = (q_local @ k_cur.transpose(-2, -1)) * scale
            if block == rank:
                tri = torch.tril(
                    torch.ones(t_local, t_local, dtype=torch.bool, device=device)
                )
                scores = scores.masked_fill(~tri.view(1, 1, t_local, t_local), float("-inf"))
            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            rescale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max)
            running_sum = running_sum * rescale + weights.sum(dim=-1, keepdim=True)
            acc = acc * rescale + weights @ v_cur
            running_max = new_max

        if step < world_size - 1:
            send_dst = (rank + 1) % world_size
            recv_src = (rank - 1) % world_size
            k_next = torch.empty_like(k_cur)
            v_next = torch.empty_like(v_cur)
            reqs = [
                comms.isend(k_cur, dst=send_dst),
                comms.isend(v_cur, dst=send_dst),
            ]
            comms.recv(k_next, src=recv_src)
            comms.recv(v_next, src=recv_src)
            for r in reqs:
                r.wait()
            k_cur, v_cur = k_next, v_next
            block = (block - 1) % world_size

    return acc / running_sum


def _project(attn_module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused qkv projection and reshape into heads."""
    b, t, _ = x.shape
    n_head = attn_module.n_head
    kv_dim = attn_module.kv_dim
    qkv = attn_module.c_attn(x)
    q, k, v = qkv.split([attn_module.n_embd, kv_dim, kv_dim], dim=2)
    q = q.view(b, t, n_head, attn_module.head_dim).transpose(1, 2)
    k = k.view(b, t, attn_module.n_kv_head, attn_module.head_dim).transpose(1, 2)
    v = v.view(b, t, attn_module.n_kv_head, attn_module.head_dim).transpose(1, 2)
    return q, k, v


def sequence_parallel_worker(
    rank: int,
    world_size: int,
    batch: int = 2,
    seq: int = 32,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check both context-parallel attentions against single-process attention.

    The reference is the repository's own :class:`CausalSelfAttention` run on the
    whole sequence. Both sharded paths must reproduce this rank's rows of it.
    The all-gather path is also checked in the backward direction, on two
    different kinds of gradient:

    * the gradient with respect to this rank's input slice, which is local; and
    * the gradient with respect to the projection weight, which is *not* local
      -- every rank sees every weight, so each rank's weight gradient is a
      partial sum and the ranks have to all-reduce it, exactly as in data
      parallelism. Comparing the all-reduced weight gradient against the
      single-process one is what proves the sequence shard did not lose a term.
    """
    config = parallel_config(**(config_kwargs or {}))
    if seq % world_size:
        raise ValueError(f"seq={seq} must be divisible by world_size={world_size}")
    t_local = seq // world_size

    ref_attn = identical_block(config, seed=11).attn
    cp_attn = identical_block(config, seed=11).attn

    torch.manual_seed(21)
    x = torch.randn(batch, seq, config.n_embd).to(current_device())
    x_ref = x.clone().requires_grad_(True)
    x_local = x[:, rank * t_local : (rank + 1) * t_local].clone().requires_grad_(True)

    # --- reference -------------------------------------------------------
    q_r, k_r, v_r = _project(ref_attn, x_ref)
    mask = ref_attn.causal_mask[:seq, :seq]
    att = (q_r @ k_r.transpose(-2, -1)) / math.sqrt(config.head_dim)
    att = att.masked_fill(~mask.view(1, 1, seq, seq), float("-inf"))
    ref_out = F.softmax(att, dim=-1) @ v_r
    ref_slice = ref_out[:, :, rank * t_local : (rank + 1) * t_local]

    # --- all-gather KV ---------------------------------------------------
    q_l, k_l, v_l = _project(cp_attn, x_local)
    ag_out = all_gather_kv_attention(q_l, k_l, v_l, rank, world_size, ref_attn.causal_mask)
    ag_error = float((ag_out - ref_slice).abs().max())

    # --- ring ------------------------------------------------------------
    with torch.no_grad():
        q_d, k_d, v_d = _project(cp_attn, x_local.detach())
        ring_out = ring_attention(q_d, k_d, v_d, rank, world_size)
    ring_error = float((ring_out - ref_slice.detach()).abs().max())

    # --- backward through the all-gather path ----------------------------
    torch.manual_seed(31)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ag_out.backward(grad_out[:, :, rank * t_local : (rank + 1) * t_local].contiguous())

    input_grad_error = float(
        (x_local.grad - x_ref.grad[:, rank * t_local : (rank + 1) * t_local]).abs().max()
    )
    weight_grad = cp_attn.c_attn.weight.grad.clone()
    comms.all_reduce(weight_grad)
    weight_grad_error = float((weight_grad - ref_attn.c_attn.weight.grad).abs().max())

    # One attention, forward and backward: context parallelism's collectives are
    # per attention layer, not per optimiser step.
    comms.get_counter().steps += 1

    return {
        "rank": rank,
        "all_gather_forward_error": ag_error,
        "ring_forward_error": ring_error,
        "input_grad_error": input_grad_error,
        "weight_grad_error": weight_grad_error,
        "output_scale": float(ref_slice.abs().max()),
        "weight_grad_scale": float(ref_attn.c_attn.weight.grad.abs().max()),
        "local_positions": t_local,
        "total_positions": seq,
    }
