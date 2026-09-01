"""Tensor parallelism: split the GEMMs inside a block across ranks.

Megatron-LM's layout (Shoeybi et al., "Megatron-LM: Training Multi-Billion
Parameter Language Models Using Model Parallelism", arXiv:1909.08053). The
residual stream stays replicated on every rank; only the two GEMM pairs inside a
block are split, and they are split so that the *pair* needs exactly one
collective, not two.

For the MLP, ``y = (act(x A) B)``:

* ``A`` is split by **columns**, ``A = [A_0 | A_1]``. Each rank computes
  ``act(x A_i)``, which is legitimate only because the activation is elementwise
  -- a non-elementwise activation would need the full hidden vector and would
  force a collective in the middle.
* ``B`` is split by **rows** to match, ``B = [B_0 ; B_1]``. Each rank computes
  the partial product ``act(x A_i) B_i``, and the outputs are summed.

So the whole MLP costs one all-reduce forward, one all-reduce backward, and the
``4 * n_embd`` hidden activation never exists in one place.

The two operators that make this work are conjugates of each other, and getting
them the wrong way round is the classic tensor-parallel bug -- it produces a
forward pass that is numerically perfect and a backward pass that is silently
wrong by a factor of the world size on part of the model:

* ``f`` (:class:`CopyToTensorParallel`): identity forward, all-reduce backward.
  It sits where a replicated tensor enters sharded compute. Every rank reads the
  same ``x``, so ``x`` collects a gradient contribution from every rank, and
  those contributions have to be summed.
* ``g`` (:class:`ReduceFromTensorParallel`): all-reduce forward, identity
  backward. It sits where partial sums leave sharded compute. The forward needs
  the sum; the backward does not, because each rank's partial output already
  received the full incoming gradient.

The same pair explains why the replicated LayerNorm needs no special handling:
its gradient arrives through ``f``'s backward all-reduce, so it is already the
full-model gradient on every rank.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_internals.config import GPTConfig
from transformer_internals.model import Block
from transformer_internals.parallel import comms
from transformer_internals.parallel.common import (
    current_device,
    identical_block,
    parallel_config,
)
from transformer_internals.perf.activation_memory import ActivationMeter

__all__ = [
    "ColumnParallelLinear",
    "CopyToTensorParallel",
    "ReduceFromTensorParallel",
    "RowParallelLinear",
    "TensorParallelBlock",
    "tp_equivalence_worker",
]


class CopyToTensorParallel(torch.autograd.Function):
    """``f``: identity in the forward pass, all-reduce in the backward pass."""

    @staticmethod
    def forward(_ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x

    @staticmethod
    def backward(_ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return comms.all_reduce(grad.contiguous())


class ReduceFromTensorParallel(torch.autograd.Function):
    """``g``: all-reduce in the forward pass, identity in the backward pass."""

    @staticmethod
    def forward(_ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return comms.all_reduce(x.contiguous())

    @staticmethod
    def backward(_ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad


def _shard(tensor: torch.Tensor, rank: int, world_size: int, dim: int) -> torch.Tensor:
    n = tensor.shape[dim]
    if n % world_size:
        raise ValueError(f"dim {dim} of size {n} is not divisible by {world_size} ranks")
    per = n // world_size
    return tensor.narrow(dim, rank * per, per).detach().clone()


class ColumnParallelLinear(nn.Module):
    """``y = x A_i^T + b_i``: the output features are split across ranks.

    The input is replicated and passes through ``f``, so the gradient reaching
    it is summed over ranks. The output is sharded; whoever consumes it must
    either be a :class:`RowParallelLinear` or all-gather it.

    Args:
        linear: The full layer to shard. Its weights are copied, not referenced.
        rank / world_size: Position in the group.
        apply_f: Insert the ``f`` operator on the input. Off when the caller has
            already applied it for a fused group of column-parallel layers.
    """

    def __init__(
        self, linear: nn.Linear, rank: int, world_size: int, apply_f: bool = True
    ) -> None:
        super().__init__()
        self.world_size = world_size
        self.apply_f = apply_f
        self.weight = nn.Parameter(_shard(linear.weight.data, rank, world_size, dim=0))
        if linear.bias is not None:
            self.bias: nn.Parameter | None = nn.Parameter(
                _shard(linear.bias.data, rank, world_size, dim=0)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_f:
            x = CopyToTensorParallel.apply(x)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """``y = sum_i (x_i A_i^T) + b``: the input features are split across ranks.

    The bias is added *after* the reduction and only once. Adding it before
    would add it ``p`` times -- a bug that scales with the world size, so it
    passes on one GPU and fails on eight.

    Args:
        linear: The full layer to shard.
        rank / world_size: Position in the group.
    """

    def __init__(self, linear: nn.Linear, rank: int, world_size: int) -> None:
        super().__init__()
        self.world_size = world_size
        self.weight = nn.Parameter(_shard(linear.weight.data, rank, world_size, dim=1))
        # Replicated, not sharded: every rank holds the whole bias and adds it
        # after the all-reduce.
        self.bias = (
            nn.Parameter(linear.bias.data.detach().clone()) if linear.bias is not None else None
        )

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x_shard, self.weight)
        out = ReduceFromTensorParallel.apply(partial)
        if self.bias is not None:
            out = out + self.bias
        return out


class TensorParallelBlock(nn.Module):
    """One transformer block with head-parallel attention and a split MLP.

    Attention is split by head, which is the natural cut: heads are independent
    all the way from the q/k/v projection to the concatenation, so no collective
    is needed inside attention at all. The fused ``c_attn`` matrix that the
    GPT-2 checkpoint stores holds ``[q ; k ; v]`` stacked in that order, so
    sharding it by head means slicing each of the three sections separately and
    restacking -- slicing the fused matrix into ``p`` contiguous pieces would
    give rank 0 all of q and rank 1 all of k and v, which is not a head split.

    The output projection ``c_proj`` is row-parallel over the same head
    partition, so the attention sub-layer, like the MLP, costs one all-reduce.

    Args:
        block: The reference block, with full weights. Copied, not referenced.
        rank / world_size: Position in the group.
    """

    def __init__(self, block: Block, rank: int, world_size: int) -> None:
        super().__init__()
        config: GPTConfig = block.config
        if config.norm_position != "pre":
            raise ValueError("TensorParallelBlock implements the pre-LN block only")
        if config.n_head % world_size or config.kv_heads % world_size:
            raise ValueError(
                f"n_head={config.n_head} and kv_heads={config.kv_heads} must both be "
                f"divisible by world_size={world_size}"
            )
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.n_head_local = config.n_head // world_size
        self.n_kv_head_local = config.kv_heads // world_size
        self.head_dim = config.head_dim

        # LayerNorms are replicated. Their gradients arrive already summed,
        # through f's backward all-reduce.
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_1.load_state_dict(block.ln_1.state_dict())
        self.ln_2.load_state_dict(block.ln_2.state_dict())

        # --- attention: shard the fused qkv by head ------------------------
        c_attn = block.attn.c_attn
        n_embd, kv_dim = config.n_embd, config.kv_heads * config.head_dim
        wq, wk, wv = c_attn.weight.data.split([n_embd, kv_dim, kv_dim], dim=0)
        q_rows = self.n_head_local * self.head_dim
        kv_rows = self.n_kv_head_local * self.head_dim
        weight = torch.cat(
            [
                wq[rank * q_rows : (rank + 1) * q_rows],
                wk[rank * kv_rows : (rank + 1) * kv_rows],
                wv[rank * kv_rows : (rank + 1) * kv_rows],
            ],
            dim=0,
        )
        self.c_attn_weight = nn.Parameter(weight.clone())
        if c_attn.bias is not None:
            bq, bk, bv = c_attn.bias.data.split([n_embd, kv_dim, kv_dim], dim=0)
            self.c_attn_bias: nn.Parameter | None = nn.Parameter(
                torch.cat(
                    [
                        bq[rank * q_rows : (rank + 1) * q_rows],
                        bk[rank * kv_rows : (rank + 1) * kv_rows],
                        bv[rank * kv_rows : (rank + 1) * kv_rows],
                    ]
                ).clone()
            )
        else:
            self.c_attn_bias = None
        self.attn_proj = RowParallelLinear(block.attn.c_proj, rank, world_size)

        # --- MLP: column-parallel then row-parallel ------------------------
        self.mlp_fc = ColumnParallelLinear(block.mlp.c_fc, rank, world_size)
        self.mlp_proj = RowParallelLinear(block.mlp.c_proj, rank, world_size)

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.n_positions,
                    config.n_positions,
                    dtype=torch.bool,
                    device=block.ln_1.weight.device,
                )
            ),
            persistent=False,
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Head-parallel attention over this rank's slice of the heads."""
        b, t, _ = x.shape
        x = CopyToTensorParallel.apply(x)
        qkv = F.linear(x, self.c_attn_weight, self.c_attn_bias)
        q_dim = self.n_head_local * self.head_dim
        kv_dim = self.n_kv_head_local * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)
        q = q.view(b, t, self.n_head_local, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head_local, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head_local, self.head_dim).transpose(1, 2)
        if self.n_kv_head_local != self.n_head_local:
            groups = self.n_head_local // self.n_kv_head_local
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:t, :t]
        att = att.masked_fill(~mask.view(1, 1, t, t), float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, q_dim)
        # Row-parallel: this rank holds the columns of c_proj matching its heads.
        return self.attn_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attention(self.ln_1(x))
        h = self.mlp_fc(self.ln_2(x))
        h = self._act(h)
        return x + self.mlp_proj(h)

    def _act(self, h: torch.Tensor) -> torch.Tensor:
        from transformer_internals.model import gelu_tanh

        if self.config.activation == "gelu_tanh":
            return gelu_tanh(h)
        if self.config.activation == "gelu_exact":
            return F.gelu(h)
        return F.relu(h)


def tp_equivalence_worker(
    rank: int,
    world_size: int,
    batch: int = 4,
    seq: int = 12,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a tensor-parallel block against the unsharded block, both ways.

    Forward equivalence is the easy half. The backward comparison covers three
    kinds of parameter, because each one fails differently when ``f`` and ``g``
    are swapped:

    * the input gradient, which is wrong by a factor of ``p`` if ``f``'s
      all-reduce is missing;
    * the replicated LayerNorm gradients, which are *partial* -- only this
      rank's heads -- if ``f``'s all-reduce is missing;
    * this rank's slice of the sharded weight gradients, which are the only
      ones that would still look right.
    """
    config = parallel_config(**(config_kwargs or {}))
    ref = identical_block(config, seed=7).train()
    tp = TensorParallelBlock(ref, rank=rank, world_size=world_size)

    torch.manual_seed(99)
    x = torch.randn(batch, seq, config.n_embd).to(current_device())
    x_ref = x.clone().requires_grad_(True)
    x_tp = x.clone().requires_grad_(True)

    ref_meter = ActivationMeter(exclude=[*ref.parameters(), *ref.buffers()])
    with ref_meter:
        ref_out, _ = ref(x_ref)
    ref_activations = ref_meter.snapshot()

    tp_meter = ActivationMeter(exclude=[*tp.parameters(), *tp.buffers()])
    with tp_meter:
        tp_out = tp(x_tp)
    tp_activations = tp_meter.snapshot()
    forward_error = float((tp_out - ref_out).abs().max())

    # A fixed, non-uniform upstream gradient. A uniform one would hide a bug
    # that only shows up when the incoming gradient varies over the sequence.
    # Drawn on the CPU and moved, like every other tensor in this package: a
    # randn_like on a CUDA tensor draws from the CUDA RNG, and the comparison
    # would then be against a different upstream gradient on a different device.
    torch.manual_seed(100)
    grad_out = torch.randn(ref_out.shape, dtype=ref_out.dtype).to(ref_out.device)
    ref_out.backward(grad_out)
    tp_out.backward(grad_out.clone())

    input_grad_error = float((x_tp.grad - x_ref.grad).abs().max())

    per = config.n_embd // world_size
    n_embd, kv_dim = config.n_embd, config.kv_heads * config.head_dim
    q_rows = (config.n_head // world_size) * config.head_dim
    kv_rows = (config.kv_heads // world_size) * config.head_dim
    wq, wk, wv = ref.attn.c_attn.weight.grad.split([n_embd, kv_dim, kv_dim], dim=0)
    want_c_attn = torch.cat(
        [
            wq[rank * q_rows : (rank + 1) * q_rows],
            wk[rank * kv_rows : (rank + 1) * kv_rows],
            wv[rank * kv_rows : (rank + 1) * kv_rows],
        ],
        dim=0,
    )

    errors = {
        "c_attn_weight": float((tp.c_attn_weight.grad - want_c_attn).abs().max()),
        "attn_c_proj_weight": float(
            (
                tp.attn_proj.weight.grad
                - ref.attn.c_proj.weight.grad[:, rank * per : (rank + 1) * per]
            )
            .abs()
            .max()
        ),
        "mlp_c_fc_weight": float(
            (
                tp.mlp_fc.weight.grad
                - ref.mlp.c_fc.weight.grad[
                    rank * (4 * per) : (rank + 1) * (4 * per)
                ]
            )
            .abs()
            .max()
        ),
        "mlp_c_proj_weight": float(
            (
                tp.mlp_proj.weight.grad
                - ref.mlp.c_proj.weight.grad[:, rank * (4 * per) : (rank + 1) * (4 * per)]
            )
            .abs()
            .max()
        ),
        "ln_1_weight": float((tp.ln_1.weight.grad - ref.ln_1.weight.grad).abs().max()),
        "ln_2_weight": float((tp.ln_2.weight.grad - ref.ln_2.weight.grad).abs().max()),
        "attn_c_proj_bias": float(
            (tp.attn_proj.bias.grad - ref.attn.c_proj.bias.grad).abs().max()
        ),
        "mlp_c_proj_bias": float((tp.mlp_proj.bias.grad - ref.mlp.c_proj.bias.grad).abs().max()),
    }

    # One forward and one backward through one block is this strategy's "step":
    # tensor parallelism issues its collectives per block, not per optimiser step.
    comms.get_counter().steps += 1

    # The scale the errors above should be read against: an absolute error is
    # meaningless without the magnitude of the thing it is an error in.
    ref_grad_scale = max(
        float(p.grad.abs().max()) for p in ref.parameters() if p.grad is not None
    )

    return {
        "rank": rank,
        "forward_error": forward_error,
        "input_grad_error": input_grad_error,
        "grad_errors": errors,
        "max_grad_error": max([input_grad_error, *errors.values()]),
        "reference_grad_scale": ref_grad_scale,
        "output_scale": float(ref_out.abs().max()),
        "local_params": sum(p.numel() for p in tp.parameters()),
        "reference_params": sum(p.numel() for p in ref.parameters()),
        # Unlike ZeRO, tensor parallelism genuinely shards activations: the
        # 4*n_embd MLP hidden and the per-head attention tensors are 1/p of the
        # width on each rank. It does not shard all of them, because the
        # residual stream and the LayerNorm inputs stay replicated, so the ratio
        # is above 1/p and that gap is the replicated part.
        "activation_bytes": tp_activations["activation_bytes"],
        "reference_activation_bytes": ref_activations["activation_bytes"],
        "activation_detail": tp_activations,
    }
