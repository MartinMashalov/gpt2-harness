"""The same tensor-parallel sharding, expressed as a DeviceMesh with DTensor.

:mod:`~transformer_internals.parallel.tensor_parallel` writes the collectives by
hand, because that is the only way to see what tensor parallelism actually does.
This module does the identical computation through PyTorch's DTensor API, where
the collectives are inferred from placements rather than written down:

* ``Shard(0)`` on the first weight is column parallelism. The output inherits
  ``Shard(-1)``.
* ``Shard(1)`` on the second weight is row parallelism. With a ``Shard(-1)``
  input the output is ``Partial()`` -- a sum that has not been reduced yet --
  and asking for ``Replicate()`` is what emits the all-reduce.

So the ``f``/``g`` conjugate pair does not disappear under DTensor; it becomes
the redistribution between ``Partial`` and ``Replicate``, and the framework
inserts it. The worth of having both implementations in one repository is that
the manual one can be checked against the framework's: if a hand-written
all-reduce sat in the wrong place, the two would disagree.

``torch.distributed.tensor.parallel.parallelize_module`` is exercised as well,
since that is what a real model actually uses -- a plan of
``ColwiseParallel``/``RowwiseParallel`` per submodule rather than per-tensor
placements.

API note: on torch 2.2 (this repository's pinned version) DTensor lives at
``torch.distributed._tensor``. It moved to the public ``torch.distributed.tensor``
in 2.5.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_internals.parallel.common import parallel_config
from transformer_internals.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

__all__ = ["dtensor_worker"]


def _reference_mlp(n_embd: int, seed: int = 5) -> tuple[nn.Linear, nn.Linear]:
    torch.manual_seed(seed)
    return nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd)


def dtensor_worker(
    rank: int,
    world_size: int,
    batch: int = 4,
    seq: int = 8,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the MLP three ways -- single process, DTensor, manual -- and compare.

    Returns:
        The two error figures that matter: DTensor against the single-process
        reference, and DTensor against this repository's hand-written tensor
        parallelism. The second is the interesting one, because it is a claim
        about the manual implementation and not about PyTorch.
    """
    from torch.distributed._tensor import DeviceMesh, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    config = parallel_config(**(config_kwargs or {}))
    n_embd = config.n_embd
    fc, proj = _reference_mlp(n_embd)

    torch.manual_seed(6)
    x = torch.randn(batch, seq, n_embd)
    reference = proj(F.gelu(fc(x)))

    # --- device mesh, placements written out by hand ---------------------
    mesh = DeviceMesh("cpu", torch.arange(world_size))
    d_fc_w = distribute_tensor(fc.weight, mesh, [Shard(0)])
    d_fc_b = distribute_tensor(fc.bias, mesh, [Shard(0)])
    d_proj_w = distribute_tensor(proj.weight, mesh, [Shard(1)])
    d_proj_b = distribute_tensor(proj.bias, mesh, [Replicate()])
    d_x = distribute_tensor(x, mesh, [Replicate()])

    h = F.gelu(F.linear(d_x, d_fc_w, d_fc_b))  # Shard(-1): the 4*n_embd axis
    y = F.linear(h, d_proj_w)  # Partial(): a sum nobody has reduced yet
    y = y.redistribute(mesh, [Replicate()]) + d_proj_b  # the all-reduce happens here
    dtensor_error = float((y.to_local() - reference).abs().max())

    # --- the same thing through parallelize_module ------------------------
    plan_mlp = nn.Sequential()
    plan_mlp.add_module("fc", nn.Linear(n_embd, 4 * n_embd))
    plan_mlp.add_module("proj", nn.Linear(4 * n_embd, n_embd))
    plan_mlp.fc.load_state_dict(fc.state_dict())
    plan_mlp.proj.load_state_dict(proj.state_dict())
    parallelize_module(
        plan_mlp, mesh, {"fc": ColwiseParallel(), "proj": RowwiseParallel()}
    )
    # No activation between the two layers here: parallelize_module hands the
    # sharded hidden tensor straight from one submodule to the next, and the
    # point of this arm is the plan, not the activation.
    plan_out = plan_mlp(x)
    plan_out = plan_out.to_local() if hasattr(plan_out, "to_local") else plan_out
    plan_error = float((plan_out - proj(fc(x))).abs().max())

    # --- and against the hand-written implementation ----------------------
    manual_fc = ColumnParallelLinear(fc, rank, world_size)
    manual_proj = RowParallelLinear(proj, rank, world_size)
    manual = manual_proj(F.gelu(manual_fc(x)))
    manual_vs_dtensor = float((manual - y.to_local()).abs().max())
    manual_error = float((manual - reference).abs().max())

    return {
        "rank": rank,
        "mesh": [int(v) for v in mesh.mesh.flatten()],
        "dtensor_error": dtensor_error,
        "parallelize_module_error": plan_error,
        "manual_error": manual_error,
        "manual_vs_dtensor": manual_vs_dtensor,
        "output_scale": float(reference.abs().max()),
        "local_fc_weight_shape": list(d_fc_w.to_local().shape),
        "full_fc_weight_shape": list(fc.weight.shape),
        "local_proj_weight_shape": list(d_proj_w.to_local().shape),
        "full_proj_weight_shape": list(proj.weight.shape),
    }
