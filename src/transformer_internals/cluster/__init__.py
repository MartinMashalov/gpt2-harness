"""Training-infrastructure half of the repository: the machinery around the model.

Four things live here, and all four are exercised by ``tests/test_cluster.py``
on this laptop, on CPU, with real ``torch.distributed`` processes on the gloo
backend:

* :mod:`~transformer_internals.cluster.checkpoint` -- sharded checkpoints that
  can be written under one parallel layout and read back under a different one.
* :mod:`~transformer_internals.cluster.streaming` -- a sharded, resumable
  streaming dataloader whose iterator position is part of the checkpoint.
* :mod:`~transformer_internals.cluster.failure` -- kill a rank mid-run, restart
  from the last checkpoint, and compare the resumed loss trajectory against the
  uninterrupted one.
* :mod:`~transformer_internals.cluster.fabric` -- an analytic interconnect cost
  model. Everything it prints is *modelled* from published bandwidths, not
  measured; the module says so in its own output.

:mod:`~transformer_internals.cluster.cgroups` is a diagnostic, not a library:
it reads the cgroup v2 limits the current process is running under.
"""

from transformer_internals.cluster.checkpoint import (
    AsyncCheckpointer,
    ShardSpec,
    gpt2_tp_plan,
    load_full,
    load_reshard,
    merge_pieces,
    save_sharded,
    split_tensor,
)
from transformer_internals.cluster.streaming import (
    ShardedStream,
    StreamState,
    TokenShardSource,
)

__all__ = [
    "AsyncCheckpointer",
    "ShardSpec",
    "ShardedStream",
    "StreamState",
    "TokenShardSource",
    "gpt2_tp_plan",
    "load_full",
    "load_reshard",
    "merge_pieces",
    "save_sharded",
    "split_tensor",
]
