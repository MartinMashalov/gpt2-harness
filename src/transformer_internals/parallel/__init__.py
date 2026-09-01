"""Training parallelism, implemented on ``torch.distributed`` and proven correct.

Five strategies, each written from scratch against the gloo backend so it runs
multi-process on a CPU, and each shipped with a test that asserts numerical
equivalence to a single-process reference:

* :mod:`~transformer_internals.parallel.data_parallel` -- replicate, split the
  batch, all-reduce the gradients.
* :mod:`~transformer_internals.parallel.zero` -- shard gradients, optimizer
  state and (ZeRO-3) parameters.
* :mod:`~transformer_internals.parallel.tensor_parallel` -- split the GEMMs
  inside a block; column-parallel then row-parallel MLP, head-parallel
  attention.
* :mod:`~transformer_internals.parallel.pipeline_parallel` -- blocks on
  different ranks, GPipe and 1F1B micro-batch schedules, measured bubble.
* :mod:`~transformer_internals.parallel.sequence_parallel` -- shard the sequence
  axis; all-gather-KV and ring attention.

Plus :mod:`~transformer_internals.parallel.dtensor_demo`, which expresses the
tensor-parallel sharding through a device mesh instead of hand-written
collectives, and :mod:`~transformer_internals.parallel.comms`, which counts the
exact bytes every collective moves.
"""

from __future__ import annotations

__all__ = [
    "common",
    "comms",
    "data_parallel",
    "dtensor_demo",
    "pipeline_parallel",
    "sequence_parallel",
    "tensor_parallel",
    "zero",
]
