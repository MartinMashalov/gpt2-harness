"""Training-performance tooling: roofline, model-FLOPs utilisation, profiling, diagnosis.

Four modules, in the order you would use them on a real run:

* :mod:`~transformer_internals.perf.roofline` measures what the machine can
  actually do (peak achievable FLOP/s from a GEMM sweep, peak achievable memory
  bandwidth from a STREAM triad) and places every transformer operator on the
  resulting roofline by its arithmetic intensity.
* :mod:`~transformer_internals.perf.mfu` counts the model FLOPs of a training
  step two ways (the 6ND rule of thumb and an exact per-layer count that keeps
  attention's sequence-quadratic term) and divides by measured wall clock.
* :mod:`~transformer_internals.perf.profiling` captures a real training step
  with ``torch.profiler`` and reduces it to a top-kernel table, a category
  breakdown, and a Chrome trace.
* :mod:`~transformer_internals.perf.diagnose` answers the operational question:
  this run is slower than it should be, why. It measures dataloader stall,
  exposed collective time and whether communication overlaps compute, MFU
  against the roofline ceiling, the memory-bound fraction of operator time, and
  whether the batch is large enough to saturate the machine, then ranks the
  findings by how much step time each one accounts for.

Everything here measures the machine it is run on. Numbers for hardware that is
not present (an A100, an H100) are computed by substituting a published peak
into the same arithmetic and are labelled ``modelled`` wherever they appear.
"""

# The ``diagnose`` *function* is deliberately not re-exported here. It would
# shadow the ``diagnose`` *module* on this package, so
# ``transformer_internals.perf.diagnose`` would resolve to the function and
# ``import transformer_internals.perf.diagnose as d`` would silently hand back
# something that is not a module. Import it from its module:
# ``from transformer_internals.perf.diagnose import diagnose``.
from transformer_internals.perf.diagnose import DiagnosisReport, Finding
from transformer_internals.perf.mfu import (
    MFUReport,
    flops_6nd,
    flops_per_token_exact,
    measure_step_mfu,
    mfu_on_published_gpu,
)
from transformer_internals.perf.profiling import (
    ProfileReport,
    profile_training_step,
)
from transformer_internals.perf.roofline import (
    MachinePeak,
    OpRoofline,
    measure_machine_peak,
    measure_peak_bandwidth,
    measure_peak_flops,
    op_roofline_table,
)

__all__ = [
    "DiagnosisReport",
    "Finding",
    "MFUReport",
    "MachinePeak",
    "OpRoofline",
    "ProfileReport",
    "flops_6nd",
    "flops_per_token_exact",
    "measure_machine_peak",
    "measure_peak_bandwidth",
    "measure_peak_flops",
    "measure_step_mfu",
    "mfu_on_published_gpu",
    "op_roofline_table",
    "profile_training_step",
]
