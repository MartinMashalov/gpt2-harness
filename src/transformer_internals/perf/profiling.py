"""``torch.profiler`` around a real training step, reduced to a table and a trace.

What this produces:

* a **top-kernel table** sorted by self time, which is the time spent inside an
  operator and not inside its children. Total time double-counts: ``aten::linear``
  contains ``aten::addmm``, so summing total time over a call tree gives a number
  larger than the wall clock. Self time partitions the step, and is the only one
  of the two that can be turned into fractions.
* a **category breakdown**, which folds the hundred-odd distinct operators into
  the handful of classes that a decision can be made about: matmul, softmax,
  normalisation, elementwise, reduction, data movement, optimiser.
* a **Chrome trace**, written to ``results/`` and committed, so the claim can be
  checked by anyone with ``chrome://tracing`` or Perfetto instead of taken on
  faith.

Profiling runs on CPU. ``torch.profiler`` in torch 2.2 exposes CPU, CUDA, XPU and
MTIA activity sets and has no MPS backend, so on this machine an MPS run would
report only the host-side dispatch time, which is not the kernel time and would
be actively misleading. On CPU the operator self time *is* the kernel time, so
the breakdown is real. The consequence to keep in mind when reading the numbers:
they are the operator mix of this model on this CPU, and the mix on a GPU differs
mainly in that GEMMs get relatively faster, which makes the memory-bound tail
*larger* as a fraction, not smaller.

The profiler is scheduled rather than left running: ``wait`` steps let the
allocator settle, ``warmup`` steps prime the caches, and only the ``active``
steps are recorded. Profiling the first step of anything measures allocation.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT

__all__ = [
    "OP_CATEGORIES",
    "ProfileReport",
    "categorize_op",
    "profile_training_step",
]


#: Operator-name prefixes and the category each belongs to. Order matters: the
#: first matching substring wins, so the specific patterns come before the
#: general ones. Categories are chosen to line up with the roofline classes:
#: ``matmul`` is the compute-bound family, everything from ``softmax`` down is
#: the memory-bound family.
OP_CATEGORIES: list[tuple[str, str]] = [
    ("addmm", "matmul"),
    ("mm", "matmul"),
    ("bmm", "matmul"),
    ("matmul", "matmul"),
    ("linear", "matmul"),
    ("einsum", "matmul"),
    ("softmax", "softmax"),
    ("layer_norm", "normalization"),
    ("native_layer_norm", "normalization"),
    ("gelu", "elementwise"),
    ("tanh", "elementwise"),
    ("add", "elementwise"),
    ("mul", "elementwise"),
    ("div", "elementwise"),
    ("pow", "elementwise"),
    ("sub", "elementwise"),
    ("neg", "elementwise"),
    ("masked_fill", "elementwise"),
    ("where", "elementwise"),
    ("clamp", "elementwise"),
    ("sum", "reduction"),
    ("mean", "reduction"),
    ("max", "reduction"),
    ("nll_loss", "loss"),
    ("cross_entropy", "loss"),
    ("log_softmax", "softmax"),
    ("adamw", "optimizer"),
    ("foreach", "optimizer"),
    ("zero", "optimizer"),
    ("copy", "data movement"),
    ("contiguous", "data movement"),
    ("cat", "data movement"),
    ("view", "data movement"),
    ("reshape", "data movement"),
    ("transpose", "data movement"),
    ("permute", "data movement"),
    ("clone", "data movement"),
    ("empty", "allocation"),
    ("resize", "allocation"),
    ("fill", "allocation"),
    ("to", "data movement"),
    ("index", "embedding"),
    ("embedding", "embedding"),
    ("gather", "embedding"),
    ("scatter", "embedding"),
]


def categorize_op(name: str) -> str:
    """Map an operator name onto one of :data:`OP_CATEGORIES`.

    The name is lowercased and stripped of the ``aten::`` / ``autograd::``
    namespace and of the trailing ``Backward0`` that autograd node names carry,
    so a backward GEMM lands in ``matmul`` alongside its forward.
    """
    n = name.lower()
    for prefix in ("aten::", "autograd::engine::evaluate_function: ", "torch::", "profilerstep"):
        n = n.replace(prefix, "")
    n = n.replace("backward0", "").replace("backward", "").strip("_ ")
    for pattern, category in OP_CATEGORIES:
        if pattern in n:
            return category
    return "other"


@dataclass
class ProfileReport:
    """The reduced output of one profiled training step.

    Attributes:
        device: Where it ran.
        batch / seq: Step shape.
        active_steps: Steps the profiler recorded.
        total_self_us: Sum of self time over every operator, which is the
            profiler's partition of the recorded steps.
        top_ops: Top operators by self time, each with its share.
        categories: Self time folded into categories, each with its share.
        trace_path: Where the Chrome trace was written.
        profiler_flops: FLOPs the profiler attributed to matmul-family ops, when
            ``with_flops`` is on. Useful as an independent check on the analytic
            count in :mod:`~transformer_internals.perf.mfu`.
    """

    device: str
    batch: int
    seq: int
    active_steps: int
    total_self_us: float
    top_ops: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)
    trace_path: str | None = None
    profiler_flops: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def memory_bound_self_time_fraction(self) -> float:
        """Share of operator self time outside the matmul family.

        The roofline says these ops cannot reach peak arithmetic. This says how
        much of the step they take. A high number here with a low MFU is the
        signature of a model that is bandwidth-limited rather than badly
        scheduled.
        """
        non_matmul = sum(
            c["self_us"] for c in self.categories if c["category"] not in ("matmul", "other")
        )
        return non_matmul / self.total_self_us if self.total_self_us else 0.0

    def table(self, limit: int = 15) -> str:
        """Render the top-kernel table as fixed-width text."""
        lines = [f"{'operator':<42}{'self ms':>10}{'share':>9}{'calls':>8}  category"]
        lines.append("-" * 84)
        for row in self.top_ops[:limit]:
            lines.append(
                f"{row['name'][:41]:<42}{row['self_us'] / 1000:>10.2f}"
                f"{row['self_fraction'] * 100:>8.1f}%{row['count']:>8}  {row['category']}"
            )
        lines.append("")
        lines.append(f"{'category':<42}{'self ms':>10}{'share':>9}")
        lines.append("-" * 61)
        for row in self.categories:
            lines.append(
                f"{row['category']:<42}{row['self_us'] / 1000:>10.2f}"
                f"{row['self_fraction'] * 100:>8.1f}%"
            )
        return "\n".join(lines)


def profile_training_step(
    cfg: GPTConfig | None = None,
    batch: int = 4,
    seq: int = 256,
    device: torch.device | str = "cpu",
    active_steps: int = 2,
    wait_steps: int = 1,
    warmup_steps: int = 1,
    trace_path: str | Path | None = None,
    top_k: int = 20,
    model: GPT | None = None,
) -> ProfileReport:
    """Profile forward, backward and the optimiser update, and reduce the result.

    Args:
        cfg: Model config. Defaults to GPT-2 124M.
        batch / seq: Step shape.
        device: Where to run. CPU is the meaningful choice here, see the module
            docstring.
        active_steps: Steps actually recorded.
        wait_steps / warmup_steps: Steps skipped and steps run un-recorded first.
        trace_path: Where to write the Chrome trace. ``None`` skips it.
        top_k: How many operators to keep in the table.
        model: Reuse an existing model.

    Returns:
        A :class:`ProfileReport`.
    """
    device = torch.device(device)
    cfg = cfg or GPTConfig(n_positions=max(seq, 64))
    model = model if model is not None else GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sched = schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1)
    total_steps = wait_steps + warmup_steps + active_steps
    with profile(activities=activities, schedule=sched, record_shapes=True, with_flops=True) as prof:
        for _ in range(total_steps):
            opt.zero_grad(set_to_none=True)
            loss = model(x, targets=y)["loss"]
            loss.backward()
            opt.step()
            prof.step()

    events = [e for e in prof.key_averages() if e.self_cpu_time_total > 0]
    total_self = float(sum(e.self_cpu_time_total for e in events))
    events.sort(key=lambda e: e.self_cpu_time_total, reverse=True)

    top_ops = [
        {
            "name": e.key,
            "self_us": float(e.self_cpu_time_total),
            "self_fraction": float(e.self_cpu_time_total) / total_self if total_self else 0.0,
            "total_us": float(e.cpu_time_total),
            "count": int(e.count),
            "category": categorize_op(e.key),
        }
        for e in events[:top_k]
    ]

    by_cat: dict[str, float] = {}
    for e in events:
        by_cat[categorize_op(e.key)] = by_cat.get(categorize_op(e.key), 0.0) + float(
            e.self_cpu_time_total
        )
    categories = [
        {
            "category": k,
            "self_us": v,
            "self_fraction": v / total_self if total_self else 0.0,
        }
        for k, v in sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)
    ]

    written: str | None = None
    if trace_path is not None:
        p = Path(trace_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(p))
        written = str(p)

    flops = float(sum(getattr(e, "flops", 0) or 0 for e in events))

    return ProfileReport(
        device=str(device),
        batch=batch,
        seq=seq,
        active_steps=active_steps,
        total_self_us=total_self,
        top_ops=top_ops,
        categories=categories,
        trace_path=written,
        profiler_flops=flops,
        meta={
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "model_config": cfg.to_dict(),
            "note": "self time partitions the step; total time double counts parents",
        },
    )
