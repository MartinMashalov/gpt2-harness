"""Roofline analysis: what this machine can do, and which transformer ops get near it.

The roofline model (Williams, Waterman & Patterson, CACM 2009) says a kernel's
attainable rate is bounded by two things at once::

    attainable FLOP/s = min(peak FLOP/s, arithmetic intensity * peak bytes/s)

where arithmetic intensity is FLOPs performed per byte moved to and from main
memory. The two bounds cross at the *ridge point*, ``peak FLOP/s / peak bytes/s``,
in FLOPs per byte. A kernel whose intensity sits left of the ridge cannot reach
peak arithmetic no matter how good its inner loop is, because it runs out of
bandwidth first. Right of the ridge, bandwidth is not the binding constraint.

Two halves to this module:

* **Measured.** :func:`measure_peak_flops` sweeps square GEMMs and takes the
  best rate any of them achieves. :func:`measure_peak_bandwidth` runs a
  STREAM-style triad (``a = b + s * c``) over arrays far larger than last-level
  cache and takes the best rate. Both are *achievable* peaks, not datasheet
  peaks, which is the honest denominator for a utilisation number.
* **Analytic.** :func:`op_roofline_table` counts FLOPs and compulsory memory
  traffic for every operator in a transformer block at a given shape, divides
  to get intensity, and classifies each one against the measured ridge point.

Byte counts are *compulsory* traffic: each input read once, each output written
once, at ``dtype_bytes`` per element. That is a lower bound on real traffic, so
the intensities here are upper bounds and the memory-bound verdicts are
conservative. Where an op is fused in practice (softmax over a tile that never
leaves cache) the real intensity is higher; the table says what the unfused
dataflow costs, which is what an eager-mode PyTorch model actually does.

Elementwise FLOP costs are counted explicitly in :data:`ELEMENTWISE_COST` rather
than waved at, because "how many FLOPs is a tanh" is a choice, not a fact. The
classification is insensitive to it: LayerNorm and softmax sit two orders of
magnitude below the ridge point, so doubling their FLOP count does not move them
across it.
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from transformer_internals.config import GPTConfig

__all__ = [
    "ELEMENTWISE_COST",
    "MachinePeak",
    "OpRoofline",
    "measure_machine_peak",
    "measure_op_rates",
    "measure_peak_bandwidth",
    "measure_peak_flops",
    "op_roofline_table",
    "roofline_payload",
]


#: FLOPs charged per element for the non-GEMM operators. Counted, not guessed:
#:
#: * ``gelu_tanh``: x^3 (2 mul), 0.044715 * x^3 (1), + x (1), * sqrt(2/pi) (1),
#:   tanh (counted as 1, it is a single hardware-or-libm transcendental),
#:   1 + t (1), 0.5 * x (1), * (1) = 9.
#: * ``layernorm``: sum for the mean (1), subtract (1), square (1), sum for the
#:   variance (1), rsqrt (1, amortised over the row and counted per element as
#:   part of it), multiply by rstd (1), scale (1), shift (1) = 8.
#: * ``softmax``: max reduction (1), subtract (1), exp (1), sum reduction (1),
#:   divide (1) = 5.
#: * ``residual_add``: one add = 1.
ELEMENTWISE_COST: dict[str, int] = {
    "gelu_tanh": 9,
    "layernorm": 8,
    "softmax": 5,
    "residual_add": 1,
}


def _synchronize(device: torch.device) -> None:
    """Block until queued device work has finished."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_best(fn: Any, device: torch.device, repeats: int, warmup: int) -> float:
    """Return the *minimum* wall clock over ``repeats`` calls, in seconds.

    The minimum, not the mean, on purpose. For a peak measurement the slow
    samples are contamination (a scheduler preemption, a thermal blip, another
    process), and the fastest observed run is the closest thing to what the
    hardware can do when nothing is in the way.
    """
    for _ in range(warmup):
        fn()
    _synchronize(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _synchronize(device)
        best = min(best, time.perf_counter() - t0)
    return best


@dataclass
class MachinePeak:
    """Achievable peaks measured on this machine, and the ridge point they imply.

    Attributes:
        device: Device string the measurement ran on.
        dtype: Element type used.
        peak_flops_per_s: Best rate any GEMM in the sweep achieved.
        peak_bytes_per_s: Best rate the STREAM triad achieved.
        ridge_flops_per_byte: ``peak_flops_per_s / peak_bytes_per_s``. Ops with a
            lower arithmetic intensity than this cannot be compute-bound here.
        gemm_sweep: Per-size GEMM results.
        bandwidth_sweep: Per-size triad results.
        meta: Platform and torch version.
    """

    device: str
    dtype: str
    peak_flops_per_s: float
    peak_bytes_per_s: float
    ridge_flops_per_byte: float
    gemm_sweep: list[dict[str, float]] = field(default_factory=list)
    bandwidth_sweep: list[dict[str, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def attainable_flops_per_s(self, intensity: float) -> float:
        """The roofline bound at a given arithmetic intensity."""
        return min(self.peak_flops_per_s, intensity * self.peak_bytes_per_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_peak_flops(
    device: torch.device | str = "cpu",
    sizes: tuple[int, ...] = (256, 512, 1024, 2048, 3072, 4096),
    dtype: torch.dtype = torch.float32,
    repeats: int = 5,
    warmup: int = 2,
) -> tuple[float, list[dict[str, float]]]:
    """Peak achievable FLOP/s, from a sweep of square GEMMs.

    A dense ``(N, N) @ (N, N)`` matmul is ``2 * N^3`` FLOPs and moves
    ``3 * N^2`` elements, so its arithmetic intensity is ``2N/3`` elements, i.e.
    it walks to the right along the roofline as N grows. Sweeping N and taking
    the best rate finds the point where the GEMM is large enough to amortise
    launch overhead and small enough to still tile well.

    Args:
        device: Where to run.
        sizes: Square GEMM dimensions to try.
        dtype: Element type.
        repeats: Timed repetitions per size.
        warmup: Untimed repetitions per size.

    Returns:
        ``(peak_flops_per_s, per_size_records)``.
    """
    device = torch.device(device)
    records: list[dict[str, float]] = []
    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        best = _time_best(lambda a=a, b=b: torch.matmul(a, b), device, repeats, warmup)
        flops = 2.0 * n**3
        records.append({"n": float(n), "seconds": best, "flops_per_s": flops / best})
        del a, b
    peak = max(r["flops_per_s"] for r in records)
    return peak, records


def measure_peak_bandwidth(
    device: torch.device | str = "cpu",
    sizes_mib: tuple[int, ...] = (16, 64, 256, 512),
    dtype: torch.dtype = torch.float32,
    repeats: int = 7,
    warmup: int = 2,
) -> tuple[float, list[dict[str, float]]]:
    """Peak achievable bytes/s, from a STREAM-style triad.

    ``a = b + s * c`` over three arrays: two read, one written, so the kernel
    moves ``3 * n * dtype_bytes`` bytes and does ``2 * n`` FLOPs. Its intensity
    is 1/6 FLOP per byte, which is so far left of any machine's ridge point that
    the measured rate is a bandwidth measurement and nothing else.

    Arrays are sized in MiB and swept upward so the result can be read for the
    cache cliff: the small sizes fit in cache and report a number that is not
    DRAM bandwidth. The peak returned is the best of the sweep, which on a cached
    machine is optimistic; the per-size records are kept so the cliff is visible.

    Args:
        device: Where to run.
        sizes_mib: Size of *each* of the three arrays, in MiB.
        dtype: Element type.
        repeats: Timed repetitions per size.
        warmup: Untimed repetitions per size.

    Returns:
        ``(peak_bytes_per_s, per_size_records)``.
    """
    device = torch.device(device)
    itemsize = torch.empty((), dtype=dtype).element_size()
    records: list[dict[str, float]] = []
    scalar = 3.0
    for mib in sizes_mib:
        n = (mib * 1024 * 1024) // itemsize
        b = torch.randn(n, device=device, dtype=dtype)
        c = torch.randn(n, device=device, dtype=dtype)
        a = torch.empty(n, device=device, dtype=dtype)

        def triad(a=a, b=b, c=c) -> None:
            torch.add(b, c, alpha=scalar, out=a)

        best = _time_best(triad, device, repeats, warmup)
        moved = 3.0 * n * itemsize
        records.append(
            {
                "array_mib": float(mib),
                "elements": float(n),
                "seconds": best,
                "bytes_per_s": moved / best,
            }
        )
        del a, b, c
    peak = max(r["bytes_per_s"] for r in records)
    return peak, records


def measure_machine_peak(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    gemm_sizes: tuple[int, ...] = (256, 512, 1024, 2048, 3072, 4096),
    stream_sizes_mib: tuple[int, ...] = (16, 64, 256, 512),
) -> MachinePeak:
    """Run both peak measurements and package them with the ridge point."""
    device = torch.device(device)
    peak_flops, gemm = measure_peak_flops(device, sizes=gemm_sizes, dtype=dtype)
    peak_bw, bw = measure_peak_bandwidth(device, sizes_mib=stream_sizes_mib, dtype=dtype)
    return MachinePeak(
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        peak_flops_per_s=peak_flops,
        peak_bytes_per_s=peak_bw,
        ridge_flops_per_byte=peak_flops / peak_bw,
        gemm_sweep=gemm,
        bandwidth_sweep=bw,
        meta={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
    )


@dataclass
class OpRoofline:
    """One transformer operator placed on the roofline.

    Attributes:
        name: Operator name.
        kind: ``"gemm"``, ``"batched_gemm"`` or ``"elementwise"``.
        flops: Analytic FLOP count at this shape.
        bytes_moved: Compulsory traffic in bytes (inputs read once, outputs
            written once, weights read once).
        intensity: ``flops / bytes_moved``, FLOPs per byte.
        bound: ``"compute"`` or ``"memory"``, decided against the ridge point.
        attainable_flops_per_s: The roofline bound at this intensity.
        roofline_seconds: ``flops / attainable_flops_per_s``, the fastest this op
            could run on this machine given its dataflow.
    """

    name: str
    kind: str
    flops: float
    bytes_moved: float
    intensity: float
    bound: str
    attainable_flops_per_s: float
    roofline_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _op_costs(
    cfg: GPTConfig, batch: int, seq: int, dtype_bytes: int = 4
) -> list[tuple[str, str, float, float]]:
    """Analytic ``(name, kind, flops, bytes)`` for one transformer block, forward.

    Shapes: ``B = batch``, ``T = seq``, ``C = n_embd``, ``H = n_head``,
    ``hd = C / H``. A matmul of ``(M, K) @ (K, N)`` is ``2 * M * K * N`` FLOPs;
    the 2 is one multiply and one add per accumulation.

    Weight traffic is counted once per operator call. For the shapes used here
    the activations dominate anyway, but at batch 1 the weights are most of the
    traffic, which is exactly why decoding a single sequence is memory-bound.
    """
    b, t, c = float(batch), float(seq), float(cfg.n_embd)
    h = float(cfg.n_head)
    kvh = float(cfg.kv_heads)
    hd = c / h
    d = float(dtype_bytes)
    act = b * t * c  # elements in one residual-stream-shaped activation
    scores = b * h * t * t  # elements in the attention score matrix
    kv_c = kvh * hd  # width of the k and v projections

    ops: list[tuple[str, str, float, float]] = []

    # LayerNorm before attention. Reads the stream, writes a normalised copy,
    # reads gain and bias (2C elements, negligible but counted).
    ops.append(
        (
            "ln_1 (LayerNorm)",
            "elementwise",
            ELEMENTWISE_COST["layernorm"] * act,
            (2 * act + 2 * c) * d,
        )
    )

    # Fused q/k/v projection: (B*T, C) @ (C, C + 2*kv_c).
    qkv_out = c + 2 * kv_c
    ops.append(
        (
            "qkv projection (GEMM)",
            "gemm",
            2 * b * t * c * qkv_out,
            (act + c * qkv_out + b * t * qkv_out) * d,
        )
    )

    # Attention scores: per head, (T, hd) @ (hd, T). Batched over B*H.
    ops.append(
        (
            "attention scores QK^T (batched GEMM)",
            "batched_gemm",
            2 * b * h * t * t * hd,
            (b * t * c + b * t * kv_c + scores) * d,
        )
    )

    # Causal mask + scale, then softmax over the last dim.
    ops.append(
        (
            "softmax (+ causal mask)",
            "elementwise",
            (ELEMENTWISE_COST["softmax"] + 2) * scores,
            (2 * scores) * d,
        )
    )

    # Attention-weighted value sum: per head, (T, T) @ (T, hd).
    ops.append(
        (
            "attention x V (batched GEMM)",
            "batched_gemm",
            2 * b * h * t * t * hd,
            (scores + b * t * kv_c + act) * d,
        )
    )

    ops.append(
        (
            "output projection (GEMM)",
            "gemm",
            2 * b * t * c * c,
            (act + c * c + act) * d,
        )
    )
    ops.append(
        (
            "residual add (attn)",
            "elementwise",
            ELEMENTWISE_COST["residual_add"] * act,
            (3 * act) * d,
        )
    )
    ops.append(
        (
            "ln_2 (LayerNorm)",
            "elementwise",
            ELEMENTWISE_COST["layernorm"] * act,
            (2 * act + 2 * c) * d,
        )
    )
    ops.append(
        (
            "MLP up 4x (GEMM)",
            "gemm",
            2 * b * t * c * 4 * c,
            (act + 4 * c * c + 4 * act) * d,
        )
    )
    ops.append(
        (
            "GELU (tanh)",
            "elementwise",
            ELEMENTWISE_COST["gelu_tanh"] * 4 * act,
            (2 * 4 * act) * d,
        )
    )
    ops.append(
        (
            "MLP down (GEMM)",
            "gemm",
            2 * b * t * 4 * c * c,
            (4 * act + 4 * c * c + act) * d,
        )
    )
    ops.append(
        (
            "residual add (mlp)",
            "elementwise",
            ELEMENTWISE_COST["residual_add"] * act,
            (3 * act) * d,
        )
    )
    return ops


def op_roofline_table(
    peak: MachinePeak,
    cfg: GPTConfig | None = None,
    batch: int = 8,
    seq: int = 512,
    dtype_bytes: int = 4,
) -> list[OpRoofline]:
    """Place every operator of one transformer block on the roofline.

    Args:
        peak: Measured machine peaks, which set the ridge point.
        cfg: Model config. Defaults to GPT-2 124M.
        batch: Sequences in the batch.
        seq: Sequence length.
        dtype_bytes: Bytes per activation element.

    Returns:
        One :class:`OpRoofline` per operator, in execution order.
    """
    cfg = cfg or GPTConfig()
    rows: list[OpRoofline] = []
    for name, kind, flops, byts in _op_costs(cfg, batch, seq, dtype_bytes):
        intensity = flops / byts
        attainable = peak.attainable_flops_per_s(intensity)
        rows.append(
            OpRoofline(
                name=name,
                kind=kind,
                flops=flops,
                bytes_moved=byts,
                intensity=intensity,
                bound="compute" if intensity >= peak.ridge_flops_per_byte else "memory",
                attainable_flops_per_s=attainable,
                roofline_seconds=flops / attainable,
            )
        )
    return rows


def measure_op_rates(
    cfg: GPTConfig | None = None,
    batch: int = 8,
    seq: int = 512,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    repeats: int = 5,
) -> list[dict[str, float]]:
    """Time the real kernels behind the analytic table, to check the model's shape.

    The analytic table above is arithmetic. This runs the actual PyTorch op at
    the same shape and reports the FLOP/s it achieves, so the two can be plotted
    together. A model whose predicted ordering does not survive contact with the
    measurement is a model that should not be trusted for the parts that cannot
    be measured here.

    Args:
        cfg: Model config. Defaults to GPT-2 124M.
        batch / seq: Shape.
        device: Where to run.
        dtype: Element type.
        repeats: Timed repetitions.

    Returns:
        One record per measured op with ``seconds`` and ``flops_per_s``.
    """
    cfg = cfg or GPTConfig()
    device = torch.device(device)
    b, t, c, h = batch, seq, cfg.n_embd, cfg.n_head
    hd = c // h
    x = torch.randn(b, t, c, device=device, dtype=dtype)
    w_qkv = torch.randn(c, 3 * c, device=device, dtype=dtype)
    w_mlp = torch.randn(c, 4 * c, device=device, dtype=dtype)
    q = torch.randn(b, h, t, hd, device=device, dtype=dtype)
    k = torch.randn(b, h, t, hd, device=device, dtype=dtype)
    scores = torch.randn(b, h, t, t, device=device, dtype=dtype)
    ln = torch.nn.LayerNorm(c, device=device, dtype=dtype)

    w_out = torch.randn(c, c, device=device, dtype=dtype)
    w_down = torch.randn(4 * c, c, device=device, dtype=dtype)
    h4 = torch.randn(b, t, 4 * c, device=device, dtype=dtype)
    v = torch.randn(b, h, t, hd, device=device, dtype=dtype)
    probs = torch.softmax(scores, dim=-1)

    cases: list[tuple[str, Any, float]] = [
        ("qkv projection (GEMM)", lambda: x @ w_qkv, 2.0 * b * t * c * 3 * c),
        ("output projection (GEMM)", lambda: x @ w_out, 2.0 * b * t * c * c),
        ("MLP down (GEMM)", lambda: h4 @ w_down, 2.0 * b * t * 4 * c * c),
        ("attention x V (batched GEMM)", lambda: probs @ v, 2.0 * b * h * t * t * hd),
        ("MLP up 4x (GEMM)", lambda: x @ w_mlp, 2.0 * b * t * c * 4 * c),
        (
            "attention scores QK^T (batched GEMM)",
            lambda: q @ k.transpose(-2, -1),
            2.0 * b * h * t * t * hd,
        ),
        (
            "softmax (+ causal mask)",
            lambda: torch.softmax(scores, dim=-1),
            (ELEMENTWISE_COST["softmax"] + 2) * float(b * h * t * t),
        ),
        ("ln_1 (LayerNorm)", lambda: ln(x), ELEMENTWISE_COST["layernorm"] * float(b * t * c)),
        (
            "GELU (tanh)",
            lambda: torch.nn.functional.gelu(x, approximate="tanh"),
            ELEMENTWISE_COST["gelu_tanh"] * float(b * t * c),
        ),
        ("residual add (attn)", lambda: x + x, float(b * t * c)),
    ]
    out: list[dict[str, float]] = []
    for name, fn, flops in cases:
        best = _time_best(fn, device, repeats=repeats, warmup=2)
        out.append({"op": name, "seconds": best, "flops_per_s": flops / best})
    return out


def roofline_payload(
    peak: MachinePeak,
    cfg: GPTConfig | None = None,
    batch: int = 8,
    seq: int = 512,
    dtype_bytes: int = 4,
    measured_ops: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON payload written to ``results/roofline.json``."""
    cfg = cfg or GPTConfig()
    rows = op_roofline_table(peak, cfg, batch=batch, seq=seq, dtype_bytes=dtype_bytes)
    total = sum(r.roofline_seconds for r in rows)
    mem = sum(r.roofline_seconds for r in rows if r.bound == "memory")
    return {
        "machine_peak": peak.to_dict(),
        "shape": {
            "batch": batch,
            "seq": seq,
            "dtype_bytes": dtype_bytes,
            "model_config": cfg.to_dict(),
        },
        "elementwise_flop_costs": ELEMENTWISE_COST,
        "ops": [r.to_dict() for r in rows],
        "summary": {
            "n_ops": len(rows),
            "n_memory_bound": sum(1 for r in rows if r.bound == "memory"),
            "n_compute_bound": sum(1 for r in rows if r.bound == "compute"),
            "flops_in_memory_bound_ops_fraction": sum(
                r.flops for r in rows if r.bound == "memory"
            )
            / sum(r.flops for r in rows),
            "roofline_time_in_memory_bound_ops_fraction": mem / total,
            "roofline_block_seconds": total,
        },
        "measured_op_rates": measured_ops or [],
    }


def summarize_sweep(records: list[dict[str, float]], key: str) -> dict[str, float]:
    """Min/median/max of one field across a sweep, for compact reporting."""
    vals = [r[key] for r in records]
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }
