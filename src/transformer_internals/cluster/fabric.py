"""An analytic cost model for the interconnect, and what it implies about placement.

EVERY NUMBER THIS MODULE PRODUCES IS MODELLED, NOT MEASURED. There is no GPU in
this repository. The inputs are published peak bandwidths with their sources
recorded in :data:`LINKS`, plus two clearly-labelled efficiency assumptions. The
output is a per-step communication time for each parallelism strategy on each
fabric, and from those numbers -- not from an assertion -- the standard
placement rule falls out: tensor parallelism stays inside a node, data/sharded
data and pipeline parallelism cross between nodes.

The shape of the model is validated locally. ``cluster/collbench.py`` measures
gloo all-reduce on this laptop across message sizes and fits the same
``time = latency + bytes / bandwidth`` form; the fit is in ``docs/CLUSTER.md``.
That does not validate the H100 numbers -- nothing here can -- it validates that
the functional form used to extrapolate them describes a real collective.

The model
---------
Ring collectives over ``N`` ranks, each holding ``S`` bytes of the buffer:

* all-reduce  : ``2(N-1)/N * S / B  +  2(N-1) * lat``  (reduce-scatter then all-gather)
* all-gather  : ``(N-1)/N * S / B  +  (N-1) * lat``
* reduce-scatter: same as all-gather
* point-to-point: ``S / B + lat``

``B`` is the *achievable* per-GPU unidirectional bandwidth: published peak times
an efficiency factor. NCCL reaches roughly 80-90% of peak bus bandwidth on large
messages; 0.85 is used and is an assumption, not a citation.

What GPUDirect RDMA removes
---------------------------
Without it, a tensor leaving GPU 0 on node A for GPU 0 on node B is copied
device -> host bounce buffer (over PCIe), host -> NIC (over PCIe again), then
across the wire, then NIC -> host -> device on the far side. Two extra PCIe
crossings and two host memory copies per direction, plus the CPU is now in the
data path, so the latency floor is a kernel round-trip rather than a wire
round-trip. GPUDirect RDMA lets the NIC DMA straight out of (and into) the GPU's
BAR-mapped memory: the payload crosses PCIe once per side, never touches host
memory, and the CPU only posts the work request. That is why it matters most for
small messages -- the fixed cost is what it removes -- and why the NIC has to sit
under the same PCIe switch as the GPU for it to be fast. The model exposes this
as :data:`GPUDIRECT_PENALTY`, applied to the inter-node fabrics when it is off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "GPUDIRECT_PENALTY",
    "LINKS",
    "Link",
    "ModelShape",
    "ParallelConfig",
    "all_reduce_time",
    "compute_time_s",
    "format_report",
    "link_from_measurement",
    "predicted_vs_measured",
    "step_costs",
]


@dataclass(frozen=True)
class Link:
    """A published interconnect figure and where it comes from.

    Attributes:
        name: Human name.
        gbytes_per_s: Peak *unidirectional* bandwidth available to one GPU, in
            GB/s (10^9 bytes/s, which is how vendors quote it).
        latency_us: Small-message latency. ASSUMED, order-of-magnitude only;
            vendors do not publish a single comparable number.
        source: Where ``gbytes_per_s`` comes from.
        inter_node: Whether traffic on this link leaves the chassis.
        measured: True when the numbers came from
            :mod:`transformer_internals.cluster.collbench` running on this
            machine rather than from a datasheet. Every report prints it,
            because a modelled link and a measured one are different claims and
            the difference must survive being copied into a table.
        efficiency_applied: True when ``gbytes_per_s`` is already an achieved
            rate and :data:`LINK_EFFICIENCY` must therefore *not* be applied to
            it again. Measured links set this; published peaks do not.
    """

    name: str
    gbytes_per_s: float
    latency_us: float
    source: str
    inter_node: bool
    measured: bool = False
    efficiency_applied: bool = False


#: Published peak bandwidths. The per-GPU figures are unidirectional: NVIDIA
#: quotes NVLink as 900 GB/s "bidirectional aggregate", which is 450 GB/s each
#: way, and that is the number a ring collective actually gets per direction.
LINKS: dict[str, Link] = {
    "nvlink4": Link(
        "NVLink 4 / NVSwitch (H100 SXM)", 450.0, 2.0,
        "NVIDIA H100 datasheet: 18 NVLink 4 links x 25 GB/s per direction = "
        "900 GB/s bidirectional aggregate per GPU; NVSwitch gives every GPU in "
        "the 8-GPU node that bandwidth all-to-all.",
        inter_node=False,
    ),
    "nvlink3": Link(
        "NVLink 3 (A100 SXM)", 300.0, 2.0,
        "NVIDIA A100 datasheet: 12 NVLink 3 links x 25 GB/s per direction = "
        "600 GB/s bidirectional aggregate per GPU.",
        inter_node=False,
    ),
    "pcie5": Link(
        "PCIe Gen5 x16", 63.0, 5.0,
        "PCI-SIG: Gen5 is 32 GT/s per lane, 128b/130b encoded; x16 is ~63 GB/s "
        "per direction. This is the ceiling for a PCIe-attached GPU with no NVLink.",
        inter_node=False,
    ),
    "ib_ndr": Link(
        "InfiniBand NDR 400G (per GPU)", 50.0, 2.0,
        "NVIDIA DGX H100: 8 x ConnectX-7 NDR400 compute NICs, 400 Gb/s = 50 GB/s "
        "each, one per GPU, so 50 GB/s per GPU and 400 GB/s per node.",
        inter_node=True,
    ),
    "ib_hdr": Link(
        "InfiniBand HDR 200G (per GPU)", 25.0, 2.0,
        "NVIDIA DGX A100: 8 x HDR200 compute NICs, 200 Gb/s = 25 GB/s each.",
        inter_node=True,
    ),
    "roce200": Link(
        "RoCE v2 over 200 GbE", 25.0, 5.0,
        "200 GbE line rate = 25 GB/s. RoCE v2 puts the same RDMA verbs on "
        "Ethernet; the wire rate matches HDR but it needs a lossless fabric "
        "(PFC/ECN) to get near it, and its small-message latency is higher.",
        inter_node=True,
    ),
}

#: Fraction of peak a large-message ring collective actually achieves.
#: ASSUMPTION, not a published figure. NCCL bus bandwidth typically lands at
#: 80-90% of peak for buffers of tens of MB and well below it under 1 MB.
LINK_EFFICIENCY = 0.85

#: Multiplier applied to inter-node time when GPUDirect RDMA is unavailable, so
#: the payload has to be staged through host memory. ASSUMPTION: the extra PCIe
#: hop and host copy roughly halve achievable bandwidth and add several
#: microseconds. Modelled as 1.6x time and +5 us.
GPUDIRECT_PENALTY = (1.6, 5.0)

#: H100 SXM dense BF16 tensor-core throughput, TFLOP/s. NVIDIA's datasheet
#: headline of 1979 TFLOPS is with structured sparsity; dense is half of it.
PEAK_BF16_TFLOPS = 989.0
#: Fraction of peak a good transformer training step reaches. ASSUMPTION.
#: Published large-scale runs report 35-55% MFU.
ASSUMED_MFU = 0.45


@dataclass(frozen=True)
class ModelShape:
    """The model and per-step token count the cost is computed for."""

    n_layer: int
    n_embd: int
    n_head: int
    vocab_size: int
    seq_len: int
    micro_batch: int
    grad_accum: int = 1
    bytes_per_elem: int = 2  # bf16 activations and gradients

    @property
    def params(self) -> int:
        """Parameter count of a decoder-only transformer of this shape.

        ``12 * L * h^2`` for the blocks (4h^2 attention + 8h^2 MLP) plus the
        embedding and the untied head.
        """
        h = self.n_embd
        return 12 * self.n_layer * h * h + 2 * self.vocab_size * h

    @property
    def tokens_per_microbatch(self) -> int:
        return self.micro_batch * self.seq_len


@dataclass(frozen=True)
class ParallelConfig:
    """A 4-D parallel layout. ``tp * pp * cp * dp`` is the GPU count."""

    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.cp * self.dp


def link_from_measurement(
    name: str,
    fit: dict[str, float],
    ranks: int,
    op: str = "all_reduce",
    peak_bus_gbytes_per_s: float | None = None,
    inter_node: bool = False,
    detail: str = "",
) -> Link:
    """Build a :class:`Link` from a measured collective sweep.

    This is how the datasheet numbers get replaced. Run
    :func:`transformer_internals.cluster.collbench.run` on the hardware, hand
    its all-reduce fit here, and every cost in this module is then computed from
    a number that came off a wire.

    The bandwidth taken is the *fitted* one, not the peak of the sweep, because
    the model's form is ``time = latency + bytes / bandwidth`` and the fitted
    slope is that model's own bandwidth parameter. The peak is carried in the
    source string so the two can be compared.

    ``efficiency_applied`` is set, so :data:`LINK_EFFICIENCY` is not applied on
    top: the measurement already *is* the achieved rate, and multiplying an
    achieved rate by an efficiency assumption would be counting the same
    derating twice.

    The fitted intercept is divided by the number of ring hops, and this matters
    by a factor of ``2(n-1)``. ``Link.latency_us`` is a **per-hop** latency,
    because :func:`all_reduce_time` multiplies it by the hop count; the
    intercept of a fit to whole-collective times is the latency of the whole
    collective. Carrying one across as the other makes the model overpredict
    small messages by six times at eight ranks, which is exactly the regime
    where latency is the only thing that matters.

    Args:
        name: Human name for the link.
        fit: A fit from :func:`collbench.fit_link_model`, carrying
            ``bandwidth_gbytes_per_s``, ``latency_us`` and ``r_squared``.
        ranks: World size the sweep ran at, needed for the hop count.
        op: Which collective was fitted, also for the hop count.
        peak_bus_gbytes_per_s: Best bus bandwidth seen in the sweep, for the
            record.
        inter_node: Whether this link leaves the chassis.
        detail: Extra provenance, such as the backend and the world size.

    Returns:
        A :class:`Link` with ``measured=True``.
    """
    hops = max(_ring_hops(op, ranks), 1)
    per_hop_us = max(float(fit["latency_us"]), 0.0) / hops
    peak = f", peak bus {peak_bus_gbytes_per_s:.2f} GB/s" if peak_bus_gbytes_per_s else ""
    return Link(
        name=name,
        gbytes_per_s=float(fit["bandwidth_gbytes_per_s"]),
        latency_us=per_hop_us,
        source=(
            f"MEASURED by transformer_internals.cluster.collbench, {op} at {ranks} ranks. "
            f"Fitted t = {fit['latency_us']:.1f}us + "
            f"bytes/{fit['bandwidth_gbytes_per_s']:.2f} GB/s, "
            f"R^2 = {fit['r_squared']:.4f}{peak}. Latency divided by {hops} ring hops "
            f"to give the per-hop figure this model uses. {detail}".strip()
        ),
        inter_node=inter_node,
        measured=True,
        efficiency_applied=True,
    )


def _ring_hops(op: str, ranks: int) -> int:
    """Ring hops a collective takes: ``2(n-1)`` for all-reduce, ``n-1`` otherwise."""
    if ranks < 2:
        return 0
    return 2 * (ranks - 1) if op == "all_reduce" else ranks - 1


def predicted_vs_measured(
    points: list[dict[str, Any]],
    ranks: int,
    link: Link,
    op: str = "all_reduce",
) -> list[dict[str, Any]]:
    """Check the cost model against the collective sweep it was fitted to.

    A model fitted to data and then evaluated on the same data is not a
    prediction, so the useful reading is the *shape* of the residual: an affine
    model should track a bandwidth-bound collective closely and should be worst
    at the smallest messages, where the real cost is scheduling rather than
    either latency or bandwidth. That failure is the model's own point, and this
    function makes it visible rather than averaging it away.

    Args:
        points: Measurement records from
            :func:`transformer_internals.cluster.collbench.run`, each with
            ``bytes`` and ``median_s``.
        ranks: World size the sweep ran at.
        link: The link to price against.
        op: Which collective the points are.

    Returns:
        One row per point: the measured time, the modelled time and their ratio.
    """
    timer = {
        "all_reduce": all_reduce_time,
        "all_gather": all_gather_time,
        "reduce_scatter": all_gather_time,
    }[op]
    rows: list[dict[str, Any]] = []
    for point in points:
        modelled = timer(float(point["bytes"]), ranks, link)
        measured = float(point["median_s"])
        rows.append(
            {
                "op": op,
                "bytes": point["bytes"],
                "measured_s": measured,
                "modelled_s": modelled,
                "modelled_over_measured": modelled / measured if measured else float("nan"),
            }
        )
    return rows


def _eff_bw(link: Link, gpudirect: bool = True) -> float:
    # A measured link already carries an achieved rate; applying the efficiency
    # assumption to it as well would derate the same number twice.
    factor = 1.0 if link.efficiency_applied else LINK_EFFICIENCY
    bw = link.gbytes_per_s * factor * 1e9
    if link.inter_node and not gpudirect:
        bw /= GPUDIRECT_PENALTY[0]
    return bw


def _lat_s(link: Link, gpudirect: bool = True) -> float:
    lat = link.latency_us
    if link.inter_node and not gpudirect:
        lat += GPUDIRECT_PENALTY[1]
    return lat * 1e-6


def all_reduce_time(nbytes: float, ranks: int, link: Link, *, gpudirect: bool = True) -> float:
    """Ring all-reduce: reduce-scatter then all-gather, so 2(N-1)/N of the buffer."""
    if ranks < 2:
        return 0.0
    return 2 * (ranks - 1) / ranks * nbytes / _eff_bw(link, gpudirect) + 2 * (ranks - 1) * _lat_s(link, gpudirect)


def all_gather_time(nbytes: float, ranks: int, link: Link, *, gpudirect: bool = True) -> float:
    if ranks < 2:
        return 0.0
    return (ranks - 1) / ranks * nbytes / _eff_bw(link, gpudirect) + (ranks - 1) * _lat_s(link, gpudirect)


def p2p_time(nbytes: float, link: Link, *, gpudirect: bool = True) -> float:
    return nbytes / _eff_bw(link, gpudirect) + _lat_s(link, gpudirect)


def compute_time_s(shape: ModelShape, cfg: ParallelConfig, *, mfu: float = ASSUMED_MFU) -> float:
    """Modelled compute time for one optimiser step across the whole job.

    ``6 * P * T`` for forward and backward through the weights, plus
    ``12 * L * h * T * s`` for the attention score/value matmuls, which the
    parameter term does not cover and which is what makes long context
    expensive. ``T`` is the global tokens in the step.
    """
    tokens = shape.tokens_per_microbatch * shape.grad_accum * cfg.dp
    flops = 6 * shape.params * tokens + 12 * shape.n_layer * shape.n_embd * tokens * shape.seq_len
    return flops / (cfg.world_size * PEAK_BF16_TFLOPS * 1e12 * mfu)


def step_costs(
    shape: ModelShape, cfg: ParallelConfig, link_key: str, *, gpudirect: bool = True
) -> dict[str, float]:
    """Modelled seconds of communication per optimiser step, by strategy.

    Each entry answers "if *this* axis of parallelism ran over *this* link, what
    would it cost per step?" Volumes:

    * ``tp``: 2 all-reduces of the ``[tokens, h]`` activation per layer in the
      forward pass (after attention, after the MLP) and 2 more in the backward.
      This is per *microbatch*, so it is paid ``grad_accum`` times per step, and
      it scales with tokens, not with parameters. That is the whole story of why
      TP cannot leave the node.
    * ``fsdp``: ZeRO-3. All-gather the parameters in the forward, all-gather
      again in the backward, reduce-scatter the gradients: three passes over the
      parameter bytes per step, once per step regardless of ``grad_accum``.
    * ``ddp``: one all-reduce of the gradients per step.
    * ``pp``: activations across each stage boundary, forward and backward, once
      per microbatch. Point-to-point, not a collective.
    * ``cp``: ring attention. Each layer passes K and V around the context group,
      forward and backward.
    """
    link = LINKS[link_key]
    h = shape.n_embd
    b = shape.bytes_per_elem
    out: dict[str, float] = {}

    # The all-reduce is over the full [tokens, h] activation: each TP rank
    # computed a partial sum of the whole tensor, not a slice of it.
    act_bytes = shape.tokens_per_microbatch * h * b
    out["tp"] = (
        4 * shape.n_layer * shape.grad_accum
        * all_reduce_time(act_bytes, cfg.tp, link, gpudirect=gpudirect)
        if cfg.tp > 1 else 0.0
    )

    # Parameters actually held by this rank: sharded by tp and pp.
    local_params = shape.params / max(cfg.tp * cfg.pp, 1)
    param_bytes = local_params * b
    out["fsdp"] = 3 * all_gather_time(param_bytes, cfg.dp, link, gpudirect=gpudirect) if cfg.dp > 1 else 0.0
    out["ddp"] = all_reduce_time(param_bytes, cfg.dp, link, gpudirect=gpudirect) if cfg.dp > 1 else 0.0

    boundary_bytes = shape.tokens_per_microbatch * h * b / max(cfg.tp, 1)
    out["pp"] = (
        2 * (cfg.pp - 1) * shape.grad_accum * p2p_time(boundary_bytes, link, gpudirect=gpudirect)
        if cfg.pp > 1 else 0.0
    )

    kv_bytes = shape.tokens_per_microbatch * h * b / (max(cfg.cp, 1) * max(cfg.tp, 1)) * 2
    out["cp"] = (
        2 * shape.n_layer * shape.grad_accum * (cfg.cp - 1)
        * p2p_time(kv_bytes, link, gpudirect=gpudirect)
        if cfg.cp > 1 else 0.0
    )
    return out


# ------------------------------------------------------------------- report


LLAMA70B = ModelShape(
    n_layer=80, n_embd=8192, n_head=64, vocab_size=128256, seq_len=8192,
    micro_batch=1, grad_accum=8,
)
GPT2_124M = ModelShape(
    n_layer=12, n_embd=768, n_head=12, vocab_size=50257, seq_len=1024,
    micro_batch=8, grad_accum=8,
)


def format_report(shape: ModelShape = LLAMA70B, label: str = "70B-class, 8k context") -> str:
    """The table that makes the placement argument from the numbers."""
    lines: list[str] = []
    w = lines.append
    w("MODELLED, NOT MEASURED. Bandwidths are published peaks (sources below);")
    w(f"efficiency {LINK_EFFICIENCY}, MFU {ASSUMED_MFU} and the latencies are assumptions.")
    w("")
    w(f"Model: {label}")
    w(f"  layers {shape.n_layer}, d_model {shape.n_embd}, vocab {shape.vocab_size}, "
      f"seq {shape.seq_len}")
    w(f"  parameters {shape.params/1e9:.1f}e9, bf16")
    w(f"  micro-batch {shape.micro_batch} x {shape.grad_accum} accumulation steps")
    w("")

    cfg = ParallelConfig(tp=8, pp=1, cp=1, dp=8)
    compute = compute_time_s(shape, cfg)
    w(f"Layout under test: tp={cfg.tp} pp={cfg.pp} cp={cfg.cp} dp={cfg.dp} "
      f"-> {cfg.world_size} GPUs")
    w(f"Modelled compute per step: {compute*1e3:.1f} ms "
      f"(H100 SXM dense bf16 {PEAK_BF16_TFLOPS} TFLOP/s x MFU {ASSUMED_MFU})")
    w("")
    header = f"{'strategy':<10}" + "".join(f"{k:>14}" for k in ("nvlink4", "pcie5", "ib_ndr", "roce200"))
    w("Per-step communication time, milliseconds (and as a fraction of compute):")
    w(header)
    for strategy in ("tp", "fsdp", "ddp", "pp", "cp"):
        row = f"{strategy:<10}"
        for key in ("nvlink4", "pcie5", "ib_ndr", "roce200"):
            t = step_costs(shape, cfg, key)[strategy]
            row += f"{t*1e3:>9.1f} ({t/compute:>4.2f}x)" if t else f"{'-':>14}"
        w(row)
    w("")

    tp_nv = step_costs(shape, cfg, "nvlink4")["tp"]
    tp_ib = step_costs(shape, cfg, "ib_ndr")["tp"]
    fsdp_ib = step_costs(shape, cfg, "ib_ndr")["fsdp"]
    w("What the numbers say:")
    w(f"  * Tensor parallel moves {4*shape.n_layer*shape.grad_accum} all-reduces of "
      f"{shape.tokens_per_microbatch*shape.n_embd*shape.bytes_per_elem/1e6:.1f} MB per step.")
    w(f"    On NVLink that is {tp_nv*1e3:.1f} ms ({tp_nv/compute:.2f}x compute).")
    w(f"    On InfiniBand NDR it is {tp_ib*1e3:.1f} ms ({tp_ib/compute:.2f}x compute), "
      f"{tp_ib/tp_nv:.1f}x worse.")
    w(f"    TP therefore stays inside the NVLink domain. Not a rule of thumb: a "
      f"{tp_ib/compute:.1f}x")
    w("    communication-to-compute ratio means the GPUs would be idle most of the step.")
    w(f"  * Sharded data parallel moves 3 passes over the {shape.params/1e9/cfg.tp:.1f}e9 "
      "parameters this rank")
    w(f"    holds, once per step. On InfiniBand NDR: {fsdp_ib*1e3:.1f} ms "
      f"({fsdp_ib/compute:.2f}x compute), and it")
    w("    overlaps with the backward pass, so it crosses nodes without stalling them.")
    w(f"  * The ratio TP:FSDP per step here is {tp_ib/fsdp_ib:.1f}:1 on the same fabric. "
      "That ratio,")
    w("    not tradition, is what pins TP inside the node.")
    w("")
    w("Same TP layout with GPUDirect RDMA switched off (staged through host memory):")
    off = step_costs(shape, cfg, "ib_ndr", gpudirect=False)["tp"]
    w(f"  tp over ib_ndr: {tp_ib*1e3:.1f} ms -> {off*1e3:.1f} ms ({off/tp_ib:.2f}x)")
    w("")
    w("Published bandwidths used (peak, unidirectional, per GPU):")
    for key, link in LINKS.items():
        w(f"  {key:<9} {link.gbytes_per_s:>6.1f} GB/s  {link.name}")
        w(f"            source: {link.source}")
    return "\n".join(lines)


def crossover_ranks(shape: ModelShape, link_key: str, *, max_tp: int = 64) -> int | None:
    """Smallest TP degree at which TP communication exceeds compute on this link.

    Returned as a number rather than argued for: it is the point past which the
    step is bound by the fabric.
    """
    for tp in (2, 4, 8, 16, 32, 64):
        if tp > max_tp:
            break
        cfg = ParallelConfig(tp=tp, dp=max(1, 64 // tp))
        if step_costs(shape, cfg, link_key)["tp"] > compute_time_s(shape, cfg):
            return tp
    return None


if __name__ == "__main__":
    print(format_report())
    print()
    for key in ("nvlink4", "ib_ndr", "roce200"):
        n = crossover_ranks(LLAMA70B, key)
        print(f"TP becomes communication-bound on {key:<8} at tp={n}" if n
              else f"TP stays compute-bound on {key} up to tp=64")
