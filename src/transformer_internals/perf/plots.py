"""Figures for the performance work. One roofline, one step-time breakdown.

The house style lives in :mod:`transformer_internals.viz`; this module borrows it
so the performance figures sit next to the rest of the repository's plots without
looking like they came from somewhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from transformer_internals.viz import ACCENT, GRID, INK, MUTED, save, use_style

__all__ = ["SHORT_NAMES", "fig_roofline", "fig_step_breakdown"]


#: Compact labels, so twelve annotated points fit on one log-log axis.
SHORT_NAMES: dict[str, str] = {
    "ln_1 (LayerNorm)": "LayerNorm",
    "qkv projection (GEMM)": "QKV proj",
    "attention scores QK^T (batched GEMM)": "QK^T",
    "softmax (+ causal mask)": "softmax",
    "attention x V (batched GEMM)": "A·V",
    "output projection (GEMM)": "out proj",
    "residual add (attn)": "residual add",
    "ln_2 (LayerNorm)": "LayerNorm",
    "MLP up 4x (GEMM)": "MLP up",
    "GELU (tanh)": "GELU",
    "MLP down (GEMM)": "MLP down",
    "residual add (mlp)": "residual add",
}


def fig_roofline(payload: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """The roofline, with every transformer operator placed on it.

    The roof is two measured numbers: a slanted bandwidth bound of measured
    triad bytes per second, and a flat compute bound of the best rate any GEMM
    in the sweep reached. Filled markers are where the roofline says each
    operator can get to. Hollow markers are what the real PyTorch kernel
    achieved when it was timed at the same shape, which is what stops this being
    a drawing of an equation.

    Args:
        payload: Output of
            :func:`~transformer_internals.perf.roofline.roofline_payload`.
        path: Destination.
        web: Narrow-column variant.

    Returns:
        The path written.
    """
    use_style(1.7 if web else 1.0)
    peak = payload["machine_peak"]
    ops = payload["ops"]
    pf = peak["peak_flops_per_s"]
    pb = peak["peak_bytes_per_s"]
    ridge = peak["ridge_flops_per_byte"]

    fig, ax = plt.subplots(figsize=(5.2, 4.2) if web else (8.2, 5.0))

    xs = np.logspace(-1.2, 3.2, 400)
    roof = np.minimum(pf, xs * pb)
    ax.plot(xs, roof, color=INK, linewidth=1.6, zorder=3)
    ax.axvline(ridge, color=GRID, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        ridge * 0.92,
        pf * 5e-4,
        f"ridge {ridge:.1f} FLOP/byte",
        color=MUTED,
        fontsize=7.5,
        rotation=90,
        ha="right",
        va="bottom",
    )

    # One marker per distinct (intensity, name). Labels are stacked within a
    # cluster of nearby intensities, because six of the twelve operators land
    # within a factor of two of each other and unstacked text is unreadable.
    points: list[tuple[float, float, str, str]] = []
    seen: set[str] = set()
    for op in ops:
        label = SHORT_NAMES.get(op["name"], op["name"])
        if label in seen:
            continue
        seen.add(label)
        points.append((op["intensity"], op["attainable_flops_per_s"], label, op["bound"]))
    points.sort(key=lambda t: t[0])

    clusters: list[list[tuple[float, float, str, str]]] = []
    for pt in points:
        if clusters and abs(np.log10(pt[0]) - np.log10(clusters[-1][-1][0])) < 0.25:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    for cluster in clusters:
        for x, y, _label, bound in cluster:
            ax.plot(x, y, "o", color=ACCENT if bound == "compute" else INK, markersize=6, zorder=5)
        # Anchor the whole stack past the rightmost marker of the cluster, so a
        # label never lands on top of a neighbouring point.
        anchor_x = max(p[0] for p in cluster)
        anchor_y = max(p[1] for p in cluster)
        for i, (_x, _y, label, bound) in enumerate(cluster):
            ax.annotate(
                label,
                (anchor_x, anchor_y),
                textcoords="offset points",
                xytext=(11, (6 + 11 * i) if bound == "memory" else (-13 - 11 * i)),
                fontsize=7.5,
                color=ACCENT if bound == "compute" else INK,
                zorder=7,
            )

    by_name = {op["name"]: op for op in ops}
    measured = payload.get("measured_op_rates") or []
    first = True
    for m in measured:
        op = by_name.get(m["op"])
        if op is None:
            continue
        ax.plot(
            op["intensity"],
            m["flops_per_s"],
            marker="x",
            color=MUTED,
            markersize=7,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=6,
            label="measured kernel rate" if first else None,
        )
        first = False

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs per byte moved)")
    ax.set_ylabel("attainable rate (FLOP/s)")
    ax.set_title(
        f"Roofline, {peak['device']}: {pf / 1e12:.2f} TFLOP/s over {pb / 1e9:.0f} GB/s"
        if web
        else (
            f"Roofline, {peak['device']} / {peak['dtype']}: "
            f"{pf / 1e12:.2f} TFLOP/s over {pb / 1e9:.0f} GB/s measured"
        )
    )
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(pf * 3e-4, pf * 2.2)
    ax.grid(True, axis="y", which="both", color=GRID, linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    if measured:
        leg = ax.legend(loc="lower right", frameon=True)
        leg.get_frame().set_edgecolor(GRID)
        leg.get_frame().set_facecolor("#FFFFFF")

    shape = payload["shape"]
    if not web:
        ax.text(
            0.015,
            0.965,
            f"one transformer block, batch {shape['batch']} x seq {shape['seq']}, "
            f"fp32\nfilled: roofline bound   orange: compute-bound   dark: memory-bound",
            transform=ax.transAxes,
            fontsize=7.5,
            color=MUTED,
            va="top",
        )
    return save(fig, path)


def fig_step_breakdown(reports: list[dict[str, Any]], path: str | Path, web: bool = False) -> Path:
    """Stacked step time for each diagnosed configuration.

    One bar per configuration, split into the terms the diagnosis measured:
    dataloader stall, exposed collective time, and compute. The injected
    pathologies are visible as the bar that grew.

    Args:
        reports: Serialised :class:`~transformer_internals.perf.diagnose.DiagnosisReport`.
        path: Destination.
        web: Narrow-column variant.

    Returns:
        The path written.
    """
    use_style(1.7 if web else 1.0)
    labels = [r["label"].split(" (")[0] if web else r["label"] for r in reports]
    stall = np.array([r["throughput"]["fetch_s"] * 1e3 for r in reports])
    comm = np.array(
        [
            next(
                (
                    f["evidence"].get("exposed_comm_ms", 0.0)
                    for f in r["findings"]
                    if f["name"].startswith("exposed collective")
                ),
                0.0,
            )
            for r in reports
        ]
    )
    compute = np.array([r["throughput"]["compute_s"] * 1e3 for r in reports]) - comm
    compute = np.maximum(compute, 0.0)

    fig, ax = plt.subplots(figsize=(5.4, 3.6) if web else (7.6, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, compute, color=INK, height=0.55, label="compute")
    ax.barh(y, comm, left=compute, color=MUTED, height=0.55, label="exposed collectives")
    ax.barh(y, stall, left=compute + comm, color=ACCENT, height=0.55, label="dataloader stall")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("step time (ms)" if web else "step time (ms), fastest of the timed steps")
    ax.set_title("Where the step time goes" if web else "Where the step time goes, measured")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    for i, r in enumerate(reports):
        total = compute[i] + comm[i] + stall[i]
        ax.text(
            total * 1.01,
            i,
            f"{r['throughput']['mfu_end_to_end'] * 100:.0f}%"
            if web
            else f"{r['throughput']['mfu_end_to_end'] * 100:.1f}% MFU",
            va="center",
            fontsize=8.5 if web else 7.5,
            color=MUTED,
        )
    ax.set_xlim(0, float(max(compute + comm + stall)) * 1.22)
    return save(fig, path)
