"""Measure all-reduce, all-gather and reduce-scatter across sizes and world sizes.

Three collectives, swept over message size and over world size, on whichever
backend the machine has. Reports algorithm bandwidth and bus bandwidth, fits
``t = latency + bytes / bandwidth`` to each collective, and then prices the
fabric cost model against the same measurements so the model's predictions can
be read next to reality rather than taken on trust.

On the machine this was written on that is gloo over TCP loopback, and the
bandwidth constant is a memory copy rather than an interconnect. On a CUDA node
it is NCCL over the real fabric, and the fitted link is what replaces the
datasheet entries in ``transformer_internals.cluster.fabric``.

Writes ``results/collective_bandwidth.json``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, write_json
from transformer_internals import hardware
from transformer_internals.cluster import collbench
from transformer_internals.cluster.fabric import (
    LINKS,
    link_from_measurement,
    predicted_vs_measured,
)
from transformer_internals.hardware import HardwareError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--world-sizes",
        default="2",
        help="comma-separated world sizes to sweep, e.g. 2,4,8",
    )
    ap.add_argument(
        "--min-log2-elements", type=int, default=12, help="smallest buffer, log2 elements"
    )
    ap.add_argument(
        "--max-log2-elements", type=int, default=24, help="largest buffer, log2 elements"
    )
    ap.add_argument("--iters", type=int, default=20, help="timed calls per point")
    ap.add_argument("--backend", default="auto", help="nccl, gloo or auto")
    ap.add_argument(
        "--allow-oversubscribe",
        action="store_true",
        help="permit more ranks than GPUs; correctness survives it, timings do not",
    )
    ap.add_argument("--compare-link", default="nvlink4", help="datasheet link to price against")
    ap.add_argument("--out", default=str(RESULTS / "collective_bandwidth.json"))
    args = ap.parse_args()

    caps = hardware.Capabilities.detect()
    requested = None if args.backend == "auto" else args.backend
    backend = hardware.select_backend(caps, requested)
    world_sizes = [int(w) for w in args.world_sizes.split(",") if w.strip()]
    sizes = [1 << k for k in range(args.min_log2_elements, args.max_log2_elements + 1, 2)]

    print(hardware.describe(caps))
    print(
        f"\nbackend {backend} | world sizes {world_sizes} | "
        f"{sizes[0] * 4:,} B to {sizes[-1] * 4:,} B | {args.iters} timed calls per point"
    )

    started = time.time()
    sweeps: dict[str, Any] = {}
    for world_size in world_sizes:
        if world_size < 2:
            print(f"  skipping world size {world_size}: a collective needs two ranks")
            continue
        print(f"\nworld size {world_size}")
        data = collbench.run(
            world_size=world_size,
            sizes=sizes,
            iters=args.iters,
            ops=collbench.OPS,
            backend=requested,
            caps=caps,
            allow_oversubscribe=args.allow_oversubscribe,
        )
        sweeps[str(world_size)] = data
        print(f"  {'collective':<16}{'peak busbw GB/s':>17}{'fit GB/s':>11}{'fit us':>9}{'R^2':>8}")
        for op, entry in data["by_op"].items():
            fit = entry.get("fit", {})
            print(
                f"  {op:<16}{entry['peak_bus_gbytes_per_s']:>17.3f}"
                f"{fit.get('bandwidth_gbytes_per_s', float('nan')):>11.3f}"
                f"{fit.get('latency_us', float('nan')):>9.1f}"
                f"{fit.get('r_squared', float('nan')):>8.4f}"
            )

    if not sweeps:
        print("nothing measured: every requested world size was below 2")
        return 1

    # The largest world size measured is the one worth turning into a link: a
    # ring's per-link rate is what the model wants, and it is best estimated
    # where the ring is longest.
    largest = max(sweeps, key=lambda k: int(k))
    reference = sweeps[largest]
    fit = reference["by_op"]["all_reduce"]["fit"]
    measured_link = link_from_measurement(
        name=f"{reference['backend']} on {reference['device']}, {largest} ranks",
        fit=fit,
        ranks=int(largest),
        op="all_reduce",
        peak_bus_gbytes_per_s=reference["by_op"]["all_reduce"]["peak_bus_gbytes_per_s"],
        inter_node=False,
        detail=f"world size {largest}, {args.iters} timed calls per point",
    )
    comparison = predicted_vs_measured(
        reference["by_op"]["all_reduce"]["points"], int(largest), measured_link
    )
    print(
        f"\nthe fitted link against the measurements it came from "
        f"(world size {largest}, all-reduce):"
    )
    print(f"  {'bytes':>14}{'measured ms':>14}{'modelled ms':>14}{'ratio':>9}")
    for row in comparison:
        print(
            f"  {row['bytes']:>14,}{row['measured_s'] * 1e3:>14.3f}"
            f"{row['modelled_s'] * 1e3:>14.3f}{row['modelled_over_measured']:>9.2f}"
        )

    datasheet = LINKS.get(args.compare_link)
    payload: dict[str, Any] = {
        "status": "MEASURED on this machine",
        "backend": backend,
        "sweeps": sweeps,
        "fitted_link": {
            "name": measured_link.name,
            "gbytes_per_s": measured_link.gbytes_per_s,
            "latency_us": measured_link.latency_us,
            "source": measured_link.source,
            "measured": measured_link.measured,
        },
        "model_vs_measurement": {
            "note": (
                "The model was fitted to these same points, so this is a "
                "goodness-of-form check and not a held-out prediction. The "
                "reading that matters is where the ratio departs from 1: an "
                "affine model should track the bandwidth-bound points closely "
                "and should be worst at the smallest messages, where the real "
                "cost is scheduling rather than latency or bandwidth."
            ),
            "rows": comparison,
        },
        "bus_bandwidth_convention": (
            "algorithm bandwidth times the ring factor: 2(n-1)/n for all-reduce, "
            "(n-1)/n for all-gather and reduce-scatter, with the size taken as "
            "the full unsharded buffer in all three cases. This is NCCL's own "
            "convention, so the numbers are comparable with nccl-tests."
        ),
        "environment": hardware.environment_payload(caps),
        "meta": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "world_sizes": world_sizes,
            "element_counts": sizes,
            "iterations_per_point": args.iters,
            "runtime_seconds": time.time() - started,
        },
    }
    if datasheet is not None:
        payload["datasheet_comparison"] = {
            "link": args.compare_link,
            "published_gbytes_per_s": datasheet.gbytes_per_s,
            "published_source": datasheet.source,
            "measured_gbytes_per_s": measured_link.gbytes_per_s,
            "measured_over_published": (
                measured_link.gbytes_per_s / datasheet.gbytes_per_s
                if datasheet.gbytes_per_s
                else float("nan")
            ),
            "note": (
                "Only meaningful when the measurement ran on hardware that has "
                f"a {args.compare_link} link. On a CPU machine the measured "
                "figure is loopback TCP and the ratio means nothing."
            ),
        }

    write_json(args.out, payload)
    if backend != "nccl":
        print(
            "\nbackend was gloo, so the bandwidth constants above are a memory copy "
            "over loopback and not an interconnect. The fitted SHAPE is the result; "
            "the constant is not."
        )
    return 0


if __name__ == "__main__":
    # A hardware or placement problem is one sentence, not a traceback. It is
    # the most likely way this script fails on a machine it has not run on
    # before, and a stack trace would bury the sentence that says what to do.
    try:
        raise SystemExit(main())
    except HardwareError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None
