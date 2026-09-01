"""Check the machine before spending money on it, and exercise the CUDA logic without CUDA.

Two modes, and the second is the point.

**Live** (the default) asks the machine what it is: GPU count and names, compute
capability, driver, CUDA runtime, NCCL version, NVLink topology from
``nvidia-smi topo -m``. It then resolves every decision the measurement suite
will make -- which backend, which device per rank, which mixed-precision policy,
whether the requested world sizes are possible -- and fails loudly, once, before
anything expensive starts.

**Dry run** (``--dry-run``) does all of that against a *fabricated* eight-GPU
node. No CUDA is touched, nothing is measured, and the point is that the
decision logic which will run on rented hardware runs here first, on a laptop,
where a mistake costs nothing. Every branch it takes is a branch the real run
takes; only the two torch calls at the bottom of
:mod:`transformer_internals.hardware` are left unexercised, because they are the
only things that genuinely need a device.

The dry run also evaluates the parts of the analysis that are arithmetic rather
than measurement -- the activation-memory count and the fabric cost model at
GPU-sized shapes -- so an obviously wrong shape shows up before the box is
rented rather than at hour three.

Exit codes: 0 if the machine can run what was asked, 1 if it cannot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import write_json
from transformer_internals import hardware
from transformer_internals.cluster.fabric import (
    LLAMA70B,
    ParallelConfig,
    compute_time_s,
    step_costs,
)
from transformer_internals.config import GPTConfig
from transformer_internals.hardware import Capabilities, HardwareError
from transformer_internals.perf.activation_memory import analytic_activation_bytes
from transformer_internals.precision import resolve_amp


def _resolve(
    caps: Capabilities, world_sizes: list[int], backend_request: str | None
) -> dict[str, Any]:
    """Run every placement and precision decision, collecting failures rather than raising."""
    report: dict[str, Any] = {"backend": None, "placements": {}, "precision": {}, "problems": []}

    try:
        backend = hardware.select_backend(caps, backend_request)
        report["backend"] = backend
    except HardwareError as exc:
        report["problems"].append(f"backend: {exc}")
        return report

    for world_size in world_sizes:
        try:
            devices = hardware.check_placement(caps, world_size, backend)
            report["placements"][str(world_size)] = devices
        except HardwareError as exc:
            report["problems"].append(f"world size {world_size}: {exc}")

    device = "cuda" if backend == "nccl" else ("mps" if caps.mps_available else "cpu")
    for dtype in ("bf16", "fp16"):
        try:
            policy = resolve_amp(True, dtype, device, caps)
            report["precision"][dtype] = policy.to_dict()
        except HardwareError as exc:
            report["precision"][dtype] = {"available": False, "reason": str(exc)}
    return report


def _arithmetic_checks(caps: Capabilities) -> dict[str, Any]:
    """The parts of the analysis that need no hardware, evaluated at GPU-sized shapes.

    Activation memory is the one that would otherwise be discovered by an OOM at
    hour three. The count is exact against measurement on this machine, so
    evaluating it at a shape nobody has run is the intended use of it.
    """
    cfg = GPTConfig(dropout=0.0)  # GPT-2 124M
    shapes = []
    for batch, seq in ((8, 512), (16, 1024), (32, 1024)):
        terms = analytic_activation_bytes(cfg, batch, seq)
        row = {
            "batch": batch,
            "seq": seq,
            "activation_bytes_fp32": terms["total"],
            "activation_gib_fp32": terms["total"] / 1024**3,
            "activation_gib_bf16": analytic_activation_bytes(cfg, batch, seq, 2)["total"]
            / 1024**3,
        }
        if caps.total_memory_bytes:
            smallest = min(caps.total_memory_bytes) / 1024**3
            row["fits_in_one_gpu_bf16"] = row["activation_gib_bf16"] < 0.8 * smallest
            row["device_gib"] = smallest
        shapes.append(row)

    cfg8 = ParallelConfig(tp=8, dp=1)
    return {
        "note": (
            "Arithmetic, not measurement. The activation count is exact against "
            "measurement on this machine (tests/test_activation_memory.py), so "
            "evaluating it at an unrun shape is what it is for. The fabric rows "
            "are MODELLED from published bandwidths."
        ),
        "gpt2_124m_activation_memory": shapes,
        "modelled_70b_tp8_seconds_per_step": {
            "compute": compute_time_s(LLAMA70B, cfg8),
            "tp_over_nvlink4": step_costs(LLAMA70B, cfg8, "nvlink4")["tp"],
            "tp_over_ib_ndr": step_costs(LLAMA70B, cfg8, "ib_ndr")["tp"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve everything against a fabricated GPU node; touches no CUDA",
    )
    ap.add_argument(
        "--stub-gpus", type=int, default=8, help="how many GPUs the dry run should pretend to have"
    )
    ap.add_argument(
        "--stub-capability",
        default="8.0",
        help="compute capability for the dry run, e.g. 8.0 for A100 or 9.0 for H100",
    )
    ap.add_argument(
        "--world-sizes", default="2,4,8", help="comma-separated world sizes the run will use"
    )
    ap.add_argument("--backend", default="auto", help="nccl, gloo or auto")
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="exit non-zero when there is no usable CUDA. For CI on a GPU runner.",
    )
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    args = ap.parse_args()

    if args.dry_run:
        major, _, minor = args.stub_capability.partition(".")
        caps = Capabilities.stub(
            device_count=args.stub_gpus,
            capability=(int(major), int(minor or 0)),
        )
        topology = None
        print("=" * 72)
        print("DRY RUN. No CUDA call is made. Every decision below is resolved")
        print("against a fabricated machine, which is how the CUDA branches get")
        print("exercised on a laptop before any of them run on rented hardware.")
        print("=" * 72)
    else:
        caps = Capabilities.detect()
        topology = hardware.nvidia_smi("topo", "-m")

    print(hardware.describe(caps, topology))

    world_sizes = [int(w) for w in args.world_sizes.split(",") if w.strip()]
    requested = None if args.backend == "auto" else args.backend
    resolution = _resolve(caps, world_sizes, requested)

    print("\nresolved decisions")
    print(f"  backend                {resolution['backend'] or 'UNRESOLVED'}")
    for world_size, devices in resolution["placements"].items():
        shown = ", ".join(devices[:8]) + (" ..." if len(devices) > 8 else "")
        print(f"  world size {world_size:<11}{shown}")
    for dtype, policy in resolution["precision"].items():
        if policy.get("available") is False:
            print(f"  amp {dtype:<19}REFUSED: {policy['reason'].splitlines()[0]}")
        elif not policy["enabled"]:
            print(f"  amp {dtype:<19}not used here: {policy['reason'].splitlines()[0]}")
        else:
            print(
                f"  amp {dtype:<19}{policy['compute_dtype']} compute, "
                f"{policy['master_weight_dtype']} master, "
                f"scaler={policy['grad_scaler']}"
            )

    checks = _arithmetic_checks(caps)
    print("\nGPT-2 124M activation memory, from the exact count (arithmetic, not measured)")
    print(f"  {'batch':>6}{'seq':>6}{'fp32 GiB':>11}{'bf16 GiB':>11}{'fits':>8}")
    for row in checks["gpt2_124m_activation_memory"]:
        fits = row.get("fits_in_one_gpu_bf16")
        mark = "-" if fits is None else ("yes" if fits else "NO")
        print(
            f"  {row['batch']:>6}{row['seq']:>6}{row['activation_gib_fp32']:>11.2f}"
            f"{row['activation_gib_bf16']:>11.2f}{mark:>8}"
        )

    for problem in resolution["problems"]:
        print(f"\nPROBLEM: {problem}")

    ok = not resolution["problems"]
    if args.require_cuda and not caps.cuda_available:
        print("\nPROBLEM: --require-cuda was given and torch reports no CUDA device.")
        ok = False

    payload = {
        "mode": "dry_run" if args.dry_run else "live",
        "environment": hardware.environment_payload(caps, topology=not args.dry_run),
        "resolution": resolution,
        "arithmetic_checks": checks,
        "ok": ok,
    }
    if args.out:
        write_json(args.out, payload)
    elif args.dry_run:
        # A dry run's report is not a measurement, so it does not go in results/
        # by default. Print it instead, which is also what makes it diffable in
        # a CI log.
        print("\n" + json.dumps(payload["resolution"], indent=2))

    print(f"\npreflight: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
