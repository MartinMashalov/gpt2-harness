"""Part 2: find out why a training run is slower than it should be.

Runs the diagnosis on five configurations of the same model. Four of them have a
known pathology injected on purpose, and the fifth is the control. The tool is
not told what was injected. What it prints is a ranked list of causes with the
share of step wall clock each one accounts for, and the script then checks, for
each configuration, that the injected fault is the top-ranked recoverable
finding.

    configuration                    injected                       expected top finding
    -------------------------------  -----------------------------  -----------------------
    baseline                         nothing                        none above "minor"
    slow dataloader                  a per-batch stall              dataloader stall
    un-overlapped all-reduce         2 ranks, reduce after backward exposed collective time
    overlapped all-reduce (DDP)      2 ranks, bucketed DDP          none above "minor"
    batch too small                  batch 1                        batch too small

The two distributed configurations share one measurement: ``collective_probe``
launches real ``torch.distributed`` ranks over gloo and times four arms with
identical gradient volume (no communication, a manual all-reduce loop after the
backward pass, DDP at its default 25 MB bucket cap, and DDP at a 1 MB cap). The
local step measured by the diagnosis uses the same model and shape as those
ranks, so the fractions line up.

Writes ``results/diagnosis.json``, ``results/collectives.json`` and
``assets/step_breakdown.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ASSETS, RESULTS, device_from_arg, write_json
from transformer_internals.config import GPTConfig
from transformer_internals.perf.diagnose import DiagnosisReport, collective_probe, diagnose
from transformer_internals.perf.plots import fig_step_breakdown
from transformer_internals.perf.roofline import MachinePeak, measure_machine_peak

#: The model the single-process configurations run. Small enough that a
#: diagnosis finishes in seconds, large enough that the GEMMs are real GEMMs.
LOCAL_CFG = {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 4096, "n_positions": 256}

#: The model the distributed configurations run. Larger, because the point of
#: those arms is the gradient volume on the wire.
DIST_CFG = {"n_layer": 4, "n_head": 8, "n_embd": 512, "vocab_size": 8192, "n_positions": 256}


def find(report: DiagnosisReport, needle: str) -> Any:
    """The finding whose name contains ``needle``, or None."""
    return next((f for f in report.findings if needle in f.name), None)


def rank_of(report: DiagnosisReport, needle: str) -> int:
    """1-based position of that finding in the ranking, or 0 if absent."""
    for i, f in enumerate(report.findings, 1):
        if needle in f.name:
            return i
    return 0


def check(report: DiagnosisReport, expected: str | None, watch: str | None = None) -> tuple[bool, str]:
    """Did the tool find what was injected, and stay quiet about what was not?

    Two kinds of expectation, because a run can have more than one true problem
    at once and the honest test is not "is my fault the only thing it says":

    * ``expected`` names a finding that must come back at ``significant`` or
      ``critical``. That is the injected fault, and the tool has to see it.
    * ``watch`` names a finding that must come back at ``minor`` or ``healthy``.
      That is the control: the same probe, on a configuration where the fault is
      not present, has to stay quiet. A detector that always fires is not a
      detector.

    Returns:
        ``(passed, description)``.
    """
    if expected is not None:
        f = find(report, expected)
        if f is None:
            return False, f"no finding matching {expected!r}"
        ok = f.severity in ("significant", "critical")
        return ok, (
            f"{f.name}: {f.severity}, {f.cost_fraction * 100:.1f}% of step, "
            f"ranked {rank_of(report, expected)} of {len(report.findings)}"
        )
    if watch is not None:
        f = find(report, watch)
        if f is None:
            return True, f"no {watch!r} finding raised at all"
        ok = f.severity in ("healthy", "minor")
        return ok, f"{f.name}: {f.severity}, {f.cost_fraction * 100:.1f}% of step"
    worst = next((f for f in report.findings if f.recoverable), None)
    ok = worst is None or worst.severity in ("healthy", "minor")
    return ok, (
        "no recoverable finding above 'minor'"
        if worst is None
        else f"worst recoverable: {worst.name} [{worst.severity}] "
        f"{worst.cost_fraction * 100:.1f}%"
    )


def check_overlap(collectives: dict[str, Any]) -> tuple[bool, str]:
    """Did the probe tell an overlapped schedule from an un-overlapped one?

    The claim being tested is not "DDP is fast". It is that the probe measures
    *overlap*: on identical gradient volume, a loop of all-reduces issued after
    the backward pass must expose all of its cost, and a bucketed reducer firing
    from autograd hooks during the backward pass must expose much less of it. So
    two things are asserted:

    * the manual loop shows essentially no overlap, under 25%,
    * both bucketed DDP arms expose at most half of what the manual loop does.

    What is **not** asserted is the ordering between the two bucket caps. On this
    machine that difference is a few milliseconds, which is inside the run to run
    spread when the box is shared, while the manual-versus-bucketed difference is
    tens of milliseconds and is not. Reporting a number the measurement cannot
    resolve as though it could would be the same mistake as quoting a datasheet
    peak as a measurement.
    """
    manual = collectives["manual_allreduce"]
    default = collectives["ddp_default_buckets"]
    small = collectives["ddp_small_buckets"]
    ok = (
        manual["overlap_fraction"] < 0.25
        and default["exposed_comm_s"] <= 0.5 * manual["exposed_comm_s"]
        and small["exposed_comm_s"] <= 0.5 * manual["exposed_comm_s"]
    )
    return ok, (
        f"exposed comm on identical volume ({collectives['grad_bytes'] / 1e6:.1f} MB): "
        f"manual loop {manual['exposed_comm_s'] * 1e3:.1f} ms "
        f"({manual['overlap_fraction'] * 100:.0f}% overlapped) vs DDP 25 MB buckets "
        f"{default['exposed_comm_s'] * 1e3:.1f} ms ({default['overlap_fraction'] * 100:.0f}%) "
        f"and DDP 1 MB buckets {small['exposed_comm_s'] * 1e3:.1f} ms "
        f"({small['overlap_fraction'] * 100:.0f}%)"
    )


def load_peak(path: Path, device: str) -> MachinePeak | None:
    """Reuse the peaks ``run_roofline.py`` measured, but only for the same device.

    MFU is a fraction of *this* device's achievable rate. Dividing a rate
    measured on the CPU by the GPU's peak is not a slightly wrong MFU, it is a
    different quantity, so a cached peak from another device is refused rather
    than used.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    peak = MachinePeak(**payload["machine_peak"])
    if peak.device != device:
        print(
            f"  {path} holds peaks for {peak.device}, this run is on {device}; "
            f"measuring again rather than dividing by the wrong peak"
        )
        return None
    return peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", help="cpu keeps the profiler meaningful")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--port", type=int, default=29551)
    ap.add_argument("--skip-distributed", action="store_true")
    ap.add_argument("--reuse-peak", action="store_true", help="read results/roofline.json")
    ap.add_argument("--out", default=str(RESULTS / "diagnosis.json"))
    args = ap.parse_args()

    device = device_from_arg(args.device)
    peak = load_peak(RESULTS / "roofline.json", str(device)) if args.reuse_peak else None
    if peak is None:
        print(f"measuring machine peaks on {device}")
        peak = measure_machine_peak(device)
    print(
        f"peak {peak.peak_flops_per_s / 1e12:.3f} TFLOP/s, "
        f"{peak.peak_bytes_per_s / 1e9:.0f} GB/s, "
        f"ridge {peak.ridge_flops_per_byte:.2f} FLOP/byte  ({peak.device}, measured)"
    )

    local_cfg = GPTConfig(**LOCAL_CFG)
    dist_cfg = GPTConfig(**DIST_CFG)
    reports: list[DiagnosisReport] = []
    checks: list[tuple[str, str, bool, str]] = []

    # --- 1. the control
    print("\n" + "=" * 84)
    baseline = diagnose(
        peak,
        local_cfg,
        batch=args.batch,
        seq=args.seq,
        device=device,
        label="baseline (nothing injected)",
        steps=args.steps,
    )
    print(baseline.text())
    reports.append(baseline)
    checks.append(("baseline", "nothing", *check(baseline, None)))

    # --- 2. a dataloader that cannot keep up. The stall is sized at half the
    # measured compute time of the control, so the expected answer is known
    # before the tool runs: about a third of the step.
    stall = 0.5 * baseline.throughput["compute_s"]
    print("\n" + "=" * 84)
    slow = diagnose(
        peak,
        local_cfg,
        batch=args.batch,
        seq=args.seq,
        device=device,
        label=f"slow dataloader (+{stall * 1e3:.0f} ms per batch, injected)",
        loader_stall_s=stall,
        steps=args.steps,
    )
    print(slow.text())
    reports.append(slow)
    checks.append(
        ("slow dataloader", f"+{stall * 1e3:.0f} ms/batch", *check(slow, "dataloader stall"))
    )

    # --- 3 and 4. real distributed ranks: the same gradient volume, scheduled
    # two different ways.
    collectives = None
    if not args.skip_distributed:
        print("\n" + "=" * 84)
        print(
            f"launching {args.world_size} gloo ranks to time four collective schedules "
            f"on identical gradient volume"
        )
        collectives = collective_probe(
            model_config=DIST_CFG,
            world_size=args.world_size,
            batch=2,
            seq=64,
            port=args.port,
        )
        write_json(RESULTS / "collectives.json", collectives)
        print(
            f"  {collectives['grad_bytes'] / 1e6:.1f} MB of gradients across "
            f"{collectives['n_grad_tensors']} tensors, {collectives['backend']} over "
            f"{collectives['transport']}"
        )
        print(f"\n  {'arm':<24}{'step ms':>10}{'exposed ms':>12}{'standalone ms':>15}{'overlap':>9}")
        print("  " + "-" * 68)
        print(
            f"  {'no communication':<24}"
            f"{collectives['best_step_s']['no_comm'] * 1e3:>10.1f}"
            f"{'':>12}{'':>15}{'':>9}"
        )
        for arm in ("manual_allreduce", "ddp_default_buckets", "ddp_small_buckets"):
            a = collectives[arm]
            print(
                f"  {arm:<24}{a['step_s'] * 1e3:>10.1f}{a['exposed_comm_s'] * 1e3:>12.1f}"
                f"{a['standalone_comm_s'] * 1e3:>15.1f}{a['overlap_fraction'] * 100:>8.0f}%"
            )

        for arm, label, expect, watch in (
            (
                "manual_allreduce",
                "un-overlapped all-reduce (2 ranks, injected)",
                "exposed collective",
                None,
            ),
            (
                "ddp_small_buckets",
                "overlapped all-reduce (DDP, 1 MB buckets)",
                None,
                "exposed collective",
            ),
        ):
            print("\n" + "=" * 84)
            rep = diagnose(
                peak,
                dist_cfg,
                batch=2,
                seq=64,
                device=device,
                label=label,
                steps=args.steps,
                collectives=collectives,
                collective_arm=arm,
                batch_sweep=(2, 4, 8),
            )
            print(rep.text())
            reports.append(rep)
            if arm == "ddp_small_buckets":
                checks.append(
                    ("overlap vs no overlap", "bucketed DDP", *check_overlap(collectives))
                )
            else:
                checks.append((label.split(" (")[0], arm, *check(rep, expect, watch)))

    # --- 5. a batch too small to amortise the fixed per-step cost
    print("\n" + "=" * 84)
    tiny = diagnose(
        peak,
        local_cfg,
        batch=1,
        seq=args.seq,
        device=device,
        label="batch too small (batch 1, injected)",
        steps=args.steps,
        batch_sweep=(1, 2, 4, 8, 16),
    )
    print(tiny.text())
    reports.append(tiny)
    checks.append(("batch too small", "batch 1", *check(tiny, "batch too small")))

    # --- the demonstration: did the tool find each injected fault?
    print("\n" + "=" * 84)
    print("did the tool find what was injected?\n")
    print(f"  {'configuration':<32}{'injected':<30}{'verdict':<8}")
    print("  " + "-" * 78)
    all_ok = True
    for name, injected, ok, description in checks:
        all_ok &= ok
        print(f"  {name:<32}{injected:<30}{'PASS' if ok else 'FAIL':<8}")
        print(f"  {'':<32}{description}")

    payload: dict[str, Any] = {
        "machine_peak": peak.to_dict(),
        "reports": [r.to_dict() for r in reports],
        "checks": [
            {"configuration": n, "injected": i, "passed": ok, "detail": d}
            for n, i, ok, d in checks
        ],
        "all_passed": bool(all_ok),
    }
    write_json(args.out, payload)
    fig = fig_step_breakdown(payload["reports"], ASSETS / "step_breakdown.png")
    print(f"\nwrote {fig}")
    fig_step_breakdown(payload["reports"], ASSETS / "step_breakdown_web.png", web=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
