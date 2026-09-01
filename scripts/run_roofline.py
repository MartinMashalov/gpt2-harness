"""Part 2: measure this machine's roofline, place every transformer op on it, report MFU.

Three things, in order:

1. Measured peaks. A GEMM sweep for peak achievable FLOP/s and a STREAM triad for
   peak achievable memory bandwidth. Their ratio is the ridge point, and every
   classification below is made against it.
2. The operator table. FLOPs and compulsory bytes for each operator of a
   transformer block, its arithmetic intensity, and whether it is compute-bound
   or memory-bound here. The real kernels are then timed at the same shape so the
   analytic table can be checked against measurement rather than trusted.
3. MFU for a real training step, with the 6ND estimate and the exact per-layer
   count reported side by side, plus what the same achieved rate would score
   against published accelerator peaks (labelled as arithmetic on a datasheet).

Writes ``results/roofline.json``, ``results/mfu.json``, ``results/profile.json``,
a gzipped Chrome trace to ``results/trace_training_step.json.gz``, and
``assets/roofline.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ASSETS, RESULTS, device_from_arg, write_json
from transformer_internals.config import GPTConfig
from transformer_internals.perf.mfu import flops_6nd, measure_step_mfu
from transformer_internals.perf.plots import fig_roofline
from transformer_internals.perf.profiling import profile_training_step
from transformer_internals.perf.roofline import (
    measure_machine_peak,
    measure_op_rates,
    roofline_payload,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", help="device for the peak and MFU measurements")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mfu-batch", type=int, default=4)
    ap.add_argument("--mfu-seq", type=int, default=256)
    ap.add_argument("--mfu-steps", type=int, default=6)
    ap.add_argument("--profile-batch", type=int, default=2)
    ap.add_argument("--profile-seq", type=int, default=256)
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "roofline.json"))
    args = ap.parse_args()

    device = device_from_arg(args.device)
    cfg = GPTConfig()

    # ---- 1. measured peaks
    print(f"measuring peaks on {device} (GEMM sweep, then STREAM triad)")
    peak = measure_machine_peak(device)
    print(
        f"  peak compute   {peak.peak_flops_per_s / 1e12:9.3f} TFLOP/s  fp32, best of the sweep\n"
        f"  peak bandwidth {peak.peak_bytes_per_s / 1e9:9.1f} GB/s     triad, best of the sweep\n"
        f"  ridge point    {peak.ridge_flops_per_byte:9.2f} FLOP/byte"
    )
    print(f"\n  {'N':>6} {'GEMM GFLOP/s':>14}")
    for r in peak.gemm_sweep:
        print(f"  {int(r['n']):>6} {r['flops_per_s'] / 1e9:>14.1f}")
    print(f"\n  {'array MiB':>10} {'triad GB/s':>12}")
    for r in peak.bandwidth_sweep:
        print(f"  {int(r['array_mib']):>10} {r['bytes_per_s'] / 1e9:>12.1f}")

    # ---- 2. the operator table, analytic then measured
    print(f"\ntiming the real kernels at batch {args.batch} x seq {args.seq}")
    measured = measure_op_rates(cfg, batch=args.batch, seq=args.seq, device=device)
    payload = roofline_payload(
        peak, cfg, batch=args.batch, seq=args.seq, measured_ops=measured
    )
    by_name = {m["op"]: m for m in measured}
    print(
        f"\n  {'operator':<38}{'FLOP/byte':>10}{'bound':>8}"
        f"{'roof GFLOP/s':>14}{'measured':>11}"
    )
    print("  " + "-" * 79)
    for op in payload["ops"]:
        m = by_name.get(op["name"])
        got = f"{m['flops_per_s'] / 1e9:>11.1f}" if m else f"{'':>11}"
        print(
            f"  {op['name'][:37]:<38}{op['intensity']:>10.2f}{op['bound']:>8}"
            f"{op['attainable_flops_per_s'] / 1e9:>14.1f}{got}"
        )
    s = payload["summary"]
    print(
        f"\n  {s['n_memory_bound']} of {s['n_ops']} operators are memory-bound at this shape; "
        f"they hold {s['flops_in_memory_bound_ops_fraction'] * 100:.1f}% of the block's FLOPs "
        f"and {s['roofline_time_in_memory_bound_ops_fraction'] * 100:.1f}% of its roofline time"
    )

    # ---- 3. MFU on a real training step
    print(f"\nmeasuring MFU: {args.mfu_steps} real training steps, GPT-2 124M, {device}")
    mfu = measure_step_mfu(
        peak.peak_flops_per_s,
        GPTConfig(n_positions=max(args.mfu_seq, 1024)),
        batch=args.mfu_batch,
        seq=args.mfu_seq,
        steps=args.mfu_steps,
        device=device,
    )
    fb = mfu.flops_breakdown
    six_nd = flops_6nd(int(fb["n_params_total"]), mfu.batch * mfu.seq)
    print(
        f"  step {mfu.step_s * 1e3:.1f} ms   {mfu.tokens_per_s:,.0f} tokens/s\n"
        f"  model FLOPs/step   exact {mfu.model_flops_per_step:.4e}"
        f"   6ND {six_nd:.4e}   ratio {fb['ratio_to_6nd_total']:.4f}\n"
        f"  attention's sequence-quadratic term is "
        f"{fb['attention_quadratic_fraction'] * 100:.2f}% of the forward pass at seq {mfu.seq}\n"
        f"  achieved {mfu.achieved_flops_per_s / 1e12:.3f} TFLOP/s of "
        f"{peak.peak_flops_per_s / 1e12:.3f} measured peak -> MFU {mfu.mfu * 100:.2f}%"
    )
    print("\n  the same achieved rate against published peaks (modelled, not measured):")
    for row in mfu.modelled_gpu_mfu:
        print(
            f"    {row['accelerator']:<24} peak {row['peak_flops_per_s'] / 1e12:7.1f} TFLOP/s "
            f"({row['precision']})  ->  MFU {row['mfu_if_this_rate_were_sustained'] * 100:6.3f}%"
            f"   ridge {row['ridge_point_flops_per_byte']:.0f} FLOP/byte"
        )

    # ---- 4. profile a real step
    profile = None
    if not args.skip_profile:
        trace = RESULTS / "trace_training_step.json.gz"
        print(f"\nprofiling a training step on cpu, trace -> {trace}")
        profile = profile_training_step(
            GPTConfig(n_positions=max(args.profile_seq, 1024)),
            batch=args.profile_batch,
            seq=args.profile_seq,
            device="cpu",
            active_steps=1,
            trace_path=trace,
        )
        print()
        print(profile.table(limit=15))
        print(
            f"\n  operator self time outside the matmul family: "
            f"{profile.memory_bound_self_time_fraction() * 100:.1f}%"
        )

    payload["mfu"] = mfu.to_dict()
    payload["mfu"]["flops_6nd_total"] = six_nd
    write_json(args.out, payload)
    write_json(RESULTS / "mfu.json", mfu.to_dict())
    if profile is not None:
        write_json(RESULTS / "profile.json", profile.to_dict())

    fig = fig_roofline(payload, ASSETS / "roofline.png")
    print(f"wrote {fig}")
    fig_web = fig_roofline(payload, ASSETS / "roofline_web.png", web=True)
    print(f"wrote {fig_web}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
