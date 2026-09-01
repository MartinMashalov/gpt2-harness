"""Part 1: run every parallelism strategy, prove it correct, and count its bytes.

Spawns real ``torch.distributed`` process groups on the gloo backend and, for
each of data parallel, ZeRO-1/2/3, tensor parallel, pipeline parallel and
context parallel:

* compares the sharded result against a single-process reference and records the
  worst disagreement;
* records the exact number of collectives and the exact number of bytes handed
  to each one, per step;
* checks that count against a closed-form expression in ``N``, ``p``, ``B``,
  ``T`` and ``C``, and then evaluates the same expression at GPT-2 124M scale.

Also measures the pipeline bubble against ``(p-1)/(m+p-1)`` and draws it.

Writes ``results/parallel_comms.json`` and ``assets/parallel_bubble.png``.
Everything here runs on CPU. Nothing in the output is a guess: numbers derived
from a formula rather than measured are in blocks labelled ``modelled`` or
``projected``, and the formula is printed next to them.
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

from _common import ASSETS, RESULTS, write_json
from transformer_internals import viz
from transformer_internals.parallel.common import parallel_config
from transformer_internals.parallel.comms import spawn_workers
from transformer_internals.parallel.data_parallel import ddp_equivalence_worker
from transformer_internals.parallel.dtensor_demo import dtensor_worker
from transformer_internals.parallel.pipeline_parallel import (
    analytic_bubble,
    bubble_scan_worker,
    pipeline_worker,
    simulate,
)
from transformer_internals.parallel.sequence_parallel import sequence_parallel_worker
from transformer_internals.parallel.tensor_parallel import tp_equivalence_worker
from transformer_internals.parallel.zero import zero3_equivalence_worker, zero_equivalence_worker

FP32 = 4  # bytes per element on every path in this repository


def _per_step(payload: dict[str, Any], op: str, field: str = "payload_bytes") -> float:
    """One collective's per-step figure, or 0 if the strategy never issues it."""
    rec = payload["per_collective"].get(op)
    if rec is None:
        return 0.0
    return rec[field] / max(payload["steps"], 1)


def _calls_per_step(payload: dict[str, Any], op: str) -> float:
    rec = payload["per_collective"].get(op)
    return 0.0 if rec is None else rec["calls"] / max(payload["steps"], 1)


def _calls(payload: dict[str, Any], op: str) -> int:
    """Total calls to one collective, or 0 if the strategy never issues it."""
    rec = payload["per_collective"].get(op)
    return 0 if rec is None else int(rec["calls"])


def _bytes(payload: dict[str, Any], op: str) -> int:
    """Total payload bytes for one collective, or 0."""
    rec = payload["per_collective"].get(op)
    return 0 if rec is None else int(rec["payload_bytes"])


def _check(name: str, measured: float, formula: str, predicted: float) -> dict[str, Any]:
    """Record a measured number beside the closed form it is supposed to equal."""
    ok = abs(measured - predicted) <= 1e-9 * max(1.0, abs(predicted))
    return {
        "quantity": name,
        "measured_bytes_per_step": measured,
        "formula": formula,
        "formula_bytes_per_step": predicted,
        "exact_match": bool(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=2, help="ranks for the equivalence runs")
    ap.add_argument("--pipeline-stages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4, help="optimiser steps per equivalence run")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--bubble-batch", type=int, default=16)
    ap.add_argument("--bubble-seq", type=int, default=128)
    ap.add_argument("--bubble-repeats", type=int, default=3)
    ap.add_argument("--out", default=str(RESULTS / "parallel_comms.json"))
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    p = args.world_size
    cfg = parallel_config()
    n_embd, seq, batch = cfg.n_embd, args.seq, args.batch
    started = time.time()
    print(
        f"gloo on CPU | equivalence world size {p} | pipeline stages "
        f"{args.pipeline_stages} | torch {torch.__version__}"
    )

    equivalence: dict[str, Any] = {}
    comms_report: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- DDP
    print("\n[1/6] data parallel")
    ddp = spawn_workers(
        ddp_equivalence_worker,
        p,
        {"batch": batch, "seq": seq, "steps": args.steps},
    )
    n_params = ddp[0].value["n_params"]
    ddp_activations = ddp[0].value["activation_bytes"]
    reference_activations = ddp[0].value["reference_activation_bytes"]
    equivalence["data_parallel"] = {
        "what_is_compared": "gradient of every parameter, and parameters after an SGD step",
        "reference": "single process, full batch, no collectives",
        "max_grad_error": max(r.value["max_grad_error"] for r in ddp),
        "max_param_error": max(r.value["max_param_error"] for r in ddp),
        "steps": args.steps,
        "world_size": p,
    }
    comms_report["data_parallel"] = ddp[0].payload
    checks.append(
        _check(
            "data_parallel.all_reduce",
            _per_step(ddp[0].payload, "all_reduce"),
            "4 * N",
            FP32 * n_params,
        )
    )
    print(
        f"      max gradient error {equivalence['data_parallel']['max_grad_error']:.2e} | "
        f"{_calls_per_step(ddp[0].payload, 'all_reduce'):.0f} all-reduce/step of "
        f"{_per_step(ddp[0].payload, 'all_reduce') / 1e3:.1f} kB"
    )

    # --------------------------------------------------------------- ZeRO
    print("[2/6] sharded data parallel (ZeRO)")
    memory: dict[str, Any] = {}
    for stage in (1, 2):
        res = spawn_workers(
            zero_equivalence_worker, p, {"stage": stage, "steps": args.steps}
        )
        v = res[0].value
        equivalence[f"zero_{stage}"] = {
            "what_is_compared": "every parameter after each AdamW step",
            "reference": "torch.optim.AdamW, single process, full batch",
            "max_param_error": max(r.value["max_param_error"] for r in res),
            "per_step_param_error": v["param_errors"],
            "steps": args.steps,
            "world_size": p,
        }
        comms_report[f"zero_{stage}"] = res[0].payload
        memory[f"zero_{stage}"] = {
            "sharded_bytes_per_rank": v["sharded_bytes"],
            "replicated_bytes_per_rank": v["replicated_bytes"],
            "activation_bytes_per_rank": v["activation_bytes"],
            "shard_numel": v["shard_numel"],
            "n_params": v["n_params"],
        }
        print(
            f"      stage {stage}: max parameter error {v['max_param_error']:.2e} | "
            f"resident {v['sharded_bytes']['total'] / 1e3:.1f} kB/rank vs "
            f"{v['replicated_bytes']['total'] / 1e3:.1f} kB replicated"
        )
    checks.append(
        _check(
            "zero_1.all_reduce",
            _per_step(comms_report["zero_1"], "all_reduce"),
            "4 * N",
            FP32 * n_params,
        )
    )
    checks.append(
        _check(
            "zero_2.reduce_scatter",
            _per_step(comms_report["zero_2"], "reduce_scatter"),
            "4 * N (padded to a multiple of p)",
            FP32 * memory["zero_2"]["shard_numel"] * p,
        )
    )
    checks.append(
        _check(
            "zero_2.all_gather",
            _per_step(comms_report["zero_2"], "all_gather"),
            "4 * N (padded to a multiple of p)",
            FP32 * memory["zero_2"]["shard_numel"] * p,
        )
    )

    z3 = spawn_workers(zero3_equivalence_worker, p, {"steps": args.steps})
    v3 = z3[0].value
    equivalence["zero_3"] = {
        "what_is_compared": (
            "every parameter after each AdamW step, all-gathered so the check "
            "covers the shards this rank does not own"
        ),
        "reference": "torch.optim.AdamW, single process, full batch",
        "max_param_error": max(r.value["max_param_error"] for r in z3),
        "per_step_param_error": v3["param_errors"],
        "steps": args.steps,
        "world_size": p,
    }
    comms_report["zero_3"] = z3[0].payload
    memory["data_parallel"] = {
        "sharded_bytes_per_rank": memory["zero_1"]["replicated_bytes_per_rank"],
        "replicated_bytes_per_rank": memory["zero_1"]["replicated_bytes_per_rank"],
        "activation_bytes_per_rank": ddp_activations,
        "n_params": n_params,
    }
    memory["single_process_reference"] = {
        "activation_bytes_per_rank": reference_activations,
        "note": (
            "one process holding the whole batch. Every per-rank activation "
            "figure above should be read against this: data parallelism and ZeRO "
            "reduce it only because each rank sees 1/p of the batch, and neither "
            "shards an activation."
        ),
    }
    memory["zero_3"] = {
        "sharded_bytes_per_rank": v3["sharded_bytes"],
        "replicated_bytes_per_rank": memory["zero_2"]["replicated_bytes_per_rank"],
        "activation_bytes_per_rank": v3["activation_bytes"],
        "activation_note": v3["activation_note"],
        "n_params": v3["n_params"],
        "n_block_params": v3["n_block_params"],
        "units": v3["n_units"],
        "shard_numel": v3["shard_numel"],
    }
    print(
        f"      stage 3: max parameter error {v3['max_param_error']:.2e} | "
        f"resident {v3['sharded_bytes']['total'] / 1e3:.1f} kB/rank | "
        f"{v3['n_units']} sharded units"
    )
    print(
        f"      activations/rank: single process {reference_activations / 1e3:.1f} kB | "
        f"DDP and ZeRO-1/2 {ddp_activations / 1e3:.1f} kB | "
        f"ZeRO-3 {v3['activation_bytes'] / 1e3:.1f} kB (recomputes)"
    )

    # ---------------------------------------------------------- tensor par
    print("[3/6] tensor parallel")
    tp = spawn_workers(tp_equivalence_worker, p, {"batch": 4, "seq": 12})
    tv = tp[0].value
    equivalence["tensor_parallel"] = {
        "what_is_compared": (
            "block output, gradient w.r.t. the input, replicated LayerNorm and "
            "bias gradients, and this rank's slice of every sharded weight gradient"
        ),
        "reference": "the same block, unsharded, single process",
        "max_forward_error": max(r.value["forward_error"] for r in tp),
        "max_backward_error": max(r.value["max_grad_error"] for r in tp),
        "reference_grad_scale": tv["reference_grad_scale"],
        "output_scale": tv["output_scale"],
        "world_size": p,
    }
    comms_report["tensor_parallel"] = tp[0].payload
    memory["tensor_parallel_block"] = {
        "activation_bytes_per_rank": tv["activation_bytes"],
        "reference_activation_bytes": tv["reference_activation_bytes"],
        "local_params": tv["local_params"],
        "reference_params": tv["reference_params"],
        "note": (
            "one block, same batch on every rank. Unlike ZeRO this genuinely "
            "shards activations: the 4C MLP hidden and the per-head attention "
            "tensors are 1/p of the width. The residual stream and the LayerNorm "
            "inputs stay replicated, which is why the ratio is above 1/p."
        ),
    }
    checks.append(
        _check(
            "tensor_parallel.all_reduce",
            _per_step(tp[0].payload, "all_reduce"),
            "4 * 4 * B * T * C  (g forward and f backward, for attention and MLP)",
            FP32 * 4 * 4 * 12 * n_embd,
        )
    )
    print(
        f"      forward {equivalence['tensor_parallel']['max_forward_error']:.2e}, "
        f"backward {equivalence['tensor_parallel']['max_backward_error']:.2e} "
        f"(gradients reach {tv['reference_grad_scale']:.2f}) | "
        f"{_calls_per_step(tp[0].payload, 'all_reduce'):.0f} all-reduce per block-step"
    )

    # -------------------------------------------------------- pipeline par
    print("[4/6] pipeline parallel")
    pipe_stages = args.pipeline_stages
    pipeline: dict[str, Any] = {}
    for kind in ("gpipe", "1f1b"):
        res = spawn_workers(
            pipeline_worker,
            pipe_stages,
            {"kind": kind, "micro_batches": 4, "batch": batch, "seq": seq},
        )
        equivalence[f"pipeline_{kind}"] = {
            "what_is_compared": "every gradient of this rank's stage",
            "reference": "the whole model, single process, full batch",
            "max_grad_error": max(r.value["max_grad_error"] for r in res),
            "reference_grad_scale": max(r.value["reference_grad_scale"] for r in res),
            "stage_params": [r.value["stage_params"] for r in res],
            "peak_stash": [r.value["peak_stash"] for r in res],
            "world_size": pipe_stages,
        }
        memory[f"pipeline_{kind}"] = {
            "activation_peak_bytes_per_rank": [
                r.value["activation_peak_bytes"] for r in res
            ],
            "peak_stash_micro_batches": [r.value["peak_stash"] for r in res],
            "note": (
                "peak over the whole step, per stage. Pipeline parallelism shards "
                "activations by depth, and the schedule decides how many "
                "micro-batches of them are alive at once."
            ),
        }
        comms_report[f"pipeline_{kind}"] = res[0].payload
        print(
            f"      {kind}: max gradient error "
            f"{equivalence[f'pipeline_{kind}']['max_grad_error']:.2e} | stash "
            f"{equivalence[f'pipeline_{kind}']['peak_stash']}"
        )
    checks.append(
        _check(
            "pipeline.send",
            _per_step(comms_report["pipeline_gpipe"], "send"),
            "4 * B * T * C  (one activation per micro-batch, summed over m)",
            FP32 * batch * seq * n_embd,
        )
    )

    print("      bubble scan")
    scan = spawn_workers(
        bubble_scan_worker,
        pipe_stages,
        {
            "micro_batch_counts": (1, 2, 4, 8, 16),
            "batch": args.bubble_batch,
            "seq": args.bubble_seq,
            "repeats": args.bubble_repeats,
            "warmup": 1,
            "config_kwargs": {
                "n_layer": 8,
                "n_embd": 128,
                "n_head": 4,
                "n_positions": args.bubble_seq,
            },
        },
        timeout=900,
    )
    rows_by_rank = [r.value["rows"] for r in scan]
    bubble_rows: list[dict[str, Any]] = []
    print(f"\n      {'sched':>6} {'m':>3} {'analytic':>9} {'measured':>9} {'makespan':>10} {'stash':>6}")
    print("      " + "-" * 48)
    for i in range(len(rows_by_rank[0])):
        group = [rank_rows[i] for rank_rows in rows_by_rank]
        compute = sum(g["compute_seconds"] for g in group)
        makespan = max(g["wall_seconds"] for g in group)
        row = {
            "schedule": group[0]["kind"],
            "micro_batches": group[0]["micro_batches"],
            "micro_batch_size": group[0]["micro_batch_size"],
            "analytic_bubble": group[0]["analytic_bubble"],
            "measured_bubble": 1.0 - compute / (pipe_stages * makespan),
            "makespan_seconds": makespan,
            "compute_seconds_all_ranks": compute,
            "comm_seconds_all_ranks": sum(g["comm_seconds"] for g in group),
            "peak_stash": max(g["peak_stash"] for g in group),
        }
        bubble_rows.append(row)
        print(
            f"      {row['schedule']:>6} {row['micro_batches']:>3} "
            f"{row['analytic_bubble']:>9.3f} {row['measured_bubble']:>9.3f} "
            f"{row['makespan_seconds'] * 1e3:>9.1f}ms {row['peak_stash']:>6}"
        )
    pipeline["measured_bubble"] = bubble_rows
    pipeline["simulated_bubble"] = [
        {
            "schedule": kind,
            "n_stages": stages,
            "micro_batches": m,
            "simulated_bubble": simulate(stages, m, 1.0, 2.0, kind).bubble,
            "analytic_bubble": analytic_bubble(stages, m),
            "peak_stash": simulate(stages, m, 1.0, 2.0, kind).peak_stash,
        }
        for kind in ("gpipe", "1f1b")
        for stages in (2, 4, 8)
        for m in (1, 2, 4, 8, 16, 32)
    ]

    # ------------------------------------------ mixed precision on the wire
    print("[2b/6] what a bf16 wire costs")
    mixed: dict[str, Any] = {
        "note": (
            "MEASURED. Same trajectory comparison, run twice, differing only in "
            "the dtype the gradient collective carries. The optimizer holds an "
            "fp32 master shard on both arms, which is why the bf16 error does "
            "not compound across steps. param_dtype is the dtype the parameter "
            "all-gather carries; for ZeRO-3 that buffer is also what the unit "
            "computes with, so it sets the block's compute dtype too."
        ),
        "rows": [],
    }
    print(f"\n      {'strategy':<10}{'reduce':>8}{'param':>8}{'bytes/step':>12}{'traj error':>13}")
    print("      " + "-" * 51)
    for reduce_dtype, param_dtype in (("fp32", "fp32"), ("bf16", "fp32"), ("fp32", "bf16")):
        rd = None if reduce_dtype == "fp32" else reduce_dtype
        pd = None if param_dtype == "fp32" else param_dtype
        arms: list[tuple[str, Any, dict[str, Any], str]] = []
        if param_dtype == "fp32":
            arms.append(
                (
                    "ddp",
                    ddp_equivalence_worker,
                    {"batch": batch, "seq": seq, "steps": args.steps, "reduce_dtype": rd},
                    "max_grad_error",
                )
            )
        for stage in (1, 2):
            arms.append(
                (
                    f"zero_{stage}",
                    zero_equivalence_worker,
                    {
                        "stage": stage,
                        "steps": args.steps,
                        "reduce_dtype": rd,
                        "param_dtype": pd,
                    },
                    "max_param_error",
                )
            )
        arms.append(
            (
                "zero_3",
                zero3_equivalence_worker,
                {"steps": args.steps, "reduce_dtype": rd, "param_dtype": pd},
                "max_param_error",
            )
        )
        for name, worker, kwargs, error_key in arms:
            res = spawn_workers(worker, p, kwargs)
            payload_bytes = res[0].payload["payload_bytes_per_step"]
            row = {
                "strategy": name,
                "reduce_dtype": reduce_dtype,
                "param_dtype": param_dtype,
                "payload_bytes_per_step": payload_bytes,
                "max_error_vs_single_process": max(r.value[error_key] for r in res),
                "per_step_error": res[0].value.get(
                    "param_errors", res[0].value.get("grad_errors")
                ),
                "reference_scale": res[0].value.get("reference_param_scale"),
            }
            mixed["rows"].append(row)
            print(
                f"      {name:<10}{reduce_dtype:>8}{param_dtype:>8}"
                f"{payload_bytes:>12,.0f}{row['max_error_vs_single_process']:>13.2e}"
            )
    base = {r["strategy"]: r for r in mixed["rows"] if r["reduce_dtype"] == "fp32" and r["param_dtype"] == "fp32"}
    for row in mixed["rows"]:
        ref = base[row["strategy"]]
        row["bytes_vs_fp32"] = row["payload_bytes_per_step"] / ref["payload_bytes_per_step"]
        row["error_vs_fp32"] = (
            row["max_error_vs_single_process"] / ref["max_error_vs_single_process"]
        )
        if row["per_step_error"]:
            first, last = row["per_step_error"][0], row["per_step_error"][-1]
            row["error_growth_first_to_last_step"] = last / first if first else float("nan")

    # ------------------------------------------------- clipping under sharding
    print("\n[2c/6] a global gradient clip on a sharded gradient")
    clipping: dict[str, Any] = {
        "note": (
            "MEASURED. A global gradient-norm clip needs the norm over every "
            "parameter, and a rank holds only its slice, so the ranks must agree "
            "on one scalar before any of them updates. ZeRO-1 already holds the "
            "whole averaged gradient and pays nothing; ZeRO-2 and ZeRO-3 pay one "
            "extra all-reduce of a single fp32 scalar per step, which carries no "
            "bandwidth and is entirely latency."
        ),
        "clip": 1.0,
        "rows": [],
    }
    for name, worker, kwargs in (
        ("zero_1", zero_equivalence_worker, {"stage": 1, "steps": args.steps}),
        ("zero_2", zero_equivalence_worker, {"stage": 2, "steps": args.steps}),
        ("zero_3", zero3_equivalence_worker, {"steps": args.steps}),
    ):
        plain = spawn_workers(worker, p, kwargs)
        clipped = spawn_workers(worker, p, {**kwargs, "grad_clip": 1.0})
        steps_run = max(int(plain[0].payload["steps"]), 1)
        row = {
            "strategy": name,
            "extra_all_reduce_calls_per_step": (
                _calls(clipped[0].payload, "all_reduce") - _calls(plain[0].payload, "all_reduce")
            )
            / steps_run,
            "extra_bytes_per_step": (
                _bytes(clipped[0].payload, "all_reduce") - _bytes(plain[0].payload, "all_reduce")
            )
            / steps_run,
            "max_norm_disagreement_vs_torch": max(
                r.value["max_grad_norm_error"] for r in clipped
            ),
            "max_param_error_clipped": max(r.value["max_param_error"] for r in clipped),
            "max_param_error_unclipped": max(r.value["max_param_error"] for r in plain),
            "global_norms": clipped[0].value["grad_norms"],
        }
        clipping["rows"].append(row)
        print(
            f"      {name}: +{row['extra_all_reduce_calls_per_step']:.0f} all-reduce/step "
            f"({row['extra_bytes_per_step']:.0f} B) | norm agrees with "
            f"torch.nn.utils.clip_grad_norm_ to {row['max_norm_disagreement_vs_torch']:.1e}"
        )
    checks.append(
        _check(
            "zero_2.clip_norm_all_reduce",
            clipping["rows"][1]["extra_bytes_per_step"],
            "4 (one fp32 scalar)",
            4.0,
        )
    )
    checks.append(
        _check(
            "zero_1.clip_norm_all_reduce",
            clipping["rows"][0]["extra_bytes_per_step"],
            "0 (stage 1 holds the whole gradient already)",
            0.0,
        )
    )

    # ------------------------------------------------------- context par
    print("\n[5/6] context / sequence parallel")
    cp = spawn_workers(sequence_parallel_worker, p, {"batch": 2, "seq": 32})
    cv = cp[0].value
    equivalence["context_parallel"] = {
        "what_is_compared": (
            "this rank's rows of the attention output (all-gather-KV and ring), "
            "the gradient w.r.t. this rank's input slice, and the all-reduced "
            "projection weight gradient"
        ),
        "reference": "CausalSelfAttention over the whole sequence, single process",
        "max_all_gather_forward_error": max(
            r.value["all_gather_forward_error"] for r in cp
        ),
        "max_ring_forward_error": max(r.value["ring_forward_error"] for r in cp),
        "max_input_grad_error": max(r.value["input_grad_error"] for r in cp),
        "max_weight_grad_error": max(r.value["weight_grad_error"] for r in cp),
        "weight_grad_scale": cv["weight_grad_scale"],
        "backward_note": (
            "the all-gather path is differentiable end to end; the ring path is "
            "forward only, since its backward needs a second ring for dK/dV"
        ),
        "world_size": p,
    }
    comms_report["context_parallel"] = cp[0].payload
    checks.append(
        _check(
            "context_parallel.all_gather",
            _per_step(cp[0].payload, "all_gather"),
            "2 * 4 * B * T * C  (K and V)",
            2 * FP32 * 2 * 32 * n_embd,
        )
    )
    checks.append(
        _check(
            "context_parallel.ring_p2p",
            _per_step(cp[0].payload, "send"),
            "2 * (p-1)/p * 4 * B * T * C",
            2 * (p - 1) / p * FP32 * 2 * 32 * n_embd,
        )
    )
    print(
        f"      all-gather KV {equivalence['context_parallel']['max_all_gather_forward_error']:.2e}, "
        f"ring {equivalence['context_parallel']['max_ring_forward_error']:.2e}, "
        f"backward {equivalence['context_parallel']['max_input_grad_error']:.2e}"
    )

    # ------------------------------------------------------------ DTensor
    print("[6/6] DTensor")
    dt = spawn_workers(dtensor_worker, p, {})
    dv = dt[0].value
    equivalence["dtensor"] = {
        "what_is_compared": (
            "DTensor placements and parallelize_module against the single-process "
            "MLP, and against this repository's hand-written tensor parallelism"
        ),
        "max_dtensor_error": max(r.value["dtensor_error"] for r in dt),
        "max_parallelize_module_error": max(
            r.value["parallelize_module_error"] for r in dt
        ),
        "max_manual_vs_dtensor": max(r.value["manual_vs_dtensor"] for r in dt),
        "device_mesh": dv["mesh"],
        "world_size": p,
        "api_note": "torch.distributed._tensor on torch 2.2; public as torch.distributed.tensor from 2.5",
    }
    print(
        f"      DTensor {equivalence['dtensor']['max_dtensor_error']:.2e} | "
        f"manual vs DTensor {equivalence['dtensor']['max_manual_vs_dtensor']:.2e}"
    )

    # ---------------------------------------------------- GPT-2 projection
    gpt2_n = 124_439_808
    gpt2 = {"N": gpt2_n, "B": 512, "T": 1024, "C": 768, "L": 12, "bytes_per_element": FP32}
    b, t, c, layers = gpt2["B"], gpt2["T"], gpt2["C"], gpt2["L"]
    act = FP32 * b * t * c
    projection = {
        "note": (
            "PROJECTED, not measured. These are the same formulas that were "
            "checked against the measured byte counts above (see formula_checks, "
            "all exact), evaluated at GPT-2 124M with OpenAI's training batch of "
            "512 x 1024 tokens in fp32. They are collective payload bytes per "
            "rank per optimiser step; multiply by the ring factors in "
            "transformer_internals.parallel.comms for wire bytes."
        ),
        "config": gpt2,
        "per_step_payload_bytes": {
            "data_parallel": {"all_reduce": FP32 * gpt2_n},
            "zero_1": {"all_reduce": FP32 * gpt2_n, "all_gather": FP32 * gpt2_n},
            "zero_2": {"reduce_scatter": FP32 * gpt2_n, "all_gather": FP32 * gpt2_n},
            "zero_3": {
                "all_gather": 2 * FP32 * gpt2_n,
                "reduce_scatter": FP32 * gpt2_n,
            },
            "tensor_parallel": {"all_reduce": 4 * act * layers},
            "pipeline_parallel_per_boundary": {"send": act, "recv": act},
            "context_parallel_all_gather_kv": {
                "all_gather": 2 * act * layers,
                "reduce_scatter": 2 * act * layers,
            },
        },
    }

    # --------------------------------------------------------------- write
    payload = {
        "equivalence": equivalence,
        "tolerance": 1e-5,
        "comms": comms_report,
        "formula_checks": checks,
        "memory": memory,
        "mixed_precision": mixed,
        "gradient_clipping": clipping,
        "pipeline": pipeline,
        "projection_gpt2_124m": projection,
        "meta": {
            "backend": "gloo",
            "device": "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "equivalence_world_size": p,
            "pipeline_stages": pipe_stages,
            "model": cfg.to_dict(),
            "bubble_model": {
                "n_layer": 8,
                "n_embd": 128,
                "n_head": 4,
                "batch": args.bubble_batch,
                "seq": args.bubble_seq,
                "repeats": args.bubble_repeats,
            },
            "measured_vs_modelled": (
                "equivalence errors, collective counts, payload bytes, resident "
                "bytes and the bubble scan are MEASURED on this machine. "
                "wire_bytes_ring_model and projection_gpt2_124m are MODELLED "
                "from the formulas printed beside them."
            ),
            "runtime_seconds": time.time() - started,
        },
    }
    write_json(args.out, payload)

    worst = max(
        [
            equivalence["data_parallel"]["max_grad_error"],
            equivalence["zero_1"]["max_param_error"],
            equivalence["zero_2"]["max_param_error"],
            equivalence["zero_3"]["max_param_error"],
            equivalence["tensor_parallel"]["max_backward_error"],
            equivalence["pipeline_gpipe"]["max_grad_error"],
            equivalence["context_parallel"]["max_weight_grad_error"],
            equivalence["dtensor"]["max_dtensor_error"],
        ]
    )
    print(f"\nworst disagreement with the single-process reference anywhere: {worst:.2e}")
    print(f"every formula check exact: {all(c['exact_match'] for c in checks)}")

    if not args.no_figure:
        for web in (False, True):
            path = Path(args.assets) / (
                "parallel_bubble_web.png" if web else "parallel_bubble.png"
            )
            viz.fig_parallel_bubble(payload, path, web=web)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
