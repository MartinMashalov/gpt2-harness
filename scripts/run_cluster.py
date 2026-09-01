"""Part 3: run the training-harness measurements and commit them to one JSON.

The harness modules are exercised by ``tests/test_cluster.py``, which asserts the
properties. This script exists so the *numbers* quoted in the README and in
``docs/CLUSTER.md`` live in a committed file rather than only in test stdout.

Five measurements, all on this machine, all on CPU, ``torch.distributed`` on
gloo with one OS process per rank:

1. Sharded checkpoint of real GPT-2 124M written under 4-way tensor parallelism,
   then resharded to 1, 2 and 8 ranks. Compared with ``torch.equal``, because
   resharding moves bytes and does not compute anything.
2. Overlapped save: how long the training thread blocks for a synchronous save
   against an asynchronous one.
3. Streaming prefetch, in both regimes: a page-cached memmap where the reader
   thread is pure overhead, and 500 us reads where it is the whole point.
4. A real rank killed with SIGKILL mid-run, restarted from the last checkpoint,
   and the resumed loss trajectory compared against an uninterrupted run.
5. Measured gloo all-reduce across message sizes, fitted to the same
   ``latency + bytes/bandwidth`` form the analytic fabric model uses.

Then the fabric model itself, which is MODELLED from published bandwidths and
labelled as such in its own payload.

Writes ``results/cluster.json``.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS, write_json
from transformer_internals import hardware
from transformer_internals.cluster.checkpoint import (
    AsyncCheckpointer,
    gpt2_tp_plan,
    load_full,
    load_reshard,
    merge_pieces,
    save_sharded,
    shard_state_dict,
)
from transformer_internals.cluster.collbench import run as collbench_run
from transformer_internals.cluster.fabric import (
    LINKS,
    LLAMA70B,
    ParallelConfig,
    compute_time_s,
    crossover_ranks,
    step_costs,
)
from transformer_internals.cluster.streaming import (
    DelayedSource,
    ShardedStream,
    TokenShardSource,
    measure_throughput,
)
from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT


def measure_resharding() -> dict[str, Any]:
    """Save GPT-2 124M under tp=4, read it back under 1, 2 and 8 ranks."""
    torch.manual_seed(0)
    model = GPT(GPTConfig())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # 50257 does not divide by 4, so the vocabulary is replicated rather than
    # split. gpt2_tp_plan refuses to pad behind your back.
    plan = gpt2_tp_plan(state, vocab_parallel=False)
    shapes = {k: list(v.shape) for k, v in state.items()}

    directory = Path(tempfile.mkdtemp())
    try:
        t0 = time.perf_counter()
        for rank in range(4):
            save_sharded(
                directory,
                shard_state_dict(state, plan, rank, 4),
                plan,
                rank=rank,
                world_size=4,
                step=100,
                global_shapes=shapes,
            )
        save_s = time.perf_counter() - t0
        sizes = [p.stat().st_size for p in sorted(directory.glob("*.pt"))]

        reshards = []
        for dst in (1, 2, 8):
            t0 = time.perf_counter()
            pieces: dict[str, list[torch.Tensor]] = {k: [] for k in state}
            opens = 0
            for rank in range(dst):
                local, _, files_read = load_reshard(directory, rank=rank, world_size=dst)
                opens += files_read
                for k, v in local.items():
                    pieces[k].append(v)
            dt = time.perf_counter() - t0
            merged = {k: merge_pieces(v, plan[k]) for k, v in pieces.items()}
            reshards.append(
                {
                    "from_world_size": 4,
                    "to_world_size": dst,
                    "seconds": dt,
                    "shard_file_opens": opens,
                    "bitwise_identical": bool(
                        all(torch.equal(merged[k], state[k]) for k in state)
                    ),
                }
            )

        restored = GPT(GPTConfig())
        full, index = load_full(directory)
        restored.load_state_dict(full)
        idx = torch.randint(0, 50257, (2, 64))
        with torch.no_grad():
            a = model(idx)["logits"]
            b = restored(idx)["logits"]
        state_bytes = sum(v.numel() * v.element_size() for v in state.values())
        return {
            "parameters": model.num_parameters(),
            "tensors": len(state),
            "save_seconds": save_s,
            "shard_bytes": sizes,
            "total_on_disk_bytes": sum(sizes),
            "state_dict_bytes": state_bytes,
            "step_in_index": index.step,
            "reshards": reshards,
            "logits_max_abs_diff": float((a - b).abs().max().item()),
            "logits_torch_equal": bool(torch.equal(a, b)),
            "duplication_note": (
                "every replicated tensor is written once per shard, and the tied "
                "embedding is stored under both wte.weight and lm_head.weight, so "
                "the four shards come to more than four times a quarter of the "
                "model. Deduplicating replicated tensors to rank 0 is not implemented."
            ),
        }
    finally:
        shutil.rmtree(directory)


def measure_async_save(repeats: int = 3) -> dict[str, Any]:
    """Blocking time of a synchronous save against an overlapped one."""
    torch.manual_seed(0)
    state = {f"w{i}": torch.randn(512, 512) for i in range(24)}
    plan: dict[str, Any] = {}
    shapes = {k: list(v.shape) for k, v in state.items()}
    nbytes = sum(v.numel() * v.element_size() for v in state.values())

    directory = Path(tempfile.mkdtemp())
    try:
        sync, async_block, async_bg = [], [], []
        for i in range(repeats):
            t0 = time.perf_counter()
            save_sharded(
                directory / f"sync{i}", state, plan, rank=0, world_size=1,
                step=1, global_shapes=shapes,
            )
            sync.append(time.perf_counter() - t0)

            saver = AsyncCheckpointer()
            saver.save(
                directory / f"async{i}", state, plan=plan, rank=0, world_size=1,
                step=1, global_shapes=shapes,
            )
            async_block.append(saver.last_blocking_seconds)
            saver.wait()
            async_bg.append(saver.last_write_seconds)
        return {
            "state_bytes": nbytes,
            "repeats": repeats,
            "statistic": "minimum over repeats",
            "sync_blocking_s": min(sync),
            "async_blocking_s": min(async_block),
            "async_background_s": min(async_bg),
            "step_time_ratio": min(sync) / min(async_block),
            "sync_blocking_s_all": sync,
            "async_blocking_s_all": async_block,
            "async_background_s_all": async_bg,
            "ratio_all": [a / b for a, b in zip(sync, async_block, strict=True)],
            "note": (
                "the blocking figure is what the training step pays. The async arm "
                "blocks for a host-memory copy of the state and nothing else; the "
                "write happens on a background thread. The ratio moves a lot between "
                "runs because the synchronous arm is at the mercy of the page cache."
            ),
        }
    finally:
        shutil.rmtree(directory)


def measure_prefetch(tmp: Path, repeats: int = 3) -> dict[str, Any]:
    """Prefetch in both regimes: a page-cached memmap, and 500 us reads."""
    rng = np.random.default_rng(0)
    path = TokenShardSource.write(
        tmp / "corpus.bin", rng.integers(0, 5000, 400_000, dtype=np.uint16)
    )
    source = TokenShardSource(path, block_size=256)
    list(ShardedStream(source, rank=0, world_size=2))  # warm the page cache

    cached = {}
    for depth in (0, 2, 8):
        rates = [
            measure_throughput(source, rank=0, world_size=2, prefetch=depth, limit=400)[0]
            for _ in range(repeats)
        ]
        cached[str(depth)] = max(rates)

    slow = DelayedSource(source, read_seconds=5e-4)
    overlapped = {}
    for depth in (0, 2, 8):
        rates = [
            measure_throughput(
                slow, rank=0, world_size=2, prefetch=depth, limit=200, per_sample_work_s=5e-4
            )[0]
            for _ in range(repeats)
        ]
        overlapped[str(depth)] = max(rates)

    return {
        "repeats": repeats,
        "statistic": "best of repeats",
        "block_size": 256,
        "page_cached_samples_per_s": cached,
        "slow_read_samples_per_s": overlapped,
        "slow_read_speedup_8_vs_0": overlapped["8"] / overlapped["0"],
        "ceiling_note": (
            "the ceiling is 2.00x when a read and a step cost the same, because "
            "perfect overlap halves the total. Against a page-cached memmap the "
            "reader thread is pure overhead and prefetch is a loss."
        ),
    }


def measure_failure_restart(tmp: Path) -> dict[str, Any]:
    """Kill a rank with SIGKILL, restart from the checkpoint, compare the losses."""
    from transformer_internals.cluster.failure import (
        JobSpec,
        free_port,
        launch,
        make_corpus,
        read_log,
        run_with_recovery,
    )

    data = make_corpus(tmp / "corpus.bin", 40_000, 128, seed=0)
    common = {
        "data_path": str(data),
        "block_size": 32,
        "micro_batch": 4,
        "steps": 20,
        "ckpt_every": 5,
        "seed": 1234,
    }

    clean = JobSpec(
        ckpt_dir=str(tmp / "ck_clean"), log_path=str(tmp / "clean.jsonl"),
        master_port=free_port(), **common,
    )
    ok, codes, _ = launch(2, clean)
    if not ok:
        raise RuntimeError(f"the uninterrupted reference run failed with {codes}")
    clean_losses = {r["step"]: r["loss"] for r in read_log(clean.log_path) if r["event"] == "step"}

    crashed = JobSpec(
        ckpt_dir=str(tmp / "ck_crash"), log_path=str(tmp / "crash.jsonl"),
        master_port=free_port(), **common,
    )
    report = run_with_recovery(2, crashed, kill_rank_at_step=(1, 12))
    if report.resumed_from_step is None:
        raise RuntimeError(f"the run never restarted: {report.events}")
    records = [r for r in read_log(crashed.log_path) if r["event"] == "step"]
    resumed = {
        r["step"]: r["loss"] for r in records if r["resumed_from"] == report.resumed_from_step
    }
    if not resumed:
        raise RuntimeError(f"no steps logged after the restart: {report.events}")
    worst = max(abs(clean_losses[s] - loss) for s, loss in resumed.items())

    return {
        "world_size": 2,
        "backend": "gloo",
        "steps": 20,
        "checkpoint_every": 5,
        "killed_rank": 1,
        "killed_at_step": 12,
        "signal": "SIGKILL",
        "resumed_from_step": report.resumed_from_step,
        "launches": report.launches,
        "time_to_recover_s": report.time_to_recover_s,
        "resumed_steps": [min(resumed), max(resumed)],
        "max_abs_loss_difference": worst,
        "events": report.events,
        "recovery_note": (
            "time to recover here is almost entirely process startup and importing "
            "torch. On a real job the same interval also contains the scheduler "
            "noticing, requeueing, allocating replacement nodes and re-reading a "
            "checkpoint of hundreds of gigabytes over the storage fabric."
        ),
    }


def measure_collectives(world_size: int = 2) -> dict[str, Any]:
    """Real gloo all-reduce over a size sweep, fitted to the model's own form."""
    data = collbench_run(world_size=world_size, sizes=[1 << k for k in range(14, 24, 2)], iters=10)
    return {
        "world_size": data["world_size"],
        "transport": "loopback TCP on this machine, not a fabric",
        "points": data["points"],
        "fit": data["fit"],
        "fit_range_bytes": [1 << 14, 1 << 22],
        "note": (
            "the fitted constants move with the machine's load; R^2 is the claim. "
            "Below about 64 KiB a gloo all-reduce is latency and scheduling noise "
            "and no affine model describes it, which is the model's own point."
        ),
    }


def model_the_fabric() -> dict[str, Any]:
    """MODELLED. Published bandwidths only; nothing here touched an H100."""
    cfg = ParallelConfig(tp=8, dp=8)
    compute_s = compute_time_s(LLAMA70B, cfg)
    rows: dict[str, dict[str, float]] = {}
    for link_key in ("nvlink4", "pcie5", "ib_ndr", "roce200"):
        costs = step_costs(LLAMA70B, cfg, link_key)
        rows[link_key] = {k: float(v) for k, v in costs.items()}
    no_gdr = step_costs(LLAMA70B, cfg, "ib_ndr", gpudirect=False)
    return {
        "status": "MODELLED from published peak bandwidths, not measured",
        "links": {
            k: {"gbytes_per_s": v.gbytes_per_s, "source": v.source} for k, v in LINKS.items()
        },
        "shape": {
            "label": "70B-class, 8k context",
            "params": LLAMA70B.params,
        },
        "config": {"tp": cfg.tp, "dp": cfg.dp, "world_size": cfg.world_size},
        "modelled_compute_s_per_step": compute_s,
        "modelled_comm_s_per_step": rows,
        "tp_over_fsdp_on_ib_ndr": rows["ib_ndr"]["tp"] / rows["ib_ndr"]["fsdp"],
        "gpudirect_off_ib_ndr_tp_s": float(no_gdr["tp"]),
        "gpudirect_penalty_ratio": float(no_gdr["tp"]) / rows["ib_ndr"]["tp"],
        "tp_crossover_degree": {
            key: crossover_ranks(LLAMA70B, key)
            for key in ("nvlink4", "ib_ndr", "roce200")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "cluster.json"))
    ap.add_argument("--skip-reshard", action="store_true", help="skip the GPT-2 124M reshard")
    args = ap.parse_args()

    started = time.time()
    payload: dict[str, Any] = {}

    if not args.skip_reshard:
        print("resharding GPT-2 124M ...")
        payload["resharding"] = measure_resharding()
        for row in payload["resharding"]["reshards"]:
            print(
                f"  4 -> {row['to_world_size']}: {row['seconds'] * 1e3:.0f} ms, "
                f"{row['shard_file_opens']} shard-file opens, "
                f"bitwise identical: {row['bitwise_identical']}"
            )

    print("overlapped save ...")
    payload["async_checkpoint"] = measure_async_save()
    a = payload["async_checkpoint"]
    print(
        f"  sync {a['sync_blocking_s'] * 1e3:.1f} ms blocking, "
        f"async {a['async_blocking_s'] * 1e3:.1f} ms blocking + "
        f"{a['async_background_s'] * 1e3:.1f} ms in the background"
    )

    tmp = Path(tempfile.mkdtemp())
    try:
        print("streaming prefetch ...")
        payload["streaming"] = measure_prefetch(tmp)
        print(f"  prefetch 8 vs 0 with slow reads: "
              f"{payload['streaming']['slow_read_speedup_8_vs_0']:.2f}x")

        print("killing a rank and restarting ...")
        payload["failure_restart"] = measure_failure_restart(tmp)
        f = payload["failure_restart"]
        print(
            f"  killed rank {f['killed_rank']} at step {f['killed_at_step']}, resumed from "
            f"step {f['resumed_from_step']}, recovered in {f['time_to_recover_s']:.2f}s, "
            f"max |loss diff| = {f['max_abs_loss_difference']:.3e}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("measuring gloo all-reduce ...")
    payload["collectives"] = measure_collectives()
    fit = payload["collectives"]["fit"]
    print(
        f"  t = {fit['latency_us']:.0f}us + bytes/{fit['bandwidth_gbytes_per_s']:.2f} GB/s, "
        f"R^2 = {fit['r_squared']:.4f}"
    )

    payload["fabric_model"] = model_the_fabric()
    payload["meta"] = {
        "backend": "gloo",
        "device": "cpu",
        "environment": hardware.environment_payload(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measured_vs_modelled": (
            "resharding, async_checkpoint, streaming, failure_restart and "
            "collectives are MEASURED on this machine. fabric_model is MODELLED "
            "from the published bandwidths it carries with it."
        ),
        "runtime_seconds": time.time() - started,
    }
    write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
