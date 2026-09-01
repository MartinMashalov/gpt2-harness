"""Measure real gloo collectives on this machine, and fit the cost model's form to them.

This cannot validate an H100's NVLink bandwidth. What it can do is check that
the functional form :mod:`~transformer_internals.cluster.fabric` extrapolates
with -- ``time = latency + bytes / bandwidth``, with the ring's
``2(N-1)/N`` factor on an all-reduce -- describes an actual collective rather
than being an assumption stacked on an assumption.

Everything here is MEASURED, on CPU, over gloo on loopback. The fitted
"bandwidth" is loopback TCP plus a memory copy, not a fabric.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

__all__ = ["fit_link_model", "run"]

DEFAULT_SIZES = [1 << k for k in range(10, 23, 2)]  # 1 KiB .. 4 MiB, elements


def fit_link_model(nbytes: list[float], seconds: list[float], ranks: int) -> dict[str, float]:
    """Least-squares fit of ``t = a + b * bytes``; report latency and bandwidth.

    The all-reduce ring moves ``2(N-1)/N`` times the buffer, so the bandwidth
    implied by the slope is ``2(N-1)/N / b``.

    Returns ``latency_us``, ``bandwidth_gbytes_per_s`` and ``r_squared``.
    """
    n = len(nbytes)
    mx = sum(nbytes) / n
    my = sum(seconds) / n
    sxx = sum((x - mx) ** 2 for x in nbytes)
    sxy = sum((x - mx) * (y - my) for x, y in zip(nbytes, seconds, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in seconds)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(nbytes, seconds, strict=True))
    factor = 2 * (ranks - 1) / ranks
    return {
        "latency_us": intercept * 1e6,
        "bandwidth_gbytes_per_s": factor / slope / 1e9 if slope > 0 else float("nan"),
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "slope_s_per_byte": slope,
        "intercept_s": intercept,
    }


def _worker() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    sizes = [int(s) for s in os.environ["TI_SIZES"].split(",")]
    iters = int(os.environ.get("TI_ITERS", "20"))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    results = []
    for numel in sizes:
        buf = torch.ones(numel, dtype=torch.float32)
        for _ in range(3):  # warm the connection and the allocator
            dist.all_reduce(buf)
        dist.barrier()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            dist.all_reduce(buf)
            samples.append(time.perf_counter() - t0)
        results.append(
            {
                "numel": numel,
                "bytes": numel * 4,
                "median_s": statistics.median(samples),
                "min_s": min(samples),
            }
        )
    if rank == 0:
        Path(os.environ["TI_OUT"]).write_text(json.dumps({"world_size": world, "points": results}))
    dist.destroy_process_group()


def run(world_size: int = 2, sizes: list[int] | None = None, iters: int = 20,
        out: Path | str | None = None) -> dict[str, Any]:
    """Launch ``world_size`` gloo ranks locally and return the measurements plus the fit."""
    from transformer_internals.cluster.failure import free_port

    sizes = sizes or DEFAULT_SIZES
    out_path = Path(out) if out else Path(os.environ.get("TMPDIR", "/tmp")) / "ti_collbench.json"
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_port()),
            "WORLD_SIZE": str(world_size),
            "TI_SIZES": ",".join(str(s) for s in sizes),
            "TI_ITERS": str(iters),
            "TI_OUT": str(out_path),
            "OMP_NUM_THREADS": "1",
        }
    )
    procs = []
    for r in range(world_size):
        e = dict(env)
        e["RANK"] = str(r)
        procs.append(subprocess.Popen([sys.executable, "-m", "transformer_internals.cluster.collbench"], env=e))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"collective benchmark ranks exited with {codes}")
    data = json.loads(out_path.read_text())
    fit = fit_link_model(
        [p["bytes"] for p in data["points"]],
        [p["median_s"] for p in data["points"]],
        world_size,
    )
    data["fit"] = fit
    return data


if __name__ == "__main__":
    if "RANK" in os.environ and "TI_SIZES" in os.environ:
        _worker()
    else:
        d = run()
        print(f"MEASURED: gloo all-reduce, {d['world_size']} ranks, CPU, loopback")
        print(f"{'bytes':>12} {'median ms':>12} {'implied GB/s':>14}")
        for p in d["points"]:
            factor = 2 * (d["world_size"] - 1) / d["world_size"]
            print(f"{p['bytes']:>12} {p['median_s']*1e3:>12.3f} "
                  f"{factor*p['bytes']/p['median_s']/1e9:>14.3f}")
        f = d["fit"]
        print(f"\nfit t = {f['intercept_s']*1e6:.1f}us + bytes/{f['bandwidth_gbytes_per_s']:.2f}GB/s"
              f"   R^2 = {f['r_squared']:.4f}")
