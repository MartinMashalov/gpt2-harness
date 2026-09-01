"""Measure real collectives across message sizes and world sizes, and report bus bandwidth.

Three collectives -- all-reduce, all-gather, reduce-scatter -- swept over message
size and world size, on whichever backend the machine has. On a laptop that is
gloo over loopback and the bandwidth figure is a memory copy. On an eight-GPU
node it is NCCL over NVLink and the figure is the interconnect, which is the
number :mod:`transformer_internals.cluster.fabric` currently has to take from a
datasheet.

Algorithm bandwidth and bus bandwidth
-------------------------------------
The two are different and confusing them is the usual way a collective
benchmark reports a wrong number.

*Algorithm bandwidth* is ``S / t``: the buffer size divided by the time. It is
what the application experiences and it is not comparable across collectives or
across world sizes, because a ring all-reduce moves more than ``S`` on the wire
and an all-gather moves less.

*Bus bandwidth* is algorithm bandwidth times the ring factor for that
collective, and it *is* comparable: it estimates what each link is carrying, so
it saturates at the hardware's per-link rate and stays flat as the world size
grows. This is NCCL's own convention, and the factors are the ones ``nccl-tests``
documents:

===============  ==========================  ==================
collective       ``S`` (the reported size)   bus factor
===============  ==========================  ==================
all-reduce       the full buffer             ``2(n-1)/n``
all-gather       the full output buffer      ``(n-1)/n``
reduce-scatter   the full input buffer       ``(n-1)/n``
===============  ==========================  ==================

``S`` is the *unsharded* buffer in all three cases, so the same message size
means the same thing in every row of the table.

Those are the same factors
:mod:`transformer_internals.parallel.comms` uses for its ``wire_bytes`` model,
which is deliberate: the modelled wire volume and the measured bus bandwidth are
two halves of one cost model, and this module is the half that can be checked
against a wire.

What this can and cannot say
----------------------------
On the machine this repository was written on, everything below is gloo over TCP
loopback. The *shape* of the curve is real -- latency floor at small messages,
bandwidth-bound at large ones -- and the fit of ``t = latency + bytes/bandwidth``
to it is a real check on the functional form the fabric model extrapolates with.
The bandwidth constant is not an interconnect measurement and is never presented
as one. On a CUDA node the same code reports NCCL over the real fabric, and
:func:`transformer_internals.cluster.fabric.link_from_measurement` turns that
into a :class:`~transformer_internals.cluster.fabric.Link` that replaces the
datasheet one.
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

from transformer_internals import hardware

__all__ = [
    "BUS_FACTORS",
    "OPS",
    "bus_bandwidth",
    "fit_link_model",
    "run",
]

#: Element counts, not bytes. At fp32 these are buffers of 4 KiB to 16 MiB.
#: scripts/run_collectives.py sweeps wider, 16 KiB to 64 MiB.
DEFAULT_SIZES = [1 << k for k in range(10, 23, 2)]

#: The collectives this module measures.
OPS = ("all_reduce", "all_gather", "reduce_scatter")


def _bus_factor(op: str, world_size: int) -> float:
    if world_size < 2:
        return 0.0
    if op == "all_reduce":
        return 2.0 * (world_size - 1) / world_size
    if op in ("all_gather", "reduce_scatter"):
        return (world_size - 1) / world_size
    raise ValueError(f"unknown collective {op!r}; expected one of {OPS}")


#: Exposed as a mapping for callers that want to price a size without timing it.
BUS_FACTORS = {op: (lambda n, _op=op: _bus_factor(_op, n)) for op in OPS}


def bus_bandwidth(op: str, nbytes: float, seconds: float, world_size: int) -> float:
    """Bus bandwidth in bytes/s: ``S / t`` times the collective's ring factor.

    Args:
        op: One of :data:`OPS`.
        nbytes: The full (unsharded) buffer size.
        seconds: Measured time for one call.
        world_size: Ranks in the group.

    Returns:
        Bytes per second. Zero for a world size of one, where no collective
        moves anything.
    """
    if seconds <= 0 or world_size < 2:
        return 0.0
    return (nbytes / seconds) * _bus_factor(op, world_size)


def fit_link_model(
    nbytes: list[float], seconds: list[float], ranks: int, op: str = "all_reduce"
) -> dict[str, float]:
    """Least-squares fit of ``t = a + b * bytes``; report latency and bandwidth.

    The bandwidth implied by the slope is the *bus* bandwidth, so the ring
    factor for the collective divides the slope. That makes the fitted constant
    comparable with a link's published per-direction rate.

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
    factor = _bus_factor(op, ranks)
    return {
        "op": op,
        "latency_us": intercept * 1e6,
        "bandwidth_gbytes_per_s": factor / slope / 1e9 if slope > 0 else float("nan"),
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "slope_s_per_byte": slope,
        "intercept_s": intercept,
    }


# --------------------------------------------------------------------------- #
# the rank
# --------------------------------------------------------------------------- #


def _barrier(device: torch.device) -> None:
    """A barrier that tells NCCL which device it is on.

    Without ``device_ids`` a NCCL barrier guesses from the current device and
    warns about guessing. Gloo does not take the argument.
    """
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index or 0])
    else:
        dist.barrier()


def _time_op(op: str, numel: int, world: int, device: torch.device, iters: int) -> dict[str, Any]:
    """Time one collective at one size, returning the samples and the bandwidths.

    Every timed call is followed by a device synchronise. Without it a CUDA
    measurement records how fast Python enqueued the kernel, which on a fast
    fabric is most of the reported time.
    """
    if op == "all_reduce":
        buf = torch.ones(numel, dtype=torch.float32, device=device)
        full_bytes = numel * 4

        def call() -> None:
            dist.all_reduce(buf)

    elif op == "all_gather":
        # numel is the FULL output; each rank contributes numel / world.
        shard = numel // world
        src = torch.ones(shard, dtype=torch.float32, device=device)
        dst = torch.empty(shard * world, dtype=torch.float32, device=device)
        full_bytes = shard * world * 4

        def call() -> None:
            dist.all_gather_into_tensor(dst, src)

    elif op == "reduce_scatter":
        # numel is the FULL input; each rank keeps numel / world.
        shard = numel // world
        src = torch.ones(shard * world, dtype=torch.float32, device=device)
        dst = torch.empty(shard, dtype=torch.float32, device=device)
        full_bytes = shard * world * 4

        def call() -> None:
            dist.reduce_scatter_tensor(dst, src)

    else:
        raise ValueError(f"unknown collective {op!r}; expected one of {OPS}")

    for _ in range(3):  # warm the connection, the allocator and the NCCL channels
        call()
    hardware.synchronize(device)
    _barrier(device)

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        hardware.synchronize(device)
        samples.append(time.perf_counter() - t0)

    median = statistics.median(samples)
    best = min(samples)
    return {
        "op": op,
        "numel": numel,
        "bytes": full_bytes,
        "median_s": median,
        "min_s": best,
        "algorithm_gbytes_per_s": full_bytes / median / 1e9,
        "bus_gbytes_per_s": bus_bandwidth(op, full_bytes, median, world) / 1e9,
        "bus_gbytes_per_s_best": bus_bandwidth(op, full_bytes, best, world) / 1e9,
    }


def _worker() -> None:
    """One rank. Launched by :func:`run` with its configuration in the environment."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    sizes = [int(s) for s in os.environ["TI_SIZES"].split(",")]
    ops = os.environ.get("TI_OPS", "all_reduce").split(",")
    iters = int(os.environ.get("TI_ITERS", "20"))
    backend = os.environ.get("TI_BACKEND", "gloo")
    device = hardware.set_visible_device(os.environ.get("TI_DEVICE", "cpu"))

    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    results: list[dict[str, Any]] = []
    for op in ops:
        for numel in sizes:
            if op != "all_reduce" and numel % world:
                # A sharded collective needs the buffer to divide evenly. Skip
                # rather than pad: padding would report a size that was not the
                # size measured.
                continue
            results.append(_time_op(op, numel, world, device, iters))
    if rank == 0:
        Path(os.environ["TI_OUT"]).write_text(
            json.dumps(
                {
                    "world_size": world,
                    "backend": backend,
                    "device": str(device),
                    "points": results,
                }
            )
        )
    _barrier(device)
    dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# the launcher
# --------------------------------------------------------------------------- #


def run(
    world_size: int = 2,
    sizes: list[int] | None = None,
    iters: int = 20,
    out: Path | str | None = None,
    ops: tuple[str, ...] = ("all_reduce",),
    backend: str | None = None,
    caps: hardware.Capabilities | None = None,
    allow_oversubscribe: bool = False,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Launch ``world_size`` ranks locally and return the measurements plus the fits.

    Args:
        world_size: Ranks.
        sizes: Element counts to sweep. For the sharded collectives a size that
            does not divide by the world size is skipped rather than padded.
        iters: Timed calls per point.
        out: Where rank 0 writes its JSON.
        ops: Which collectives to measure; a subset of :data:`OPS`.
        backend: ``"nccl"``, ``"gloo"`` or ``None`` to choose from the machine.
        caps: Machine description; detected when omitted.
        allow_oversubscribe: Permit more ranks than GPUs. Refused by default,
            because two ranks sharing a GPU turn a bandwidth measurement into a
            contention measurement.
        timeout: Seconds to wait for a rank.

    Returns:
        A dict with ``points`` (every measurement), ``by_op`` (grouped, with a
        fit and a peak bus bandwidth per collective) and ``fit`` (the all-reduce
        fit, kept at the top level for callers that predate the other
        collectives).

    Raises:
        RuntimeError: If a rank exits non-zero.
        HardwareError: If the backend or the world size is impossible here.
    """
    from transformer_internals.cluster.failure import free_port

    for op in ops:
        if op not in OPS:
            raise ValueError(f"unknown collective {op!r}; expected a subset of {OPS}")

    caps = caps if caps is not None else hardware.Capabilities.detect()
    chosen = hardware.select_backend(caps, backend)
    devices = hardware.check_placement(
        caps, world_size, chosen, allow_oversubscribe=allow_oversubscribe
    )

    sizes = sizes or DEFAULT_SIZES
    out_path = Path(out) if out else Path(os.environ.get("TMPDIR", "/tmp")) / "ti_collbench.json"
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(free_port()),
            "WORLD_SIZE": str(world_size),
            "TI_SIZES": ",".join(str(s) for s in sizes),
            "TI_OPS": ",".join(ops),
            "TI_ITERS": str(iters),
            "TI_OUT": str(out_path),
            "TI_BACKEND": chosen,
            "OMP_NUM_THREADS": "1",
        }
    )
    procs = []
    for r in range(world_size):
        e = dict(env)
        e["RANK"] = str(r)
        e["TI_DEVICE"] = devices[r]
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "transformer_internals.cluster.collbench"], env=e
            )
        )
    codes = []
    for proc in procs:
        try:
            codes.append(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            for other in procs:
                if other.poll() is None:
                    other.kill()
            raise RuntimeError(
                f"a collective-benchmark rank did not finish within {timeout}s. On a "
                f"first NCCL run this is usually the rendezvous: check that "
                f"NCCL_SOCKET_IFNAME names a real interface and that no firewall "
                f"blocks {env['MASTER_PORT']} on loopback."
            ) from None
    if any(codes):
        raise RuntimeError(
            f"collective benchmark ranks exited with {codes}. Re-run with "
            f"NCCL_DEBUG=INFO to see what NCCL said."
        )

    data = json.loads(out_path.read_text())
    by_op: dict[str, Any] = {}
    for op in ops:
        points = [p for p in data["points"] if p["op"] == op]
        if not points:
            continue
        entry: dict[str, Any] = {
            "points": points,
            "peak_bus_gbytes_per_s": max(p["bus_gbytes_per_s"] for p in points),
        }
        if len(points) >= 3:
            entry["fit"] = fit_link_model(
                [p["bytes"] for p in points],
                [p["median_s"] for p in points],
                world_size,
                op=op,
            )
        by_op[op] = entry
    data["by_op"] = by_op
    data["ops"] = list(ops)
    # Kept at the top level under its original name: run_cluster.py and
    # tests/test_cluster.py both read data["fit"] and predate the other two
    # collectives.
    if "all_reduce" in by_op and "fit" in by_op["all_reduce"]:
        data["fit"] = by_op["all_reduce"]["fit"]
    data["points"] = [p for p in data["points"] if p["op"] == "all_reduce"] or data["points"]
    data["measurement_note"] = (
        f"MEASURED on this machine over {data['backend']}, device {data['device']}. "
        "Bus bandwidth follows NCCL's convention: algorithm bandwidth times the "
        "ring factor for the collective, so the three collectives are comparable "
        "with each other and across world sizes."
    )
    return data


if __name__ == "__main__":
    if "RANK" in os.environ and "TI_SIZES" in os.environ:
        _worker()
    else:
        d = run(ops=OPS)
        print(
            f"MEASURED: {d['backend']} on {d['device']}, {d['world_size']} ranks\n"
            f"{'collective':<16}{'bytes':>12}{'median ms':>12}{'algbw GB/s':>13}"
            f"{'busbw GB/s':>13}"
        )
        for op, entry in d["by_op"].items():
            for p in entry["points"]:
                print(
                    f"{op:<16}{p['bytes']:>12,}{p['median_s'] * 1e3:>12.3f}"
                    f"{p['algorithm_gbytes_per_s']:>13.3f}{p['bus_gbytes_per_s']:>13.3f}"
                )
            fit = entry.get("fit")
            if fit:
                print(
                    f"{op:<16}fit t = {fit['intercept_s'] * 1e6:.1f}us + "
                    f"bytes/{fit['bandwidth_gbytes_per_s']:.2f}GB/s   "
                    f"R^2 = {fit['r_squared']:.4f}"
                )
