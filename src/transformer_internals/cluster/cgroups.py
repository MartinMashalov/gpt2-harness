"""Read the cgroup v2 limits this process is actually running under.

Why a training harness cares
----------------------------
Every scheduler that runs training jobs -- Slurm with cgroup task affinity,
Kubernetes, Docker, Ray under either -- enforces its resource limits through
cgroups, not through anything the process can see with ``free`` or
``nproc``. Inside the container those tools still report the *host's* memory and
CPU count, so a dataloader that sizes its worker pool from ``os.cpu_count()`` or
a cache that sizes itself from total RAM will be wrong by an order of magnitude
and the job will be throttled or killed for reasons that do not appear in its own
logs. The numbers that matter are in ``/sys/fs/cgroup``.

The distinction that gets jobs killed
-------------------------------------
``memory.max`` is a hard wall. A cgroup that cannot be brought under it is
handed to the cgroup OOM killer, which kills a process inside the cgroup -- with
no ``MemoryError``, no traceback, no stack. From the launcher's point of view the
rank exits with signal 9 and the run stops. This is why an out-of-memory training
rank looks like a node failure rather than an exception, and why
``memory.events`` (which counts ``oom`` and ``oom_kill``) is the first thing to
read after an unexplained rank death.

``memory.high`` is a soft wall: exceeding it does not kill anything, it throttles
the cgroup and pushes it into reclaim. A job over ``memory.high`` does not crash,
it goes slow -- and "the run is at 30% of expected throughput" is exactly what
that looks like from outside.

Swap changes the failure mode but rarely helps a training job. With
``memory.swap.max`` above zero the kernel can page anonymous memory out instead
of OOM-killing, so the rank survives -- at disk latency, mid-step, while every
other rank waits for it at the next collective. One rank swapping stalls the
whole job, and because the collective has no timeout short enough to matter it
usually presents as a hang rather than as a slow rank. Most GPU clusters run
training cgroups with swap disabled for that reason: a fast death is easier to
diagnose and cheaper than a slow one.

``cpu.max`` is a quota (``quota period``, both in microseconds). Exceeding it does
not kill anything either: the cgroup is throttled at the end of each period, and
``cpu.stat``'s ``nr_throttled`` and ``throttled_usec`` are how you prove it. A
dataloader that has been given 2 CPUs and spawns 32 worker threads spends its
life being throttled, and the GPU waits.

Run it under a limit to see it work::

    docker run --rm --memory=512m --memory-swap=512m --cpus=1.5 \\
        -v "$PWD":/w -w /w python:3.11-slim \\
        python src/transformer_internals/cluster/cgroups.py

``deploy/cgroups_demo.sh`` does exactly that and
``deploy/cgroups_demo_output.txt`` is the captured result.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CgroupInfo", "read_cgroup", "report"]

CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _proc_cgroup_path() -> str | None:
    """The cgroup this process is in, from ``/proc/self/cgroup``.

    On cgroup v2 the file has a single line ``0::<path>``. On v1 there is one
    line per controller and the path means something different per controller,
    which is one of several reasons v2 exists.
    """
    text = _read(Path("/proc/self/cgroup"))
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return parts[2]
    return text.splitlines()[0]


@dataclass
class CgroupInfo:
    """The subset of cgroup v2 that a training job should look at."""

    available: bool
    reason: str = ""
    path: str | None = None
    memory_max: str | None = None
    memory_high: str | None = None
    memory_current: str | None = None
    memory_swap_max: str | None = None
    memory_events: str | None = None
    cpu_max: str | None = None
    cpu_stat: str | None = None
    pids_max: str | None = None


def read_cgroup() -> CgroupInfo:
    """Read this process's cgroup v2 files. Never raises."""
    if not CGROUP_ROOT.exists():
        return CgroupInfo(
            available=False,
            reason=f"no {CGROUP_ROOT} on this host (platform: {platform.system()}). "
                   "cgroups are a Linux kernel feature; on macOS the container "
                   "runtime's own Linux VM is where they exist.",
        )
    if not (CGROUP_ROOT / "cgroup.controllers").exists():
        return CgroupInfo(
            available=False,
            reason="/sys/fs/cgroup exists but has no cgroup.controllers, so this is "
                   "cgroup v1 (or a hybrid mount). The v1 layout puts each controller "
                   "under its own mount point, e.g. /sys/fs/cgroup/memory/.",
        )
    # Inside a container the cgroup namespace is usually rooted, so the files
    # for *this* cgroup are at the top of the mount.
    return CgroupInfo(
        available=True,
        path=_proc_cgroup_path(),
        memory_max=_read(CGROUP_ROOT / "memory.max"),
        memory_high=_read(CGROUP_ROOT / "memory.high"),
        memory_current=_read(CGROUP_ROOT / "memory.current"),
        memory_swap_max=_read(CGROUP_ROOT / "memory.swap.max"),
        memory_events=_read(CGROUP_ROOT / "memory.events"),
        cpu_max=_read(CGROUP_ROOT / "cpu.max"),
        cpu_stat=_read(CGROUP_ROOT / "cpu.stat"),
        pids_max=_read(CGROUP_ROOT / "pids.max"),
    )


def _fmt_bytes(v: str | None) -> str:
    if v is None:
        return "unreadable"
    if v == "max":
        return "max (no limit at this level; the real ceiling is an ancestor cgroup or the host)"
    try:
        n = int(v)
    except ValueError:
        return v
    return f"{n} bytes ({n/2**30:.2f} GiB)"


def _cpu_allowance(cpu_max: str | None) -> str:
    if not cpu_max:
        return "unreadable"
    quota, _, period = cpu_max.partition(" ")
    if quota == "max":
        return "max (unthrottled; the job can use every CPU the scheduler put in its cpuset)"
    try:
        return (
            f"{int(quota)}us of CPU time per {int(period)}us period = "
            f"{int(quota)/int(period):.2f} CPUs' worth. Going over does not fail the "
            "process, it throttles it until the next period."
        )
    except ValueError:
        return cpu_max


def report() -> str:
    """Human-readable summary, including what each limit does to a training job."""
    info = read_cgroup()
    out: list[str] = []
    w = out.append
    w("cgroup v2 limits for this process")
    w("=" * 60)
    if not info.available:
        w(f"NOT AVAILABLE: {info.reason}")
        w("")
        w("Run this inside a Linux container to see real numbers:")
        w("  docker run --rm --memory=512m --memory-swap=512m --cpus=1.5 \\")
        w("      -v \"$PWD\":/w -w /w python:3.11-slim \\")
        w("      python src/transformer_internals/cluster/cgroups.py")
        return "\n".join(out)

    w(f"cgroup path        : {info.path}")
    w(f"memory.max         : {_fmt_bytes(info.memory_max)}")
    w(f"memory.high        : {_fmt_bytes(info.memory_high)}")
    w(f"memory.current     : {_fmt_bytes(info.memory_current)}")
    w(f"memory.swap.max    : {_fmt_bytes(info.memory_swap_max)}")
    w(f"pids.max           : {info.pids_max or 'unreadable'}")
    w(f"cpu.max            : {info.cpu_max or 'unreadable'}")
    w(f"                     {_cpu_allowance(info.cpu_max)}")
    w("")
    w("memory.events (counters since this cgroup was created):")
    for line in (info.memory_events or "unreadable").splitlines():
        w(f"  {line}")
    w("cpu.stat:")
    for line in (info.cpu_stat or "unreadable").splitlines():
        if line.startswith(("nr_throttled", "throttled_usec")):
            w(f"  {line}   <- non-zero means the CPU quota is being hit")
        else:
            w(f"  {line}")
    w("")

    try:
        current = int(info.memory_current or 0)
        limit = int(info.memory_max) if info.memory_max not in (None, "max") else None
    except ValueError:
        current, limit = 0, None
    if limit:
        w(f"Headroom           : {(limit-current)/2**20:.0f} MiB "
          f"({100*current/limit:.1f}% of memory.max in use)")
    w("")
    w("What happens to a training job at each wall:")
    swap_off = info.memory_swap_max in ("0", None)
    w("  memory.max  hard. The cgroup OOM killer picks a process in this cgroup and")
    w("              SIGKILLs it. No Python exception, no traceback: the rank just")
    w("              disappears and the launcher sees exit code -9. Check")
    w("              memory.events' oom_kill counter to tell this apart from a")
    w("              hardware or network failure.")
    w("  memory.high soft. No kill; the cgroup is throttled into reclaim and the")
    w("              step time inflates. This is a throughput bug, not a crash.")
    if swap_off:
        w("  swap        memory.swap.max is 0 here, so there is no paging escape hatch:")
        w("              the job hits memory.max and dies rather than going slow. For a")
        w("              collective-synchronised job that is the better failure -- one")
        w("              rank paging to disk stalls every other rank at the next")
        w("              all-reduce, and a hang is harder to diagnose than a kill.")
    else:
        w("  swap        swap is permitted here. The rank will survive an overshoot by")
        w("              paging, at disk latency, mid-step, while every other rank waits")
        w("              for it at the next collective. Prefer swap off for training.")
    w("  cpu.max     quota, not a kill. Throttling shows up in cpu.stat and looks")
    w("              like a slow dataloader, which looks like a slow GPU.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
