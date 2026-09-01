"""Collective wrappers that count the bytes they move, and a process launcher.

Everything in :mod:`transformer_internals.parallel` calls the collectives through
this module rather than calling ``torch.distributed`` directly. The reason is
that the cost of a parallelism strategy is decided by two numbers -- how many
collectives it issues per step, and how many bytes each one carries -- and both
are properties of the *algorithm*, not of the hardware. They can therefore be
measured exactly on a laptop and then priced against any interconnect.

Two byte counts are reported, and they are not the same thing:

``payload_bytes``
    The size of the buffer handed to the collective, summed over calls. This is
    *measured*: it is ``tensor.numel() * tensor.element_size()`` of the actual
    tensors this process passed in. No modelling, no estimation.

``wire_bytes``
    What a ring implementation moves on the network per rank to service that
    payload. This is a *model*, given by the standard ring cost analysis
    (Thakur, Rabenseifner & Gropp, "Optimization of Collective Communication
    Operations in MPICH", IJHPCA 2005; the same accounting NCCL documents as
    "bus bandwidth"):

    ==================  ====================================  =========================
    collective          bytes sent per rank                   bytes received per rank
    ==================  ====================================  =========================
    all-reduce          ``2 * (p-1)/p * S``                   ``2 * (p-1)/p * S``
    all-gather          ``(p-1)/p * S``                       ``(p-1)/p * S``
    reduce-scatter      ``(p-1)/p * S``                       ``(p-1)/p * S``
    broadcast           ``(p-1)/p * S`` (amortised)           ``S`` on non-roots
    point-to-point      ``S``                                 ``S``
    ==================  ====================================  =========================

    where ``S`` is the payload in bytes -- the full buffer for all-reduce and
    broadcast, the *output* buffer for all-gather, the *input* buffer for
    reduce-scatter -- and ``p`` is the world size.

The distinction matters because gloo does not implement a ring reduce-scatter:
``dist.reduce_scatter_tensor`` on the gloo backend is an all-reduce followed by a
slice, so this machine really moves all-reduce volume for it. The payload count
is still exactly right, and the wire model still says what an NCCL ring on real
hardware would move. Both are reported; neither is presented as the other.
"""

from __future__ import annotations

import os
import queue
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "CollectiveRecord",
    "CommCounter",
    "all_gather_into",
    "all_gather_list",
    "all_reduce",
    "broadcast",
    "counter",
    "counter_scope",
    "get_counter",
    "isend",
    "recv",
    "reduce_scatter_into",
    "send",
    "spawn_workers",
]


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #


@dataclass
class CollectiveRecord:
    """Per-collective tally for one rank.

    Attributes:
        op: Collective name (``all_reduce``, ``all_gather``, ...).
        calls: Number of invocations.
        elements: Total elements passed through, summed over calls.
        payload_bytes: Total bytes passed through, summed over calls. Measured.
        wire_bytes: Bytes a ring implementation would move on the network, per
            rank, for those payloads. Modelled -- see the module docstring.
    """

    op: str
    calls: int = 0
    elements: int = 0
    payload_bytes: int = 0
    wire_bytes: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "elements": self.elements,
            "payload_bytes": self.payload_bytes,
            "wire_bytes_ring_model": self.wire_bytes,
        }


def _ring_wire_bytes(op: str, payload_bytes: int, world_size: int) -> float:
    """Bytes a ring implementation moves per rank. Model, not measurement."""
    p = max(world_size, 1)
    if p == 1:
        return 0.0
    if op == "all_reduce":
        return 2.0 * (p - 1) / p * payload_bytes
    if op in ("all_gather", "reduce_scatter", "broadcast"):
        return (p - 1) / p * payload_bytes
    # send / recv and anything else: the payload crosses one link, once.
    return float(payload_bytes)


class CommCounter:
    """Accumulates exact collective volume for one rank.

    Args:
        world_size: Ranks in the group. Used only by the ring wire model.
    """

    def __init__(self, world_size: int = 1) -> None:
        self.world_size = world_size
        self.records: dict[str, CollectiveRecord] = {}
        #: Set by the trainers so per-step volume can be reported.
        self.steps: int = 0

    def record(self, op: str, tensors: torch.Tensor | Sequence[torch.Tensor]) -> None:
        """Tally one collective call over the given payload tensor(s)."""
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        elements = sum(int(t.numel()) for t in tensors)
        payload = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
        rec = self.records.setdefault(op, CollectiveRecord(op=op))
        rec.calls += 1
        rec.elements += elements
        rec.payload_bytes += payload
        rec.wire_bytes += _ring_wire_bytes(op, payload, self.world_size)

    def reset(self) -> None:
        self.records.clear()
        self.steps = 0

    @property
    def total_payload_bytes(self) -> int:
        return sum(r.payload_bytes for r in self.records.values())

    @property
    def total_wire_bytes(self) -> float:
        return sum(r.wire_bytes for r in self.records.values())

    @property
    def total_calls(self) -> int:
        return sum(r.calls for r in self.records.values())

    def summary(self) -> dict[str, Any]:
        """A JSON-safe dict of everything counted, plus per-step figures."""
        steps = max(self.steps, 1)
        out: dict[str, Any] = {
            "world_size": self.world_size,
            "steps": self.steps,
            "per_collective": {op: r.as_dict() for op, r in sorted(self.records.items())},
            "total_calls": self.total_calls,
            "total_payload_bytes": self.total_payload_bytes,
            "total_wire_bytes_ring_model": self.total_wire_bytes,
            "calls_per_step": self.total_calls / steps,
            "payload_bytes_per_step": self.total_payload_bytes / steps,
            "wire_bytes_per_step_ring_model": self.total_wire_bytes / steps,
        }
        return out


_ACTIVE = CommCounter()


def get_counter() -> CommCounter:
    """The counter every wrapper in this module writes to."""
    return _ACTIVE


@contextmanager
def counter_scope(world_size: int):
    """Install a fresh counter for the duration of the block, and yield it."""
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = CommCounter(world_size=world_size)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = previous


# Short alias so call sites read ``comms.counter().steps += 1``.
counter = get_counter


# --------------------------------------------------------------------------- #
# collective wrappers
# --------------------------------------------------------------------------- #


def all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None) -> torch.Tensor:
    """In-place all-reduce (SUM by default), counted.

    Args:
        tensor: Contiguous buffer, reduced in place.
        op: A ``dist.ReduceOp``; SUM when omitted.
        group: Process group, or the default one.

    Returns:
        The same tensor, for chaining.
    """
    get_counter().record("all_reduce", tensor)
    dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_into(output: torch.Tensor, tensor: torch.Tensor, group: Any = None) -> torch.Tensor:
    """All-gather ``tensor`` from every rank into the flat ``output``, counted.

    The payload charged is the *output* buffer: that is the volume the
    collective has to place on every rank.
    """
    get_counter().record("all_gather", output)
    dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
    return output


def all_gather_list(tensor: torch.Tensor, world_size: int, group: Any = None) -> list[torch.Tensor]:
    """All-gather into a list of per-rank tensors, counted.

    Used where the gathered pieces are consumed separately (sequence parallel)
    rather than as one flat buffer.
    """
    out = [torch.empty_like(tensor) for _ in range(world_size)]
    get_counter().record("all_gather", out)
    dist.all_gather(out, tensor.contiguous(), group=group)
    return out


def reduce_scatter_into(
    output: torch.Tensor, tensor: torch.Tensor, op: Any = None, group: Any = None
) -> torch.Tensor:
    """Reduce-scatter ``tensor`` into this rank's shard ``output``, counted.

    The payload charged is the *input* buffer, which is the volume reduced.

    Note that gloo has no native ring reduce-scatter; PyTorch services this call
    with an all-reduce plus a slice, so on this machine the wire traffic is
    all-reduce traffic. ``payload_bytes`` is still exact and the ring wire model
    still describes what NCCL would do.
    """
    get_counter().record("reduce_scatter", tensor)
    dist.reduce_scatter_tensor(output, tensor.contiguous(), op=op or dist.ReduceOp.SUM, group=group)
    return output


def broadcast(tensor: torch.Tensor, src: int = 0, group: Any = None) -> torch.Tensor:
    """Broadcast from ``src``, counted."""
    get_counter().record("broadcast", tensor)
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def send(tensor: torch.Tensor, dst: int, tag: int = 0, group: Any = None) -> None:
    """Blocking point-to-point send, counted."""
    get_counter().record("send", tensor)
    dist.send(tensor.contiguous(), dst=dst, tag=tag, group=group)


def recv(tensor: torch.Tensor, src: int, tag: int = 0, group: Any = None) -> torch.Tensor:
    """Blocking point-to-point receive into ``tensor``, counted."""
    get_counter().record("recv", tensor)
    dist.recv(tensor, src=src, tag=tag, group=group)
    return tensor


class SendHandle:
    """A pending non-blocking send, holding its buffer alive until it completes.

    Keeping the reference is not tidiness. ``isend`` returns before the data has
    left; if the only reference to the buffer is dropped, the tensor is freed
    underneath an in-flight transfer, and what arrives at the peer is whatever
    the allocator did with that memory next. That is a race, so it shows up as a
    rare wrong number rather than a crash.
    """

    def __init__(self, work: Any, buffer: torch.Tensor) -> None:
        self._work = work
        self._buffer = buffer

    def wait(self) -> None:
        self._work.wait()
        self._buffer = None  # type: ignore[assignment]

    def is_completed(self) -> bool:
        return bool(self._work.is_completed())


def isend(tensor: torch.Tensor, dst: int, tag: int = 0, group: Any = None) -> SendHandle:
    """Non-blocking point-to-point send, counted.

    Pipeline parallelism needs the send to be non-blocking, because a schedule
    where every rank sends before it receives deadlocks the moment the ordering
    of two ranks disagrees. With asynchronous sends and blocking receives, the
    only thing a rank can block on is data it genuinely does not have yet --
    which is exactly the definition of the pipeline bubble, and is what makes it
    measurable.
    """
    get_counter().record("send", tensor)
    buf = tensor if tensor.is_contiguous() else tensor.contiguous()
    return SendHandle(dist.isend(buf, dst=dst, tag=tag, group=group), buf)


# --------------------------------------------------------------------------- #
# launcher
# --------------------------------------------------------------------------- #


@dataclass
class WorkerResult:
    """What one rank returned, or the traceback that stopped it."""

    rank: int
    value: Any = None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _entry(
    rank: int,
    world_size: int,
    store_file: str,
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    queue: Any,
) -> None:
    """Child-process entry point: init the group, run ``fn``, report back."""
    # One thread per rank. Without this, p ranks each spin up a full BLAS thread
    # pool on the same cores and the pipeline timing measurements become noise.
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_file}",
            rank=rank,
            world_size=world_size,
        )
        with counter_scope(world_size) as counted:
            value = fn(rank=rank, world_size=world_size, **kwargs)
        result = WorkerResult(rank=rank, value=value, payload=counted.summary())
    except Exception:
        result = WorkerResult(rank=rank, error=traceback.format_exc())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    queue.put(result)


def spawn_workers(
    fn: Callable[..., Any],
    world_size: int,
    kwargs: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> list[WorkerResult]:
    """Run ``fn`` on ``world_size`` gloo processes and collect what each returned.

    This is the whole harness. ``torchrun`` would do the same thing, but going
    through ``mp.spawn`` directly means the tests are ordinary pytest functions
    with no external launcher, which is what lets them run in CI.

    Args:
        fn: A module-level (picklable) callable taking ``rank`` and
            ``world_size`` keyword arguments plus whatever is in ``kwargs``. Its
            return value must be picklable; return plain Python objects rather
            than tensors, since tensors cross the queue by shared memory and the
            file descriptors do not outlive the child.
        world_size: Number of processes.
        kwargs: Extra keyword arguments forwarded to ``fn``.
        timeout: Seconds to wait for the whole group.

    Returns:
        One :class:`WorkerResult` per rank, ordered by rank. A rank that raised
        carries its traceback in ``error``.

    Raises:
        RuntimeError: If a rank did not report within ``timeout``.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    tmpdir = tempfile.mkdtemp(prefix="ti-parallel-")
    store_file = str(Path(tmpdir) / "store")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_entry,
            args=(rank, world_size, store_file, fn, kwargs or {}, q),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()

    # Poll rather than block on the queue: a rank that segfaults or is killed by
    # the OOM killer never puts anything, and blocking would then wait out the
    # whole timeout instead of failing in the second it took to die.
    results: list[WorkerResult] = []
    deadline = time.monotonic() + timeout
    try:
        while len(results) < world_size:
            try:
                results.append(q.get(timeout=0.2))
                continue
            except queue.Empty:
                pass
            dead = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
            if dead:
                raise RuntimeError(
                    f"rank process exited with code {dead[0].exitcode} without reporting"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(f"a rank did not report within {timeout}s")
    except Exception:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=30)
        for p in procs:
            if p.is_alive():
                p.terminate()
        with os.scandir(tmpdir) as it:
            for entry in it:
                Path(entry.path).unlink(missing_ok=True)
        Path(tmpdir).rmdir()

    results.sort(key=lambda r: r.rank)
    failures = [r for r in results if r.error]
    if failures:
        raise RuntimeError(
            f"rank {failures[0].rank} failed:\n{failures[0].error}"
        )
    return results
