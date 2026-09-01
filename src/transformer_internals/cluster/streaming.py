"""A sharded, resumable streaming dataloader.

Three properties, all of them asserted in ``tests/test_cluster.py``:

1. **Disjoint shards.** Every rank reads its own slice of the corpus. No rank
   sees a sample another rank saw, and between them the ranks cover the epoch
   exactly once.
2. **The iterator position is part of the checkpoint.** Not the epoch number --
   the position *within* the epoch, per rank. Restarting from step 4,000 of a
   10,000-step epoch has to resume at sample 4,000 of the shard, not at the top
   of the epoch. Resuming at the top is the single most common data bug in a
   restartable trainer, and it is invisible: the loss curve looks fine, the model
   just quietly sees a fraction of the corpus many times and the rest never.
3. **Elastic resharding of the stream.** The checkpoint stores every rank's
   position and the world size it was written under. On resume at a *different*
   world size the remaining samples are recomputed and dealt out again, so
   exactly-once coverage holds across the restart even when the number of ranks
   changed.

The design
----------
There is one global order for the epoch: a seeded permutation of ``[0, N)``.
Rank ``r`` of ``W`` takes the entries at positions ``r, r+W, r+2W, ...``. That
is a strided deal rather than contiguous blocks, and it matters for streaming:
contiguous blocks make each rank read one long region of the corpus, so ranks
that draw a region of short documents run ahead and the step time is set by the
slowest rank all epoch. A strided deal mixes every region across every rank.

Resuming is expressed as replanning rather than as seeking. Given every rank's
position, the set of already-consumed order positions is known exactly, so the
remainder of the epoch is dealt out again over however many ranks now exist.
When the world size has not changed and all ranks stopped at the same step --
the normal case, because ranks checkpoint together -- replanning reproduces the
original assignment exactly, so the resumed run sees the same samples in the
same order as an uninterrupted one. The failure test relies on that.

Prefetch
--------
A background thread reads ahead into a bounded queue. The position counter only
advances when a sample is handed to the consumer, so samples sitting in the
queue at checkpoint time are re-read after the restart rather than skipped.
That is the correct trade: re-reading is free, skipping is silent data loss.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "DelayedSource",
    "ShardedStream",
    "StreamState",
    "TokenShardSource",
    "measure_throughput",
    "plan_indices",
]


@dataclass
class StreamState:
    """Everything needed to resume the stream, and small enough to checkpoint.

    Attributes:
        num_samples: Size of the epoch.
        seed: Seed of the global permutation.
        epoch: Which epoch this position is inside.
        world_size: The world size the positions were recorded under.
        positions: ``positions[r]`` = number of samples rank ``r`` has consumed
            in this epoch. Written by rank 0 after gathering from every rank.
    """

    num_samples: int
    seed: int = 0
    epoch: int = 0
    world_size: int = 1
    positions: list[int] = field(default_factory=lambda: [0])
    shuffle: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> StreamState:
        return StreamState(
            num_samples=int(d["num_samples"]),
            seed=int(d["seed"]),
            epoch=int(d["epoch"]),
            world_size=int(d["world_size"]),
            positions=[int(x) for x in d["positions"]],
            shuffle=bool(d.get("shuffle", True)),
        )

    @property
    def consumed(self) -> int:
        return sum(self.positions)


def global_order(num_samples: int, seed: int, epoch: int, shuffle: bool) -> np.ndarray:
    """The one order the whole job agrees on for this epoch.

    Derived from ``(seed, epoch)`` alone, so every rank computes the same array
    without communicating and a restart reconstructs it exactly.
    """
    if not shuffle:
        return np.arange(num_samples, dtype=np.int64)
    rng = np.random.default_rng(seed * 1_000_003 + epoch)
    return rng.permutation(num_samples).astype(np.int64)


def plan_indices(state: StreamState, rank: int, world_size: int) -> np.ndarray:
    """Global sample indices that rank ``rank`` of ``world_size`` should read next.

    Handles the fresh case (all positions zero), the same-world resume, and the
    elastic resume where ``world_size != state.world_size``.
    """
    order = global_order(state.num_samples, state.seed, state.epoch, state.shuffle)
    if state.consumed == 0:
        return order[rank::world_size]
    consumed = np.zeros(state.num_samples, dtype=bool)
    for r, pos in enumerate(state.positions):
        if pos:
            consumed[r : r + pos * state.world_size : state.world_size] = True
    remaining = order[~consumed]
    return remaining[rank::world_size]


class TokenShardSource:
    """Fixed-length training samples cut out of a flat token file.

    Sample ``i`` is ``tokens[i*block : i*block + block + 1]``: ``block`` inputs
    and the same window shifted by one as targets. The file is memory-mapped, so
    a rank touches only the pages its own shard needs and the page cache is
    shared between ranks on the same node instead of each holding its own copy.
    The map is opened lazily and per process, because a memmap cannot be
    inherited across a fork and stay valid.
    """

    def __init__(self, path: Path | str, block_size: int, dtype: str = "uint16") -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.dtype = dtype
        self._map: np.memmap | None = None
        itemsize = np.dtype(dtype).itemsize
        total = self.path.stat().st_size // itemsize
        self.num_samples = max(0, (total - 1) // block_size)

    @staticmethod
    def write(path: Path | str, tokens: np.ndarray, dtype: str = "uint16") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens.astype(dtype).tofile(path)
        return path

    def _mm(self) -> np.memmap:
        if self._map is None:
            self._map = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._map

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        mm = self._mm()
        lo = i * self.block_size
        window = np.asarray(mm[lo : lo + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


class ShardedStream:
    """Iterate this rank's shard, with a checkpointable position.

    Args:
        source: anything with ``__len__`` and ``__getitem__``; a
            :class:`TokenShardSource` in the trainer, a list of ids in tests.
        rank, world_size: this rank's place in the job.
        state: resume point. ``None`` starts a fresh epoch.
        prefetch: queue depth of the reader thread. ``0`` reads inline.
    """

    def __init__(
        self,
        source: Sequence[Any],
        *,
        rank: int,
        world_size: int,
        state: StreamState | None = None,
        seed: int = 0,
        shuffle: bool = True,
        prefetch: int = 0,
    ) -> None:
        self.source = source
        self.rank = rank
        self.world_size = world_size
        self.prefetch = prefetch
        self.state = state or StreamState(
            num_samples=len(source), seed=seed, world_size=world_size,
            positions=[0] * world_size, shuffle=shuffle,
        )
        if len(source) != self.state.num_samples:
            raise ValueError(
                f"source has {len(source)} samples but the checkpointed stream state describes "
                f"{self.state.num_samples}; the corpus changed under the checkpoint"
            )
        self._plan = plan_indices(self.state, rank, world_size)
        #: Samples this rank has handed to the consumer since the resume point.
        self.consumed_here = 0
        #: Position at the moment of resume, so ``position`` is absolute.
        self._base = self.state.positions[rank] if world_size == self.state.world_size else 0

    @property
    def plan(self) -> np.ndarray:
        """The global sample indices this rank will read, in order."""
        return self._plan

    @property
    def position(self) -> int:
        """Samples consumed by this rank in this epoch, including before a resume."""
        return self._base + self.consumed_here

    def state_dict(self, all_positions: list[int] | None = None) -> dict[str, Any]:
        """The stream's contribution to a checkpoint.

        Args:
            all_positions: positions of every rank, gathered by the caller (a
                one-integer ``all_gather``). Without it only this rank's slot is
                filled, which is enough for a same-world resume and not enough
                for an elastic one.
        """
        positions = list(all_positions) if all_positions else list(self.state.positions)
        if not all_positions:
            positions = positions + [0] * (self.world_size - len(positions))
            positions[self.rank] = self.position
        s = StreamState(
            num_samples=self.state.num_samples,
            seed=self.state.seed,
            epoch=self.state.epoch,
            world_size=self.world_size,
            positions=positions,
            shuffle=self.state.shuffle,
        )
        return s.to_dict()

    def __iter__(self) -> Iterator[Any]:
        if self.prefetch <= 0:
            for idx in self._plan:
                item = self.source[int(idx)]
                self.consumed_here += 1
                yield item
            return
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def _reader() -> None:
            for idx in self._plan:
                if stop.is_set():
                    break
                q.put(self.source[int(idx)])
            q.put(_SENTINEL)

        thread = threading.Thread(target=_reader, name=f"prefetch-{self.rank}", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    return
                self.consumed_here += 1
                yield item
        finally:
            stop.set()
            # Drain so the reader is never blocked on a full queue at shutdown.
            while thread.is_alive():
                try:
                    q.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)
            thread.join()


_SENTINEL = object()


class DelayedSource:
    """Wraps a source with a fixed per-read delay, to stand in for slow storage.

    A memory-mapped file that is already in the page cache reads in microseconds,
    so a prefetch thread against it measures nothing but queue overhead. Real
    training data comes off network storage or a shared filesystem where a read
    is hundreds of microseconds to milliseconds, and that is the regime prefetch
    exists for. ``time.sleep`` releases the GIL, so the delay overlaps with the
    consumer exactly as a real blocking read would.
    """

    def __init__(self, source: Sequence[Any], read_seconds: float) -> None:
        self.source = source
        self.read_seconds = read_seconds

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int) -> Any:
        time.sleep(self.read_seconds)
        return self.source[i]


def measure_throughput(
    source: Sequence[Any], *, rank: int = 0, world_size: int = 1, prefetch: int = 0,
    limit: int | None = None, per_sample_work_s: float = 0.0,
) -> tuple[float, int]:
    """Samples per second for one rank's shard. Returns ``(rate, n)``.

    Prefetch buys nothing unless the consumer is doing something while the
    reader reads. ``per_sample_work_s`` stands in for the training step: without
    it this measures a thread handing objects to itself.
    """
    stream = ShardedStream(source, rank=rank, world_size=world_size, prefetch=prefetch)
    n = 0
    t0 = time.perf_counter()
    for _ in stream:
        if per_sample_work_s:
            time.sleep(per_sample_work_s)
        n += 1
        if limit is not None and n >= limit:
            break
    dt = time.perf_counter() - t0
    return (n / dt if dt > 0 else float("inf")), n
