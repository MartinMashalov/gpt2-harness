"""Kill a rank mid-training, restart from the last checkpoint, prove the run is unharmed.

This is a real multi-process job: ``torch.distributed`` on the gloo backend,
``DistributedDataParallel``, one process per rank, a real crash (``os._exit``
inside the training loop, so no cleanup runs and the collective the other ranks
are sitting in never completes), and a real restart from disk. It runs on CPU,
which is the only thing about it that is a stand-in; the control flow is what a
GPU job does.

What "unharmed" means here
--------------------------
Not "the loss still goes down". The test runs the same job twice -- once
uninterrupted, once with a rank killed at step 12 when the last checkpoint was
at step 10 -- and compares the two loss trajectories step by step. If the
restart replays the right data in the right order with the right optimiser
state, the resumed losses are identical to the uninterrupted ones, not merely
close. Anything that resumes at the top of the epoch, or drops the optimiser
moments, or reshuffles, shows up immediately as a divergence at the first step
after the resume.

Three things have to be in the checkpoint for that to hold, and all three are:

* model parameters,
* optimiser state (Adam's two moments and its step count -- restoring the
  weights but not the moments restarts the bias correction and gives a visible
  loss bump for a few hundred steps),
* the dataloader position for every rank.

What is *not* covered locally: a real node failure also takes out the NCCL
communicator, and the surviving ranks have to be torn down rather than left
hanging on a collective. On gloo with a killed peer the other ranks block until
the launcher kills them, which is exactly what ``torchrun`` does on a worker
failure. See ``docs/CLUSTER.md`` for the elastic case.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer_internals.cluster.checkpoint import (
    REPLICATE,
    CheckpointIndex,
    load_extra,
    load_full,
    save_sharded,
)
from transformer_internals.cluster.streaming import (
    ShardedStream,
    StreamState,
    TokenShardSource,
)
from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT

__all__ = ["JobSpec", "RunReport", "free_port", "read_log", "run_with_recovery", "worker_main"]


def free_port() -> int:
    """Ask the OS for a port nobody is using, then hand it to the workers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class JobSpec:
    """Everything both the launcher and the workers need to agree on."""

    data_path: str
    ckpt_dir: str
    log_path: str
    block_size: int = 32
    micro_batch: int = 4
    steps: int = 20
    lr: float = 3e-4
    seed: int = 1234
    ckpt_every: int = 5
    master_port: int = 0
    # Failure injection. ``crash_step`` is 1-based and refers to the step the
    # rank is about to run when it dies.
    crash_step: int | None = None
    crash_rank: int = 1
    resume: bool = False
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    vocab_size: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_config(self) -> GPTConfig:
        return GPTConfig(
            vocab_size=self.vocab_size,
            n_positions=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=0.0,
        )


@dataclass
class RunReport:
    """What the launcher observed."""

    launches: int = 0
    crashed_at: float | None = None
    recovered_at: float | None = None
    resumed_from_step: int | None = None
    time_to_recover_s: float | None = None
    events: list[str] = field(default_factory=list)


def _log(path: str, record: dict[str, Any]) -> None:
    with Path(path).open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_log(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _build_batch(items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([a for a, _ in items])
    y = torch.stack([b for _, b in items])
    return x, y


def worker_main(rank: int, world_size: int, spec_dict: dict[str, Any]) -> None:
    """One rank. Called by ``mp.spawn``; also the shape of a ``torchrun`` entrypoint."""
    spec = JobSpec(**spec_dict)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(spec.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        _train(rank, world_size, spec)
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _train(rank: int, world_size: int, spec: JobSpec) -> None:
    torch.manual_seed(spec.seed)
    model = GPT(spec.model_config())
    source = TokenShardSource(spec.data_path, spec.block_size)

    start_step = 0
    stream_state: StreamState | None = None
    optim_state: dict[str, Any] | None = None
    ckpt = Path(spec.ckpt_dir)
    if spec.resume and (ckpt / "index.json").exists():
        state, index = load_full(ckpt)
        model.load_state_dict(state)
        extra = load_extra(ckpt, 0)
        start_step = index.step
        optim_state = extra["optimizer"]
        stream_state = StreamState.from_dict(extra["stream"])

    ddp = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if optim_state is not None:
        opt.load_state_dict(optim_state)

    if stream_state is None:
        stream_state = StreamState(
            num_samples=len(source), seed=spec.seed, world_size=world_size,
            positions=[0] * world_size, shuffle=True,
        )
    stream = ShardedStream(source, rank=rank, world_size=world_size, state=stream_state, prefetch=2)
    it = iter(stream)

    step = start_step
    for step in range(start_step + 1, spec.steps + 1):
        if spec.crash_step is not None and step == spec.crash_step and rank == spec.crash_rank:
            _log(spec.log_path, {"event": "crash", "rank": rank, "step": step, "t": time.time()})
            os._exit(137)  # SIGKILL-shaped exit: no atexit, no communicator teardown

        items = [next(it) for _ in range(spec.micro_batch)]
        x, y = _build_batch(items)
        out = ddp(x, targets=y)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # The number every rank agrees on, which is the one worth comparing.
        reduced = loss.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
        if rank == 0:
            _log(spec.log_path, {"event": "step", "step": step, "loss": float(reduced),
                                 "resumed_from": start_step, "t": time.time()})

        if step % spec.ckpt_every == 0:
            _checkpoint(rank, world_size, spec, model, opt, stream, step)

    if rank == 0:
        _log(spec.log_path, {"event": "done", "step": step, "t": time.time()})


def _checkpoint(
    rank: int, world_size: int, spec: JobSpec, model: GPT, opt: torch.optim.Optimizer,
    stream: ShardedStream, step: int,
) -> None:
    """Gather the per-rank stream positions, then write from rank 0.

    Under DDP every rank holds the same weights, so one writer is right. The
    only genuinely per-rank state is the dataloader position, and it is gathered
    with a one-integer ``all_gather`` -- which is also the barrier that makes the
    checkpoint a consistent cut across ranks.
    """
    pos = torch.tensor([stream.position], dtype=torch.long)
    gathered = [torch.zeros_like(pos) for _ in range(world_size)]
    dist.all_gather(gathered, pos)
    positions = [int(t.item()) for t in gathered]
    if rank == 0:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_sharded(
            spec.ckpt_dir, state, dict.fromkeys(state, REPLICATE), rank=0, world_size=1, step=step,
            global_shapes={k: list(v.shape) for k, v in state.items()},
            extra={"optimizer": opt.state_dict(), "stream": stream.state_dict(positions)},
            meta={"world_size": world_size, "wrote_at": time.time()},
        )
    dist.barrier()


# ------------------------------------------------------------------ launcher


def _rank_env(rank: int, world_size: int, spec: JobSpec, spec_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(spec.master_port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "TI_JOB_SPEC": str(spec_path),
            # One thread per rank. Oversubscribing a laptop's cores across ranks
            # makes the step time noise, and on a real node it is how you end up
            # with 8 ranks each spawning 96 OpenMP threads.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def launch(
    world_size: int, spec: JobSpec, *, kill_rank_at_step: tuple[int, int] | None = None,
    timeout_s: float = 300.0,
) -> tuple[bool, list[int | None], float | None]:
    """Start ``world_size`` real OS processes, one per rank, and supervise them.

    This is what ``torchrun`` does, minus the rendezvous backend: set
    ``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT`` in each child's
    environment, start them, and watch. When one dies the rest are killed rather
    than left blocked in a collective, because a rank waiting on a peer that
    will never arrive hangs until the NCCL/gloo timeout and holds the allocation
    the whole time.

    Args:
        kill_rank_at_step: ``(rank, step)``. The launcher SIGKILLs that rank as
            soon as the log shows the job reached that step. A real
            ``SIGKILL`` from outside the process is the closest local analogue
            of a node dropping off the fabric: no unwinding, no teardown, no
            chance to flush anything.

    Returns:
        ``(all_clean, exit_codes, kill_time)``.
    """
    import subprocess

    spec_path = Path(spec.ckpt_dir).parent / f"jobspec_{spec.master_port}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec.to_dict()))
    procs = [
        subprocess.Popen(
            [__import__("sys").executable, "-m", "transformer_internals.cluster.failure"],
            env=_rank_env(r, world_size, spec, spec_path),
        )
        for r in range(world_size)
    ]
    kill_time: float | None = None
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if kill_rank_at_step is not None and kill_time is None:
                victim, at_step = kill_rank_at_step
                if _reached_step(spec.log_path, at_step):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(procs[victim].pid, 9)
                    kill_time = time.time()
                    _log(spec.log_path, {"event": "killed", "rank": victim, "t": kill_time})
            codes = [p.poll() for p in procs]
            if any(c is not None and c != 0 for c in codes):
                for p in procs:
                    with contextlib.suppress(Exception):
                        p.kill()
                for p in procs:
                    p.wait()
                return False, [p.poll() for p in procs], kill_time
            if all(c == 0 for c in codes):
                return True, codes, kill_time
            # While a kill is still pending, poll hard. A step of this job takes
            # single-digit milliseconds, so a 20 ms supervisor loop can be two or
            # three steps late, and "kill at step 12" then lands after the step-15
            # checkpoint has already been written. A supervisor that is late by
            # more than a step is not injecting the failure it says it is.
            time.sleep(0.002 if (kill_rank_at_step is not None and kill_time is None) else 0.02)
    finally:
        for p in procs:
            if p.poll() is None:
                with contextlib.suppress(Exception):
                    p.kill()
                p.wait()
    return False, [p.poll() for p in procs], kill_time


def _reached_step(log_path: str, step: int) -> bool:
    return any(rec.get("event") == "step" and rec["step"] >= step for rec in read_log(log_path))


def run_with_recovery(
    world_size: int, spec: JobSpec, *, kill_rank_at_step: tuple[int, int] | None = None,
    max_restarts: int = 3,
) -> RunReport:
    """Launch, and on a worker failure relaunch from the last checkpoint.

    This is ``torchrun --max-restarts`` in miniature. The restart is not free
    and the report says how expensive it was: elapsed time from the ``SIGKILL``
    to the first optimiser step landing after the restart, which is what a
    cluster operator means by time-to-recover. On a real job that interval also
    contains the scheduler's requeue latency and the time to re-read a
    checkpoint over the storage fabric, both of which dwarf what is measured
    here; the structure of the measurement is the same.
    """
    report = RunReport()
    attempt_spec = spec
    kill_spec = kill_rank_at_step
    for attempt in range(max_restarts + 1):
        report.launches += 1
        ok, codes, kill_time = launch(world_size, attempt_spec, kill_rank_at_step=kill_spec)
        if ok:
            report.events.append(f"attempt {attempt}: clean exit {codes}")
            break
        report.crashed_at = kill_time if kill_time is not None else time.time()
        report.events.append(f"attempt {attempt}: worker failure, exit codes {codes}")
        if not (Path(spec.ckpt_dir) / "index.json").exists():
            report.events.append("no checkpoint on disk; the run is lost")
            break
        report.resumed_from_step = CheckpointIndex.read(Path(spec.ckpt_dir)).step
        report.events.append(f"resuming from step {report.resumed_from_step}")
        attempt_spec = JobSpec(
            **{**spec.to_dict(), "resume": True, "master_port": free_port()}
        )
        kill_spec = None  # only fail once
    if report.crashed_at is not None:
        for rec in read_log(spec.log_path):
            if rec.get("event") == "step" and rec["t"] > report.crashed_at:
                report.recovered_at = rec["t"]
                report.time_to_recover_s = rec["t"] - report.crashed_at
                break
    return report


def make_corpus(path: Path | str, num_tokens: int, vocab_size: int, seed: int = 0) -> Path:
    """A deterministic synthetic corpus, so the data is not the variable under test."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    return TokenShardSource.write(path, tokens)


def _main() -> None:
    """Rank entrypoint. ``RANK``/``WORLD_SIZE``/``TI_JOB_SPEC`` come from the launcher."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    spec_dict = json.loads(Path(os.environ["TI_JOB_SPEC"]).read_text())
    worker_main(rank, world_size, spec_dict)


if __name__ == "__main__":
    _main()
