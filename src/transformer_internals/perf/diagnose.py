"""Why is this run at 30% of expected throughput?

This is the tool for the question that actually gets asked. It takes a training
step and partitions its wall clock into terms that can each be attacked:

    step wall clock = dataloader stall
                    + exposed collective time
                    + compute

    compute         = the roofline-ideal arithmetic time
                    + everything else

Each term is measured, not inferred, and the residual is never hidden. The
findings are ranked by the fraction of step time each one accounts for, so the
output is a work order rather than a dashboard. Every finding carries the
measurement it came from.

The five probes, in the order the ranking usually puts them:

1. **Dataloader stall.** Time the training loop spends blocked in ``next(loader)``
   with the device idle. Measured by timing the fetch and the compute separately
   inside the same loop. An input pipeline that cannot keep up is the single most
   common cause of a run at a fraction of expected throughput, and it is invisible
   in any metric that averages over the step.
2. **Collectives, and whether they overlap.** Two numbers, not one. *Exposed*
   comm is the step time that disappears when the communication is removed, which
   is the only part that costs anything. *Standalone* comm is what the same
   all-reduce volume takes on its own. The ratio is the overlap: DDP's bucketed
   all-reduce fires during the backward pass and hides most of its cost, while a
   hand-written all-reduce after ``loss.backward()`` hides none of it. Two
   implementations with identical communication volume can differ by the whole of
   that volume in wall clock, and this probe tells them apart.
3. **MFU against the roofline ceiling.** Not against 100%: against the ceiling
   this op mix can reach on this machine, which is lower, because LayerNorm and
   softmax are bandwidth-limited and no amount of scheduling changes that.
4. **Memory-bound operator fraction.** From the profiler, the share of operator
   self time outside the matmul family. This is the part of the MFU gap that is
   physics rather than a bug.
5. **Batch saturation.** Sweep the batch size and watch tokens per second. If
   throughput is still climbing steeply, the machine is not saturated and the
   step is paying fixed overhead it could amortise.

Pathologies can be injected on purpose (:class:`SyntheticLoader` takes a
``stall_s``; :func:`collective_probe` runs an un-overlapped arm), which is how
the tool is tested: inject a known fault, check that it comes out at the top of
the ranking with roughly the size it was injected at.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT
from transformer_internals.perf.mfu import flops_per_token_exact
from transformer_internals.perf.profiling import profile_training_step
from transformer_internals.perf.roofline import MachinePeak, op_roofline_table

__all__ = [
    "DiagnosisReport",
    "Finding",
    "StepBreakdown",
    "SyntheticLoader",
    "collective_probe",
    "diagnose",
    "measure_step_breakdown",
    "sweep_batch_throughput",
]


# Thresholds. A fraction of step time above the first number is critical, above
# the second is significant, above the third is minor, below it is healthy.
_SEVERITY_BANDS = (0.30, 0.10, 0.03)


def _severity(fraction: float) -> str:
    hi, mid, lo = _SEVERITY_BANDS
    if fraction >= hi:
        return "critical"
    if fraction >= mid:
        return "significant"
    if fraction >= lo:
        return "minor"
    return "healthy"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class SyntheticLoader:
    """A batch source with a controllable stall, for measuring pipeline overlap.

    Real dataloaders block on disk, on decompression, or on tokenisation. This one
    blocks on ``time.sleep``, which models the IO-bound case exactly: the training
    process is descheduled and the device sits idle, and no CPU is stolen from the
    model, so the compute measurement stays clean. ``stall_s=0`` is a loader that
    is never the problem, which is the control arm.

    Args:
        vocab_size: Range of the synthetic token ids.
        batch / seq: Shape of each batch.
        device: Where the batch is placed.
        stall_s: Seconds to block before returning each batch.
        seed: Generator seed, so two runs see identical data.
    """

    def __init__(
        self,
        vocab_size: int,
        batch: int,
        seq: int,
        device: torch.device | str = "cpu",
        stall_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.batch = batch
        self.seq = seq
        self.device = torch.device(device)
        self.stall_s = float(stall_s)
        self.generator = torch.Generator().manual_seed(seed)
        self.batches_served = 0

    def __iter__(self) -> SyntheticLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stall_s > 0:
            time.sleep(self.stall_s)
        ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch, self.seq + 1),
            generator=self.generator,
        )
        self.batches_served += 1
        x = ids[:, :-1].contiguous().to(self.device)
        y = ids[:, 1:].contiguous().to(self.device)
        return x, y


@dataclass
class StepBreakdown:
    """Per-step wall clock split into fetching the batch and computing on it.

    The reported statistic is the **minimum** over the timed steps, not the mean
    or the median. Contention only ever adds time: another process on the box, a
    page fault, the scheduler moving the thread. The fastest step this
    configuration managed is the closest available estimate of what it costs when
    nothing is in the way, and it is the only statistic that stays comparable
    across configurations measured minutes apart on a machine that is also doing
    other work. Every raw sample is kept in the payload, so anyone who prefers
    the median can compute it.
    """

    steps: int
    fetch_s: list[float] = field(default_factory=list)
    compute_s: list[float] = field(default_factory=list)

    @property
    def best_fetch_s(self) -> float:
        return min(self.fetch_s)

    @property
    def best_compute_s(self) -> float:
        return min(self.compute_s)

    @property
    def best_step_s(self) -> float:
        return self.best_fetch_s + self.best_compute_s

    @property
    def stall_fraction(self) -> float:
        return self.best_fetch_s / self.best_step_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "statistic": "minimum over the timed steps",
            "fetch_s": self.fetch_s,
            "compute_s": self.compute_s,
            "best_fetch_s": self.best_fetch_s,
            "best_compute_s": self.best_compute_s,
            "best_step_s": self.best_step_s,
            "median_fetch_s": statistics.median(self.fetch_s),
            "median_compute_s": statistics.median(self.compute_s),
            "stall_fraction": self.stall_fraction,
        }


def measure_step_breakdown(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    loader: SyntheticLoader,
    steps: int = 8,
    warmup: int = 2,
    device: torch.device | str = "cpu",
) -> StepBreakdown:
    """Run real training steps, timing the fetch and the compute separately.

    The device is synchronised at the end of the compute and *not* between the
    fetch and the compute, so an asynchronous backend does not get to charge its
    queued work to the fetch. On a synchronous CPU backend this makes no
    difference; on MPS it is the difference between a real number and nonsense.

    Args:
        model: The model. Left in training mode.
        optimizer: The optimiser to step.
        loader: Batch source.
        steps: Timed steps.
        warmup: Untimed steps first.
        device: Device to synchronise on.

    Returns:
        A :class:`StepBreakdown`.
    """
    device = torch.device(device)
    model.train()
    for _ in range(warmup):
        x, y = next(loader)
        optimizer.zero_grad(set_to_none=True)
        model(x, targets=y)["loss"].backward()
        optimizer.step()
    _synchronize(device)

    out = StepBreakdown(steps=steps)
    for _ in range(steps):
        t0 = time.perf_counter()
        x, y = next(loader)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        model(x, targets=y)["loss"].backward()
        optimizer.step()
        _synchronize(device)
        t2 = time.perf_counter()
        out.fetch_s.append(t1 - t0)
        out.compute_s.append(t2 - t1)
    return out


def sweep_batch_throughput(
    cfg: GPTConfig,
    batch_sizes: tuple[int, ...],
    seq: int,
    device: torch.device | str = "cpu",
    steps: int = 3,
    warmup: int = 1,
    model: GPT | None = None,
) -> list[dict[str, float]]:
    """Tokens per second as a function of batch size, on the same model.

    A step has fixed costs that do not scale with the batch: Python dispatch, the
    optimiser update over every parameter, kernel launch. At small batch those
    costs are most of the step, and throughput climbs steeply with batch size. A
    curve that has flattened means the machine is saturated and a larger batch
    buys nothing; a curve still climbing means it is not, and the run is paying
    overhead it does not have to.

    Args:
        cfg: Model architecture.
        batch_sizes: Batch sizes to try, ascending.
        seq: Sequence length, held fixed.
        device: Where to run.
        steps: Timed steps per size.
        warmup: Untimed steps per size.
        model: Reuse an existing model rather than building one per size.

    Returns:
        One record per batch size with ``tokens_per_s`` and ``step_s``.
    """
    device = torch.device(device)
    model = model if model is not None else GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rows: list[dict[str, float]] = []
    for b in batch_sizes:
        x = torch.randint(0, cfg.vocab_size, (b, seq), device=device)
        y = torch.randint(0, cfg.vocab_size, (b, seq), device=device)

        def one(x: torch.Tensor = x, y: torch.Tensor = y) -> None:
            opt.zero_grad(set_to_none=True)
            model(x, targets=y)["loss"].backward()
            opt.step()

        for _ in range(warmup):
            one()
        _synchronize(device)
        samples = []
        for _ in range(steps):
            t0 = time.perf_counter()
            one()
            _synchronize(device)
            samples.append(time.perf_counter() - t0)
        best = min(samples)
        rows.append(
            {
                "batch": float(b),
                "step_s": best,
                "tokens_per_s": (b * seq) / best,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Collectives: measured with real torch.distributed processes over gloo.
# ---------------------------------------------------------------------------


def _run_collective_rank(rank: int, world_size: int, kw: dict[str, Any], out_path: str) -> None:
    """One rank of the collective probe.

    Four training arms, all on identical data, identical model shape and
    identical gradient volume:

    * ``no_comm``: local backward, gradients never leave the process. The floor
      the other three are measured against.
    * ``manual_allreduce``: every gradient all-reduced in a loop after
      ``loss.backward()`` has returned. Correct, and hides nothing.
    * ``ddp_default_buckets``: ``DistributedDataParallel`` at its default 25 MB
      bucket cap. The reducer fires a bucket's all-reduce from an autograd hook
      as soon as that bucket's gradients are all ready, so communication happens
      during the backward pass.
    * ``ddp_small_buckets``: the same at a 1 MB cap. More, smaller buckets means
      the first collective can start earlier and there is more backward left to
      hide it behind.

    Plus three standalone references that issue the same call patterns with no
    compute to hide behind. Each training arm is compared against the reference
    whose pattern it uses, because that is the only fair denominator: a loop of
    52 small all-reduces costs more than one flat all-reduce of the same bytes,
    and charging the loop against the flat number would report as poor overlap
    what is really a latency difference.

    Every arm is timed **round robin**, one step of each per outer iteration,
    rather than in blocks. This machine is shared, and block-timed arms measure
    whatever else the box was doing during their block. Round robin gives every
    arm the same ambient load, and the reported statistic is the minimum, which
    is the closest thing to the contention-free cost.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, int(kw["threads_per_rank"])))

    cfg = GPTConfig(**kw["model_config"])
    x = torch.randint(0, cfg.vocab_size, (kw["batch"], kw["seq"]))
    y = torch.randint(0, cfg.vocab_size, (kw["batch"], kw["seq"]))
    steps, warmup = int(kw["steps"]), int(kw["warmup"])

    def fresh() -> tuple[GPT, torch.optim.Optimizer]:
        torch.manual_seed(0)
        m = GPT(cfg)
        return m, torch.optim.AdamW(m.parameters(), lr=1e-4)

    model, opt = fresh()
    n_grad_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_bytes = n_grad_elems * 4
    grad_tensor_count = sum(1 for p in model.parameters() if p.requires_grad)

    def local_step() -> None:
        opt.zero_grad(set_to_none=True)
        model(x, targets=y)["loss"].backward()
        opt.step()

    def manual_step() -> None:
        opt.zero_grad(set_to_none=True)
        model(x, targets=y)["loss"].backward()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size
        opt.step()

    arms: dict[str, Any] = {"no_comm": local_step, "manual_allreduce": manual_step}
    keep_alive = []
    for arm_name, cap_mb in (("ddp_default_buckets", 25), ("ddp_small_buckets", 1)):
        dmodel, _ = fresh()
        ddp = DistributedDataParallel(dmodel, bucket_cap_mb=cap_mb)
        dopt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
        keep_alive.append((dmodel, ddp, dopt))

        def ddp_step(ddp: Any = ddp, dopt: Any = dopt) -> None:
            dopt.zero_grad(set_to_none=True)
            ddp(x, targets=y)["loss"].backward()
            dopt.step()

        arms[arm_name] = ddp_step

    flat = torch.zeros(n_grad_elems)
    per_param = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
    chunks = list(torch.split(flat, (1 << 20) // 4))

    def flat_call() -> None:
        dist.all_reduce(flat)

    def per_param_call() -> None:
        for g in per_param:
            dist.all_reduce(g)

    def chunk_call() -> None:
        for c in chunks:
            dist.all_reduce(c)

    arms["standalone_flat"] = flat_call
    arms["standalone_per_param"] = per_param_call
    arms["standalone_chunks_1mb"] = chunk_call

    for fn in arms.values():
        for _ in range(warmup):
            fn()
    dist.barrier()

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(steps):
        for name, fn in arms.items():
            dist.barrier()
            t0 = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter() - t0)

    if rank == 0:
        best = {name: min(v) for name, v in samples.items()}
        floor = best["no_comm"]
        payload: dict[str, Any] = {
            "world_size": world_size,
            "backend": "gloo",
            "transport": "loopback TCP on one host; a shared-memory number, not an interconnect measurement",
            "statistic": "minimum over round-robin timed steps",
            "n_grad_elements": n_grad_elems,
            "n_grad_tensors": grad_tensor_count,
            "grad_bytes": grad_bytes,
            "median_step_s": {
                k: statistics.median(v) for k, v in samples.items() if not k.startswith("standalone")
            },
            "best_step_s": {k: v for k, v in best.items() if not k.startswith("standalone")},
            "standalone_s": {
                k.replace("standalone_", ""): v
                for k, v in best.items()
                if k.startswith("standalone")
            },
            "standalone_allreduce_bytes_per_s": grad_bytes / best["standalone_flat"],
            "samples": samples,
        }
        reference = {
            "manual_allreduce": "per_param",
            "ddp_default_buckets": "flat",
            "ddp_small_buckets": "chunks_1mb",
        }
        for arm_name, ref in reference.items():
            arm_s = best[arm_name]
            alone = best[f"standalone_{ref}"]
            exposed = max(0.0, arm_s - floor)
            payload[arm_name] = {
                "reference_pattern": ref,
                "standalone_comm_s": alone,
                "exposed_comm_s": exposed,
                "exposed_fraction_of_step": exposed / arm_s,
                "overlap_fraction": min(1.0, max(0.0, 1.0 - exposed / alone)),
                "cost_beyond_standalone_s": exposed - alone,
                "step_s": arm_s,
                "step_without_comm_s": floor,
            }
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


def _collective_entrypoint() -> None:
    """Child-process entry point. Reads its rank and its config from the environment.

    This is the same contract ``torchrun`` uses: the launcher sets ``RANK``,
    ``WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT``, and every rank discovers
    the others through that rendezvous. Launching by environment rather than by
    ``multiprocessing.spawn`` is not a stylistic choice: ``spawn`` re-imports the
    parent's ``__main__``, which works from a script and breaks under pytest,
    where ``__main__`` is the pytest console entry point.
    """
    kw = json.loads(Path(os.environ["COLLECTIVE_PROBE_CONFIG"]).read_text(encoding="utf-8"))
    _run_collective_rank(
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        kw,
        kw["out_path"],
    )


def collective_probe(
    model_config: dict[str, Any] | None = None,
    world_size: int = 2,
    batch: int = 2,
    seq: int = 64,
    steps: int = 15,
    warmup: int = 3,
    port: int = 29561,
    out_path: str | Path | None = None,
    threads_per_rank: int = 2,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Measure exposed collective time and overlap, with real ``torch.distributed`` ranks.

    Launches ``world_size`` processes on this machine, each running the same
    model over the gloo backend, and times three arms (no communication,
    un-overlapped manual all-reduce, DDP).

    The transport is loopback TCP, so the *bandwidth* here is a shared-memory
    number and has nothing to say about InfiniBand or NVLink. That is not what
    the probe is for. Overlap is a scheduling property of the framework: whether
    the reducer issues its collectives from autograd hooks during the backward
    pass or after it, which is identical on any wire. Absolute collective cost on
    interconnect hardware is not measurable on this machine and is not claimed
    anywhere in this repository.

    Args:
        model_config: Kwargs for :class:`GPTConfig`. Small by default, so the
            probe finishes in seconds.
        world_size: Number of ranks.
        batch / seq: Per-rank step shape.
        steps / warmup: Timed and untimed steps per arm.
        port: Rendezvous port for the process group.
        out_path: Where rank 0 writes its JSON. A temp file if ``None``.
        threads_per_rank: Intra-op threads per rank, so the ranks do not fight
            over the same cores and turn the measurement into a scheduling study.
        timeout_s: Give up on a rank that has not exited by then.

    Returns:
        The payload rank 0 wrote.

    Raises:
        RuntimeError: If any rank exits non-zero.
    """
    import subprocess
    import sys
    import tempfile

    import transformer_internals

    model_config = model_config or {
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 512,
        "vocab_size": 8192,
        "n_positions": 256,
        "dropout": 0.0,
    }
    tmp = Path(tempfile.mkdtemp(prefix="collective_probe_"))
    out_path = Path(out_path) if out_path is not None else tmp / "rank0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = {
        "model_config": model_config,
        "batch": batch,
        "seq": seq,
        "steps": steps,
        "warmup": warmup,
        "threads_per_rank": threads_per_rank,
        "out_path": str(out_path),
    }
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps(kw), encoding="utf-8")

    src_root = str(Path(transformer_internals.__file__).resolve().parents[1])
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = src_root + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env["MASTER_ADDR"] = "127.0.0.1"
    base_env["MASTER_PORT"] = str(port)
    base_env["WORLD_SIZE"] = str(world_size)
    base_env["COLLECTIVE_PROBE_CONFIG"] = str(cfg_path)
    base_env["OMP_NUM_THREADS"] = str(threads_per_rank)

    code = (
        "from transformer_internals.perf.diagnose import _collective_entrypoint; "
        "_collective_entrypoint()"
    )
    procs = []
    for rank in range(world_size):
        env = dict(base_env)
        env["RANK"] = str(rank)
        procs.append(subprocess.Popen([sys.executable, "-c", code], env=env))
    for rank, proc in enumerate(procs):
        rc = proc.wait(timeout=timeout_s)
        if rc != 0:
            for other in procs:
                if other.poll() is None:
                    other.kill()
            raise RuntimeError(f"collective probe rank {rank} exited with code {rc}")
    return json.loads(out_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One diagnosed cause, with the measurement behind it.

    Attributes:
        name: Short identifier.
        severity: ``critical`` / ``significant`` / ``minor`` / ``healthy``.
        cost_fraction: Share of step wall clock this accounts for. The ranking
            key, and the reason the output is a work order and not a dashboard.
        recoverable: Whether fixing it is expected to give the time back.
            Memory-bound operator time is real time that is not recoverable by
            scheduling, and saying so is more useful than listing it as a bug.
        headline: One line, with the number in it.
        evidence: The measurements that produced the finding.
        recommendation: What to do about it.
    """

    name: str
    severity: str
    cost_fraction: float
    recoverable: bool
    headline: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisReport:
    """Everything the diagnosis measured, plus the ranked findings."""

    label: str
    device: str
    shape: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d

    def text(self) -> str:
        """The ranked diagnosis, as printed."""
        t = self.throughput
        lines = [
            f"diagnosis: {self.label}",
            f"  device {self.device}   shape batch={self.shape.get('batch')} "
            f"seq={self.shape.get('seq')}   {t.get('tokens_per_s', 0):,.0f} tokens/s",
            f"  step {t.get('step_s', 0) * 1e3:.1f} ms (fastest of "
            f"{self.measurements.get('step_breakdown', {}).get('steps', 0)})"
            f"   model FLOPs/step {t.get('model_flops_per_step', 0):.3e}"
            f"   MFU {t.get('mfu_end_to_end', 0) * 100:.2f}%"
            f" of measured peak {t.get('peak_flops_per_s', 0) / 1e12:.2f} TFLOP/s",
        ]
        ceiling = t.get("roofline_ceiling_mfu")
        if ceiling:
            lines.append(
                f"  roofline ceiling for this op mix: {ceiling * 100:.1f}% "
                f"(the op mix cannot reach 100%, see the memory-bound finding)"
            )
        lines.append("")
        lines.append("  ranked findings (share of step wall clock):")
        for i, f in enumerate(self.findings, 1):
            flag = "" if f.recoverable else "  [not recoverable by scheduling]"
            lines.append(
                f"   {i}. [{f.severity:>11}] {f.cost_fraction * 100:5.1f}%  {f.name}{flag}"
            )
            lines.append(f"        {f.headline}")
            for k, v in f.evidence.items():
                vs = f"{v:.4g}" if isinstance(v, float) else str(v)
                lines.append(f"        - {k}: {vs}")
            if f.recommendation:
                lines.append(f"        -> {f.recommendation}")
        return "\n".join(lines)


def diagnose(
    peak: MachinePeak,
    cfg: GPTConfig | None = None,
    batch: int = 4,
    seq: int = 128,
    device: torch.device | str = "cpu",
    label: str = "run",
    loader_stall_s: float = 0.0,
    steps: int = 6,
    warmup: int = 2,
    batch_sweep: tuple[int, ...] | None = (1, 2, 4, 8),
    profile_ops: bool = True,
    collectives: dict[str, Any] | None = None,
    collective_arm: str = "ddp_default_buckets",
    model: GPT | None = None,
) -> DiagnosisReport:
    """Diagnose one training configuration and rank what is costing it time.

    Args:
        peak: Measured machine peaks, from
            :func:`~transformer_internals.perf.roofline.measure_machine_peak`.
        cfg: Model architecture.
        batch / seq: Step shape.
        device: Where to run.
        label: Name for this configuration, used in the printout.
        loader_stall_s: Inject a dataloader stall of this many seconds per batch.
            Zero is the healthy control.
        steps / warmup: Timed and untimed steps.
        batch_sweep: Batch sizes for the saturation probe. ``None`` skips it.
        profile_ops: Run the profiler for the memory-bound operator fraction.
            CPU only, see :mod:`~transformer_internals.perf.profiling`.
        collectives: A payload from :func:`collective_probe`. ``None`` means this
            is a single-process run and the collective finding is skipped.
        collective_arm: Which arm of that payload describes this run: one of
            ``"manual_allreduce"``, ``"ddp_default_buckets"``,
            ``"ddp_small_buckets"``.
        model: Reuse an existing model.

    Returns:
        A :class:`DiagnosisReport` with findings sorted by cost.
    """
    device = torch.device(device)
    cfg = cfg or GPTConfig(n_layer=4, n_head=4, n_embd=256, vocab_size=4096, n_positions=max(seq, 64))
    model = model if model is not None else GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = SyntheticLoader(cfg.vocab_size, batch, seq, device=device, stall_s=loader_stall_s)

    breakdown = measure_step_breakdown(model, opt, loader, steps=steps, warmup=warmup, device=device)

    fb = flops_per_token_exact(cfg, seq)
    tokens = batch * seq
    flops_per_step = fb["training_flops_per_token"] * tokens
    step_s = breakdown.best_step_s
    compute_s = breakdown.best_compute_s

    # The roofline-ideal time for this shape: sum over the block-level op table,
    # scaled to the whole model, for the forward pass, times three for training.
    rows = op_roofline_table(peak, cfg, batch=batch, seq=seq)
    block_ideal_s = sum(r.roofline_seconds for r in rows)
    ideal_s = 3.0 * cfg.n_layer * block_ideal_s
    block_flops = sum(r.flops for r in rows)
    ceiling_mfu = (block_flops / block_ideal_s) / peak.peak_flops_per_s

    mfu_end_to_end = (flops_per_step / step_s) / peak.peak_flops_per_s
    mfu_compute_only = (flops_per_step / compute_s) / peak.peak_flops_per_s

    findings: list[Finding] = []

    # --- 1. dataloader stall
    stall_frac = breakdown.stall_fraction
    findings.append(
        Finding(
            name="dataloader stall",
            severity=_severity(stall_frac),
            cost_fraction=stall_frac,
            recoverable=True,
            headline=(
                f"{stall_frac * 100:.1f}% of the step is spent blocked in next(loader) "
                f"with the device idle "
                f"({breakdown.best_fetch_s * 1e3:.1f} ms fetch vs "
                f"{compute_s * 1e3:.1f} ms compute)"
            ),
            evidence={
                "fetch_ms": breakdown.best_fetch_s * 1e3,
                "compute_ms": compute_s * 1e3,
                "step_ms": step_s * 1e3,
                "injected_stall_ms": loader_stall_s * 1e3,
                "batches_served": loader.batches_served,
            },
            recommendation=(
                "prefetch: move decoding and collation onto worker processes and keep a "
                "queue ahead of the trainer, so the fetch happens during the previous step"
            )
            if stall_frac >= _SEVERITY_BANDS[2]
            else "input pipeline keeps up; nothing to do",
        )
    )

    # --- 2. collectives
    if collectives is not None:
        arm = collectives[collective_arm]
        exposed_frac = arm["exposed_fraction_of_step"]
        overlap = arm["overlap_fraction"]
        findings.append(
            Finding(
                name=f"exposed collective time ({collective_arm})",
                severity=_severity(exposed_frac),
                cost_fraction=exposed_frac,
                recoverable=True,
                headline=(
                    f"{exposed_frac * 100:.1f}% of the step is all-reduce that is not hidden "
                    f"behind compute; {overlap * 100:.0f}% of this arm's collective time does "
                    f"overlap the backward pass"
                ),
                evidence={
                    "world_size": collectives["world_size"],
                    "backend": collectives["backend"],
                    "grad_bytes_all_reduced": collectives["grad_bytes"],
                    "n_grad_tensors": collectives["n_grad_tensors"],
                    "step_ms": arm["step_s"] * 1e3,
                    "step_without_comm_ms": arm["step_without_comm_s"] * 1e3,
                    "exposed_comm_ms": arm["exposed_comm_s"] * 1e3,
                    "standalone_comm_ms": arm["standalone_comm_s"] * 1e3,
                    "standalone_reference_pattern": arm["reference_pattern"],
                    "cost_beyond_standalone_ms": arm["cost_beyond_standalone_s"] * 1e3,
                    "overlap_fraction": overlap,
                },
                recommendation=(
                    "issue the all-reduce from autograd hooks in buckets, the way DDP does, so "
                    "the gradients of the last layers go on the wire while the first layers are "
                    "still being differentiated; then size the buckets so there is more than one"
                )
                if overlap < 0.5
                else "communication is already overlapped with the backward pass",
            )
        )

    # --- 3. memory-bound operator fraction
    profile_report = None
    if profile_ops and device.type == "cpu":
        profile_report = profile_training_step(
            cfg, batch=batch, seq=seq, device=device, active_steps=1, model=model
        )
        mem_frac_of_compute = profile_report.memory_bound_self_time_fraction()
        cost = mem_frac_of_compute * (compute_s / step_s)
        top_non_matmul = [
            c for c in profile_report.categories if c["category"] not in ("matmul", "other")
        ][:3]
        findings.append(
            Finding(
                name="memory-bound operator time",
                severity=_severity(cost),
                cost_fraction=cost,
                recoverable=False,
                headline=(
                    f"{mem_frac_of_compute * 100:.1f}% of operator self time is outside the "
                    f"matmul family; the roofline says those ops cannot reach peak arithmetic"
                ),
                evidence={
                    "matmul_self_fraction": next(
                        (
                            c["self_fraction"]
                            for c in profile_report.categories
                            if c["category"] == "matmul"
                        ),
                        0.0,
                    ),
                    **{f"{c['category']}_self_fraction": c["self_fraction"] for c in top_non_matmul},
                    "ridge_point_flops_per_byte": peak.ridge_flops_per_byte,
                    "n_memory_bound_ops_in_block": sum(1 for r in rows if r.bound == "memory"),
                },
                recommendation=(
                    "fuse: these ops are bandwidth-limited, so the win comes from touching "
                    "memory once (fused LayerNorm, fused attention) rather than from faster "
                    "arithmetic"
                ),
            )
        )

    # --- 4. batch saturation
    sweep: list[dict[str, float]] = []
    if batch_sweep:
        sweep = sweep_batch_throughput(cfg, batch_sweep, seq, device=device, model=model)
        here = next(
            (r["tokens_per_s"] for r in sweep if int(r["batch"]) == batch),
            tokens / compute_s,
        )
        # Only batches at least as large as this one count. A *smaller* batch
        # reaching more tokens per second is a different phenomenon (cache
        # residency, usually) and it is not evidence that this batch is too small
        # to saturate the machine.
        larger = [r for r in sweep if int(r["batch"]) >= batch] or sweep
        best_row = max(larger, key=lambda r: r["tokens_per_s"])
        best = best_row["tokens_per_s"]
        shortfall = max(0.0, 1.0 - here / best)
        cost = shortfall * (compute_s / step_s)
        findings.append(
            Finding(
                name="batch too small to saturate",
                severity=_severity(cost),
                cost_fraction=cost,
                recoverable=True,
                headline=(
                    f"at batch {batch} the step reaches {here:,.0f} tokens/s against "
                    f"{best:,.0f} tokens/s at batch {int(best_row['batch'])}, a "
                    f"{shortfall * 100:.0f}% shortfall"
                ),
                evidence={
                    "tokens_per_s_by_batch": {
                        int(r["batch"]): round(r["tokens_per_s"]) for r in sweep
                    },
                    "best_batch_at_or_above_this_one": int(best_row["batch"]),
                    "shortfall_fraction": shortfall,
                },
                recommendation=(
                    "raise the micro-batch, or use gradient accumulation to keep the "
                    "optimiser step amortised over more tokens"
                )
                if cost >= _SEVERITY_BANDS[2]
                else "batch is large enough that per-step fixed costs are amortised",
            )
        )

    findings.sort(key=lambda f: f.cost_fraction, reverse=True)

    return DiagnosisReport(
        label=label,
        device=str(device),
        shape={
            "batch": batch,
            "seq": seq,
            "tokens_per_step": tokens,
            "model_config": cfg.to_dict(),
            "n_params": model.num_parameters(),
        },
        throughput={
            "step_s": step_s,
            "compute_s": compute_s,
            "fetch_s": breakdown.best_fetch_s,
            "statistic": "minimum over the timed steps",
            "tokens_per_s": tokens / step_s,
            "model_flops_per_step": flops_per_step,
            "achieved_flops_per_s_end_to_end": flops_per_step / step_s,
            "achieved_flops_per_s_compute_only": flops_per_step / compute_s,
            "peak_flops_per_s": peak.peak_flops_per_s,
            "mfu_end_to_end": mfu_end_to_end,
            "mfu_compute_only": mfu_compute_only,
            "roofline_ceiling_mfu": ceiling_mfu,
            "roofline_ideal_step_s": ideal_s,
        },
        findings=findings,
        measurements={
            "step_breakdown": breakdown.to_dict(),
            "flops_breakdown": fb,
            "batch_sweep": sweep,
            "collectives": collectives,
            "profile": profile_report.to_dict() if profile_report else None,
        },
        meta={
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "peak_source": "measured on this machine",
            "peak_device": peak.device,
            "peak_device_matches_run_device": peak.device == str(device),
        },
    )
