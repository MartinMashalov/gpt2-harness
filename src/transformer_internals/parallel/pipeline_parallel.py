"""Pipeline parallelism: different blocks on different ranks, GPipe and 1F1B.

Splitting a model by depth is the easiest kind of sharding to implement and the
hardest to make efficient, because a depth split serialises: stage 1 cannot
start until stage 0 has produced something. With one batch, ``p - 1`` of the
``p`` stages are idle at any instant. Splitting the batch into ``m``
micro-batches fills the pipe, and the idle fraction becomes

    bubble = (p - 1) / (m + p - 1)

which follows from the makespan. Every micro-batch costs ``t_f + t_b``; the
schedule takes ``(m + p - 1)`` slots of it rather than ``m``, because the pipe
takes ``p - 1`` slots to fill and the same to drain. This module *measures* that
fraction on real processes and compares it against the formula
(:func:`bubble_scan`), and separately simulates the schedule to show the
schedule logic itself reproduces the formula exactly (:func:`simulate`).

GPipe (Huang et al., arXiv:1811.06965) runs all ``m`` forwards, then all ``m``
backwards. 1F1B (Narayanan et al., "PipeDream", SOSP'19; the flush variant in
Megatron-LM) interleaves them as soon as the pipe is full. **Both have the same
bubble** -- that is not a bug in the measurement, it is the point. What 1F1B
changes is memory: GPipe must keep the activations of all ``m`` micro-batches
alive until its backward phase starts, while 1F1B keeps at most ``p`` of them.
This module measures that too, as ``peak_stash``, and it is the reason 1F1B is
what production pipelines use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from transformer_internals.parallel import comms
from transformer_internals.parallel.common import (
    current_device,
    identical_batch,
    identical_model,
    parallel_config,
)

__all__ = [
    "PipelineStage",
    "analytic_bubble",
    "bubble_scan_worker",
    "pipeline_worker",
    "schedule_order",
    "simulate",
]


def analytic_bubble(n_stages: int, micro_batches: int) -> float:
    """``(p - 1) / (m + p - 1)``: the fraction of stage-time spent idle."""
    return (n_stages - 1) / (micro_batches + n_stages - 1)


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


class PipelineStage(nn.Module):
    """One contiguous depth slice of a GPT.

    Stage 0 owns the embeddings, the last stage owns the final LayerNorm and the
    tied head, and everything in between owns only transformer blocks. The
    pieces this stage does not own are dropped, not merely unused, so the
    parameter count reported per rank is the real one.

    Args:
        model: A constructed GPT; consumed.
        stage / n_stages: Position in the pipeline.
    """

    def __init__(self, model: nn.Module, stage: int, n_stages: int) -> None:
        super().__init__()
        n_layer = len(model.h)
        if n_layer % n_stages:
            raise ValueError(f"n_layer={n_layer} must be divisible by n_stages={n_stages}")
        per = n_layer // n_stages
        self.stage = stage
        self.n_stages = n_stages
        self.is_first = stage == 0
        self.is_last = stage == n_stages - 1
        self.blocks = nn.ModuleList(list(model.h)[stage * per : (stage + 1) * per])
        self.config = model.config
        if self.is_first:
            self.wte = model.wte
            self.wpe = model.wpe
            self.drop = model.drop
        if self.is_last:
            self.ln_f = model.ln_f
            # Weight tying spans the pipeline: with the embedding on stage 0 and
            # the head on stage p-1, a tied weight would need its own all-reduce
            # between two non-adjacent stages on every step. Real pipelines
            # either place both on the same stage or pay for that collective.
            # Here the head is untied (see _reference_model) and the model the
            # pipeline is checked against is untied the same way, so the
            # comparison stays exact.
            self.lm_head = model.lm_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Token ids in on stage 0, hidden states everywhere else, logits out at the end."""
        if self.is_first:
            t = x.shape[1]
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)
            h = self.drop(self.wte(x) + self.wpe(pos))
        else:
            h = x
        for block in self.blocks:
            h, _ = block(h)
        if self.is_last:
            return self.lm_head(self.ln_f(h))
        return h


def _reference_model(config) -> nn.Module:
    """The single-process model the pipeline is checked against.

    Built with the head untied from the embedding, matching how
    :class:`PipelineStage` places it, so that a difference in the comparison is
    a pipeline bug and not a weight-tying difference.
    """
    model = identical_model(config, seed=0)
    model.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(
        current_device()
    )
    model.lm_head.weight = nn.Parameter(model.wte.weight.detach().clone())
    return model


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #


def schedule_order(rank: int, n_stages: int, micro_batches: int, kind: str) -> list[tuple[str, int]]:
    """The op sequence one rank executes, as ``("F"|"B", micro_batch)`` pairs.

    GPipe is all forwards then all backwards in reverse order. 1F1B gives rank
    ``r`` a warmup of ``p - 1 - r`` forwards -- the deeper the stage, the fewer,
    because it has to wait for the stages above it -- then alternates one
    forward with one backward, then drains.
    """
    m = micro_batches
    if kind == "gpipe":
        return [("F", i) for i in range(m)] + [("B", i) for i in reversed(range(m))]
    if kind != "1f1b":
        raise ValueError(f"unknown schedule {kind!r}")
    warmup = min(n_stages - 1 - rank, m)
    steady = m - warmup
    ops: list[tuple[str, int]] = [("F", i) for i in range(warmup)]
    for i in range(steady):
        ops.append(("F", warmup + i))
        ops.append(("B", i))
    ops.extend(("B", i) for i in range(steady, m))
    return ops


@dataclass
class SimResult:
    """Outcome of a dependency-respecting simulation of one schedule."""

    makespan: float
    bubble: float
    peak_stash: int


def simulate(
    n_stages: int, micro_batches: int, t_f: float, t_b: float, kind: str = "gpipe"
) -> SimResult:
    """Simulate a schedule against its true dependencies.

    Not a formula -- an event simulation. Forward ``i`` on stage ``r`` cannot
    start until forward ``i`` on stage ``r-1`` has finished and stage ``r`` is
    free; backward ``i`` on stage ``r`` waits for backward ``i`` on stage
    ``r+1`` (or, on the last stage, for its own forward). Running it and getting
    ``(p-1)/(m+p-1)`` out is a check on the schedule, independent of any timing
    noise on the machine.

    Args:
        n_stages / micro_batches: Pipeline shape.
        t_f / t_b: Per-micro-batch forward and backward cost.
        kind: ``"gpipe"`` or ``"1f1b"``.

    Returns:
        Makespan, bubble fraction, and the peak number of micro-batch
        activations any one stage has to keep alive.
    """
    orders = [schedule_order(r, n_stages, micro_batches, kind) for r in range(n_stages)]
    finish: dict[tuple[int, str, int], float] = {}
    rank_free = [0.0] * n_stages
    cursor = [0] * n_stages
    peak_stash = 0
    live = [0] * n_stages
    remaining = sum(len(o) for o in orders)

    while remaining:
        progressed = False
        for r in range(n_stages):
            if cursor[r] >= len(orders[r]):
                continue
            op, i = orders[r][cursor[r]]
            if op == "F":
                dep = 0.0 if r == 0 else finish.get((r - 1, "F", i))
            elif r == n_stages - 1:
                dep = finish.get((r, "F", i))
            else:
                dep = finish.get((r + 1, "B", i))
            if dep is None:
                continue
            start = max(rank_free[r], dep)
            end = start + (t_f if op == "F" else t_b)
            finish[(r, op, i)] = end
            rank_free[r] = end
            cursor[r] += 1
            remaining -= 1
            progressed = True
            live[r] += 1 if op == "F" else -1
            peak_stash = max(peak_stash, live[r])
        if not progressed:
            raise RuntimeError("schedule deadlocked in simulation")

    makespan = max(rank_free)
    ideal = micro_batches * (t_f + t_b)
    return SimResult(makespan=makespan, bubble=1.0 - ideal / makespan, peak_stash=peak_stash)


# --------------------------------------------------------------------------- #
# the real distributed run
# --------------------------------------------------------------------------- #


class _Timer:
    """Splits a rank's wall clock into compute and waiting-for-a-peer."""

    def __init__(self) -> None:
        self.compute = 0.0
        self.comm = 0.0

    def __call__(self, kind: str):
        timer = self

        class _Ctx:
            def __enter__(self) -> None:
                self.t0 = time.perf_counter()

            def __exit__(self, *exc: Any) -> None:
                dt = time.perf_counter() - self.t0
                if kind == "compute":
                    timer.compute += dt
                else:
                    timer.comm += dt

        return _Ctx()


def _run_pipeline_step(
    stage: PipelineStage,
    rank: int,
    world_size: int,
    micro_inputs: list[torch.Tensor],
    micro_targets: list[torch.Tensor],
    hidden_shape: tuple[int, ...],
    kind: str,
    timer: _Timer,
) -> tuple[float, int]:
    """One optimiser step's worth of pipeline: forwards, backwards, p2p.

    Sends are non-blocking and receives are blocking, which is both deadlock-free
    for any schedule and exactly the instrumentation we want: time spent in a
    receive *is* the bubble, measured rather than inferred.

    Returns:
        ``(loss_sum, peak_stash)`` where the loss is only meaningful on the last
        stage.
    """
    m = len(micro_inputs)
    device = micro_inputs[0].device
    ops = schedule_order(rank, world_size, m, kind)
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    pending: list[Any] = []
    loss_sum = 0.0
    peak_stash = 0

    for op, i in ops:
        if op == "F":
            if stage.is_first:
                x = micro_inputs[i]
            else:
                buf = torch.empty(hidden_shape, device=device)
                with timer("comm"):
                    comms.recv(buf, src=rank - 1)
                x = buf.requires_grad_(True)
            with timer("compute"):
                out = stage(x)
            inputs[i] = x
            outputs[i] = out
            peak_stash = max(peak_stash, len(outputs))
            if not stage.is_last:
                with timer("comm"):
                    pending.append((comms.isend(out.detach(), dst=rank + 1), out))
            else:
                with timer("compute"):
                    loss = (
                        nn.functional.cross_entropy(
                            out.reshape(-1, out.size(-1)), micro_targets[i].reshape(-1)
                        )
                        / m
                    )
                outputs[i] = loss
                loss_sum += float(loss) * m
        else:
            out = outputs.pop(i)
            x = inputs.pop(i)
            if stage.is_last:
                with timer("compute"):
                    out.backward()
            else:
                grad = torch.empty(hidden_shape, device=device)
                with timer("comm"):
                    comms.recv(grad, src=rank + 1)
                with timer("compute"):
                    out.backward(grad)
            if not stage.is_first:
                with timer("comm"):
                    pending.append((comms.isend(x.grad, dst=rank - 1), x.grad))

    for req, _keep in pending:
        req.wait()
    return loss_sum, peak_stash


def pipeline_worker(
    rank: int,
    world_size: int,
    micro_batches: int = 4,
    batch: int = 8,
    seq: int = 16,
    kind: str = "gpipe",
    check_grads: bool = True,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one pipelined step and compare its gradients to the single-process run.

    The comparison is per stage: this rank checks its own blocks' gradients
    against the same blocks in a model that saw the whole batch at once. A
    micro-batching bug -- forgetting the ``1/m`` on the loss is the usual one --
    shows up immediately as a factor-``m`` gradient error.
    """
    config = parallel_config(**(config_kwargs or {}))
    inputs, targets = identical_batch(config, batch, seq)
    if batch % micro_batches:
        raise ValueError(f"batch={batch} must be divisible by micro_batches={micro_batches}")
    micro = batch // micro_batches
    micro_inputs = list(inputs.split(micro))
    micro_targets = list(targets.split(micro))
    hidden_shape = (micro, seq, config.n_embd)

    stage = PipelineStage(_reference_model(config), rank, world_size)
    timer = _Timer()
    dist.barrier()
    t0 = time.perf_counter()
    loss_sum, peak_stash = _run_pipeline_step(
        stage, rank, world_size, micro_inputs, micro_targets, hidden_shape, kind, timer
    )
    wall = time.perf_counter() - t0
    comms.get_counter().steps += 1

    result: dict[str, Any] = {
        "rank": rank,
        "kind": kind,
        "micro_batches": micro_batches,
        "loss": loss_sum,
        "peak_stash": peak_stash,
        "compute_seconds": timer.compute,
        "comm_seconds": timer.comm,
        "wall_seconds": wall,
        "stage_params": sum(p.numel() for p in stage.parameters()),
    }

    if check_grads:
        ref = _reference_model(config)
        ref(inputs, targets=targets)["loss"].backward()
        ref_stage = PipelineStage(ref, rank, world_size)
        errors = [
            float((a.grad - b.grad).abs().max())
            for a, b in zip(stage.parameters(), ref_stage.parameters(), strict=True)
            if a.grad is not None and b.grad is not None
        ]
        scales = [
            float(b.grad.abs().max())
            for b in ref_stage.parameters()
            if b.grad is not None
        ]
        result["max_grad_error"] = max(errors) if errors else 0.0
        result["reference_grad_scale"] = max(scales) if scales else 0.0
        result["compared_tensors"] = len(errors)
    return result


def bubble_scan_worker(
    rank: int,
    world_size: int,
    micro_batch_counts: tuple[int, ...] = (1, 2, 4, 8),
    batch: int = 8,
    seq: int = 32,
    repeats: int = 3,
    warmup: int = 1,
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the bubble as a function of micro-batch count, for both schedules.

    Method. Every rank splits its own wall clock into time spent computing and
    time spent blocked in a receive. Summed over ranks, the second is the
    bubble. The reported fraction is

        ``1 - sum_r(compute_r) / (p * makespan)``

    with the makespan taken from a barrier before the step to the last rank's
    finish. That is a direct measurement, and it charges the pipeline for
    everything real -- the gloo transfers, the Python dispatch, the scheduler on
    a 10-core laptop running ``p`` processes -- which is why it sits above the
    formula rather than on it. The gap is the honest part of the result.

    ``batch`` is held fixed as ``m`` varies, so more micro-batches means smaller
    ones: the same trade a real run makes.
    """
    config = parallel_config(**(config_kwargs or {}))
    inputs, targets = identical_batch(config, batch, seq)
    stage = PipelineStage(_reference_model(config), rank, world_size)

    rows: list[dict[str, Any]] = []
    for kind in ("gpipe", "1f1b"):
        for m in micro_batch_counts:
            if batch % m:
                continue
            micro = batch // m
            micro_inputs = list(inputs.split(micro))
            micro_targets = list(targets.split(micro))
            hidden_shape = (micro, seq, config.n_embd)
            best: dict[str, Any] | None = None
            for rep in range(warmup + repeats):
                for p in stage.parameters():
                    p.grad = None
                timer = _Timer()
                dist.barrier()
                t0 = time.perf_counter()
                _loss, peak = _run_pipeline_step(
                    stage,
                    rank,
                    world_size,
                    micro_inputs,
                    micro_targets,
                    hidden_shape,
                    kind,
                    timer,
                )
                wall = time.perf_counter() - t0
                if rep < warmup:
                    continue
                row = {
                    "compute_seconds": timer.compute,
                    "comm_seconds": timer.comm,
                    "wall_seconds": wall,
                    "peak_stash": peak,
                }
                # Keep the fastest repeat: a slow one is the OS scheduler
                # descheduling a rank, which is noise about this laptop rather
                # than about the schedule.
                if best is None or row["wall_seconds"] < best["wall_seconds"]:
                    best = row
            assert best is not None
            rows.append(
                {
                    "kind": kind,
                    "micro_batches": m,
                    "micro_batch_size": micro,
                    "analytic_bubble": analytic_bubble(world_size, m),
                    **best,
                }
            )
    return {"rank": rank, "world_size": world_size, "rows": rows}
