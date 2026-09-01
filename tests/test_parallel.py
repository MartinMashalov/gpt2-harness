"""Equivalence proofs for the five parallelism strategies.

Every test here spawns real ``torch.distributed`` processes and compares the
sharded result against a single-process reference computed in the same process.
The backend is chosen from the machine -- gloo here, NCCL on a CUDA box -- and
the assertions are the same either way, which is the point: whether a sharded
implementation computes the right function is a property of the algorithm, not
of the wire. There is no GPU and no cluster involved in CI, and none is needed:
correctness is the property that is hardest to get right and easiest to fake,
and it is exactly the one that does not need special hardware to check.

The tolerance is 1e-5 throughout, which is loose compared to what is actually
measured (see the numbers in ``results/parallel_comms.json``). It is set by the
one place where float non-associativity genuinely bites: reducing gradients in a
different order changes them by ~3e-8, and Adam's ``g / sqrt(g^2)``
normalisation amplifies that on parameters whose gradient is near zero. At world
size 1 the same code reproduces ``torch.optim.AdamW`` bit for bit, which is how
we know that residual is the reduction and not the optimizer.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from transformer_internals.hardware import Capabilities, HardwareError
from transformer_internals.parallel import comms
from transformer_internals.parallel.data_parallel import (
    DEFAULT_BUCKET_BYTES,
    build_buckets,
    ddp_equivalence_worker,
)
from transformer_internals.parallel.dtensor_demo import dtensor_worker
from transformer_internals.parallel.pipeline_parallel import (
    analytic_bubble,
    bubble_scan_worker,
    pipeline_worker,
    schedule_order,
    simulate,
)
from transformer_internals.parallel.sequence_parallel import sequence_parallel_worker
from transformer_internals.parallel.tensor_parallel import tp_equivalence_worker
from transformer_internals.parallel.zero import zero3_equivalence_worker, zero_equivalence_worker

TOL = 1e-5

_CACHE: dict[tuple, list[Any]] = {}


def run(fn, world_size: int, **kwargs) -> list[Any]:
    """Spawn once per distinct configuration and share the result between tests.

    Process startup dominates the runtime of these tests -- the arithmetic is
    milliseconds -- so several assertions ride on one spawn.
    """
    key = (fn.__module__, fn.__name__, world_size, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = comms.spawn_workers(fn, world_size, kwargs, timeout=300)
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# launcher: backend and placement
# --------------------------------------------------------------------------- #


def test_workers_report_the_backend_and_device_they_actually_ran_on():
    """Not what was asked for. What ran."""
    results = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3)
    for r in results:
        assert r.backend in ("gloo", "nccl")
        assert r.device.startswith("cpu") or r.device.startswith("cuda")
    if not torch.cuda.is_available():
        assert all(r.backend == "gloo" and r.device == "cpu" for r in results)


def test_an_impossible_placement_fails_in_the_parent_before_any_process_starts():
    """Eight NCCL ranks on a two-GPU box is one sentence, not eight tracebacks."""
    two_gpus = Capabilities.stub(device_count=2)
    with pytest.raises(HardwareError, match="exceeds the 2 visible"):
        comms.spawn_workers(ddp_equivalence_worker, 8, {}, caps=two_gpus)


def test_requesting_nccl_on_this_machine_says_why_it_cannot():
    caps = Capabilities.detect()
    if caps.cuda_available and caps.nccl_available:
        pytest.skip("this machine has CUDA and NCCL, so the request succeeds")
    with pytest.raises(HardwareError):
        comms.spawn_workers(ddp_equivalence_worker, 2, {}, backend="nccl", caps=caps)


# --------------------------------------------------------------------------- #
# byte counting
# --------------------------------------------------------------------------- #


def test_comm_counter_counts_exact_payload_bytes():
    counter = comms.CommCounter(world_size=4)
    counter.record("all_reduce", torch.zeros(1000, dtype=torch.float32))
    counter.record("all_gather", [torch.zeros(250) for _ in range(4)])
    assert counter.records["all_reduce"].payload_bytes == 4000
    assert counter.records["all_gather"].payload_bytes == 4000
    # Ring model: all-reduce moves 2(p-1)/p, all-gather (p-1)/p.
    assert counter.records["all_reduce"].wire_bytes == pytest.approx(2 * 0.75 * 4000)
    assert counter.records["all_gather"].wire_bytes == pytest.approx(0.75 * 4000)


# --------------------------------------------------------------------------- #
# 1. data parallel
# --------------------------------------------------------------------------- #


def test_ddp_gradient_equals_single_process_full_batch():
    for r in run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3):
        assert r.value["max_grad_error"] < TOL
        assert r.value["max_param_error"] < TOL


def test_ddp_moves_exactly_one_gradient_per_step():
    results = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3)
    payload = results[0].payload
    n_params = results[0].value["n_params"]
    assert payload["per_collective"]["all_reduce"]["calls"] == 3
    # One fp32 copy of every gradient, per step. Not an estimate: this is the
    # number of bytes the collective was handed.
    assert payload["payload_bytes_per_step"] == n_params * 4


def test_ddp_bucketing_does_not_change_the_answer():
    unbucketed = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=2, bucket_bytes=0)
    bucketed = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=2)
    assert unbucketed[0].value["max_grad_error"] < TOL
    assert bucketed[0].value["max_grad_error"] < TOL
    assert unbucketed[0].payload["per_collective"]["all_reduce"]["calls"] > (
        bucketed[0].payload["per_collective"]["all_reduce"]["calls"]
    )
    # Same bytes either way; fewer, larger collectives.
    assert (
        unbucketed[0].payload["payload_bytes_per_step"]
        == bucketed[0].payload["payload_bytes_per_step"]
    )


def test_buckets_respect_the_size_cap_and_run_in_reverse():
    params = [torch.nn.Parameter(torch.zeros(1024)) for _ in range(10)]  # 4 KB each
    buckets = build_buckets(params, bucket_bytes=10 * 1024)
    assert all(sum(p.numel() * 4 for p in b) <= 12 * 1024 for b in buckets)
    assert buckets[0][0] is params[-1]
    assert len(build_buckets(params, DEFAULT_BUCKET_BYTES)) == 1


# --------------------------------------------------------------------------- #
# 2. sharded data parallel (ZeRO)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", [1, 2])
def test_zero_matches_single_process_adamw_trajectory(stage: int):
    for r in run(zero_equivalence_worker, 2, stage=stage, steps=4):
        assert r.value["max_param_error"] < TOL, r.value["param_errors"]


def test_zero_shards_optimizer_state():
    r = run(zero_equivalence_worker, 2, stage=2, steps=4)[0].value
    sharded, replicated = r["sharded_bytes"], r["replicated_bytes"]
    # Two Adam moments over N/p elements instead of N, measured from the actual
    # tensor storages rather than from the formula.
    assert sharded["optimizer"] <= replicated["optimizer"] / 2 * 1.05
    assert sharded["grads"] <= replicated["grads"] / 2 * 1.05
    assert sharded["total"] < replicated["total"]


def test_zero_stage2_replaces_the_all_reduce_with_a_reduce_scatter():
    stage1 = run(zero_equivalence_worker, 2, stage=1, steps=4)[0].payload["per_collective"]
    stage2 = run(zero_equivalence_worker, 2, stage=2, steps=4)[0].payload["per_collective"]
    assert "reduce_scatter" not in stage1
    assert stage2["reduce_scatter"]["calls"] == 4
    assert stage1["all_gather"]["calls"] == stage2["all_gather"]["calls"] == 4


def test_zero3_matches_single_process_adamw_trajectory():
    for r in run(zero3_equivalence_worker, 2, steps=3):
        assert r.value["max_param_error"] < TOL, r.value["param_errors"]


def test_zero3_actually_shards_the_parameters():
    r = run(zero3_equivalence_worker, 2, steps=3)[0].value
    block_bytes = r["n_block_params"] * 4
    # Every rank holds half the block parameters plus the replicated root.
    assert r["sharded_bytes"]["params"] < block_bytes
    assert sum(r["shard_numel"]) * 2 >= r["n_block_params"]


def test_zero3_gathers_once_per_unit_per_pass():
    r = run(zero3_equivalence_worker, 2, steps=3)[0]
    units, steps = r.value["n_units"], 3
    # One all-gather per unit in the forward pass and one in the backward pass.
    assert r.payload["per_collective"]["all_gather"]["calls"] == 2 * units * steps
    assert r.payload["per_collective"]["reduce_scatter"]["calls"] == units * steps


# --------------------------------------------------------------------------- #
# 2b. what a bf16 wire costs
# --------------------------------------------------------------------------- #


def test_a_bf16_gradient_reduction_moves_exactly_half_the_bytes():
    """Not an estimate. The counter records the buffer the collective was given."""
    fp32 = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3)[0]
    bf16 = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3, reduce_dtype="bf16")[0]
    assert bf16.payload["payload_bytes_per_step"] * 2 == (
        fp32.payload["payload_bytes_per_step"]
    )
    # And the same number of collectives, so the saving is bytes and not calls.
    assert (
        bf16.payload["per_collective"]["all_reduce"]["calls"]
        == fp32.payload["per_collective"]["all_reduce"]["calls"]
    )


def test_a_bf16_gradient_reduction_costs_accuracy_and_the_cost_is_bounded():
    """Halving the wire is not free, and the size of the bill is the result.

    An fp32 all-reduce reproduces the single-process gradient to 3e-08, which is
    float non-associativity and nothing else. A bf16 one rounds each rank's
    contribution to 8 significand bits before summing, so the error rises by
    about four orders of magnitude, to a few parts in ten thousand of the
    gradient. Both bounds are asserted: the loose one because a bf16 reduction
    that was accidentally still fp32 would pass a one-sided test, and the tight
    one because that is the claim.
    """
    fp32 = run(ddp_equivalence_worker, 2, batch=8, seq=16, steps=3)[0].value
    bf16 = run(
        ddp_equivalence_worker, 2, batch=8, seq=16, steps=3, reduce_dtype="bf16"
    )[0].value
    assert fp32["max_grad_error"] < 1e-06
    assert 1e-05 < bf16["max_grad_error"] < 1e-02


@pytest.mark.parametrize("stage", [1, 2])
def test_a_bf16_reduction_injects_a_fresh_error_each_step_and_none_accumulates(
    stage: int,
):
    """The bf16 gradient reduction perturbs the trajectory without compounding.

    The optimiser's state is fp32 on this path and the collective is the only
    narrow thing, so each step gets a new rounding error and none of it is
    carried in the moments. The trajectory error should therefore be flat across
    steps rather than growing. The assertion is that step 4's error is no more
    than 1.5x step 1's; a run that let the rounding into its optimiser state
    would grow far faster than that.

    This path holds no separate master shard, and does not need one: with an
    fp32 parameter all-gather the replicated parameters *are* the master copy.
    The path that does need one is below.
    """
    r = run(zero_equivalence_worker, 2, stage=stage, steps=4, reduce_dtype="bf16")[0].value
    errors = r["param_errors"]
    assert errors[0] > 1e-05, "the bf16 reduction did not perturb anything"
    assert errors[-1] <= 1.5 * errors[0], errors
    # No narrow parameter gather, so no second copy of anything.
    assert r["sharded_bytes"]["master_shard"] == 0


def test_a_bf16_parameter_gather_allocates_the_fp32_master_shard():
    """The copy exists exactly when it is needed, and not otherwise.

    With an fp32 all-gather the replicated parameters are lossless and are
    already the master, so a second copy would be 4N/p bytes per rank of
    duplication. With a bf16 all-gather they are not, and re-deriving the shard
    from them each step would feed the gather's rounding back into the optimiser.
    """
    without = run(zero_equivalence_worker, 2, stage=2, steps=4)[0].value
    with_bf16 = run(zero_equivalence_worker, 2, stage=2, steps=4, param_dtype="bf16")[0].value

    assert without["sharded_bytes"]["master_shard"] == 0
    assert with_bf16["sharded_bytes"]["master_shard"] > 0
    # One fp32 copy of this rank's shard of the flat parameter vector.
    assert with_bf16["sharded_bytes"]["master_shard"] == with_bf16["shard_numel"] * 4


def test_the_bf16_parameter_gather_error_saturates_at_bf16s_own_resolution():
    """It grows for a few steps and then stops, and the ceiling is checkable.

    bf16's grid spacing at 1.0 is 2^-7, so its worst rounding error there is
    2^-8 = 3.906e-03. The largest parameter in the tested model is a LayerNorm
    gain sitting at 1.0, which is what caps this column. Over eight steps the
    error goes 1.00, 2.00, 3.00, 3.91, 3.82, 3.91, 3.89, 3.89 e-03: it reaches
    the ceiling and stays there rather than compounding, which is the fp32
    master shard doing its job on a path where the parameters themselves are
    rounded every step.
    """
    r = run(zero_equivalence_worker, 2, stage=2, steps=8, param_dtype="bf16")[0].value
    errors = r["param_errors"]
    ceiling = 2.0**-8
    assert max(errors) < ceiling * 1.05, errors
    # It really does grow first, so the plateau is a plateau and not a constant.
    assert errors[0] < 0.5 * max(errors)
    # And the last four steps are flat, not still climbing.
    tail = errors[4:]
    assert max(tail) - min(tail) < 0.1 * max(tail), tail


# --------------------------------------------------------------------------- #
# 2c. gradient clipping under sharding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", [1, 2])
def test_a_sharded_clip_reproduces_a_single_process_clip(stage: int):
    """Rank r holds a slice of the gradient and still gets the global norm right.

    Compared two ways, because either alone is weak. The norm itself must match
    what ``torch.nn.utils.clip_grad_norm_`` computed over the whole model, and
    the parameters after the clipped step must still match the reference
    trajectory at the same tolerance as the unclipped one.
    """
    for r in run(zero_equivalence_worker, 2, stage=stage, steps=4, grad_clip=1.0):
        assert r.value["max_grad_norm_error"] < 1e-05, r.value["grad_norms"]
        assert r.value["max_param_error"] < TOL, r.value["param_errors"]


def test_the_clip_actually_bites_on_at_least_one_step():
    """A clip that never triggers proves nothing about the clip."""
    norms = run(zero_equivalence_worker, 2, stage=2, steps=4, grad_clip=1.0)[0].value[
        "grad_norms"
    ]
    assert any(n["reference"] > 1.0 for n in norms), norms


def test_zero3_clips_shards_and_replicated_roots_consistently():
    """ZeRO-3's two kinds of parameter need different treatment in one norm.

    The unit shards are disjoint across ranks and must be summed; the root
    parameters have already been all-reduced and would be counted p times if
    they went into the same collective.
    """
    for r in run(zero3_equivalence_worker, 2, steps=3, grad_clip=1.0):
        assert r.value["max_grad_norm_error"] < 1e-05, r.value["grad_norms"]
        assert r.value["max_param_error"] < TOL, r.value["param_errors"]


def test_clipping_costs_one_extra_collective_per_step_under_zero2_and_none_under_zero1():
    """The whole reason to count bytes: the cost of clipping is not the same at
    every stage, and it falls out of who holds the gradient.

    ZeRO-1 all-reduces the full gradient anyway, so every rank can take the norm
    locally. ZeRO-2 reduce-scatters, so each rank holds a disjoint slice and the
    ranks have to agree on one scalar before any of them updates. That is one
    extra all-reduce of four bytes per step: no bandwidth at all, pure latency,
    which is the expensive kind.
    """
    z1_plain = run(zero_equivalence_worker, 2, stage=1, steps=4)[0].payload
    z1_clip = run(zero_equivalence_worker, 2, stage=1, steps=4, grad_clip=1.0)[0].payload
    z2_plain = run(zero_equivalence_worker, 2, stage=2, steps=4)[0].payload
    z2_clip = run(zero_equivalence_worker, 2, stage=2, steps=4, grad_clip=1.0)[0].payload

    def calls(payload, op):
        rec = payload["per_collective"].get(op)
        return 0 if rec is None else rec["calls"]

    assert calls(z1_clip, "all_reduce") == calls(z1_plain, "all_reduce")
    assert calls(z2_clip, "all_reduce") == calls(z2_plain, "all_reduce") + 4
    # Four bytes a step, one fp32 scalar, and nothing else changed.
    extra = z2_clip["per_collective"]["all_reduce"]["payload_bytes"]
    assert extra == 4 * 4


def test_zero3_clipping_adds_one_all_reduce_per_step():
    plain = run(zero3_equivalence_worker, 2, steps=3)[0].payload
    clipped = run(zero3_equivalence_worker, 2, steps=3, grad_clip=1.0)[0].payload
    assert (
        clipped["per_collective"]["all_reduce"]["calls"]
        == plain["per_collective"]["all_reduce"]["calls"] + 3
    )
    assert (
        clipped["per_collective"]["all_reduce"]["payload_bytes"]
        == plain["per_collective"]["all_reduce"]["payload_bytes"] + 3 * 4
    )


# --------------------------------------------------------------------------- #
# 3. tensor parallel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world_size", [2, 4])
def test_tensor_parallel_forward_and_backward_match(world_size: int):
    for r in run(tp_equivalence_worker, world_size):
        assert r.value["forward_error"] < TOL
        assert r.value["input_grad_error"] < TOL
        for name, err in r.value["grad_errors"].items():
            assert err < TOL, f"{name} off by {err}"


def test_tensor_parallel_costs_two_all_reduces_each_way():
    r = run(tp_equivalence_worker, 2)[0]
    # attention g and MLP g in the forward pass; attention f and MLP f in the
    # backward pass. Four collectives per block, independent of world size.
    assert r.payload["per_collective"]["all_reduce"]["calls"] == 4
    assert set(r.payload["per_collective"]) == {"all_reduce"}


def test_tensor_parallel_holds_a_fraction_of_the_parameters():
    for world_size in (2, 4):
        r = run(tp_equivalence_worker, world_size)[0].value
        # Not exactly 1/p: the LayerNorms and the two output-projection biases
        # stay replicated on every rank.
        assert r["local_params"] < r["reference_params"] / world_size * 1.4


# --------------------------------------------------------------------------- #
# 4. pipeline parallel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["gpipe", "1f1b"])
def test_pipeline_gradients_match_single_process(kind: str):
    for r in run(pipeline_worker, 4, kind=kind, micro_batches=4, batch=8, seq=16):
        assert r.value["max_grad_error"] < TOL
        assert r.value["compared_tensors"] > 0


def test_pipeline_stages_partition_the_model():
    results = run(pipeline_worker, 4, kind="gpipe", micro_batches=4, batch=8, seq=16)
    per_stage = [r.value["stage_params"] for r in results]
    assert len(set(per_stage)) > 1  # the end stages carry the embeddings
    assert all(n > 0 for n in per_stage)


@pytest.mark.parametrize("n_stages", [2, 4, 8])
@pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16])
def test_schedule_simulation_reproduces_the_bubble_formula(n_stages, micro_batches):
    for kind in ("gpipe", "1f1b"):
        sim = simulate(n_stages, micro_batches, t_f=1.0, t_b=2.0, kind=kind)
        assert sim.bubble == pytest.approx(analytic_bubble(n_stages, micro_batches), abs=1e-12)


@pytest.mark.parametrize("n_stages", [2, 4, 8])
@pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16])
def test_1f1b_bounds_the_activation_stash_and_gpipe_does_not(n_stages, micro_batches):
    gpipe = simulate(n_stages, micro_batches, 1.0, 2.0, "gpipe")
    onef = simulate(n_stages, micro_batches, 1.0, 2.0, "1f1b")
    assert gpipe.peak_stash == micro_batches
    assert onef.peak_stash == min(n_stages, micro_batches)


def test_schedule_order_is_a_permutation_of_the_work():
    for kind in ("gpipe", "1f1b"):
        for rank in range(4):
            ops = schedule_order(rank, 4, 8, kind)
            assert sorted(ops) == sorted(
                [("F", i) for i in range(8)] + [("B", i) for i in range(8)]
            )
            # No micro-batch may be backward-ed before it is forward-ed.
            seen: set[int] = set()
            for op, i in ops:
                if op == "F":
                    seen.add(i)
                else:
                    assert i in seen


@pytest.mark.slow
def test_measured_bubble_falls_as_micro_batches_rise():
    """The measured bubble must track ``(p-1)/(m+p-1)``, not merely decrease.

    Measured on real processes, so it sits *above* the formula: the formula
    charges nothing for the gloo transfers or for the Python dispatch of a
    smaller micro-batch, and both grow as ``m`` grows. The assertion is
    therefore one-sided and generous. ``scripts/run_parallel.py`` runs the same
    scan at a size where the two curves can be plotted against each other.
    """
    world_size = 2
    results = run(
        bubble_scan_worker,
        world_size,
        micro_batch_counts=(1, 2, 4),
        batch=4,
        seq=32,
        repeats=2,
        warmup=1,
    )
    rows = [r.value["rows"] for r in results]
    measured = []
    for i in range(len(rows[0])):
        group = [rank_rows[i] for rank_rows in rows]
        compute = sum(g["compute_seconds"] for g in group)
        makespan = max(g["wall_seconds"] for g in group)
        measured.append((rows[0][i], 1.0 - compute / (world_size * makespan)))

    for row, value in measured:
        assert value >= row["analytic_bubble"] - 0.10, row
        assert value <= row["analytic_bubble"] + 0.30, row
    gpipe = [(row["micro_batches"], v) for row, v in measured if row["kind"] == "gpipe"]
    assert gpipe[0][1] > gpipe[-1][1]


# --------------------------------------------------------------------------- #
# 5. context / sequence parallel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world_size", [2, 4])
def test_sequence_parallel_attention_matches_single_process(world_size: int):
    for r in run(sequence_parallel_worker, world_size, batch=2, seq=32):
        assert r.value["all_gather_forward_error"] < TOL
        assert r.value["ring_forward_error"] < TOL
        assert r.value["input_grad_error"] < TOL
        assert r.value["weight_grad_error"] < TOL


def test_ring_attention_rotates_p_minus_1_blocks():
    world_size = 4
    r = run(sequence_parallel_worker, world_size, batch=2, seq=32)[0]
    # Two tensors (K and V) exchanged on each of p-1 hops.
    assert r.payload["per_collective"]["send"]["calls"] == 2 * (world_size - 1)
    assert r.payload["per_collective"]["recv"]["calls"] == 2 * (world_size - 1)


# --------------------------------------------------------------------------- #
# DTensor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world_size", [2, 4])
def test_dtensor_expresses_the_same_sharding(world_size: int):
    for r in run(dtensor_worker, world_size):
        assert r.value["dtensor_error"] < TOL
        assert r.value["parallelize_module_error"] < TOL
        assert r.value["manual_error"] < TOL
        # The hand-written collectives and DTensor's inferred ones agree
        # exactly, not merely to tolerance.
        assert r.value["manual_vs_dtensor"] == 0.0


def test_dtensor_shards_the_weights_the_way_megatron_does():
    r = run(dtensor_worker, 2)[0].value
    full_fc, local_fc = r["full_fc_weight_shape"], r["local_fc_weight_shape"]
    full_proj, local_proj = r["full_proj_weight_shape"], r["local_proj_weight_shape"]
    assert local_fc == [full_fc[0] // 2, full_fc[1]]  # column parallel: rows split
    assert local_proj == [full_proj[0], full_proj[1] // 2]  # row parallel: columns split
