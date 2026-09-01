"""Equivalence proofs for the five parallelism strategies.

Every test here spawns real ``torch.distributed`` processes on the gloo backend
and compares the sharded result against a single-process reference computed in
the same process. There is no GPU and no cluster involved, and none is needed:
whether a sharded implementation computes the right function is a property of
the algorithm, and it is the property that is hardest to get right and easiest
to fake.

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
