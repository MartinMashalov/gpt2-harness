"""Tests for the cluster half: checkpoint resharding, streaming, failure recovery, cost model.

The tests that matter here are the equivalence ones. A sharded checkpoint that
loads without error proves nothing -- a wrong split loads fine and produces a
subtly different model. So the assertions are on *logits*, on *coverage*, and on
the *loss trajectory*, at exact equality wherever exact equality is what the
implementation should deliver.
"""

from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from transformer_internals.cluster.checkpoint import (
    REPLICATE,
    AsyncCheckpointer,
    CheckpointIndex,
    ShardSpec,
    gpt2_tp_plan,
    load_full,
    load_reshard,
    merge_pieces,
    save_sharded,
    shard_state_dict,
    split_tensor,
)
from transformer_internals.cluster.fabric import (
    LINKS,
    LLAMA70B,
    ModelShape,
    ParallelConfig,
    all_reduce_time,
    compute_time_s,
    crossover_ranks,
    step_costs,
)
from transformer_internals.cluster.streaming import (
    DelayedSource,
    ShardedStream,
    StreamState,
    TokenShardSource,
    measure_throughput,
    plan_indices,
)
from transformer_internals.config import GPTConfig
from transformer_internals.model import GPT

SMALL = GPTConfig(vocab_size=128, n_positions=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


def _model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(SMALL)


def _state(model: GPT) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _save_under(directory: Path, state: dict[str, torch.Tensor], plan, world_size: int, step: int = 7) -> None:
    shapes = {k: list(v.shape) for k, v in state.items()}
    for rank in range(world_size):
        save_sharded(
            directory, shard_state_dict(state, plan, rank, world_size), plan,
            rank=rank, world_size=world_size, step=step, global_shapes=shapes,
        )


# ------------------------------------------------------------------ sharding


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("sections", [1, 3])
def test_split_then_merge_is_the_identity(world_size: int, sections: int) -> None:
    t = torch.randn(48, 16)
    spec = ShardSpec("shard", dim=0, sections=sections)
    pieces = split_tensor(t, spec, world_size)
    assert all(p.shape[0] == 48 // world_size for p in pieces)
    assert torch.equal(merge_pieces(pieces, spec), t)


def test_fused_qkv_split_keeps_whole_heads_of_each_projection() -> None:
    """The bug this guards: splitting the (3C, C) QKV matrix into contiguous blocks.

    With ``sections=1`` and world_size=3, rank 0 would own all of Q and none of
    K or V, and every rank's attention would be nonsense. With ``sections=3``
    each rank owns head-group ``r`` of Q, of K and of V.
    """
    c = SMALL.n_embd
    head_dim = SMALL.head_dim
    # Mark each row with which projection and which head it belongs to.
    tag = torch.zeros(3 * c, 1)
    for i in range(3 * c):
        proj, within = divmod(i, c)
        tag[i, 0] = proj * 1000 + within // head_dim

    world_size = 2
    pieces = split_tensor(tag, ShardSpec("shard", 0, sections=3), world_size)
    heads_per_rank = SMALL.n_head // world_size
    for rank, piece in enumerate(pieces):
        assert piece.shape[0] == 3 * c // world_size
        for proj in range(3):
            block = piece[proj * c // world_size : (proj + 1) * c // world_size, 0]
            heads = sorted({int(v) % 1000 for v in block})
            projections = {int(v) // 1000 for v in block}
            assert projections == {proj}, "a rank's Q block must contain only Q rows"
            assert heads == list(range(rank * heads_per_rank, (rank + 1) * heads_per_rank))
            # Whole heads, not fragments.
            for h in heads:
                assert (block == proj * 1000 + h).sum().item() == head_dim

    # The naive plan gets this wrong, which is the point of the sections field.
    naive = split_tensor(tag, ShardSpec("shard", 0, sections=1), world_size)
    assert {int(v) // 1000 for v in naive[0][:, 0]} != {0, 1, 2}


def test_column_parallel_shards_reproduce_the_full_matmul() -> None:
    """A shard plan is only valid if the sharded computation equals the whole one."""
    layer = torch.nn.Linear(64, 96)
    x = torch.randn(5, 64)
    spec = ShardSpec("shard", dim=0)
    ws = split_tensor(layer.weight.detach(), spec, 4)
    bs = split_tensor(layer.bias.detach(), spec, 4)
    parts = [torch.nn.functional.linear(x, w, b) for w, b in zip(ws, bs, strict=True)]
    assert torch.allclose(torch.cat(parts, dim=-1), layer(x), atol=0, rtol=0)


def test_row_parallel_shards_sum_to_the_full_matmul() -> None:
    """Row parallel: each rank computes a partial sum, the all-reduce adds them."""
    layer = torch.nn.Linear(96, 64, bias=False)
    x = torch.randn(5, 96)
    ws = split_tensor(layer.weight.detach(), ShardSpec("shard", dim=1), 4)
    xs = split_tensor(x, ShardSpec("shard", dim=1), 4)
    partial = sum(torch.nn.functional.linear(xi, wi) for xi, wi in zip(xs, ws, strict=True))
    assert torch.allclose(partial, layer(x), atol=1e-6)


def test_uneven_split_is_refused_rather_than_padded() -> None:
    with pytest.raises(ValueError, match="does not divide evenly"):
        split_tensor(torch.randn(10, 4), ShardSpec("shard", 0), 4)


def test_real_gpt2_vocabulary_cannot_be_split_four_ways() -> None:
    """50257 is not divisible by 4. The plan says so instead of padding silently.

    This is the constraint behind Megatron-LM's ``make_vocab_size_divisible_by``:
    a vocabulary-parallel embedding forces the vocabulary size to be a multiple
    of every tensor-parallel degree the run might ever use, so the vocabulary is
    padded with unreachable rows at construction time.
    """
    state = {"wte.weight": torch.zeros(50257, 768)}
    plan = gpt2_tp_plan(state)
    with pytest.raises(ValueError, match="does not divide evenly"):
        shard_state_dict(state, plan, rank=0, world_size=4)
    # Replicating the embedding is the other legitimate answer, and it works.
    replicated = gpt2_tp_plan(state, vocab_parallel=False)
    assert replicated["wte.weight"] == REPLICATE
    assert shard_state_dict(state, replicated, rank=0, world_size=4)["wte.weight"].shape == (50257, 768)


def test_plan_covers_every_parameter_and_shards_the_right_ones() -> None:
    state = _state(_model())
    plan = gpt2_tp_plan(state)
    assert set(plan) == set(state)
    assert plan["h.0.attn.c_attn.weight"] == ShardSpec("shard", 0, 3)
    assert plan["h.0.attn.c_proj.weight"] == ShardSpec("shard", 1)
    assert plan["h.0.mlp.c_fc.weight"] == ShardSpec("shard", 0)
    assert plan["h.0.ln_1.weight"] == REPLICATE
    assert plan["wte.weight"] == ShardSpec("shard", 0)
    # Tied weights must get the same treatment or the two views disagree.
    assert plan["lm_head.weight"] == plan["wte.weight"]


# -------------------------------------------------------------- resharding


@pytest.mark.parametrize(("src_world", "dst_world"), [(4, 2), (4, 1), (2, 8), (4, 8), (8, 4), (1, 4)])
def test_reshard_between_layouts_preserves_the_model_exactly(
    tmp_path: Path, src_world: int, dst_world: int
) -> None:
    """Save under one tensor-parallel width, restore under another, compare logits.

    Exact equality, not a tolerance: resharding moves bytes, it does not compute
    anything, so any difference at all is a bug.
    """
    model = _model(seed=3)
    state = _state(model)
    plan = gpt2_tp_plan(state)
    _save_under(tmp_path, state, plan, src_world)

    # Every destination rank rebuilds its slice, then we reassemble and compare.
    rebuilt: dict[str, list[torch.Tensor]] = {k: [] for k in state}
    for rank in range(dst_world):
        local, index, _ = load_reshard(tmp_path, rank=rank, world_size=dst_world)
        assert index.step == 7
        for name, tensor in local.items():
            rebuilt[name].append(tensor)
    merged = {name: merge_pieces(pieces, plan[name]) for name, pieces in rebuilt.items()}

    for name in state:
        assert torch.equal(merged[name], state[name]), f"{name} changed across the reshard"

    restored = GPT(SMALL)
    restored.load_state_dict(merged)
    idx = torch.randint(0, SMALL.vocab_size, (2, 16))
    with torch.no_grad():
        before = model(idx)["logits"]
        after = restored(idx)["logits"]
    assert torch.equal(before, after)


def test_reshard_reads_only_the_shards_it_needs(tmp_path: Path) -> None:
    """4 -> 8 splits each source shard in two, so a target rank touches exactly one file."""
    state = _state(_model(seed=4))
    plan = gpt2_tp_plan(state)
    _save_under(tmp_path, state, plan, 4)
    _, _, files_read = load_reshard(tmp_path, rank=5, world_size=8)
    assert files_read == 1

    # 4 -> 2 merges two source shards into each target, so a target rank reads
    # exactly the two it is made of and nothing else.
    _, _, files_read = load_reshard(tmp_path, rank=1, world_size=2)
    assert files_read == 2

    # And restoring the whole model into one process does have to read all four.
    _, _, files_read = load_reshard(
        tmp_path, rank=0, world_size=1, plan=dict.fromkeys(state, REPLICATE)
    )
    assert files_read == 4


def test_sharded_checkpoint_loads_into_a_single_unsharded_process(tmp_path: Path) -> None:
    model = _model(seed=5)
    state = _state(model)
    _save_under(tmp_path, state, gpt2_tp_plan(state), 4)
    full, index = load_full(tmp_path)
    assert index.world_size == 4
    restored = GPT(SMALL)
    restored.load_state_dict(full)
    idx = torch.randint(0, SMALL.vocab_size, (2, 16))
    with torch.no_grad():
        assert torch.equal(model(idx)["logits"], restored(idx)["logits"])


def test_index_is_self_describing(tmp_path: Path) -> None:
    state = _state(_model())
    _save_under(tmp_path, state, gpt2_tp_plan(state), 2)
    index = CheckpointIndex.read(tmp_path)
    assert index.world_size == 2
    assert index.global_shapes["wte.weight"] == [SMALL.vocab_size, SMALL.n_embd]
    raw = json.loads((tmp_path / "index.json").read_text())
    assert raw["plan"]["h.0.attn.c_attn.weight"] == {"kind": "shard", "dim": 0, "sections": 3}


def test_async_save_matches_sync_save_and_blocks_for_less_time(tmp_path: Path) -> None:
    """Overlapped save: the step pays for the copy, not for the write."""
    torch.manual_seed(0)
    state = {f"w{i}": torch.randn(512, 512) for i in range(24)}  # ~25 MB
    plan = dict.fromkeys(state, REPLICATE)
    shapes = {k: list(v.shape) for k, v in state.items()}

    t0 = time.perf_counter()
    save_sharded(tmp_path / "sync", state, plan, rank=0, world_size=1, step=1, global_shapes=shapes)
    sync_s = time.perf_counter() - t0

    saver = AsyncCheckpointer()
    saver.save(tmp_path / "async", state, plan=plan, rank=0, world_size=1, step=1, global_shapes=shapes)
    blocking_s = saver.last_blocking_seconds
    saver.wait()

    a, _ = load_full(tmp_path / "sync")
    b, _ = load_full(tmp_path / "async")
    for k in state:
        assert torch.equal(a[k], b[k])
    assert blocking_s < sync_s, (
        f"async save blocked {blocking_s*1e3:.1f} ms, sync save took {sync_s*1e3:.1f} ms"
    )
    print(f"\n[async ckpt] sync {sync_s*1e3:.1f} ms blocking, "
          f"async {blocking_s*1e3:.1f} ms blocking + {saver.last_write_seconds*1e3:.1f} ms "
          f"in the background ({sync_s/blocking_s:.1f}x less step time)")


def test_async_save_surfaces_an_error_from_the_writer_thread(tmp_path: Path) -> None:
    saver = AsyncCheckpointer()
    # No global_shapes on rank 0 is a programming error; it must not be swallowed.
    saver.save(tmp_path / "bad", {"w": torch.zeros(2)}, plan={}, rank=0, world_size=1, step=1)
    with pytest.raises(ValueError, match="global_shapes"):
        saver.wait()


# --------------------------------------------------------------- streaming


def test_ranks_read_disjoint_shards_that_cover_the_epoch_exactly_once() -> None:
    n, world = 97 * 4, 4
    state = StreamState(num_samples=n, world_size=world, positions=[0] * world)
    plans = [plan_indices(state, r, world) for r in range(world)]
    allocated = np.concatenate(plans)
    assert len(allocated) == n
    assert sorted(allocated.tolist()) == list(range(n))
    for a in range(world):
        for b in range(a + 1, world):
            assert not set(plans[a].tolist()) & set(plans[b].tolist())


def test_resume_mid_epoch_neither_repeats_nor_skips_a_sample() -> None:
    n, world, stop_after = 240, 4, 17
    source = list(range(n))
    seen: list[int] = []
    positions = []
    for rank in range(world):
        stream = ShardedStream(source, rank=rank, world_size=world, seed=11)
        seen.extend(islice(stream, stop_after))
        positions.append(stream.position)
    assert positions == [stop_after] * world

    state = StreamState(num_samples=n, seed=11, world_size=world, positions=positions)
    for rank in range(world):
        seen.extend(list(ShardedStream(source, rank=rank, world_size=world, state=state)))

    assert sorted(seen) == list(range(n)), "resume must not skip"
    assert len(seen) == len(set(seen)), "resume must not repeat"


def test_resume_replays_the_same_order_an_uninterrupted_run_would_have_seen() -> None:
    """This is what makes the loss trajectory after a restart identical."""
    n, world, cut = 200, 2, 23
    source = list(range(n))
    for rank in range(world):
        whole = list(ShardedStream(source, rank=rank, world_size=world, seed=5))
        state = StreamState(num_samples=n, seed=5, world_size=world, positions=[cut] * world)
        resumed = list(ShardedStream(source, rank=rank, world_size=world, state=state))
        assert whole[cut:] == resumed


def test_stream_reshards_across_a_change_of_world_size() -> None:
    """Elastic restart: 4 ranks stop, 3 come back, coverage is still exactly once."""
    n, old_world, new_world, cut = 240, 4, 3, 13
    source = list(range(n))
    seen: list[int] = []
    for rank in range(old_world):
        seen.extend(islice(ShardedStream(source, rank=rank, world_size=old_world, seed=7), cut))
    state = StreamState(num_samples=n, seed=7, world_size=old_world, positions=[cut] * old_world)
    for rank in range(new_world):
        seen.extend(list(ShardedStream(source, rank=rank, world_size=new_world, state=state)))
    assert sorted(seen) == list(range(n))
    assert len(seen) == len(set(seen))


def test_prefetched_but_unconsumed_samples_are_re_read_not_skipped() -> None:
    """The queue is not the position.

    With a prefetch depth of 8 the reader thread has pulled samples out of the
    source that the consumer never received. If the position counter followed
    the *reader* those samples would be lost at the next restart -- silently,
    because nothing errors and the loss curve looks normal. It follows the
    consumer instead, so they are re-read.
    """

    class CountingSource(list):
        reads = 0

        def __getitem__(self, i):  # type: ignore[override]
            CountingSource.reads += 1
            return list.__getitem__(self, i)

    n, world, cut = 120, 2, 9
    source = CountingSource(range(n))
    streams = [ShardedStream(source, rank=r, world_size=world, seed=3, prefetch=8) for r in range(world)]
    seen = [list(islice(s, cut)) for s in streams]
    assert [s.position for s in streams] == [cut] * world
    assert CountingSource.reads > cut * world, (
        "the prefetch thread should have read ahead of the consumer; "
        f"read {CountingSource.reads} for {cut * world} consumed"
    )

    state = StreamState.from_dict(streams[0].state_dict([s.position for s in streams]))
    resumed = [list(ShardedStream(source, rank=r, world_size=world, state=state, prefetch=8))
               for r in range(world)]
    for rank in range(world):
        plain = list(ShardedStream(source, rank=rank, world_size=world, seed=3))
        assert seen[rank] + resumed[rank] == plain, "a prefetched sample went missing"
    everything = [x for part in seen + resumed for x in part]
    assert sorted(everything) == list(range(n))


def test_token_source_cuts_non_overlapping_windows(tmp_path: Path) -> None:
    tokens = np.arange(1000, dtype=np.uint16)
    path = TokenShardSource.write(tmp_path / "corpus.bin", tokens)
    source = TokenShardSource(path, block_size=16)
    assert len(source) == (1000 - 1) // 16
    x, y = source[3]
    assert x.tolist() == list(range(48, 64))
    assert y.tolist() == list(range(49, 65)), "targets are inputs shifted by one"


@pytest.mark.slow
def test_prefetch_hides_read_latency_behind_the_training_step(tmp_path: Path) -> None:
    """Measured on this machine, and the mechanism is the thing being measured.

    Against a memory-mapped file that is already in the page cache, prefetch
    buys nothing: the read is a memcpy and the reader thread only adds queue
    overhead. That case is measured too, and reported, because it is the honest
    answer for a local corpus. The case prefetch exists for is slow storage plus
    a busy consumer, which is what the second table shows.
    """
    rng = np.random.default_rng(0)
    path = TokenShardSource.write(
        tmp_path / "corpus.bin", rng.integers(0, 5000, 400_000, dtype=np.uint16)
    )
    source = TokenShardSource(path, block_size=256)
    list(ShardedStream(source, rank=0, world_size=2))  # warm the page cache

    cached = {}
    for depth in (0, 2, 8):
        rates = [measure_throughput(source, rank=0, world_size=2, prefetch=depth, limit=400)[0]
                 for _ in range(3)]
        cached[depth] = max(rates)
    print("\n[stream throughput] page-cached memmap, no consumer work, samples/s "
          "(best of 3): " + ", ".join(f"depth {d}: {r:,.0f}" for d, r in cached.items()))

    # Slow storage (500us per read) and a consumer that takes 500us per sample.
    slow = DelayedSource(source, read_seconds=5e-4)
    overlapped = {}
    for depth in (0, 2, 8):
        rates = [measure_throughput(slow, rank=0, world_size=2, prefetch=depth,
                                    limit=200, per_sample_work_s=5e-4)[0] for _ in range(3)]
        overlapped[depth] = max(rates)
    print("[stream throughput] 500us reads + 500us step, samples/s (best of 3): "
          + ", ".join(f"depth {d}: {r:,.0f}" for d, r in overlapped.items()))
    speedup = overlapped[8] / overlapped[0]
    print(f"[stream throughput] prefetch 8 vs 0 with slow reads: {speedup:.2f}x "
          f"(ceiling is 2.00x when read and step cost the same)")

    assert speedup > 1.3, (
        "prefetch should overlap a slow read with the step; measured "
        f"{overlapped[0]:.0f} -> {overlapped[8]:.0f} samples/s"
    )


# ------------------------------------------------------- failure and restart


@pytest.mark.slow
def test_killing_a_rank_and_resuming_reproduces_the_uninterrupted_loss_curve(tmp_path: Path) -> None:
    """Real processes, real gloo, real SIGKILL, real restart from disk.

    The clean run and the interrupted run must agree step for step. They are
    compared at exact equality: same weights, same optimiser moments, same data
    in the same order means the same float.
    """
    from transformer_internals.cluster.failure import (
        JobSpec,
        free_port,
        launch,
        make_corpus,
        read_log,
        run_with_recovery,
    )

    data = make_corpus(tmp_path / "corpus.bin", 40_000, 128, seed=0)
    common = {
        "data_path": str(data), "block_size": 32, "micro_batch": 4, "steps": 20,
        "ckpt_every": 5, "seed": 1234,
    }

    clean = JobSpec(ckpt_dir=str(tmp_path / "ck_clean"), log_path=str(tmp_path / "clean.jsonl"),
                    master_port=free_port(), **common)
    ok, codes, _ = launch(2, clean)
    assert ok, f"uninterrupted run failed with {codes}"
    clean_losses = {r["step"]: r["loss"] for r in read_log(clean.log_path) if r["event"] == "step"}
    assert len(clean_losses) == 20

    crashed = JobSpec(ckpt_dir=str(tmp_path / "ck_crash"), log_path=str(tmp_path / "crash.jsonl"),
                      master_port=free_port(), **common)
    report = run_with_recovery(2, crashed, kill_rank_at_step=(1, 12))

    assert report.launches == 2, f"expected one restart, got {report.events}"
    assert report.time_to_recover_s is not None

    # Which checkpoint the restart lands on is decided by a race between the
    # supervisor's SIGKILL and the ranks, which on a fast quiet machine run a
    # step in single-digit milliseconds. So the assertions below pin the
    # invariants rather than one arbitrary outcome of that race: the resume must
    # come from a real checkpoint boundary, before the end of the run, and the
    # resumed trajectory must be bit-identical to the uninterrupted one. That
    # last assertion is *stronger* the earlier the resume happens, so nothing is
    # given away by not naming the step.
    resumed_from = report.resumed_from_step
    assert resumed_from is not None, f"the run never restarted: {report.events}"
    assert resumed_from % 5 == 0, f"resumed from step {resumed_from}, not a checkpoint boundary"
    assert 0 < resumed_from < 20, f"resumed from step {resumed_from}, outside the run"

    records = [r for r in read_log(crashed.log_path) if r["event"] == "step"]
    resumed = {r["step"]: r["loss"] for r in records if r["resumed_from"] == resumed_from}
    assert max(resumed) == 20, "the resumed run must reach the end"
    assert min(resumed) == resumed_from + 1, "it must restart at the step after the checkpoint"
    assert set(resumed) == set(range(resumed_from + 1, 21)), "no step may be skipped or repeated"

    diffs = {s: abs(clean_losses[s] - loss) for s, loss in resumed.items()}
    worst = max(diffs.values())
    print(f"\n[failure/restart] killed rank 1 at step 12, checkpoint was step "
          f"{resumed_from}, time to recover "
          f"{report.time_to_recover_s:.2f}s, launches {report.launches}")
    print(f"[failure/restart] max |loss(resumed) - loss(uninterrupted)| over steps "
          f"{resumed_from + 1}-20 = {worst:.3e}")
    assert worst == 0.0, f"resumed trajectory diverged by {worst}"


@pytest.mark.slow
def test_measured_gloo_allreduce_fits_the_cost_model_form() -> None:
    """The cost model's shape, checked against a real collective on this machine."""
    from transformer_internals.cluster.collbench import run

    # Fit over the bandwidth-bound regime, 64 KiB to 16 MiB. Below about 64 KiB
    # a gloo all-reduce is entirely latency and scheduling noise, and no affine
    # model describes it well -- which is exactly why the fabric model's small-
    # message behaviour is driven by its latency term rather than its slope.
    data = run(world_size=2, sizes=[1 << k for k in range(14, 24, 2)], iters=10)
    fit = data["fit"]
    print(f"\n[collective fit] MEASURED gloo all-reduce, 2 ranks, CPU/loopback: "
          f"t = {fit['latency_us']:.0f}us + bytes/{fit['bandwidth_gbytes_per_s']:.2f} GB/s, "
          f"R^2 = {fit['r_squared']:.4f}")
    assert fit["r_squared"] > 0.9, "latency + bytes/bandwidth should describe a real collective"
    assert fit["bandwidth_gbytes_per_s"] > 0


# -------------------------------------------------------------- cost model


def test_tensor_parallel_is_the_communication_heavy_axis() -> None:
    """The placement rule has to come out of the arithmetic, not be asserted."""
    cfg = ParallelConfig(tp=8, dp=8)
    ib = step_costs(LLAMA70B, cfg, "ib_ndr")
    nvlink = step_costs(LLAMA70B, cfg, "nvlink4")
    compute = compute_time_s(LLAMA70B, cfg)

    assert nvlink["tp"] < compute, "TP inside an NVLink domain hides under compute"
    assert ib["tp"] > compute, "TP across InfiniBand does not"
    assert ib["fsdp"] < compute, "sharded data parallel does fit across nodes"
    assert ib["tp"] / ib["fsdp"] > 5, "TP is the heavier axis by a wide margin"
    assert ib["tp"] / nvlink["tp"] > 5


def test_pipeline_traffic_is_tiny_compared_with_tensor_parallel() -> None:
    tp_only = step_costs(LLAMA70B, ParallelConfig(tp=8, dp=8), "ib_ndr")["tp"]
    pp_only = step_costs(LLAMA70B, ParallelConfig(tp=8, pp=8), "ib_ndr")["pp"]
    assert pp_only < tp_only / 100


def test_crossover_degree_falls_with_bandwidth() -> None:
    nv = crossover_ranks(LLAMA70B, "nvlink4") or 999
    ib = crossover_ranks(LLAMA70B, "ib_ndr") or 999
    roce = crossover_ranks(LLAMA70B, "roce200") or 999
    assert nv > ib >= roce


def test_disabling_gpudirect_makes_inter_node_slower_and_leaves_nvlink_alone() -> None:
    cfg = ParallelConfig(tp=8, dp=8)
    assert step_costs(LLAMA70B, cfg, "ib_ndr", gpudirect=False)["tp"] > \
        step_costs(LLAMA70B, cfg, "ib_ndr")["tp"]
    assert step_costs(LLAMA70B, cfg, "nvlink4", gpudirect=False)["tp"] == \
        step_costs(LLAMA70B, cfg, "nvlink4")["tp"]


def test_ring_allreduce_model_has_the_right_limits() -> None:
    link = LINKS["ib_ndr"]
    assert all_reduce_time(1e9, 1, link) == 0.0
    # Doubling the ranks moves at most 2x the data, never more.
    t2 = all_reduce_time(1e9, 2, link)
    t64 = all_reduce_time(1e9, 64, link)
    assert 1.9 < t64 / t2 < 2.05


def test_parameter_count_formula_matches_a_real_model() -> None:
    """The cost model's parameter count has to be the same quantity the repo counts."""
    shape = ModelShape(n_layer=12, n_embd=768, n_head=12, vocab_size=50257,
                       seq_len=1024, micro_batch=1)
    real = GPT(GPTConfig()).num_parameters()  # 124,439,808 with tied weights
    # The formula counts an untied head, so it is one embedding matrix heavier.
    assert abs(shape.params - (real + 50257 * 768)) / shape.params < 0.02
