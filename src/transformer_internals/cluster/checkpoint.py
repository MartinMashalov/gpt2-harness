"""Sharded checkpoints that survive a change of parallel layout.

The operational problem this solves
-----------------------------------
A job is launched on 4 GPUs with 4-way tensor parallelism. Each rank owns a
slice of every parallel weight and writes only its own slice, because writing
the full model from rank 0 means a gather of the whole model over the fabric and
a single writer for hundreds of gigabytes. Two weeks later the job has to come
back on 2 GPUs, or on 8, or be evaluated on one. The bytes on disk are laid out
for a world size that no longer exists.

Resharding is the operation that fixes that: read the pieces written by the old
layout, and hand each rank of the new layout exactly the slice it needs, without
ever materialising the full model on any one rank when it can be avoided.

What is actually implemented
----------------------------
* A **shard plan**: one :class:`ShardSpec` per parameter saying whether that
  tensor is replicated across ranks or split along a dimension, and -- the part
  that is easy to get wrong -- how many *fused sections* the tensor has.
* :func:`save_sharded`, which writes one file per rank plus a JSON index.
* :func:`load_reshard`, which rebuilds rank ``r`` of a *new* world size from the
  files of the old one, reading only the shard files that actually overlap the
  slice being rebuilt.
* :class:`AsyncCheckpointer`, which snapshots to host memory synchronously and
  serialises on a background thread, so the training step blocks for the copy
  and not for the write.

The fused-section subtlety
--------------------------
GPT-2's ``c_attn`` weight is one ``(3C, C)`` matrix holding query, key and value
projections stacked along dim 0. Tensor parallelism splits it *column-wise* into
per-head groups, so rank ``r`` must own head-group ``r`` of Q **and** of K
**and** of V. Splitting the ``(3C, C)`` matrix into ``world`` contiguous blocks
does not do that: with ``world=3`` rank 0 would own all of Q and none of K or V,
and the model would still load, still run, and be silently wrong. ``sections=3``
in the :class:`ShardSpec` is what makes the split per-section, and
:func:`gpt2_tp_plan` sets it. ``tests/test_cluster.py`` asserts that the split
lands on attention-head boundaries and that the sharded matmul reproduces the
unsharded one.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "REPLICATE",
    "AsyncCheckpointer",
    "CheckpointIndex",
    "ShardSpec",
    "gpt2_tp_plan",
    "load_full",
    "load_reshard",
    "merge_pieces",
    "save_sharded",
    "shard_state_dict",
    "split_tensor",
]

FORMAT = "transformer-internals-sharded-v1"


@dataclass(frozen=True)
class ShardSpec:
    """How one tensor is distributed across the ranks of a parallel layout.

    Attributes:
        kind: ``"replicate"`` (every rank holds the whole tensor) or ``"shard"``.
        dim: Dimension the tensor is split along. Ignored when replicated.
        sections: Number of independent fused sub-matrices concatenated along
            ``dim``. Each section is split across the world separately and the
            local pieces are re-concatenated in section order, so rank ``r``
            owns slice ``r`` of *every* section. ``1`` for an ordinary weight,
            ``3`` for GPT-2's fused QKV.
    """

    kind: str = "replicate"
    dim: int = 0
    sections: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("replicate", "shard"):
            raise ValueError(f"unknown shard kind {self.kind!r}")
        if self.sections < 1:
            raise ValueError(f"sections must be >= 1, got {self.sections}")

    @property
    def replicated(self) -> bool:
        return self.kind == "replicate"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ShardSpec:
        return ShardSpec(kind=d["kind"], dim=int(d.get("dim", 0)), sections=int(d.get("sections", 1)))


REPLICATE = ShardSpec("replicate")


def split_tensor(tensor: torch.Tensor, spec: ShardSpec, world_size: int) -> list[torch.Tensor]:
    """Split ``tensor`` into the ``world_size`` local pieces the plan calls for.

    Returns ``world_size`` contiguous clones. Replicated tensors return the same
    values ``world_size`` times.

    Raises:
        ValueError: if the split would not be even, or would cut a fused section.
    """
    if spec.replicated:
        return [tensor.clone() for _ in range(world_size)]
    size = tensor.shape[spec.dim]
    if size % spec.sections != 0:
        raise ValueError(f"dim {spec.dim} of size {size} is not divisible into {spec.sections} sections")
    section = size // spec.sections
    if section % world_size != 0:
        raise ValueError(
            f"section of size {section} does not divide evenly across world_size={world_size}; "
            "an uneven tensor-parallel split is a configuration error, not something to pad around"
        )
    per = section // world_size
    pieces: list[torch.Tensor] = []
    for rank in range(world_size):
        parts = [
            tensor.narrow(spec.dim, s * section + rank * per, per) for s in range(spec.sections)
        ]
        pieces.append(torch.cat(parts, dim=spec.dim).contiguous() if len(parts) > 1 else parts[0].clone())
    return pieces


def merge_pieces(pieces: list[torch.Tensor], spec: ShardSpec) -> torch.Tensor:
    """Inverse of :func:`split_tensor`: rebuild the global tensor from all pieces."""
    if spec.replicated:
        return pieces[0].clone()
    world_size = len(pieces)
    per = pieces[0].shape[spec.dim] // spec.sections
    parts = [
        pieces[rank].narrow(spec.dim, s * per, per)
        for s in range(spec.sections)
        for rank in range(world_size)
    ]
    return torch.cat(parts, dim=spec.dim).contiguous()


# --------------------------------------------------------------------- plan


def gpt2_tp_plan(state_dict: dict[str, torch.Tensor], *, vocab_parallel: bool = True) -> dict[str, ShardSpec]:
    """The Megatron-style tensor-parallel plan for this repository's GPT.

    ``nn.Linear`` stores its weight as ``(out_features, in_features)``, so a
    *column-parallel* layer (split the output) is dim 0 and a *row-parallel*
    layer (split the input, then all-reduce the partial sums) is dim 1. Biases
    follow the output dimension, which means a row-parallel layer's bias is
    replicated and added once after the reduction, not sharded.

    * ``attn.c_attn`` -- column parallel, ``sections=3`` for fused QKV.
    * ``attn.c_proj`` -- row parallel.
    * ``mlp.c_fc`` -- column parallel.
    * ``mlp.c_proj`` -- row parallel.
    * ``wte`` / ``lm_head`` -- vocabulary-parallel, split on the vocab dim. They
      are tied (one Parameter, two names), so both entries get the same spec and
      resharding keeps them consistent.
    * LayerNorms, position embeddings and row-parallel biases -- replicated.

    A note on the vocabulary. Real GPT-2 has 50257 tokens, which is not
    divisible by 4, so a vocabulary-parallel split across 4 ranks is impossible
    and :func:`split_tensor` refuses it rather than padding behind your back.
    This is not an implementation limit, it is the reason Megatron-LM has a
    ``--make-vocab-size-divisible-by 128`` flag: the vocabulary is padded at
    model-construction time with rows that can never be selected, so that it
    divides by any tensor-parallel degree you might later want. Pass
    ``vocab_parallel=False`` to replicate the embedding instead, which costs
    ``vocab * d_model`` of memory per rank and is what a job with a small
    vocabulary or a large TP degree usually does.
    """
    plan: dict[str, ShardSpec] = {}
    for name in state_dict:
        if name.endswith(("attn.c_attn.weight", "attn.c_attn.bias")):
            # Fused QKV: three sections, each split per rank on head boundaries.
            plan[name] = ShardSpec("shard", dim=0, sections=3)
        elif name.endswith(("attn.c_proj.weight", "mlp.c_proj.weight")):
            plan[name] = ShardSpec("shard", dim=1)
        elif name.endswith(("mlp.c_fc.weight", "mlp.c_fc.bias")) or (vocab_parallel and name in ("wte.weight", "lm_head.weight")):
            plan[name] = ShardSpec("shard", dim=0)
        else:
            plan[name] = REPLICATE
    return plan


def shard_state_dict(
    state_dict: dict[str, torch.Tensor], plan: dict[str, ShardSpec], rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    """The slice of ``state_dict`` that rank ``rank`` of ``world_size`` owns."""
    out = {}
    for name, tensor in state_dict.items():
        spec = plan.get(name, REPLICATE)
        out[name] = split_tensor(tensor, spec, world_size)[rank]
    return out


# ------------------------------------------------------------------- on disk


@dataclass
class CheckpointIndex:
    """The JSON file that makes a directory of shard files self-describing."""

    format: str
    world_size: int
    step: int
    plan: dict[str, ShardSpec]
    global_shapes: dict[str, list[int]]
    meta: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": self.format,
                "world_size": self.world_size,
                "step": self.step,
                "plan": {k: asdict(v) for k, v in self.plan.items()},
                "global_shapes": self.global_shapes,
                "meta": self.meta,
            },
            indent=2,
            sort_keys=True,
        )

    @staticmethod
    def read(directory: Path) -> CheckpointIndex:
        d = json.loads((Path(directory) / "index.json").read_text())
        if d["format"] != FORMAT:
            raise ValueError(f"unknown checkpoint format {d['format']!r}")
        return CheckpointIndex(
            format=d["format"],
            world_size=int(d["world_size"]),
            step=int(d["step"]),
            plan={k: ShardSpec.from_dict(v) for k, v in d["plan"].items()},
            global_shapes={k: list(v) for k, v in d["global_shapes"].items()},
            meta=d["meta"],
        )


def shard_path(directory: Path | str, rank: int, world_size: int) -> Path:
    return Path(directory) / f"shard_{rank:04d}_of_{world_size:04d}.pt"


def save_sharded(
    directory: Path | str,
    local_state: dict[str, torch.Tensor],
    plan: dict[str, ShardSpec],
    *,
    rank: int,
    world_size: int,
    step: int = 0,
    global_shapes: dict[str, list[int]] | None = None,
    extra: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write rank ``rank``'s shard, and (from rank 0) the index.

    Args:
        local_state: this rank's slice of the model state dict.
        plan: the shard plan the slice was produced with.
        extra: anything else this rank owns and needs back on resume -- optimiser
            state, RNG state, the dataloader position. Written into the same file
            so a rank's state cannot be half-restored.
        global_shapes: the unsharded shapes. Required on rank 0 (they are what
            makes resharding possible without loading every file).

    Returns:
        The path of the shard file this rank wrote.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = shard_path(directory, rank, world_size)
    payload = {
        "rank": rank,
        "world_size": world_size,
        "step": step,
        "tensors": local_state,
        "extra": extra or {},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a torn file is never visible under the real name
    if rank == 0:
        if global_shapes is None:
            raise ValueError("rank 0 must pass global_shapes so the index can describe the checkpoint")
        index = CheckpointIndex(
            format=FORMAT,
            world_size=world_size,
            step=step,
            plan={k: plan.get(k, REPLICATE) for k in local_state},
            global_shapes=global_shapes,
            meta=meta or {},
        )
        itmp = directory / "index.json.tmp"
        itmp.write_text(index.to_json())
        itmp.replace(directory / "index.json")
    return path


class _ShardReader:
    """Lazily opens shard files and remembers which ones it had to touch.

    The count is the point: resharding 4 -> 2 should read every source file
    (each target rank needs half of each source), but resharding 4 -> 8 should
    have each target rank touch exactly one. ``files_read`` is asserted in the
    tests so the claim is not just a comment.
    """

    def __init__(self, directory: Path, world_size: int) -> None:
        self.directory = Path(directory)
        self.world_size = world_size
        self._cache: dict[int, dict[str, Any]] = {}
        self.files_read = 0

    def payload(self, rank: int) -> dict[str, Any]:
        if rank not in self._cache:
            path = shard_path(self.directory, rank, self.world_size)
            try:
                self._cache[rank] = torch.load(str(path), map_location="cpu", mmap=True)
            except (RuntimeError, TypeError):
                # mmap needs the zipfile serialisation; fall back for old files.
                self._cache[rank] = torch.load(path, map_location="cpu")
            self.files_read += 1
        return self._cache[rank]

    def tensor(self, rank: int, name: str) -> torch.Tensor:
        return self.payload(rank)["tensors"][name]


def _reshard_one(
    reader: _ShardReader,
    name: str,
    spec: ShardSpec,
    global_size: int,
    src_world: int,
    dst_world: int,
    dst_rank: int,
) -> torch.Tensor:
    """Rebuild one tensor's ``dst_rank`` slice from the source shards it overlaps."""
    if spec.replicated:
        return reader.tensor(0, name).clone()  # unreachable: handled by the caller

    section = global_size // spec.sections
    src_per = section // src_world
    dst_per = section // dst_world
    parts: list[torch.Tensor] = []
    for s in range(spec.sections):
        lo = dst_rank * dst_per
        hi = lo + dst_per
        # Source rank r holds section-local rows [r*src_per, (r+1)*src_per).
        first = lo // src_per
        last = (hi - 1) // src_per
        for r in range(first, last + 1):
            src_lo = r * src_per
            take_lo = max(lo, src_lo)
            take_hi = min(hi, src_lo + src_per)
            local = reader.tensor(r, name)
            # Inside the source shard, section s starts at s * src_per.
            offset = s * src_per + (take_lo - src_lo)
            parts.append(local.narrow(spec.dim, offset, take_hi - take_lo))
    return torch.cat(parts, dim=spec.dim).contiguous() if len(parts) > 1 else parts[0].clone()


def load_reshard(
    directory: Path | str,
    *,
    rank: int,
    world_size: int,
    plan: dict[str, ShardSpec] | None = None,
) -> tuple[dict[str, torch.Tensor], CheckpointIndex, int]:
    """Load the slice rank ``rank`` of a ``world_size``-way layout needs.

    The checkpoint may have been written by any world size. Nothing here ever
    materialises a full parameter on a rank that does not need one: each tensor
    is assembled straight into the destination slice from the overlapping source
    slices.

    Args:
        plan: override the plan stored in the index. Passing an all-replicate
            plan is how you restore a sharded checkpoint into a single
            unsharded process.

    Returns:
        ``(local_state, index, files_read)``.
    """
    directory = Path(directory)
    index = CheckpointIndex.read(directory)
    plan = plan or index.plan
    reader = _ShardReader(directory, index.world_size)
    # Replicated tensors are identical in every source shard, so any one will
    # do. Picking the source that this destination rank already has to open for
    # its sharded slices means the reshard usually touches one file, and it
    # spreads the read load instead of having every rank in the job hammer
    # shard 0 at the same instant.
    replica_src = min(index.world_size - 1, (rank * index.world_size) // max(world_size, 1))
    out: dict[str, torch.Tensor] = {}
    for name, saved_spec in index.plan.items():
        want = plan.get(name, REPLICATE)
        global_size = index.global_shapes[name][saved_spec.dim] if not saved_spec.replicated else 0
        if want.replicated and not saved_spec.replicated:
            # Gather: every source shard contributes.
            pieces = [reader.tensor(r, name) for r in range(index.world_size)]
            out[name] = merge_pieces(pieces, saved_spec)
        elif want.replicated:
            out[name] = reader.tensor(replica_src, name).clone()
        else:
            if saved_spec.replicated:
                out[name] = split_tensor(reader.tensor(replica_src, name), want, world_size)[rank]
            else:
                if (want.dim, want.sections) != (saved_spec.dim, saved_spec.sections):
                    raise ValueError(
                        f"{name}: cannot reshard from {saved_spec} to {want}; the shard axis "
                        "would change, which is a re-layout of the weight, not a re-split"
                    )
                out[name] = _reshard_one(
                    reader, name, saved_spec, global_size, index.world_size, world_size, rank
                )
    return out, index, reader.files_read


def load_full(directory: Path | str) -> tuple[dict[str, torch.Tensor], CheckpointIndex]:
    """Rebuild the complete, unsharded state dict. Used by eval and by export."""
    index = CheckpointIndex.read(directory)
    plan = dict.fromkeys(index.plan, REPLICATE)
    state, index, _ = load_reshard(directory, rank=0, world_size=1, plan=plan)
    return state, index


def load_extra(directory: Path | str, rank: int) -> dict[str, Any]:
    """Read the non-tensor payload (optimiser step, RNG, dataloader position)."""
    index = CheckpointIndex.read(directory)
    src = rank if rank < index.world_size else 0
    payload = torch.load(shard_path(Path(directory), src, index.world_size), map_location="cpu")
    return payload["extra"]


# ----------------------------------------------------------- overlapped save


class AsyncCheckpointer:
    """Snapshot on the training thread, serialise on a background thread.

    A synchronous save holds the training step for as long as the write takes.
    Splitting it in two -- a copy of every tensor into a separate buffer, then a
    write from that buffer -- means the step only pays for the copy, and the
    write overlaps the next steps. The copy still has to be synchronous: the
    optimiser is about to overwrite those tensors in place.

    One save is in flight at a time. :meth:`save` waits for the previous one, so
    a checkpoint interval shorter than the write time degrades to synchronous
    instead of piling up threads and running the host out of memory.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        #: Seconds the training thread was blocked by the most recent save.
        self.last_blocking_seconds = 0.0
        #: Wall-clock seconds the most recent background write took.
        self.last_write_seconds = 0.0

    def save(self, directory: Path | str, local_state: dict[str, torch.Tensor], **kwargs: Any) -> None:
        """Snapshot ``local_state`` and return; the write happens in the background."""
        t0 = time.perf_counter()
        self.wait()
        snapshot = {k: v.detach().to("cpu", copy=True) for k, v in local_state.items()}
        self.last_blocking_seconds = time.perf_counter() - t0

        def _run() -> None:
            w0 = time.perf_counter()
            try:
                save_sharded(directory, snapshot, **kwargs)
            except BaseException as exc:  # surfaced on the next wait()
                self._error = exc
            self.last_write_seconds = time.perf_counter() - w0

        self._thread = threading.Thread(target=_run, name="ckpt-writer", daemon=False)
        self._thread.start()

    def wait(self) -> None:
        """Block until the in-flight write finishes, re-raising anything it hit."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err


def _demo() -> None:
    """Save real GPT-2 124M under 4-way TP, reshard it to 1, 2 and 8, time it.

    ``python -m transformer_internals.cluster.checkpoint``. This is where the
    resharding numbers in ``docs/CLUSTER.md`` come from.
    """
    import shutil
    import tempfile

    from transformer_internals.config import GPTConfig
    from transformer_internals.model import GPT

    torch.manual_seed(0)
    model = GPT(GPTConfig())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Replicated vocabulary: 50257 does not divide by 4. See gpt2_tp_plan.
    plan = gpt2_tp_plan(state, vocab_parallel=False)
    shapes = {k: list(v.shape) for k, v in state.items()}
    print(f"GPT-2 124M: {model.num_parameters():,} parameters in {len(state)} tensors, fp32")

    directory = Path(tempfile.mkdtemp())
    try:
        t0 = time.perf_counter()
        for rank in range(4):
            save_sharded(
                directory, shard_state_dict(state, plan, rank, 4), plan,
                rank=rank, world_size=4, step=100, global_shapes=shapes,
            )
        save_s = time.perf_counter() - t0
        sizes = [p.stat().st_size for p in sorted(directory.glob("*.pt"))]
        print(f"save under tp=4: {save_s*1e3:.0f} ms, {len(sizes)} shards of "
              f"{sizes[0]/1e6:.0f} MB, {sum(sizes)/1e6:.0f} MB total")
        wte = state["wte.weight"].numel() * 4
        model_bytes = sum(v.numel() * v.element_size() for v in state.values())
        print(f"  (a {model_bytes/1e6:.0f} MB state dict becomes {sum(sizes)/1e6:.0f} MB on disk: every")
        print("   replicated tensor is written once per shard, and the tied embedding is under both")
        print(f"   wte.weight and lm_head.weight, so {2*wte/1e6:.0f} MB of each {sizes[0]/1e6:.0f} MB shard is")
        print(f"   one {wte/1e6:.0f} MB matrix written twice. Deduplicating replicated tensors to")
        print("   rank 0 is the obvious fix and is not implemented here.)")

        for dst in (1, 2, 8):
            t0 = time.perf_counter()
            pieces: dict[str, list[torch.Tensor]] = {k: [] for k in state}
            opens = 0
            for rank in range(dst):
                local, _, files_read = load_reshard(directory, rank=rank, world_size=dst)
                opens += files_read
                for k, v in local.items():
                    pieces[k].append(v)
            dt = time.perf_counter() - t0
            merged = {k: merge_pieces(v, plan[k]) for k, v in pieces.items()}
            same = all(torch.equal(merged[k], state[k]) for k in state)
            print(f"reshard 4 -> {dst}: {dt*1e3:.0f} ms for all {dst} ranks, "
                  f"{opens} shard-file opens, bitwise identical: {same}")

        restored = GPT(GPTConfig())
        full, _ = load_full(directory)
        restored.load_state_dict(full)
        idx = torch.randint(0, 50257, (2, 64))
        with torch.no_grad():
            a, b = model(idx)["logits"], restored(idx)["logits"]
        print(f"logits after 4 -> 1 reshard: max abs diff {(a-b).abs().max().item():g}, "
              f"torch.equal: {torch.equal(a, b)}")
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    _demo()
