"""The same job under Ray Train instead of Slurm.

Ray is a different bargain from Slurm. Slurm gives you a fixed allocation and
gets out of the way; Ray gives you a cluster object that can grow and shrink,
an actor per worker, and a failure model where the driver survives a dead worker
and can rebuild the group. What you give up is the scheduler's understanding of
the network: Ray places by resource request, so if you want the eight ranks of a
tensor-parallel group on one machine you have to say so with a placement group,
not hope for it.

The parts that matter for a real run are marked below. Run with::

    python deploy/ray_train.py --num-workers 64 --tp 8

This file is a launcher, not a library: it is written to be read alongside
``deploy/slurm_train.sbatch`` and ``deploy/k8s/``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def train_loop_per_worker(config: dict) -> None:
    """Runs inside each Ray worker actor. One actor per GPU."""
    import ray.train
    import torch
    from ray.train import Checkpoint

    from transformer_internals.cluster.checkpoint import REPLICATE, load_full, save_sharded
    from transformer_internals.cluster.streaming import ShardedStream, StreamState, TokenShardSource
    from transformer_internals.config import GPTConfig
    from transformer_internals.model import GPT

    ctx = ray.train.get_context()
    rank, world_size = ctx.get_world_rank(), ctx.get_world_size()

    model = GPT(GPTConfig(**config["model"]))
    # ray.train.torch.prepare_model does the DDP wrap and the device move.
    model = ray.train.torch.prepare_model(model)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    start_step, stream_state = 0, None
    # A resumed run gets the checkpoint the driver kept. This is the whole
    # reason Ray's failure handling is usable: the driver outlives the workers,
    # so the checkpoint is still addressable after every worker has died.
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            state, index = load_full(ckpt_dir)
            model.module.load_state_dict(state)
            extra = torch.load(Path(ckpt_dir) / "extra.pt", map_location="cpu")
            opt.load_state_dict(extra["optimizer"])
            stream_state = StreamState.from_dict(extra["stream"])
            start_step = index.step

    source = TokenShardSource(config["data_path"], config["block_size"])
    stream = ShardedStream(source, rank=rank, world_size=world_size,
                           state=stream_state, seed=config["seed"], prefetch=4)
    batches = iter(stream)

    for step in range(start_step + 1, config["steps"] + 1):
        items = [next(batches) for _ in range(config["micro_batch"])]
        x = torch.stack([a for a, _ in items])
        y = torch.stack([b for _, b in items])
        loss = model(x, targets=y)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % config["ckpt_every"] == 0:
            metrics = {"loss": float(loss.detach()), "step": step}
            with tempfile.TemporaryDirectory() as tmp:
                if rank == 0:
                    state = {k: v.cpu() for k, v in model.module.state_dict().items()}
                    save_sharded(
                        tmp, state, dict.fromkeys(state, REPLICATE), rank=0, world_size=1,
                        step=step, global_shapes={k: list(v.shape) for k, v in state.items()},
                    )
                    torch.save(
                        {"optimizer": opt.state_dict(), "stream": stream.state_dict()},
                        Path(tmp) / "extra.pt",
                    )
                    ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))
                else:
                    ray.train.report(metrics)
        elif rank == 0:
            ray.train.report({"loss": float(loss.detach()), "step": step})


def main() -> None:
    import ray
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=64)
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel degree, kept inside a node")
    ap.add_argument("--data-path", default=os.environ.get("TI_DATA", "data/tokens.bin"))
    ap.add_argument("--storage-path", default=os.environ.get("TI_STORAGE", "/mnt/shared/ray-runs"))
    args = ap.parse_args()

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    scaling = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 12},
        # STRICT_PACK forces every worker of a bundle onto one node. That is how
        # a tensor-parallel group is kept inside the NVLink domain under Ray:
        # the scheduler has no idea what NVLink is, so the constraint has to be
        # expressed as placement. Without it Ray will happily spread a TP group
        # across four machines and the run will be fabric-bound at a tenth of
        # the throughput, with no error anywhere.
        placement_strategy="STRICT_PACK" if args.tp > 1 else "PACK",
    )

    run = RunConfig(
        name="gpt-train",
        # Shared filesystem or object store. A checkpoint on a worker's local
        # disk is gone with the worker.
        storage_path=args.storage_path,
        failure_config=FailureConfig(
            # Rebuild the worker group and resume from the last checkpoint. This
            # is the Ray equivalent of torchrun --max-restarts, and it covers
            # the same class of failure: a rank dies, the rest are unusable
            # because the communicator is broken, everything restarts.
            max_failures=3,
        ),
        checkpoint_config=CheckpointConfig(num_to_keep=3,
                                           checkpoint_score_attribute="loss",
                                           checkpoint_score_order="min"),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model": {"vocab_size": 50257, "n_positions": 1024, "n_layer": 12,
                      "n_head": 12, "n_embd": 768, "dropout": 0.0},
            "lr": 3e-4, "steps": 100_000, "micro_batch": 8, "block_size": 1024,
            "ckpt_every": 1000, "seed": 1234, "data_path": args.data_path,
        },
        scaling_config=scaling,
        run_config=run,
    )
    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    main()
