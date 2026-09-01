# The training harness

The rest of this repository is about whether a transformer is *correct*. This
half is about whether a training run *survives*: sharded checkpoints that can be
read back under a different parallel layout, a job that comes back after a rank
dies, a dataloader that resumes mid-epoch without repeating or losing a sample,
and a cost model that says which axis of parallelism belongs inside a node.

There is no GPU cluster behind this document. Pretending otherwise would be the
easiest thing to catch, so every number below is labelled:

* **MEASURED** — produced by running the code in this repository on the machine
  it was written on: an Apple M1 Max, 10 cores, 32 GiB, macOS 15.6.1, PyTorch
  2.2.2, CPU only, `torch.distributed` on the gloo backend with one OS process
  per rank. Every measured number below is read from
  [`results/cluster.json`](../results/cluster.json), which `make cluster`
  (`scripts/run_cluster.py`) writes in one 28-second run.
* **MODELLED** — computed from published interconnect bandwidths by
  `src/transformer_internals/cluster/fabric.py`, which prints its sources.
  Nothing here was measured on an H100.

The bet is that correctness of a sharded implementation is hardware-independent.
A resharded checkpoint either reproduces the original logits or it does not, and
that answer is the same on a laptop and on 512 GPUs.

---

## 1. Checkpoints that survive a change of layout

**The problem.** A job runs on 4 GPUs with 4-way tensor parallelism. Each rank
owns a slice of every parallel weight and writes only its own slice, because
gathering the whole model onto rank 0 means moving hundreds of gigabytes over
the fabric to a single writer. Two weeks later the job has to come back on 2
GPUs, or 8, or be evaluated in one process. The bytes on disk are laid out for a
world size that no longer exists.

`cluster/checkpoint.py` implements the resharding. A **shard plan** gives each
parameter a `ShardSpec`: replicated, or split along a dimension. `save_sharded`
writes one file per rank plus a JSON index that makes the directory
self-describing. `load_reshard` rebuilds rank *r* of a *new* world size, reading
only the source shards that overlap the slice it is building.

### The detail that makes it a real plan and not a byte-shuffle

GPT-2's `c_attn` weight is one `(3C, C)` matrix holding query, key and value
projections stacked along dim 0. Tensor parallelism splits it column-wise into
per-head groups, so rank *r* must own head-group *r* of Q **and** of K **and** of
V. Splitting the `(3C, C)` matrix into `world_size` contiguous blocks does not do
that. With 3 ranks, rank 0 would get all of Q and none of K or V — and the model
would still load, still run, and be silently wrong. `ShardSpec.sections = 3` is
what makes the split per-section.

`tests/test_cluster.py::test_fused_qkv_split_keeps_whole_heads_of_each_projection`
tags every row of the matrix with which projection and which head it belongs to,
then asserts that each rank's slice contains only whole heads of each of the
three projections — and that the naive plan does not.

Two more tests assert the plan is a *valid tensor-parallel* plan rather than
merely reversible: the column-parallel shards concatenate to the full matmul
output exactly, and the row-parallel shards' partial sums add to it.

### What the resharding actually delivers

`make cluster` — **MEASURED**, `results/cluster.json` → `resharding`:

```
GPT-2 124M: 124,439,808 parameters in 149 tensors, fp32
save under tp=4: 3420 ms, 4 shards of 397 MB, 1589 MB total
reshard 4 -> 1:  247 ms for all 1 ranks, 4 shard-file opens, bitwise identical: True
reshard 4 -> 2:  326 ms for all 2 ranks, 4 shard-file opens, bitwise identical: True
reshard 4 -> 8: 1073 ms for all 8 ranks, 8 shard-file opens, bitwise identical: True
logits after 4 -> 1 reshard: max abs diff 0, torch.equal: True
```

One run, on a laptop that was doing other things. The wall-clock figures move by
tens of percent between runs and are here for order of magnitude; the
`bitwise identical: True` and the `torch.equal: True` do not move at all.

`torch.equal`, not `allclose`. Resharding moves bytes; it does not compute
anything, so any difference at all is a bug. The test matrix covers 4→2, 4→1,
2→8, 4→8, 8→4 and 1→4 and asserts exact logit equality in every direction.

**Reads only what it needs.** Going 4→8 splits each source shard in two, so a
destination rank opens exactly one file — asserted, not claimed. Going 4→2 opens
two. Restoring the full model into one process opens all four. Replicated
tensors are read from the source shard the rank is already opening rather than
always from shard 0, so a 512-rank restore does not turn into 512 ranks
stampeding one file.

**Overlapped save.** `AsyncCheckpointer` snapshots to host memory on the
training thread and serialises on a background thread. The copy has to be
synchronous, because the optimiser is about to overwrite those tensors in place;
the write does not. **MEASURED** on a 25 MB state, three repeats
(`results/cluster.json` → `async_checkpoint`):

```
sync  42.3 ms blocking   (42.3, 51.3, 44.0 across the three)
async  2.3 ms blocking + 42.6 ms on the background thread
                          (2.3-3.1 ms blocking, 42.6-54.0 ms in the background)
```

The training step pays the blocking figure and nothing else, so the step-time
ratio was 13.7x, 22.1x and 14.5x across the three repeats. The ratio moves that
much because the synchronous arm is at the mercy of the page cache; the test
asserts only the direction, which is the part that is a property of the design
rather than of the disk.

**Known cost, not fixed.** GPT-2 124M is 498 MB of parameters in fp32, but its
state dict serialises 652 MB because the tied embedding appears under two names
(`wte.weight` and `lm_head.weight`), and four shards come to 1589 MB because
every replicated tensor is written once per shard. 309 MB of each 397 MB shard
is one 154 MB matrix stored twice. Deduplicating replicated tensors to rank 0 is
the obvious fix and is not implemented.

**A constraint worth naming.** Real GPT-2 has 50257 vocabulary entries, which is
not divisible by 4, so a vocabulary-parallel embedding cannot be split 4 ways.
`split_tensor` refuses rather than padding behind your back. This is exactly why
Megatron-LM has `--make-vocab-size-divisible-by 128`: the vocabulary is padded
with unreachable rows at construction time so it divides by any tensor-parallel
degree the run might later use. The alternative — replicating the embedding — is
supported with `gpt2_tp_plan(state, vocab_parallel=False)`.

---

## 2. Killing a rank and getting the run back

`cluster/failure.py` runs a real multi-process job: one OS process per rank
(`python -m transformer_internals.cluster.failure`, with `RANK`, `WORLD_SIZE`,
`MASTER_ADDR` and `MASTER_PORT` in the environment, which is the same contract
`torchrun` has with its workers), gloo, `DistributedDataParallel`, checkpoints
every 5 steps. The launcher then sends a real `SIGKILL` to rank 1 — no
unwinding, no teardown, no chance to flush — waits for the survivors to be torn
down, and relaunches from the last checkpoint.

The assertion is not "the loss still goes down". The same job is run twice, once
uninterrupted and once with rank 1 killed at step 12 when the last checkpoint
was step 10, and the two loss trajectories are compared step by step.

**MEASURED** (`results/cluster.json` → `failure_restart`):

```
killed rank 1 at step 12, checkpoint was step 10,
time to recover 2.20s, launches 2
max |loss(resumed) - loss(uninterrupted)| over steps 11-20 = 0.000e+00
```

Zero, not "small". Same weights, same optimiser moments, same data in the same
order gives the same float. Three things have to be in the checkpoint for that
to hold and all three are: parameters, optimiser state (Adam's two moments and
its step count — restoring weights but not moments restarts the bias correction
and puts a visible bump in the loss for a few hundred steps), and the dataloader
position for every rank.

Time-to-recover here is 2.20 s, and almost all of it is process startup and
importing torch. On a real job that interval also contains the scheduler
noticing, requeueing, allocating replacement nodes, and re-reading a checkpoint
of hundreds of gigabytes over the storage fabric — minutes, not seconds. The
structure of the measurement is the same: the clock starts when the rank dies
and stops when the first optimiser step lands after the restart.

### The elastic case

Not implemented locally, because it needs a rendezvous backend and more than one
machine to be worth anything. What it means in practice:

`torchrun --max-restarts=3 --nnodes=8 --rdzv-backend=c10d` puts an elastic agent
on each node. The agent owns the local ranks; when one dies, the agent kills the
rest of its local ranks, re-enters the rendezvous, and the whole group restarts
from the last checkpoint. With `--nnodes=6:8` the rendezvous accepts a range,
and the job continues on whatever nodes are present — which is only usable if
the world size is not baked into the data plan. That is the reason the streaming
dataloader below reshards across a change of world size: with a fixed
`i % world_size` assignment, coming back on 6 nodes instead of 8 silently
changes which samples each rank reads and, without the replanning, would repeat
some and drop others.

Two layers of restart, because they fail at different granularities. `torchrun`
can restart ranks in place after a transient failure. Only the scheduler can
replace a node that has actually died, which is what `--requeue` and the
`SIGUSR1` trap in `deploy/slurm_train.sbatch` are for.

The single most expensive mistake in this area is *not* the restart: it is a
hung collective. A rank waiting on a peer that will never arrive blocks until
the NCCL timeout, holding the whole allocation. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
turns that hang into a failed rank, which the agent can act on. It is set in
every launcher in `deploy/`.

---

## 3. Streaming data that resumes correctly

`cluster/streaming.py`. Three properties, all asserted:

**Disjoint shards covering the epoch exactly once.** One global order for the
epoch — a permutation seeded from `(seed, epoch)`, so every rank computes it
without communicating. Rank *r* of *W* takes positions `r, r+W, r+2W, ...`. A
strided deal rather than contiguous blocks, deliberately: contiguous blocks give
each rank one long region of the corpus, so a rank that draws a region of short
documents runs ahead and every step is set by the slowest rank for the whole
epoch.

**The position is part of the checkpoint.** Not the epoch number — the position
*within* the epoch, per rank. Resuming at the top of the epoch is the most
common data bug in a restartable trainer and it is invisible: nothing errors,
the loss curve looks normal, and the model quietly sees a fraction of the corpus
many times and the rest never.

**Resuming is replanning, not seeking.** Given every rank's position, the set of
already-consumed order positions is known exactly, so the remainder of the epoch
is dealt out again over however many ranks now exist. When the world size has
not changed and all ranks stopped at the same step — the normal case, because
ranks checkpoint together at a barrier — replanning reproduces the original
assignment exactly. That is what makes section 2's loss trajectory match. When
the world size *has* changed, coverage is still exactly once, which is what
makes elastic restart safe.

The prefetch thread is a trap and is tested as one. With a depth of 8 the reader
has pulled samples the consumer never received; if the position counter followed
the reader, those samples would be lost at the next restart. It follows the
consumer, so they are re-read. Re-reading is free; skipping is silent data loss.

### Prefetch, measured honestly

Prefetch is usually presented as free throughput. It is not: it buys the overlap
of a slow read with the training step, and if the read is not slow it costs.
Both cases, **MEASURED** (`results/cluster.json` → `streaming`), one rank of
two, 256-token samples, best of three:

| prefetch depth | page-cached memmap, no consumer work | 500 µs reads + 500 µs step |
|---:|---:|---:|
| 0 | 117,055 samples/s | 708 samples/s |
| 2 | 25,523 samples/s | 1,354 samples/s |
| 8 | 59,969 samples/s | 1,356 samples/s |

Against a memory-mapped file already in the page cache, a read is a memcpy and
the reader thread is pure overhead: prefetch makes it **2x to 4.6x slower**.
Against storage that takes 500 µs a read, with a consumer that takes 500 µs a
sample, prefetch 8 is **1.92x** faster than no prefetch — against a ceiling of
exactly 2.00x, because when the read and the step cost the same, perfect overlap
halves the total.

The first column is the reason to measure rather than to set the depth to 32 and
move on. The right depth is the one that covers the jitter in read latency for
*this* storage, and past that point a deeper queue only adds memory and
contention.

---

## 4. Which parallelism goes where, from the numbers

`cluster/fabric.py`. **EVERYTHING IN THIS SECTION IS MODELLED.** The inputs are
published peak bandwidths, each carrying its source in the code; the two
efficiency factors are labelled assumptions.

Ring collectives, `time = latency + bytes / bandwidth`:

* all-reduce: `2(N-1)/N · S/B + 2(N-1)·lat` (reduce-scatter then all-gather)
* all-gather / reduce-scatter: `(N-1)/N · S/B + (N-1)·lat`
* point-to-point: `S/B + lat`

Published bandwidths used (peak, unidirectional, per GPU — NVIDIA quotes NVLink
as 900 GB/s *bidirectional aggregate*, which is 450 GB/s each way, and each way
is what a ring gets):

| key | GB/s | source |
|---|---:|---|
| `nvlink4` | 450 | H100 datasheet: 18 NVLink 4 links × 25 GB/s per direction |
| `nvlink3` | 300 | A100 datasheet: 12 NVLink 3 links × 25 GB/s per direction |
| `pcie5` | 63 | PCI-SIG Gen5 x16, 32 GT/s/lane, 128b/130b |
| `ib_ndr` | 50 | DGX H100: 8 × ConnectX-7 NDR400, 400 Gb/s each, one per GPU |
| `ib_hdr` | 25 | DGX A100: 8 × HDR200 compute NICs |
| `roce200` | 25 | 200 GbE line rate; RDMA verbs over Ethernet |

Assumptions, not citations: link efficiency 0.85 (NCCL bus bandwidth typically
reaches 80–90% of peak on large messages), MFU 0.45, and the small-message
latencies, which no vendor publishes in a comparable form.

### The output

`python -m transformer_internals.cluster.fabric`, 70B-class model, 8k context,
64 H100s as tp=8 × dp=8:

```
Modelled compute per step: 8533.1 ms

Per-step communication time, milliseconds (and as a fraction of compute):
strategy         nvlink4         pcie5        ib_ndr       roce200
tp           1643.7 (0.19x)  11407.9 (1.34x)  14219.8 (1.67x)  28475.5 (3.34x)
fsdp          114.2 (0.01x)    815.4 (0.10x)   1027.3 (0.12x)   2054.6 (0.24x)
ddp            76.1 (0.01x)    543.6 (0.06x)    684.9 (0.08x)   1369.7 (0.16x)
```

Read the `tp` row across. Tensor parallelism moves 2560 all-reduces of 134 MB
per step, because it communicates *per layer, per microbatch, in both
directions*, and the volume scales with tokens rather than with parameters. On
NVLink that is 0.19x the compute time and hides under it. On InfiniBand NDR it
is 1.67x the compute time — the GPUs would spend most of the step idle. On RoCE
it is 3.3x.

Sharded data parallelism moves three passes over the parameters this rank holds,
**once per step**, not once per microbatch. Across InfiniBand that is 0.12x
compute, and it overlaps with the backward pass.

The ratio is what matters: on the same fabric, TP costs **13.8x** what FSDP
costs per step. That number, not tradition, is what pins tensor parallelism
inside the node. Pipeline parallelism is smaller still — point-to-point
activations at stage boundaries only — and the test asserts it is under 1% of
the TP traffic, which is why the stage boundaries are the right place to put a
node boundary.

The crossover degree, computed rather than argued:

```
TP becomes communication-bound on nvlink4  at tp=64
TP becomes communication-bound on ib_ndr   at tp=8
TP becomes communication-bound on roce200  at tp=4
```

So the standard layout — TP inside the 8-GPU NVLink domain, FSDP or pipeline
across the InfiniBand fabric — is not a convention. It is where these two curves
cross.

### GPUDirect RDMA, and what it removes

Without it, a tensor leaving GPU 0 on node A for GPU 0 on node B goes: device →
host bounce buffer (across PCIe), host → NIC (across PCIe again), the wire, then
NIC → host → device on the far side. Two extra PCIe crossings and two host
memory copies per direction, and the CPU is in the data path, so the latency
floor becomes a kernel round-trip rather than a wire round-trip.

GPUDirect RDMA lets the NIC DMA straight into and out of the GPU's BAR-mapped
memory. The payload crosses PCIe once per side, never touches host memory, and
the CPU only posts the work request. That is why it matters most for small and
medium messages — the fixed cost is what it removes — and why the NIC has to sit
under the same PCIe switch as the GPU, which is what `NCCL_NET_GDR_LEVEL=2`
means. **MODELLED** effect of turning it off, as a 1.6x bandwidth penalty and
+5 µs:

```
tp over ib_ndr: 14219.8 ms -> 22887.9 ms (1.61x)
```

Two operational notes that are not in the model. RDMA has to pin the memory it
registers with the NIC, so a container without `CAP_IPC_LOCK` fails
registration and NCCL falls back to TCP — the job runs, at a fraction of the
speed, with nothing in the logs but a line in `NCCL_DEBUG=INFO` output saying
`via NET/Socket` instead of `via NET/IB`. And on RoCE the same verbs run over
Ethernet, which needs PFC/ECN configured to be lossless; without it the fabric
connects and then behaves like a congested network.

### Validating the shape of the model

The H100 numbers cannot be checked here. The *functional form* can.
`cluster/collbench.py` measures gloo all-reduce on this laptop across message
sizes and fits `t = latency + bytes/bandwidth` with the same `2(N-1)/N` ring
factor. **MEASURED** (`results/cluster.json` → `collectives`), 2 ranks, CPU, loopback,
64 KiB to 4 MiB:

```
t = 408 us + bytes / 3.16 GB/s     R^2 = 0.9911
```

Runs on a laptop that was doing other things gave 180 µs / 2.95 GB/s,
408 µs / 3.16 GB/s, 450 µs / 3.85 GB/s and 474 µs / 1.97 GB/s. The fitted
constants move with the machine's load; **R² stayed above 0.99 in every one of
them**, and R² is the claim. The affine form describes a real collective. The
2-4 GB/s is loopback TCP plus a memory copy, not a fabric, and means nothing
beyond this machine.

Below about 64 KiB the fit degrades badly — R² of 0.84 over 16 KiB–4 MiB in one
run — which is the model's own point: small messages are latency, not bandwidth,
and that is the regime where GPUDirect and the number of hops decide everything.

The full sweep is three collectives across message sizes and world sizes, in
NCCL's bus-bandwidth convention, written to `results/collective_bandwidth.json`
by `make collectives`. The README's Part 4 carries the table. Two findings from
it: the fitted latency term is identical for all three collectives at world size
2 (143.8, 143.8, 143.9 µs), so it really is a per-call fixed cost; and
reduce-scatter comes in at 1.193 GB/s of bus bandwidth against an all-reduce's
2.923, which is within 20% of the 2x ratio that gloo servicing reduce-scatter as
an all-reduce plus a slice would produce. `collbench.py` said that in a docstring
before it was measured.

On a CUDA node the same command measures NCCL over the real fabric, and
`fabric.link_from_measurement` turns the fit into a `Link` that replaces the
datasheet NVLink entry. See `docs/GPU_RUN.md`.

---

## 5. cgroups

Every scheduler that runs training jobs — Slurm's cgroup plugin, Kubernetes,
Docker, Ray under either — enforces its limits through cgroups, and nothing the
process can see with `free` or `nproc` reflects them. Inside a container those
still report the host's memory and CPU count, so a dataloader sizing its worker
pool from `os.cpu_count()` is wrong by an order of magnitude and gets throttled
for reasons that never appear in its own logs.

`cluster/cgroups.py` reads `memory.max`, `memory.high`, `memory.current`,
`memory.swap.max`, `memory.events`, `cpu.max`, `cpu.stat` and `pids.max` for the
current process and explains each one. `deploy/cgroups_demo.sh` runs it in
Docker under a 512 MiB limit and 1.5 CPUs; the full captured output is
`deploy/cgroups_demo_output.txt`. **MEASURED**:

```
memory.max         : 536870912 bytes (0.50 GiB)
memory.high        : max (no limit at this level; ...)
memory.current     : 23445504 bytes (0.02 GiB)
memory.swap.max    : 0 bytes (0.00 GiB)
cpu.max            : 150000 100000
                     150000us of CPU time per 100000us period = 1.50 CPUs' worth.
```

Then the same container allocates past the limit:

```
allocated 384 MiB
allocated 448 MiB
### exit code: 137
```

No traceback, no `MemoryError`. 137 is 128 + SIGKILL: the cgroup OOM killer took
the process. **This is why an out-of-memory training rank looks like a node
failure rather than an exception**, and why `memory.events`' `oom_kill` counter
is the first thing to read after an unexplained rank death.

The three walls behave differently, and only one of them kills:

* `memory.max` — hard. OOM kill, signal 9, no Python-level anything.
* `memory.high` — soft. No kill; the cgroup is throttled into reclaim and the
  step time inflates. A throughput bug, not a crash, and "the run is at 30% of
  expected throughput" is exactly what it looks like from outside.
* `cpu.max` — a quota. Also no kill: the cgroup is throttled at the end of each
  period, and `cpu.stat`'s `nr_throttled` and `throttled_usec` are the proof. A
  dataloader given 2 CPUs that spawns 32 workers spends its life throttled, and
  the GPU waits.

**Swap changes the failure mode and is usually the wrong trade.** With swap
allowed, a rank survives an overshoot by paging — at disk latency, mid-step,
while every other rank waits for it at the next collective. One rank swapping
stalls the whole job, and it presents as a hang rather than as a slow rank. A
fast death is cheaper to diagnose. Most GPU clusters run training cgroups with
swap off for exactly this reason.

The demo tried to show the swapping case and could not, which is itself worth
recording: raising the cgroup's swap allowance to 512 MiB changed nothing
because the kernel had no swap device (`SwapTotal: 0 kB`), and the process was
OOM-killed again with exit 137. `memory.swap.max` is permission, not capacity.

---

## 6. Launchers

In `deploy/`, with the reasoning inline. Four schedulers, one job.

**Slurm** (`slurm_train.sbatch`). One task per GPU, `--gpus-per-task=1`,
`--cpus-per-task` sized to the node, `--gres-flags=enforce-binding` so the CPU
cgroup follows the GPU's NUMA node. `srun` starts one `torchrun` per node and
`torchrun` owns the ranks under it — two layers, because a transient rank
failure is `torchrun`'s job and a dead node is Slurm's. `--requeue` plus
`--signal=B:USR1@180` gives the job three minutes to checkpoint before the wall
clock expires and hand itself back to the scheduler. `--open-mode=append` so a
requeued attempt does not erase the log of the one that died. The NCCL block
names the HCAs and the socket interface explicitly: letting NCCL choose is how a
job ends up bootstrapping over the management NIC or a `docker0` bridge and
running at a twentieth of the expected all-reduce bandwidth.

**Ray** (`ray_train.py`). `TorchTrainer` with `FailureConfig(max_failures=3)`,
which is Ray's equivalent of `--max-restarts`, and checkpoints reported through
`ray.train.report` so the driver — which outlives the workers — still holds them
after every worker has died. The line that matters is
`placement_strategy="STRICT_PACK"`: Ray schedules by resource request and has no
idea what NVLink is, so if you want a tensor-parallel group on one machine you
have to say so. Without it Ray will spread a TP group across four machines and
the run will be fabric-bound at a tenth of the throughput, with no error
anywhere.

**Kubernetes** (`k8s/job-indexed.yaml`, `k8s/statefulset.yaml`). An Indexed Job
gives each pod a stable `JOB_COMPLETION_INDEX` to use as the node rank, and a
headless Service gives the rendezvous host a DNS name. Two things go wrong here
and nowhere else. `/dev/shm` defaults to 64 MB in a container, which kills
DataLoader workers and NCCL's shared-memory transport with a bus error minutes
into the run; it is sized explicitly with a memory-backed `emptyDir`. And
Kubernetes spreads pods across failure domains by default, which is right for a
service and wrong for a training job — the affinity rules pin every pod into one
network topology block and one pod per node. Requests equal limits so the pod is
Guaranteed QoS and not evictable, `rdma/hca_shared_devices_a` requests the RDMA
device, and `CAP_IPC_LOCK` lets RDMA pin its memory regions. The StatefulSet
variant exists for the case where each rank writes its own checkpoint shard to
its own volume: a rescheduled Job pod gets a new name and cannot find the shard
its predecessor wrote, while `gpt-train-3` always comes back as `gpt-train-3`.

**Dask** (`dask_note.md`). Not in the training loop — a dynamic work-stealing
scheduler is the opposite of what a synchronous collective needs. It belongs in
corpus preparation: tokenisation, MinHash deduplication as a groupby on band
hashes, the global shuffle that stops each rank seeing a source-correlated
stream, and the aggregate statistics that decide what goes into the run. The
handoff to training is the file layout and nothing else.

---

## Running it

```bash
# everything, including the multi-process failure test
.venv/bin/python -m pytest tests/test_cluster.py -q

# every measured number quoted above, into results/cluster.json (28 s)
make cluster

# the individual demos
.venv/bin/python -m transformer_internals.cluster.checkpoint   # resharding
.venv/bin/python -m transformer_internals.cluster.fabric       # cost model
.venv/bin/python -m transformer_internals.cluster.collbench    # measured collectives
.venv/bin/python scripts/run_collectives.py --world-sizes 2,4  # the full sweep
bash deploy/cgroups_demo.sh                                    # needs Docker
```

## What is not here

* No GPU, so no NCCL, no measured NVLink or InfiniBand number, and no profile of
  a real distributed program. The fabric model is the substitute and it says so.
* Elastic rendezvous (`--nnodes=6:8`) is described, not run: it needs more than
  one machine to mean anything.
* Replicated tensors are duplicated across shard files rather than deduplicated
  to rank 0.
* The failure test covers a killed rank. It does not cover a *hung* rank, which
  is the harder and more common production failure, and which needs a collective
  timeout to detect rather than a process exit.
