# Running this on a rented GPU box

Every number in this repository was measured on an Apple M1 Max with no CUDA.
The gaps that leaves are named in the README's Limitations, and all of them
close on one eight-GPU node for a few hours. This is the procedure.

Read this once before booking. The single most expensive mistake available here
is renting the box and then discovering something that could have been found on
a laptop, which is what `--dry-run` and `--smoke` exist to prevent.

---

## 0. Before you rent anything

Two commands, on the laptop, costing nothing:

```bash
scripts/gpu_preflight.py --dry-run --stub-gpus 8 --stub-capability 8.0
./scripts/run_on_gpu.sh --smoke --infra-only --skip-install
```

The first resolves every CUDA decision against a fabricated eight-GPU node:
which backend, which device each rank gets at every world size, which
mixed-precision policy, and whether the shapes fit in 80 GB. No CUDA call is
made. Change `--stub-capability` to `9.0` to see what an H100 node resolves to,
or `7.0` to watch bf16 be refused with fp16 named as the alternative.

The second runs the whole training-infrastructure pipeline at tiny sizes into a
scratch tree. **Measured on the development laptop over three runs: 5m29s, 6m17s
and 6m54s**, the last with the machine at a load average of 176 from unrelated
work. Per-stage times land in `smoke/results/.run_state/timings.tsv`. Adding the
weight-dependent stages (drop `--infra-only`) took about twelve minutes there.

If either fails, fix it before renting. Neither needs a GPU.

---

## 1. Which box

**Recommended: 8x A100 80GB SXM.** SXM rather than PCIe, and this matters more
than the choice between A100 and H100. The measurement this trip is for is the
NCCL collective sweep, and on an SXM node that runs over NVLink and NVSwitch,
while on a PCIe node it runs over PCIe and through the host. `nvidia-smi topo -m`
prints `NV*` between every pair on an SXM node and `PIX`, `PXB` or `SYS` on a
PCIe one; the preflight prints that table, and it is the first thing to read.

| shape | why | why not |
|---|---|---|
| **8x A100 80GB SXM** | NVLink 3 between all eight; 80 GB leaves room for every shape here; the cheapest node that answers all the questions | none for this workload |
| 8x H100 80GB SXM | NVLink 4, roughly 1.5x the NVLink bandwidth and 3x the dense bf16 rate | costs more per hour for measurements that are about the *shape* of the curves |
| 8x A100 40GB | works; every shape in this repository fits | halves the headroom for the activation-memory sweep, which is the one measurement that wants a big batch |
| 8x A100 **PCIe** | cheapest eight-GPU node | the collective benchmark then measures PCIe, which is a different and less interesting number. Acceptable only if the goal is the correctness proofs |
| 2x or 4x anything | fine for the equivalence proofs and mixed precision | the collective sweep wants a long ring; world size 8 is the point |
| 1x anything | roofline, MFU, profiler traces and activation memory all work | no collectives at all, so half the run is skipped |

**Provider.** These instructions are written for RunPod because that is the
cheapest way to get an eight-GPU SXM node by the hour, but nothing here is
RunPod-specific: any box with eight GPUs, a CUDA build of torch and SSH works.
On RunPod, prefer Secure Cloud over Community Cloud for a multi-GPU pod;
Community Cloud hosts are individually configured and the NVLink topology is
the thing being measured.

**Disk.** 60 GB of container disk is comfortable. The Python dependencies are
about 8 GB with a CUDA torch already in the image, and the published GPT-2
checkpoint and the tokenized corpus are under 2 GB.

**Image.** Any recent `runpod/pytorch` template. Check what it has before
installing anything:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

This repository supports torch 2.2 through 2.5. If the image has 2.6 or newer,
either pin it down or read `pyproject.toml`'s note first: `torch.load` changed
its `weights_only` default in 2.6 and the checkpoint loader is written against
the older behaviour.

---

## 2. What it costs

Two numbers multiply: the hourly rate and the hours. The hours are the part this
repository controls.

**Hourly rate.** At the time of writing, an 8x A100 80GB SXM node on RunPod
Secure Cloud lists at roughly $12 to $16 per hour for the whole node, and an 8x
H100 node at roughly $20 to $30. **Verify the current price before booking**;
these move, and spot or community pricing is lower and less reliable. This
paragraph is the only place in this repository quoting a number nobody here
measured, and it is here because "roughly what does it cost" is a fair question.

**Hours.** Budget as follows, and see the next section for where the numbers
come from:

| what | wall clock | node-hours on 8x A100 |
|---|---|---|
| setup, install, smoke run | 20 to 30 min | 0.5 |
| the training-infrastructure stages (`--infra-only`) | 45 to 90 min | 1.0 |
| the weight-dependent stages (Parts 6 to 8) | 30 to 60 min | 0.75 |
| re-runs, a second look at anything surprising | 60 min | 1.0 |
| **total** | | **about 3 node-hours** |

So roughly **$40 to $50 on an 8x A100 node**, or **$70 to $90 on H100**, at the
rates above. Rent for four hours, not three: the cost of overrunning is another
hour, and the cost of being cut off mid-sweep is the whole session. Everything
is checkpointed per stage, so an interrupted run resumes, but a terminated pod
does not come back.

**How to spend less.** Run `--infra-only` first. The training-infrastructure
half is the part that cannot be measured anywhere else; the Part 6 to 8
measurements are single-GPU work that a cheap one-GPU box does just as well and
for a tenth of the price.

---

## 3. The procedure on the box

```bash
# 1. get the code
git clone https://github.com/MartinMashalov/gpt2-harness
cd gpt2-harness

# 2. look at the machine. Prints GPU count and names, compute capability,
#    driver, CUDA runtime, NCCL version and the full nvidia-smi topology
#    matrix, then resolves the backend, the per-rank device placement for
#    every world size, and the mixed-precision policy. Exits non-zero if any
#    of that is impossible.
python scripts/gpu_preflight.py --world-sizes 2,4,8 --require-cuda

# 3. prove the pipeline at tiny sizes before spending an hour on the real one.
#    Writes to smoke/, so it cannot touch anything committed.
./scripts/run_on_gpu.sh --smoke

# 4. the real sweep
./scripts/run_on_gpu.sh 2>&1 | tee run.log
```

If step 4 dies, run it again. Completed stages are skipped:

```bash
./scripts/run_on_gpu.sh --resume
```

To redo one stage, delete its marker:

```bash
rm results/.run_state/collectives.done
./scripts/run_on_gpu.sh --stages collectives
```

Useful flags:

| flag | effect |
|---|---|
| `--infra-only` | only the stages that need no published checkpoint |
| `--list` | print the stages and exit |
| `--stages a,b` | run a subset |
| `--skip-install` | the environment is already set up |
| `--skip-tests` | skip the correctness gate. Do not, on the first run |
| `--keep-going` | do not stop at the first failure |
| `--world-sizes 2,4,8` | which world sizes the collective sweep uses |
| `--fresh` | ignore the markers and re-run everything |

**Bring the results home before the pod dies.** From the laptop:

```bash
scp -r -P <port> root@<host>:/workspace/gpt2-harness/results ./results-gpu
scp -r -P <port> root@<host>:/workspace/gpt2-harness/assets ./assets-gpu
```

---

## 4. The stages, and how long each takes

Two columns of times, and they mean different things.

**Measured on the M1 Max** is what the same stage takes on the machine this
repository was written on. Those are real numbers: the top rows are from this
session's runs and from the `results/.run_state/timings.tsv` that a run writes
(generated, not committed, so it appears only after you run something), and the
Part 6 to 8 rows are the runtimes the `Makefile` records for the same targets.
One row says "not recorded" rather than guessing.

**Two stages do not use the GPU at all, by design.** `cluster` measures
resharding, an overlapped checkpoint, a streaming dataloader and a killed rank,
all of which are disk and process behaviour; `diagnose` profiles a training
step, and torch 2.2's profiler has no MPS backend, which is why the committed
profile is a CPU one. Both hardcode gloo. Their budgets below are the same CPU
work on a faster host CPU, not GPU work, which is why they are small.

**Budget on 8x A100** is an *estimate*, not a measurement. Nothing in this
repository has run on a GPU. The estimates come from what each stage is bound
by: a stage dominated by process startup and tiny tensors will not speed up
much, a stage dominated by GPT-2 forward and backward passes should speed up a
lot, and a stage that does not exist on CPU has no CPU time to scale from.
**When the run finishes, replace this column with the measured one from
`results/.run_state/timings.tsv`.** That is the point of writing the timings to
a file.

| stage | what it measures | measured on M1 Max | budget on 8x A100 | bound by |
|---|---|---|---|---|
| `preflight` | the machine, and every decision made from it | 4 s | under 1 min | nothing |
| `tests` | 298 tests, including every equivalence proof | 2 min 9 s | 3 to 6 min | process startup for ~30 distributed spawns |
| `parallel` | equivalence, byte counts, bf16 wire cost, sharded clipping, activation memory | 1 min 47 s | 5 to 15 min | NCCL group setup per spawn, which is slower than gloo's |
| `collectives` | all-reduce, all-gather, reduce-scatter across sizes and world sizes 2, 4, 8 | seconds at 2 ranks | 5 to 15 min | the sweep itself; this is the stage the trip is for |
| `cluster` | resharding, async checkpoints, streaming, a killed rank, the fitted link | 28 s | 2 to 5 min | disk, not the GPU |
| `roofline` | GEMM and STREAM peaks, the operator table, MFU, a profiler trace | about 3 min | 5 to 10 min | the sweeps, plus a first CUDA context |
| `diagnose` | five injected pathologies, found or not | about 5 min | 5 to 15 min | five full diagnoses plus a four-arm collective probe |
| `verify` | our GPT-2 against HuggingFace, layer by layer | about 2 min | 2 to 5 min | downloading the checkpoint the first time |
| `ablate` | 9 architectures x 3 seeds | 18 min | 3 to 8 min | training, so this is the stage that gains most |
| `induction` | 144 heads scored, plus causal ablation | about 5 min | 1 to 3 min | 144 forward passes with attention weights |
| `kv` | KV-cache latency, throughput, memory | about 2 min | 1 to 3 min | generation, which is latency-bound |
| `quantize` | int8 and int4, per-tensor and per-channel | about 4 min | 1 to 3 min | evaluation passes |
| `prune` | structured head and neuron pruning | about 5 min | 2 to 4 min | evaluation passes |
| `distill` | GPT-2 into a 4-layer student, 2 seeds | 20 min | 3 to 8 min | training |
| `pareto` | every configuration on one quality and size frontier | not recorded | 1 to 3 min | evaluation passes |
| `figures` | all twelve figures redrawn from the committed JSONs | 13 s | under 1 min | matplotlib |
| `summary` | every result diffed against the committed baseline | 3 s | under 1 min | nothing |

Sum of the budgets: **40 to 100 minutes for the infrastructure half**, and
**15 to 35 more** for Parts 6 to 8.

---

## 5. What changes from modelled to measured

This is the list to check off. Everything here is currently labelled *modelled*
or *gloo* in the repository, and everything here becomes a measurement.

**Becomes measured:**

1. **Collective bandwidth and latency.** `results/collective_bandwidth.json`
   currently holds a gloo-over-loopback sweep. It becomes NCCL over NVLink:
   bus bandwidth per collective, per size, at world sizes 2, 4 and 8.
2. **The fabric model's link constants.** `cluster/fabric.py`'s `LINKS` are
   published datasheet peaks with an assumed 0.85 efficiency factor.
   `link_from_measurement` replaces the NVLink entry with a fitted one, and the
   assumed efficiency stops being assumed: the ratio of measured bus bandwidth
   to the datasheet peak *is* the efficiency, measured.
3. **Whether the ring model describes NCCL.** The `2(n-1)/n` factor and the
   affine `latency + bytes/bandwidth` form are checked against gloo today. On
   NCCL they get checked against the thing they were written for.
4. **Every equivalence proof, on NCCL.** The proofs are backend-independent by
   construction, and running them on NCCL is what turns "by construction" into
   "and also observed". The `summary` stage fails if any equivalence error or
   byte count moves, so this is a real check and not a formality.
5. **Mixed precision on hardware that has bf16 tensor cores.** The bf16 path is
   implemented and tested on CPU, where it is numerically identical and offers
   no speedup. On an A100 it is the same code on tensor cores.
6. **The roofline, against a GPU's roof.** The current ridge point is 21.07
   FLOP/byte, measured on an M1 Max. The A100's is 153 by datasheet arithmetic.
   The whole operator table gets reclassified against a measured GPU ridge, and
   the README's argument that `QK^T` and `attention x V` change sides stops
   being derived from a datasheet.
7. **MFU against a measured GPU peak.** Currently 47.71% of an M1 Max's measured
   6.489 TFLOP/s. The published-peak division in `mfu.json` is labelled
   modelled and stays labelled; what replaces it is a measured A100 GEMM peak
   and a measured step time.
8. **Activation memory from the CUDA allocator.** The saved-tensor accounting
   already works on any device and is exact in fp32. On CUDA the meter
   additionally reports `torch.cuda.max_memory_allocated`, which includes the
   transient workspaces the stash does not, and that is the number that decides
   an OOM. The bf16 figure is measured rather than computed, here and there,
   because autocast's cast policy is torch's and is not the same list on CPU and
   CUDA. Measuring it on an A100 is worth doing for exactly that reason: the
   0.70x measured on CPU is a CPU number.
9. **Comm and compute overlap, from a real profiler trace.** torch 2.2's
   profiler has no MPS backend, so the committed trace is CPU-only and the
   overlap claims in Part 3 are about scheduling rather than about a wire. On
   CUDA the trace has kernel and NCCL streams in it.

**Stays modelled, and must stay labelled:**

- Inter-node fabrics. One node has no InfiniBand traffic on it, so `ib_ndr`,
  `ib_hdr` and `roce200` stay datasheet numbers.
- The GPUDirect RDMA penalty, which is an inter-node effect.
- The 70B-class cost table, which is a model of a job nobody is running here.
- Anything about an H100 measured on an A100, and the reverse.

---

## 6. Things that go wrong on a first NCCL run

Collected here because each of them costs twenty minutes to diagnose from
scratch and one minute to recognise.

**The first collective hangs and nothing is printed.** Almost always two ranks
on the same GPU, which happens when `torch.cuda.set_device` is not called before
`init_process_group`. This repository calls it in the right order in
`parallel/comms.py` and says why, and `gpu_preflight.py` prints the placement it
will use, so check that first. Second most likely: `NCCL_SOCKET_IFNAME` pointing
at an interface that does not exist. `export NCCL_DEBUG=INFO` and read the
first twenty lines.

**`is_nccl_available() == False`.** The installed torch is a CPU or ROCm wheel.
`select_backend` refuses rather than silently using gloo, precisely so this is
found in the preflight rather than in the results.

**`CUDA_VISIBLE_DEVICES` set to an empty string** hides every GPU and produces
"zero visible CUDA devices" with CUDA apparently available. The preflight prints
the variable when it is set.

**bf16 refused.** The node is pre-Ampere. Pass `amp_dtype="fp16"`, which routes
through a real `GradScaler`; the message says so.

**Out of memory in the parallel stage.** Run
`python scripts/gpu_preflight.py --dry-run` and read the activation table. It is
the fp32 count, which is exact against measurement, and it is deliberately the
fp32 one: bf16 autocast can only shrink the activation stash, so fp32 is the
safe side of "will it fit". It does not shrink it by half. Autocast keeps
LayerNorm, its saved statistics and the cross-entropy log-softmax in fp32, and
measured on the CPU test model the bf16 stash is 0.70x of the fp32 one, not
0.50x. Autocast also adds a weight cache of `2N` bytes that the activation
column does not carry.

**A stage dies and takes the run with it.** It does not: markers survive.
`./scripts/run_on_gpu.sh --resume`.

---

## 7. After the run

```bash
# what changed, sorted into correctness invariants, timings and everything else
python scripts/compare_results.py --baseline-git HEAD --current results
```

Then, in order:

1. **Check that no correctness invariant broke.** `compare_results.py` exits
   non-zero if one did. A byte count, a collective count or a formula check
   that differs at all between gloo and NCCL is a bug, and finding it is worth
   the trip on its own. Equivalence errors are floats and *will* move, because
   NCCL does not reduce in gloo's order; what the gate checks there is that
   they stay under the 1e-5 the tests assert, and it prints the ratio so a move
   inside the tolerance is still visible.

   Expect a block of `NEW KEYS` too, and it is not a problem.
   `results/roofline.json` and `results/cluster.json` were written before those
   scripts started recording an `environment` block, so the first run adds one
   to each. That is provenance, not a measurement.
2. **Replace the budget column in section 4** with the measured one from
   `results/.run_state/timings.tsv`.
3. **Update the README's Limitations.** "No GPU cluster" and "Mixed precision is
   implemented and tested, but only on CPU" both need rewriting, and the parts
   that are still true (inter-node fabrics, no fused kernel) need to stay.
4. **Relabel what is no longer modelled**, and only what is no longer modelled.
   The measured NVLink number replaces the datasheet NVLink number. It does not
   license any claim about InfiniBand.
5. **Keep the CPU results too.** A gloo-versus-NCCL comparison of the same
   equivalence proofs is worth more than either alone, and `results/` before
   this run is in git history.
