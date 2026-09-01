#!/bin/bash
# Multi-node training under Slurm.
#
# The layout this asks for is one task per GPU, one GPU per task, and the CPUs
# and memory that go with it. Getting that shape right is most of what makes a
# job run at the speed the hardware allows: Slurm's cgroup plugin binds each
# task to the cores and memory it was allocated, and if the binding does not
# follow the PCIe topology the GPU spends the run waiting on a NUMA-remote
# dataloader.
#
#   sbatch deploy/slurm_train.sbatch
#
#SBATCH --job-name=gpt-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8          # one rank per GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12           # 96-core node / 8 ranks, leaves headroom
#SBATCH --mem=0                      # all of the node's memory; do not let the
                                     # cgroup memory.max sit below what the
                                     # dataloader's page cache needs
#SBATCH --exclusive                  # no other job's cgroup sharing our NICs
#SBATCH --gres-flags=enforce-binding # bind CPUs to the GPU's NUMA node
#SBATCH --time=12:00:00
#SBATCH --requeue                    # eligible for automatic requeue
#SBATCH --signal=B:USR1@180          # SIGUSR1 to the batch script 180s before
                                     # the wall clock runs out, so we can
                                     # checkpoint and requeue ourselves
#SBATCH --open-mode=append           # a requeued job appends, it does not erase
                                     # the log of the attempt that died
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- rendezvous
# The first node in the allocation is the rendezvous host. Read it from Slurm
# rather than hardcoding: after a requeue the allocation is usually different.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# ----------------------------------------------------------------- NCCL / IB
# NCCL_DEBUG=INFO prints the rings and the transport it chose. Read it once per
# cluster and once per topology change: if it says "via NET/Socket" instead of
# "via NET/IB", the job is running over TCP and will be at a fraction of the
# throughput you expected, with nothing else to indicate it.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH

# Which HCAs to use, and which interface for the bootstrap. Both are cluster
# specific. Naming them explicitly stops NCCL picking the management NIC or a
# docker0 bridge, which is the single most common cause of a job that runs at
# 1/20th of the expected all-reduce bandwidth.
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

# GPUDirect RDMA: let the NIC DMA straight into GPU memory instead of bouncing
# through the host. Level 2 means "use it when the GPU and the NIC are under the
# same PCIe switch", which is where it is a win; across a CPU root complex it
# can be slower than staging through host memory.
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_IB_PCI_RELAXED_ORDERING=1

# On a RoCE fabric instead of InfiniBand, the GID index selects the RoCEv2
# entry. Getting it wrong gives a fabric that connects and then runs at
# Ethernet-with-drops speed.
#   export NCCL_IB_GID_INDEX=3
#   export NCCL_IB_TC=106            # traffic class for the lossless queue

# Turn a hung collective into a failed rank. Without this, one dead peer parks
# every other rank in a spin-wait until the job hits its wall clock, burning the
# whole allocation on a job that is already dead.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=1800

# One rank per GPU means one process per GPU: do not also let each of them
# spawn a thread per core.
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))

# ------------------------------------------------------- requeue on SIGUSR1
# The trap runs in the batch script. It tells the ranks to checkpoint (they
# handle SIGUSR1 by saving at the next step boundary), waits for the write, and
# then hands the job back to the scheduler. A requeued job restarts from the
# checkpoint rather than from step 0.
checkpoint_and_requeue() {
    echo "$(date -Is) SIGUSR1: wall clock nearly up, checkpointing and requeueing"
    scancel --signal=USR1 --batch "$SLURM_JOB_ID" || true
    sleep 120
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap checkpoint_and_requeue USR1

echo "job $SLURM_JOB_ID  attempt $SLURM_RESTART_COUNT  nodes $SLURM_JOB_NODELIST"
echo "master $MASTER_ADDR:$MASTER_PORT  world $WORLD_SIZE"

# --------------------------------------------------------------------- launch
# srun starts one torchrun per node; torchrun starts one rank per GPU under it
# and owns the local restart. Two layers, because they fail at different
# granularities: torchrun can restart a rank in place after a transient CUDA
# error, and only Slurm can replace a node that has actually died.
srun --kill-on-bad-exit=1 --wait=60 \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc-per-node="$SLURM_NTASKS_PER_NODE" \
        --node-rank="$SLURM_NODEID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --rdzv-id="$SLURM_JOB_ID" \
        --max-restarts=3 \
        -m transformer_internals.cluster.failure \
        &
wait
