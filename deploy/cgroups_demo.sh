#!/usr/bin/env bash
# Run the cgroup reader inside a container with a real memory limit, then prove
# the OOM-kill path by asking for more memory than the limit allows.
#
# Everything in deploy/cgroups_demo_output.txt is the captured output of this
# script on the machine this repository was built on.
set -u
cd "$(dirname "$0")/.."

LIMIT=${LIMIT:-512m}
CPUS=${CPUS:-1.5}
IMAGE=${IMAGE:-python:3.11-slim}

echo "### docker run --memory=$LIMIT --memory-swap=$LIMIT --cpus=$CPUS"
echo "### (memory-swap == memory means swap is disabled for the cgroup)"
docker run --rm --memory="$LIMIT" --memory-swap="$LIMIT" --cpus="$CPUS" \
    -v "$PWD":/w -w /w "$IMAGE" \
    python src/transformer_internals/cluster/cgroups.py

echo
echo "### Now allocate past memory.max inside the same limits."
echo "### Expect: no Python traceback, exit code 137 (128 + SIGKILL)."
docker run --rm --memory="$LIMIT" --memory-swap="$LIMIT" --cpus="$CPUS" \
    -v "$PWD":/w -w /w "$IMAGE" \
    python -c "
import sys
chunks = []
for i in range(64):
    chunks.append(bytearray(64 * 1024 * 1024))   # 64 MiB at a time
    print(f'allocated {(i+1)*64} MiB', flush=True)
"
echo "### exit code: $?"

echo
echo "### Does swap change the failure mode? First, is there a swap device at all"
echo "### in the kernel this container runs on?"
docker run --rm --memory="$LIMIT" --memory-swap=1g --cpus="$CPUS" "$IMAGE" \
    sh -c 'echo -n "cgroup memory.swap.max: "; cat /sys/fs/cgroup/memory.swap.max; grep SwapTotal /proc/meminfo'

echo
echo "### Same overshoot, cgroup now allowed 512m swap on top of 512m memory."
docker run --rm --memory="$LIMIT" --memory-swap=1g --cpus="$CPUS" \
    -v "$PWD":/w -w /w "$IMAGE" \
    python -c "
chunks = []
for i in range(24):
    chunks.append(bytearray(64 * 1024 * 1024))
    for j in range(0, len(chunks[-1]), 4096):
        chunks[-1][j] = 1
    print(f'touched {(i+1)*64} MiB', flush=True)
print('survived')
"
echo "### exit code: $?"
echo "### Read that against SwapTotal above: raising the cgroup's swap allowance"
echo "### does nothing if the kernel has no swap device to page to. The limit is"
echo "### permission, not capacity."
