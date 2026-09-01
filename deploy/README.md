# deploy/

Launchers for the same training job under four different schedulers, plus the
cgroups demonstration.

| file | what it is |
|---|---|
| `slurm_train.sbatch` | Slurm batch script: 8 nodes x 8 GPUs, NCCL/InfiniBand environment, SIGUSR1 checkpoint-and-requeue, `srun` + `torchrun` two-layer launch |
| `ray_train.py` | The same job under Ray Train, with the placement group that keeps a tensor-parallel group inside one node |
| `k8s/job-indexed.yaml` | Indexed Job + headless Service: shared-memory sizing, Guaranteed QoS, RDMA device request, topology-aware affinity |
| `k8s/statefulset.yaml` | StatefulSet variant, for when each rank needs a stable identity and its own checkpoint volume |
| `dask_note.md` | Where Dask belongs (corpus preparation) and why it does not belong in the training loop |
| `cgroups_demo.sh` | Runs the cgroup v2 reader in Docker under a memory limit, then triggers a real OOM kill |
| `cgroups_demo_output.txt` | Captured output of the above, from the machine this repository was built on |

None of these were run against a real cluster: there is no cluster here. They
are written to be read by someone who runs one. The cgroups demo *was* run, in
Docker, and its output is the real thing.
