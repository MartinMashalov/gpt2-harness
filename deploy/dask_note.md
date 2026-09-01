# Where Dask fits, and where it does not

Dask does not belong in the training loop. It has no place in a job that runs
collectives: its scheduler is dynamic and work-stealing, which is the opposite
of what a synchronous all-reduce needs, and there is no equivalent of a
communicator to break when a worker dies. Training wants a fixed set of ranks
that all take the same number of steps. Dask wants a task graph it can rearrange.

It fits in the step before, which is usually the larger engineering problem:

**Corpus preparation.** Tokenising a few terabytes of text is embarrassingly
parallel, IO-bound, and produces output that has to be sharded deterministically
so the training job's dataloader can address it. `dask.bag` over the raw files,
map the tokenizer, and write fixed-size shards. The reason to reach for Dask
rather than a Slurm array job is the shuffle: deduplication and mixing across
sources are joins, and `dask.dataframe` does joins that do not fit in memory
without you writing the spill logic.

**Deduplication.** MinHash/LSH over documents is a groupby on band hashes.
`dask.dataframe.groupby` on a column of hashes, over parquet, with a partition
count you choose from the data size. This is the workload Dask is actually good
at and it is genuinely painful to write by hand.

**Global shuffle before sharding.** A training corpus that is ordered by source
gives every rank a correlated stream, and the loss curve shows it as an
oscillation with a period of one source. Shuffling globally once, offline, is
much cheaper than trying to shuffle at read time inside the dataloader.
`dask.dataframe.shuffle` with `shuffle="disk"` or `"p2p"` does the out-of-core
version.

**Statistics and filtering.** Token-length histograms, language ID counts,
quality-score cutoffs: aggregations over the whole corpus that decide what goes
into the run.

The handoff to training is the file layout. Dask writes N shards; the streaming
dataloader in `src/transformer_internals/cluster/streaming.py` reads them with a
deterministic assignment from `(seed, epoch)` and a per-rank position that goes
into the checkpoint. Nothing about the training job depends on Dask having been
the thing that wrote the files, which is the property to preserve: the data
pipeline and the training harness should share a file format and nothing else.

**Why not Ray for this too?** You can, and if the cluster already runs Ray it is
one system instead of two. Dask wins when the work is dataframe-shaped, because
`dask.dataframe` implements the out-of-core relational operators and Ray Data's
equivalents are younger. Ray wins when the per-record work is a Python function
holding a model (a quality classifier, an embedding model), because Ray actors
keep that model resident and Dask's task model reloads it.
