PY ?= .venv/bin/python
export PYTHONPATH := src
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: help install test test-fast lint fmt verify ablate induction kv quantize prune distill pareto parallel roofline diagnose cluster figures all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create the venv and install the package with dev + verify extras
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev,verify]"

test:  ## full suite, including the weight-dependent verification tests
	$(PY) -m pytest -q

test-fast:  ## what CI runs: everything that needs neither weights nor network
	$(PY) -m pytest -q -m "not weights"

lint:  ## ruff check
	$(PY) -m ruff check .

fmt:  ## ruff autofix + format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

verify:  ## PART 5 -- prove the implementation equals HuggingFace GPT-2 (~2 min, CPU)
	$(PY) scripts/run_verification.py

ablate:  ## PART 6 -- 9 configurations x 3 seeds (18 min measured)
	$(PY) scripts/run_ablations.py

induction:  ## PART 6 -- find induction heads in GPT-2 small (~5 min, CPU)
	$(PY) scripts/run_induction.py

kv:  ## PART 7 -- KV cache latency, throughput and memory (~2 min)
	$(PY) scripts/run_kv_cache.py

quantize:  ## PART 7 -- int8/int4, per-tensor vs per-channel (~4 min, CPU)
	$(PY) scripts/run_quantization.py

prune:  ## PART 7 -- structured head and neuron pruning (~5 min, CPU)
	$(PY) scripts/run_pruning.py

distill:  ## PART 7 -- distil GPT-2 into a 4-layer student (20 min measured, CPU)
	$(PY) scripts/run_distillation.py

pareto:  ## PART 7 -- collect every configuration into one quality/size frontier
	$(PY) scripts/make_pareto.py

parallel:  ## PART 1 -- every parallelism strategy, proven against a single-process reference (49 s measured, CPU)
	$(PY) scripts/run_parallel.py

roofline:  ## PART 2 -- measure the machine's roofline, place every op on it, report MFU (~3 min)
	$(PY) scripts/run_roofline.py --device mps

diagnose:  ## PART 2 -- inject four throughput pathologies and show the tool find each one (~5 min)
	$(PY) scripts/diagnose_run.py --reuse-peak

cluster:  ## PART 3 -- reshard a checkpoint, kill a rank, restart, fit the collective model (28 s measured, CPU)
	$(PY) scripts/run_cluster.py

figures:  ## redraw every figure from the committed results
	$(PY) scripts/make_figures.py

all: parallel roofline diagnose cluster verify ablate induction kv quantize prune distill pareto figures  ## the whole pipeline

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
