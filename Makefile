PY ?= .venv/bin/python
export PYTHONPATH := src
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: help install test test-fast lint fmt verify ablate induction kv quantize prune distill pareto figures all clean

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

verify:  ## PART 2 -- prove the implementation equals HuggingFace GPT-2 (~2 min, CPU)
	$(PY) scripts/run_verification.py

ablate:  ## PART 3 -- 9 configurations x 3 seeds (18 min measured)
	$(PY) scripts/run_ablations.py

induction:  ## PART 3 -- find induction heads in GPT-2 small (~5 min, CPU)
	$(PY) scripts/run_induction.py

kv:  ## PART 4 -- KV cache latency, throughput and memory (~2 min)
	$(PY) scripts/run_kv_cache.py

quantize:  ## PART 4 -- int8/int4, per-tensor vs per-channel (~4 min, CPU)
	$(PY) scripts/run_quantization.py

prune:  ## PART 4 -- structured head and neuron pruning (~5 min, CPU)
	$(PY) scripts/run_pruning.py

distill:  ## PART 4 -- distil GPT-2 into a 4-layer student (20 min measured, CPU)
	$(PY) scripts/run_distillation.py

pareto:  ## PART 4 -- collect every configuration into one quality/size frontier
	$(PY) scripts/make_pareto.py

figures:  ## redraw every figure from the committed results
	$(PY) scripts/make_figures.py

all: verify induction ablate kv quantize prune distill pareto figures  ## the whole pipeline

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
