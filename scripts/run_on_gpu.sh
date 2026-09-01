#!/usr/bin/env bash
#
# One command for a rented GPU box: check it, install into it, measure it, and
# say what changed.
#
# The design constraint is that a paid GPU hour should be spent measuring rather
# than debugging, so:
#
#   * every stage writes a marker when it succeeds and the run skips completed
#     stages on a re-invocation, which makes it resumable after a crash, an
#     out-of-memory kill or a dropped SSH session;
#   * every stage is timed, and the timings accumulate in a TSV so the second
#     run can be planned from the first;
#   * every stage's stdout and stderr go to their own log, so a failure at stage
#     nine does not mean scrolling past eight stages of output;
#   * --smoke runs the whole pipeline at sizes that finish in a few minutes,
#     writing to a scratch directory, so the shape of the run is proved before
#     the real sweep starts;
#   * the last stage diffs every result against the committed baseline and
#     fails if a correctness invariant moved.
#
# Usage:
#   scripts/run_on_gpu.sh --smoke              prove the pipeline, scratch output
#   scripts/run_on_gpu.sh --smoke --infra-only the same, infrastructure stages only
#   scripts/run_on_gpu.sh                      the real sweep, writes results/
#   scripts/run_on_gpu.sh --no-resume          re-run every stage, ignoring markers
#   scripts/run_on_gpu.sh --stages parallel,collectives
#   scripts/run_on_gpu.sh --list               show the stages and exit
#
# Resuming is the DEFAULT: a stage with a marker under results/.run_state is
# skipped. That is what makes a crashed run cheap to continue, and it also means
# a re-run after a code change will report stale stages as done. Use --no-resume
# for that, or delete the one marker you want redone.
#
# The stages that need the published GPT-2 checkpoint are marked optional: a
# failure in one is reported and the run continues, so a box with no network
# still produces every training-infrastructure measurement. --infra-only drops
# them entirely, which is the fastest way to prove the pipeline.
#
set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- defaults

SMOKE=0
RESUME=1
DO_INSTALL=1
DO_TESTS=1
KEEP_GOING=0
LIST_ONLY=0
NEEDS_WEIGHTS=1
INFRA_ONLY=0
ONLY_STAGES=""
WORLD_SIZES="2,4,8"
DEVICE="auto"
PYTHON_BIN="${PYTHON:-}"
RESULTS_DIR="results"
ASSETS_DIR="assets"
BASELINE_REF="HEAD"

usage() {
    # Every comment line of the header, so --help cannot end mid-sentence.
    sed -n '2,/^set -o errexit/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)          SMOKE=1 ;;
        --resume)         RESUME=1 ;;
        --no-resume)      RESUME=0 ;;
        --fresh)          RESUME=0 ;;
        --skip-install)   DO_INSTALL=0 ;;
        --skip-tests)     DO_TESTS=0 ;;
        --no-weights)     NEEDS_WEIGHTS=0 ;;
        --infra-only)     INFRA_ONLY=1; NEEDS_WEIGHTS=0 ;;
        --keep-going)     KEEP_GOING=1 ;;
        --list)           LIST_ONLY=1 ;;
        --stages)         ONLY_STAGES="$2"; shift ;;
        --world-sizes)    WORLD_SIZES="$2"; shift ;;
        --device)         DEVICE="$2"; shift ;;
        --python)         PYTHON_BIN="$2"; shift ;;
        --results)        RESULTS_DIR="$2"; shift ;;
        --assets)         ASSETS_DIR="$2"; shift ;;
        --baseline)       BASELINE_REF="$2"; shift ;;
        -h|--help)        usage ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

if [[ $SMOKE -eq 1 ]]; then
    # A smoke run must not touch the committed measurements. Everything it
    # writes goes to a scratch tree, which is also what lets it be run on the
    # development laptop as a pipeline check.
    RESULTS_DIR="${RESULTS_DIR%/}"
    [[ "$RESULTS_DIR" == "results" ]] && RESULTS_DIR="smoke/results"
    [[ "$ASSETS_DIR" == "assets" ]] && ASSETS_DIR="smoke/assets"
fi

STATE_DIR="${RESULTS_DIR}/.run_state"
LOG_DIR="${STATE_DIR}/logs"
TIMINGS="${STATE_DIR}/timings.tsv"

# ------------------------------------------------------------- interpreter

pick_python() {
    if [[ -n "$PYTHON_BIN" ]]; then echo "$PYTHON_BIN"; return; fi
    if [[ -x .venv/bin/python ]]; then echo ".venv/bin/python"; return; fi
    if command -v python3 >/dev/null 2>&1; then echo "python3"; return; fi
    echo "python"
}
PY="$(pick_python)"

# --------------------------------------------------------------- reporting

BOLD=""; DIM=""; RED=""; GREEN=""; RESET=""
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; RESET=$'\033[0m'
fi

banner() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"; printf '%s\n' "$(printf '=%.0s' $(seq 1 ${#1}))"; }
say()    { printf '%s\n' "$1"; }
warn()   { printf '%s%s%s\n' "$RED" "$1" "$RESET" >&2; }

hhmmss() {
    local total=${1%.*}
    printf '%dh%02dm%02ds' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

# ------------------------------------------------------------------ stages
#
# Each stage is "name|optional|command". "optional" means a failure is recorded
# and the run continues: the weight-dependent stages need the published GPT-2
# checkpoint and a network, and a box without either should still produce every
# infrastructure measurement.

R="$RESULTS_DIR"
A="$ASSETS_DIR"

if [[ $SMOKE -eq 1 ]]; then
    STAGES=(
        "preflight|no|$PY scripts/gpu_preflight.py --world-sizes 2 --out $R/gpu_preflight.json"
        "tests|no|$PY -m pytest -q -m 'not weights and not slow' -x"
        "parallel|no|$PY scripts/run_parallel.py --world-size 2 --pipeline-stages 2 --steps 1 --batch 4 --seq 8 --bubble-batch 4 --bubble-seq 32 --bubble-repeats 1 --out $R/parallel_comms.json --assets $A"
        "collectives|no|$PY scripts/run_collectives.py --world-sizes 2 --iters 4 --min-log2-elements 12 --max-log2-elements 18 --out $R/collective_bandwidth.json"
        "cluster|no|$PY scripts/run_cluster.py --skip-reshard --out $R/cluster.json"
        "roofline|no|$PY scripts/run_roofline.py --device $DEVICE --gemm-sizes 256 512 --stream-sizes-mib 16 --batch 2 --seq 64 --mfu-batch 1 --mfu-seq 128 --mfu-steps 2 --profile-batch 1 --profile-seq 64 --out $R/roofline.json --assets $A"
        "diagnose|no|$PY scripts/diagnose_run.py --batch 2 --seq 32 --steps 2 --world-size 2 --report-only --out $R/diagnosis.json --assets $A"
        "verify|yes|$PY scripts/run_verification.py --max-new-tokens 20 --no-perplexity --out $R/verification.json"
        "ablate|yes|$PY scripts/run_ablations.py --device $DEVICE --seeds 0 --steps 5 --arms baseline --out $R/ablations.json"
        "induction|yes|$PY scripts/run_induction.py --seq-len 24 --batch-size 2 --no-ablation --out $R/induction.json"
        "kv|yes|$PY scripts/run_kv_cache.py --device $DEVICE --prompt-lens 16 64 --new-tokens 4 --repeats 1 --out $R/kv_cache.json"
        "quantize|yes|$PY scripts/run_quantization.py --chunks 1 --batches-per-chunk 1 --batch-size 2 --out $R/quantization.json"
        "prune|yes|$PY scripts/run_pruning.py --out $R/pruning.json --induction $R/induction.json"
        "distill|yes|$PY scripts/run_distillation.py --device $DEVICE --steps 5 --seeds 0 --batch-size 2 --block-size 32 --out $R/distillation.json"
        "pareto|yes|$PY scripts/make_pareto.py --eval-batches 2 --batch-size 2 --out $R/pareto.json"
        "figures|no|$PY scripts/make_figures.py --results $R --assets $A"
        "summary|no|$PY scripts/compare_results.py --baseline-git $BASELINE_REF --current $R --allow-changes"
    )
else
    STAGES=(
        "preflight|no|$PY scripts/gpu_preflight.py --world-sizes $WORLD_SIZES --out $R/gpu_preflight.json"
        "tests|no|$PY -m pytest -q -m 'not weights' --durations=10"
        "parallel|no|$PY scripts/run_parallel.py --out $R/parallel_comms.json --assets $A"
        "collectives|no|$PY scripts/run_collectives.py --world-sizes $WORLD_SIZES --out $R/collective_bandwidth.json"
        "cluster|no|$PY scripts/run_cluster.py --out $R/cluster.json"
        "roofline|no|$PY scripts/run_roofline.py --device $DEVICE --out $R/roofline.json --assets $A"
        "diagnose|no|$PY scripts/diagnose_run.py --reuse-peak --out $R/diagnosis.json --assets $A"
        "verify|yes|$PY scripts/run_verification.py --out $R/verification.json"
        "ablate|yes|$PY scripts/run_ablations.py --device $DEVICE --out $R/ablations.json"
        "induction|yes|$PY scripts/run_induction.py --out $R/induction.json"
        "kv|yes|$PY scripts/run_kv_cache.py --device $DEVICE --out $R/kv_cache.json"
        "quantize|yes|$PY scripts/run_quantization.py --out $R/quantization.json"
        "prune|yes|$PY scripts/run_pruning.py --out $R/pruning.json --induction $R/induction.json"
        "distill|yes|$PY scripts/run_distillation.py --device $DEVICE --out $R/distillation.json"
        "pareto|yes|$PY scripts/make_pareto.py --out $R/pareto.json"
        "figures|no|$PY scripts/make_figures.py --results $R --assets $A"
        "summary|no|$PY scripts/compare_results.py --baseline-git $BASELINE_REF --current $R"
    )
fi

if [[ $INFRA_ONLY -eq 1 ]]; then
    # Keep only the stages that need no published checkpoint. Those are exactly
    # the training-infrastructure half, which is what a GPU session is for.
    KEPT=()
    for entry in "${STAGES[@]}"; do
        IFS='|' read -r _name optional _cmd <<< "$entry"
        [[ "$optional" == "no" ]] && KEPT+=("$entry")
    done
    STAGES=("${KEPT[@]}")
fi

if [[ $LIST_ONLY -eq 1 ]]; then
    banner "stages"
    for entry in "${STAGES[@]}"; do
        IFS='|' read -r name optional _cmd <<< "$entry"
        printf '  %-14s %s\n' "$name" "$([[ $optional == yes ]] && echo '(optional: needs the GPT-2 checkpoint)' || echo '')"
    done
    exit 0
fi

wanted() {
    [[ -z "$ONLY_STAGES" ]] && return 0
    [[ ",$ONLY_STAGES," == *",$1,"* ]]
}

# ------------------------------------------------------------------ set up

mkdir -p "$R" "$A" "$STATE_DIR" "$LOG_DIR"
[[ -f "$TIMINGS" ]] || printf 'stage\tstatus\tseconds\tfinished_at\n' > "$TIMINGS"

banner "gpt2-harness measurement run"
say "repository   $REPO_ROOT"
say "python       $PY"
say "mode         $([[ $SMOKE -eq 1 ]] && echo 'SMOKE (tiny sizes, scratch output)' || echo 'FULL')$([[ $INFRA_ONLY -eq 1 ]] && echo ', infrastructure stages only' || echo '')"
say "results      $R"
say "assets       $A"
say "state        $STATE_DIR"
say "resume       $([[ $RESUME -eq 1 ]] && echo 'on: completed stages are skipped' || echo 'off: every stage re-runs')"
say "started      $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# ---------------------------------------------------------------- install

if [[ $DO_INSTALL -eq 1 ]] && wanted install; then
    marker="${STATE_DIR}/install.done"
    if [[ $RESUME -eq 1 && -f "$marker" ]]; then
        say "install      already done (delete $marker to redo)"
    else
        banner "installing"
        # torch is deliberately NOT installed here. On a rented GPU box the
        # image already carries a torch built against that box's CUDA, and
        # reinstalling from PyPI is the fastest way to end up with a CPU wheel
        # and a confusing NCCL error an hour later.
        if ! "$PY" -c "import torch" >/dev/null 2>&1; then
            warn "torch is not importable with $PY."
            warn "Install the CUDA build that matches this box first, for example:"
            warn "  pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.2,<2.6'"
            exit 1
        fi
        say "torch       $("$PY" -c 'import torch; print(torch.__version__)')"
        "$PY" -m pip install --quiet --upgrade pip
        install_extras="dev"
        [[ $NEEDS_WEIGHTS -eq 1 ]] && install_extras="dev,verify"
        "$PY" -m pip install --quiet -e ".[${install_extras}]"
        say "installed the package with extras: ${install_extras}"
        touch "$marker"
    fi
fi

# ------------------------------------------------------------- the stages

run_stage() {
    local name="$1" optional="$2" cmd="$3"
    local marker="${STATE_DIR}/${name}.done"
    local log="${LOG_DIR}/${name}.log"

    if ! wanted "$name"; then return 0; fi
    if [[ "$name" == "tests" && $DO_TESTS -eq 0 ]]; then
        say "  ${DIM}skip${RESET}   $name (--skip-tests)"
        return 0
    fi
    if [[ $RESUME -eq 1 && -f "$marker" ]]; then
        local recorded
        recorded="$(cat "$marker" 2>/dev/null || echo '?')"
        say "  ${DIM}skip${RESET}   ${name} (done in ${recorded}, delete ${marker} to redo)"
        return 0
    fi

    printf '  %-6s %s\n' "run" "$name"
    printf '         %s%s%s\n' "$DIM" "$cmd" "$RESET"
    local start end elapsed status
    start=$(date +%s)
    set +o errexit
    ( eval "$cmd" ) > "$log" 2>&1
    status=$?
    set -o errexit
    end=$(date +%s)
    elapsed=$((end - start))

    if [[ $status -eq 0 ]]; then
        printf '%s\n' "$(hhmmss "$elapsed")" > "$marker"
        printf '%s\tok\t%d\t%s\n' "$name" "$elapsed" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$TIMINGS"
        printf '  %s%-6s%s %s in %s\n' "$GREEN" "done" "$RESET" "$name" "$(hhmmss "$elapsed")"
        return 0
    fi

    printf '%s\tfailed\t%d\t%s\n' "$name" "$elapsed" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$TIMINGS"
    warn "  FAILED $name after $(hhmmss "$elapsed") (exit $status). Last 25 lines of $log:"
    tail -n 25 "$log" | sed 's/^/    /' >&2
    if [[ "$optional" == "yes" ]]; then
        warn "  $name is optional (it needs the published GPT-2 checkpoint); continuing."
        return 0
    fi
    if [[ $KEEP_GOING -eq 1 ]]; then
        warn "  --keep-going is set; continuing."
        FAILURES+=("$name")
        return 0
    fi
    warn ""
    warn "  Stopping. Every stage that already succeeded is checkpointed, so"
    warn "  re-running this script picks up from here:"
    # Not ${SMOKE:+...}: that expands on the string "0" because "0" is non-null,
    # so a FULL run that died at hour two would print advice that resumes into
    # smoke mode, write to smoke/ and report "Smoke run complete."
    warn "      scripts/run_on_gpu.sh$([[ $SMOKE -eq 1 ]] && echo ' --smoke')$([[ $INFRA_ONLY -eq 1 ]] && echo ' --infra-only') --resume"
    exit "$status"
}

FAILURES=()
banner "stages"
RUN_START=$(date +%s)
for entry in "${STAGES[@]}"; do
    IFS='|' read -r name optional cmd <<< "$entry"
    run_stage "$name" "$optional" "$cmd"
done
RUN_END=$(date +%s)

# ------------------------------------------------------------------ report

banner "summary"
printf '%-14s %-8s %10s\n' "stage" "status" "seconds"
tail -n +2 "$TIMINGS" | awk -F'\t' '{ last[$1]=$0 } END { for (k in last) print last[k] }' \
    | sort | while IFS=$'\t' read -r name status seconds _finished; do
        printf '%-14s %-8s %10s\n' "$name" "$status" "$seconds"
    done
say ""
say "total wall clock: $(hhmmss $((RUN_END - RUN_START)))"
say "logs:             $LOG_DIR"
say "results:          $R"
say "timings:          $TIMINGS"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    warn ""
    warn "failed stages: ${FAILURES[*]}"
    exit 1
fi

say ""
if [[ $SMOKE -eq 1 ]]; then
    say "${GREEN}Smoke run complete.${RESET} The pipeline ran end to end at tiny sizes and"
    say "wrote to ${R}, leaving the committed measurements untouched."
    say "The real sweep is:  scripts/run_on_gpu.sh"
else
    say "${GREEN}Run complete.${RESET} What changed against the committed baseline is in the"
    say "'summary' stage log: ${LOG_DIR}/summary.log"
fi
