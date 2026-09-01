"""The results diff, which is what the GPU run's last stage exits on.

This is the gate: after a run on new hardware, `compare_results.py` decides
whether anything that cannot legitimately depend on the machine changed. Getting
that decision wrong in either direction wastes the trip. Too strict and every
cross-backend run fails for a reason that is float non-associativity; too loose
and a real sharding bug arrives home labelled "expected".

So the two directions are tested against each other: a perturbation of every
equivalence error by a factor of three must pass, and a single error at 0.5 must
fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_results import (
    EQUIVALENCE_TOLERANCE,
    compare,
    flatten,
    load_tree,
)

BASE = {
    "parallel_comms.json": {
        "equivalence": {
            "data_parallel": {"max_grad_error": 7.8e-08},
            "zero_2": {"max_param_error": 1.44e-06},
        },
        "comms": {"data_parallel": {"per_collective": {"all_reduce": {"calls": 3}}}},
        "formula_checks": [
            {"quantity": "data_parallel.all_reduce", "exact_match": True,
             "formula_bytes_per_step": 228096},
        ],
        "meta": {"runtime_seconds": 104.0},
    }
}


def _write(tmp_path: Path, payload: dict, name: str = "current") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, body in payload.items():
        (directory / filename).write_text(json.dumps(body), encoding="utf-8")
    return directory


def _bucket(tmp_path: Path, mutate) -> dict:
    import copy

    after = copy.deepcopy(BASE)
    mutate(after)
    baseline = load_tree(_write(tmp_path, BASE, "baseline"))
    current = load_tree(_write(tmp_path, after, "current"))
    return compare(baseline, current, threshold=0.01)


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #


def test_an_unchanged_run_reports_every_invariant_as_held(tmp_path: Path):
    buckets = _bucket(tmp_path, lambda _d: None)
    assert not buckets["invariant_moved"]
    assert not buckets["within_tolerance"]
    # Held invariants are reported even though nothing moved: "it did not move"
    # is the result the GPU run is looking for.
    assert len(buckets["invariant_held"]) >= 4


def test_equivalence_errors_may_move_between_backends_and_that_is_not_a_failure(
    tmp_path: Path,
):
    """NCCL does not reduce in gloo's order, and float addition is not associative.

    Demanding bit-identical floats would fail every cross-backend run for a
    reason that has nothing to do with correctness, which is exactly what this
    gate must not do.
    """

    def triple(d):
        eq = d["parallel_comms.json"]["equivalence"]
        eq["data_parallel"]["max_grad_error"] *= 3
        eq["zero_2"]["max_param_error"] *= 3

    buckets = _bucket(tmp_path, triple)
    assert not buckets["invariant_moved"]
    assert len(buckets["within_tolerance"]) == 2


def test_an_equivalence_error_leaving_the_tolerance_is_a_failure(tmp_path: Path):
    """The criterion is the one the tests assert and the result file records."""

    def brk(d):
        d["parallel_comms.json"]["equivalence"]["data_parallel"]["max_grad_error"] = 0.5

    buckets = _bucket(tmp_path, brk)
    assert len(buckets["invariant_moved"]) == 1
    path, _before, after = buckets["invariant_moved"][0]
    assert path.endswith("max_grad_error")
    assert after > EQUIVALENCE_TOLERANCE


def test_the_boundary_is_the_stated_tolerance_and_not_a_ratio(tmp_path: Path):
    """Just under passes, just over fails, whatever the ratio to the baseline is."""

    def just_under(d):
        d["parallel_comms.json"]["equivalence"]["zero_2"]["max_param_error"] = (
            EQUIVALENCE_TOLERANCE * 0.99
        )

    def just_over(d):
        d["parallel_comms.json"]["equivalence"]["zero_2"]["max_param_error"] = (
            EQUIVALENCE_TOLERANCE * 1.01
        )

    assert not _bucket(tmp_path, just_under)["invariant_moved"]
    assert len(_bucket(tmp_path, just_over)["invariant_moved"]) == 1


def test_a_byte_count_or_a_collective_count_may_not_move_at_all(tmp_path: Path):
    """Integer properties of the algorithm. One extra byte is a bug."""

    def one_more_call(d):
        d["parallel_comms.json"]["comms"]["data_parallel"]["per_collective"][
            "all_reduce"
        ]["calls"] = 4

    def one_more_byte(d):
        d["parallel_comms.json"]["formula_checks"][0]["formula_bytes_per_step"] = 228097

    assert len(_bucket(tmp_path, one_more_call)["invariant_moved"]) == 1
    assert len(_bucket(tmp_path, one_more_byte)["invariant_moved"]) == 1


def test_a_timing_change_is_reported_and_is_never_a_failure(tmp_path: Path):
    def faster(d):
        d["parallel_comms.json"]["meta"]["runtime_seconds"] = 12.0

    buckets = _bucket(tmp_path, faster)
    assert not buckets["invariant_moved"]
    assert len(buckets["timing"]) == 1


# --------------------------------------------------------------------------- #
# path handling
# --------------------------------------------------------------------------- #


def test_list_elements_are_keyed_by_name_so_inserting_a_row_is_not_ten_changes():
    """Keying by index made two new formula checks look like ten moved numbers."""
    before = flatten({"checks": [{"quantity": "a", "v": 1}, {"quantity": "b", "v": 2}]})
    after = flatten(
        {"checks": [{"quantity": "new", "v": 9}, {"quantity": "a", "v": 1},
                    {"quantity": "b", "v": 2}]}
    )
    assert before["checks[a].v"] == after["checks[a].v"]
    assert before["checks[b].v"] == after["checks[b].v"]
    assert "checks[new].v" in after


def test_elements_with_colliding_names_fall_back_to_indices():
    """Five rows per schedule would collapse into one if names were trusted blindly."""
    flat = flatten(
        {"rows": [{"schedule": "gpipe", "v": 1}, {"schedule": "gpipe", "v": 2}]},
    )
    # The discriminator fields disambiguate where they exist; here nothing does,
    # so both rows must survive under their indices rather than one overwriting
    # the other.
    assert len([k for k in flat if k.endswith(".v")]) == 2


def test_the_baseline_git_path_is_not_taken_from_the_current_directorys_name(
    tmp_path: Path,
):
    """A smoke run compares smoke/results against the committed results/.

    Deriving the repository path from the basename of --current worked for
    'smoke/results' by luck and looked for a directory that does not exist under
    any other name.
    """
    from compare_results import git_baseline

    # The real committed path resolves.
    directory = git_baseline("HEAD", "results")
    assert list(directory.glob("*.json"))
    # A path that is not in the repository fails loudly, naming the flag.
    with pytest.raises(RuntimeError, match="baseline-git-path"):
        git_baseline("HEAD", "not_a_directory_in_this_repo")
