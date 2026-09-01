"""Diff two trees of result JSONs and print what changed, and by how much.

A GPU run rewrites the same result files a CPU run produced. Without a diff, the
only way to know what the new hardware changed is to read two 280 KB JSONs side
by side, so this walks both trees, pairs up every leaf number by its path, and
reports the ones that moved.

The output is deliberately blunt about three categories, because they mean
different things:

**Correctness numbers must not move**, and there are two kinds of "must not".

A byte count, a collective count or a formula check is an integer property of
the algorithm. It must be **bit-identical** between a gloo run and a NCCL run,
and any change at all is a bug. Those are :data:`EXACT_INVARIANTS`.

An equivalence error is a float. It *cannot* be bit-identical across backends,
because NCCL reduces in a different order from gloo and float addition is not
associative, so demanding equality would fail every cross-backend run for the
wrong reason. What must hold is the property the tests assert and
``parallel_comms.json`` records: every error stays under
:data:`EQUIVALENCE_TOLERANCE`, which is 1e-5. Those are
:data:`TOLERANT_INVARIANTS`, and they are reported with their ratio so a change
that stays inside the tolerance is still visible.

Both kinds are reported first, under their own heading, whether or not they
moved, because "it did not move" is the result.

**Timings are expected to move**, by a lot, and are reported with the ratio
rather than the difference.

**Everything else** is listed if it moved by more than the threshold.

Usage::

    python scripts/compare_results.py --baseline /tmp/before --current results
    python scripts/compare_results.py --baseline-git HEAD --current results
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RESULTS

#: The tolerance every equivalence proof in this repository is asserted at, in
#: ``tests/test_parallel.py`` and recorded as ``tolerance`` in
#: ``results/parallel_comms.json``. An error under it is a correct
#: implementation; an error over it is a bug, whatever backend produced it.
EQUIVALENCE_TOLERANCE = 1e-5

#: Integer properties of the algorithm. Bit-identical or it is a bug.
EXACT_INVARIANTS = [
    re.compile(r"formula_checks.*(exact_match|formula_bytes_per_step)$"),
    re.compile(r"\.payload_bytes(_per_step)?$"),
    re.compile(r"\.calls$"),
    re.compile(r"resharding\.reshards\..*\.bitwise_identical$"),
]

#: Float properties of the algorithm. They move between backends because float
#: addition is not associative and NCCL does not reduce in gloo's order; what
#: must hold is that they stay under :data:`EQUIVALENCE_TOLERANCE`.
TOLERANT_INVARIANTS = [
    re.compile(r"equivalence\..*error$"),
    re.compile(r"failure_restart\.max_abs_loss_difference$"),
]

#: Paths that are timings. Reported as ratios, and never as regressions.
TIMING_PATTERNS = [
    re.compile(r"(seconds|_s|_ms|wall_clock|runtime|latency)$"),
    re.compile(r"(step_s|median_s|min_s|makespan_seconds)$"),
    re.compile(r"flops_per_s$"),
    re.compile(r"bytes_per_s$"),
    re.compile(r"gbytes_per_s$"),
    re.compile(r"tokens_per_s$"),
]


def _matches(path: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(path) for p in patterns)


#: Fields that identify a list element. When every element of a list of dicts
#: can be given a *distinct* key from these, the elements are keyed by that
#: instead of by their index, so inserting a row does not make every row after
#: it look changed. That happened the first time this was run: two new entries
#: in ``formula_checks`` shifted the list and produced ten spurious differences.
#:
#: The uniqueness check is what makes this safe. ``measured_bubble`` has five
#: rows per schedule, so ``schedule`` alone would collide; the discriminators
#: below separate them, and where nothing separates them the index is used.
KEY_FIELDS = (
    "quantity",
    "strategy",
    "op",
    "name",
    "accelerator",
    "schedule",
    "arm",
    "kind",
    "stage",
    "reduce_dtype",
    "param_dtype",
    "micro_batches",
    "n_stages",
    "world_size",
    "to_world_size",
    "batch",
    "seq",
)


def _element_key(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = [
        f"{value[key]}"
        for key in KEY_FIELDS
        if isinstance(value.get(key), (str, int)) and not isinstance(value.get(key), bool)
    ]
    return "|".join(parts) if parts else None


def _list_keys(items: list[Any]) -> list[str]:
    """Names for a list's elements, falling back to indices unless all are distinct."""
    keys = [_element_key(v) for v in items]
    if all(k is not None for k in keys) and len(set(keys)) == len(keys):
        return [str(k) for k in keys]
    return [str(i) for i in range(len(items))]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a nested JSON structure, keyed by its dotted path.

    List elements are keyed by name where they have one (see :data:`NAME_KEYS`)
    and by index otherwise, so a per-rank list stays comparable element by
    element while a list of named records survives being reordered.

    Lists longer than 64 entries are summarised by their length alone: a raw
    sample vector has nothing to say in a diff and would bury everything that
    does.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        if len(obj) > 64:
            out[f"{prefix}.__len__"] = len(obj)
        else:
            for key, value in zip(_list_keys(obj), obj, strict=True):
                out.update(flatten(value, f"{prefix}[{key}]"))
    else:
        out[prefix] = obj
    return out


def load_tree(root: Path) -> dict[str, Any]:
    """Flatten every ``*.json`` under ``root`` into one path -> value mapping."""
    flat: dict[str, Any] = {}
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        flat.update(flatten(payload, path.name))
    return flat


def git_baseline(ref: str, git_path: str) -> Path:
    """Extract the committed result files at ``ref`` into a temporary directory.

    Uses ``git show`` per file rather than a checkout, so nothing in the working
    tree is touched by taking a baseline.

    ``git_path`` is the path *inside the repository*, and is deliberately not
    derived from ``--current``. A smoke run compares ``smoke/results`` against
    the committed ``results``, and taking the basename of the current directory
    would have looked for ``HEAD:results`` by luck and for ``HEAD:sres`` under
    any other name.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ti-baseline-"))
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:{git_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(
            f"could not list {git_path}/ at {ref}: {listing.stderr.strip()}. "
            f"Pass --baseline-git-path if the committed results live elsewhere."
        )
    for name in listing.stdout.split():
        if not name.endswith(".json"):
            continue
        blob = subprocess.run(
            ["git", "show", f"{ref}:{git_path}/{name}"],
            capture_output=True,
            check=False,
        )
        if blob.returncode == 0:
            (tmp / name).write_bytes(blob.stdout)
    return tmp


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def compare(
    baseline: dict[str, Any], current: dict[str, Any], threshold: float
) -> dict[str, list[tuple[str, Any, Any]]]:
    """Bucket every path into invariants, timings, moved, added and removed."""
    buckets: dict[str, list[tuple[str, Any, Any]]] = {
        "invariant_held": [],
        "invariant_moved": [],
        "within_tolerance": [],
        "timing": [],
        "moved": [],
        "added": [],
        "removed": [],
    }
    for path in sorted(set(baseline) | set(current)):
        before, after = baseline.get(path), current.get(path)
        if path not in current:
            buckets["removed"].append((path, before, None))
            continue
        if path not in baseline:
            buckets["added"].append((path, None, after))
            continue
        exact = _matches(path, EXACT_INVARIANTS)
        tolerant = _matches(path, TOLERANT_INVARIANTS)
        if before == after:
            if exact or tolerant:
                buckets["invariant_held"].append((path, before, after))
            continue
        if exact:
            buckets["invariant_moved"].append((path, before, after))
            continue
        if tolerant:
            # A float equivalence error. It is allowed to move; it is not
            # allowed to leave the tolerance the tests assert.
            if isinstance(after, (int, float)) and after <= EQUIVALENCE_TOLERANCE:
                buckets["within_tolerance"].append((path, before, after))
            else:
                buckets["invariant_moved"].append((path, before, after))
            continue
        if _matches(path, TIMING_PATTERNS):
            buckets["timing"].append((path, before, after))
            continue
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            scale = max(abs(before), abs(after), 1e-30)
            if abs(after - before) / scale < threshold:
                continue
        buckets["moved"].append((path, before, after))
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=None, help="directory of result JSONs to compare against")
    ap.add_argument(
        "--baseline-git",
        default=None,
        help="git ref whose committed results/ to compare against, e.g. HEAD",
    )
    ap.add_argument("--current", default=str(RESULTS))
    ap.add_argument(
        "--baseline-git-path",
        default=str(RESULTS),
        help="path inside the repository whose committed JSONs are the baseline",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="relative change below which a number is not worth mentioning",
    )
    ap.add_argument("--limit", type=int, default=40, help="rows per section")
    ap.add_argument(
        "--allow-changes",
        action="store_true",
        help=(
            "report differences but always exit 0. For the smoke run, whose "
            "results come from deliberately tiny configurations and therefore "
            "differ from the committed full-size ones by design."
        ),
    )
    args = ap.parse_args()

    current_dir = Path(args.current)
    if args.baseline:
        baseline_dir = Path(args.baseline)
    elif args.baseline_git:
        baseline_dir = git_baseline(args.baseline_git, args.baseline_git_path)
    else:
        baseline_dir = git_baseline("HEAD", args.baseline_git_path)

    baseline = load_tree(baseline_dir)
    current = load_tree(current_dir)
    if not baseline:
        print(f"no baseline results found under {baseline_dir}")
        return 1

    buckets = compare(baseline, current, args.threshold)

    print(f"baseline {baseline_dir}  ({len(baseline)} leaf values)")
    print(f"current  {current_dir}  ({len(current)} leaf values)")

    held = buckets["invariant_held"]
    moved_inv = buckets["invariant_moved"]
    tolerated = buckets["within_tolerance"]
    print(
        f"\nCORRECTNESS INVARIANTS: {len(held)} identical, "
        f"{len(tolerated)} moved but within {EQUIVALENCE_TOLERANCE:g}, "
        f"{len(moved_inv)} BROKEN"
    )
    if moved_inv:
        print("  These are properties of the algorithm, not of the hardware.")
        print("  Any row here is a finding and probably a bug.")
        for path, before, after in moved_inv[: args.limit]:
            print(f"    {path}\n      {_fmt(before)}  ->  {_fmt(after)}")
    else:
        print("  Every byte count and formula check is identical, and every")
        print(f"  equivalence error is still under {EQUIVALENCE_TOLERANCE:g}.")
    for path, before, after in tolerated[: args.limit]:
        ratio = ""
        if isinstance(before, (int, float)) and before:
            ratio = f"   ({after / before:.2f}x)"
        print(f"  ~ {path}\n      {_fmt(before)}  ->  {_fmt(after)}{ratio}")
    if len(tolerated) > args.limit:
        print(f"  ... and {len(tolerated) - args.limit} more within tolerance")

    print(f"\nTIMINGS AND RATES: {len(buckets['timing'])} changed")
    for path, before, after in buckets["timing"][: args.limit]:
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
            print(f"  {path}\n    {_fmt(before)}  ->  {_fmt(after)}   ({after / before:.2f}x)")
        else:
            print(f"  {path}\n    {_fmt(before)}  ->  {_fmt(after)}")
    if len(buckets["timing"]) > args.limit:
        print(f"  ... and {len(buckets['timing']) - args.limit} more")

    print(f"\nOTHER VALUES THAT MOVED BY MORE THAN {args.threshold:.0%}: {len(buckets['moved'])}")
    for path, before, after in buckets["moved"][: args.limit]:
        print(f"  {path}\n    {_fmt(before)}  ->  {_fmt(after)}")
    if len(buckets["moved"]) > args.limit:
        print(f"  ... and {len(buckets['moved']) - args.limit} more")

    print(f"\nNEW KEYS: {len(buckets['added'])}   REMOVED KEYS: {len(buckets['removed'])}")
    for path, _before, after in buckets["added"][: args.limit]:
        print(f"  + {path} = {_fmt(after)}")
    if len(buckets["added"]) > args.limit:
        print(f"  ... and {len(buckets['added']) - args.limit} more")
    for path, before, _after in buckets["removed"][: args.limit]:
        print(f"  - {path} (was {_fmt(before)})")
    if len(buckets["removed"]) > args.limit:
        print(f"  ... and {len(buckets['removed']) - args.limit} more")

    # A moved invariant is the one thing that should make this exit non-zero: it
    # means a number that cannot legitimately depend on the hardware did.
    if moved_inv:
        if args.allow_changes:
            print(
                f"\n--allow-changes: {len(moved_inv)} correctness invariant(s) differ "
                f"from the baseline. Expected when the two runs used different "
                f"configurations, as a smoke run does. Exiting 0."
            )
            return 0
        print(f"\nFAILED: {len(moved_inv)} correctness invariant(s) broke")
        return 1
    print(
        f"\nOK: every exact invariant is identical and every equivalence error "
        f"is under {EQUIVALENCE_TOLERANCE:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
