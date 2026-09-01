"""Diff two trees of result JSONs and print what changed, and by how much.

A GPU run rewrites the same result files a CPU run produced. Without a diff, the
only way to know what the new hardware changed is to read two 280 KB JSONs side
by side, so this walks both trees, pairs up every leaf number by its path, and
reports the ones that moved.

The output is deliberately blunt about three categories, because they mean
different things:

**Correctness numbers must not move.** Equivalence errors, formula checks and
byte counts are properties of the algorithm, not of the hardware. An
equivalence error that changes by orders of magnitude between backends is a
finding and probably a bug; a byte count that changes at all is definitely one.
Those paths are matched by :data:`INVARIANT_PATTERNS` and are reported first,
under their own heading, whether or not they moved.

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

#: Paths whose value is a property of the algorithm and not of the machine.
#: Anything matching these is reported even when it did not move, because "it
#: did not move" is the result.
INVARIANT_PATTERNS = [
    re.compile(r"formula_checks.*(exact_match|formula_bytes_per_step)$"),
    re.compile(r"equivalence\..*error$"),
    re.compile(r"\.payload_bytes(_per_step)?$"),
    re.compile(r"\.calls$"),
    re.compile(r"resharding\.reshards\..*\.bitwise_identical$"),
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


def git_baseline(ref: str, results_dir: Path) -> Path:
    """Extract the committed result files at ``ref`` into a temporary directory.

    Uses ``git show`` per file rather than a checkout, so nothing in the working
    tree is touched by taking a baseline.
    """
    tmp = Path(tempfile.mkdtemp(prefix="ti-baseline-"))
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:{results_dir.name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(
            f"could not list {results_dir.name}/ at {ref}: {listing.stderr.strip()}"
        )
    for name in listing.stdout.split():
        if not name.endswith(".json"):
            continue
        blob = subprocess.run(
            ["git", "show", f"{ref}:{results_dir.name}/{name}"],
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
        if before == after:
            if _matches(path, INVARIANT_PATTERNS):
                buckets["invariant_held"].append((path, before, after))
            continue
        if _matches(path, INVARIANT_PATTERNS):
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
        "--threshold",
        type=float,
        default=0.01,
        help="relative change below which a number is not worth mentioning",
    )
    ap.add_argument("--limit", type=int, default=40, help="rows per section")
    args = ap.parse_args()

    current_dir = Path(args.current)
    if args.baseline:
        baseline_dir = Path(args.baseline)
    elif args.baseline_git:
        baseline_dir = git_baseline(args.baseline_git, current_dir)
    else:
        baseline_dir = git_baseline("HEAD", current_dir)

    baseline = load_tree(baseline_dir)
    current = load_tree(current_dir)
    if not baseline:
        print(f"no baseline results found under {baseline_dir}")
        return 1

    buckets = compare(baseline, current, args.threshold)

    print(f"baseline {baseline_dir}  ({len(baseline)} leaf values)")
    print(f"current  {current_dir}  ({len(current)} leaf values)")

    held, moved_inv = buckets["invariant_held"], buckets["invariant_moved"]
    print(f"\nCORRECTNESS INVARIANTS: {len(held)} unchanged, {len(moved_inv)} MOVED")
    if moved_inv:
        print("  These are properties of the algorithm, not of the hardware.")
        print("  Any row here is a finding and probably a bug.")
        for path, before, after in moved_inv[: args.limit]:
            print(f"    {path}\n      {_fmt(before)}  ->  {_fmt(after)}")
    else:
        print("  Every equivalence error, byte count and formula check is identical.")

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
        print(f"\nFAILED: {len(moved_inv)} correctness invariant(s) changed")
        return 1
    print("\nOK: every correctness invariant held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
