"""Redraw every figure from the committed results JSONs.

Takes no measurements of its own: if a result file is missing, the figures it
would feed are skipped with a note. That is the invariant the repository relies
on -- a chart can only show numbers that are already in ``results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ASSETS, RESULTS, read_json
from transformer_internals import viz
from transformer_internals.perf.plots import fig_roofline, fig_step_breakdown


def _step_breakdown(payload: dict, path, web: bool = False):
    """Adapter: the breakdown draws the ``reports`` list inside the diagnosis payload."""
    return fig_step_breakdown(payload["reports"], path, web=web)

# result file -> (figure function, output stem)
FIGURES = [
    ("verification.json", viz.fig_verification, "verification"),
    ("induction.json", viz.fig_induction_heatmap, "induction_heads"),
    ("ablations.json", viz.fig_ablations, "ablations"),
    ("kv_cache.json", viz.fig_kv_latency, "kv_cache_latency"),
    ("kv_cache.json", viz.fig_kv_memory, "kv_cache_memory"),
    ("quantization.json", viz.fig_quantization, "quantization"),
    ("pruning.json", viz.fig_pruning_pareto, "pruning_pareto"),
    ("distillation.json", viz.fig_distillation, "distillation"),
    ("pareto.json", viz.fig_pareto, "pareto"),
    ("parallel_comms.json", viz.fig_parallel_bubble, "parallel_bubble"),
    ("roofline.json", fig_roofline, "roofline"),
    ("diagnosis.json", _step_breakdown, "step_breakdown"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--assets", default=str(ASSETS))
    args = ap.parse_args()

    results = Path(args.results)
    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)

    made, skipped = 0, []
    for filename, fn, stem in FIGURES:
        path = results / filename
        if not path.exists():
            skipped.append(f"{stem} (missing {filename})")
            continue
        payload = read_json(path)
        fn(payload, assets / f"{stem}.png", web=False)
        fn(payload, assets / f"{stem}_web.png", web=True)
        print(f"  {stem}.png + {stem}_web.png")
        made += 2

    print(f"\n{made} files written to {assets}")
    for s in skipped:
        print(f"  skipped {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
