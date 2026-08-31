"""Figures, drawn only from committed result JSONs.

None of these functions can train, sample or measure anything. They take a loaded
results dict and write a PNG. If a number appears on a chart in this repository,
it came out of a file in ``results/``.

Design rules, set once in :func:`use_style` and not overridden per figure:

* White ground, ink ``#101114`` for text and data, ``#6A6F78`` for anything
  secondary, hairline ``#D8DBE1`` grid drawn **horizontally only and behind** the
  data, top and right spines off.
* Exactly one accent, ``#FF3B00``, reserved for the single thing the eye should
  land on. If everything is highlighted, nothing is.
* No categorical default palettes. Series that need to be distinguished are
  separated in *lightness* as well as hue so the figures survive greyscale.
* Direct labels in preference to a legend wherever the chart has room.

Every headline figure is written twice: ``<name>.png`` at full detail, and
``<name>_web.png`` sized and simplified to stay legible at 340px wide, where a
long title or rotated tick labels become a smudge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "ACCENT",
    "GRID",
    "INK",
    "MUTED",
    "fig_ablations",
    "fig_distillation",
    "fig_induction_heatmap",
    "fig_kv_latency",
    "fig_kv_memory",
    "fig_pareto",
    "fig_pruning_pareto",
    "fig_quantization",
    "fig_verification",
    "save",
    "use_style",
]

INK = "#101114"
MUTED = "#6A6F78"
GRID = "#D8DBE1"
ACCENT = "#FF3B00"
SURFACE = "#FFFFFF"

#: A light-to-ink sequential ramp for heatmaps, ending at the accent so the
#: strongest cells are the ones the eye finds first. Sequential and monotone in
#: lightness, so it also reads correctly in greyscale and for colour-blind
#: readers -- unlike the rainbow maps that heatmaps usually get.
HEAT = LinearSegmentedColormap.from_list(
    "ink_accent", ["#FFFFFF", "#DCE0E6", "#9AA1AC", "#4A4F58", "#101114", ACCENT], N=256
)


def use_style(scale: float = 1.0) -> None:
    """Install the house rcParams. Call once before drawing."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9 * scale,
            "axes.titlesize": 11 * scale,
            "axes.titleweight": "bold",
            "axes.labelsize": 9 * scale,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8 * scale,
            "ytick.labelsize": 8 * scale,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8 * scale,
            "text.color": INK,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _hgrid(ax: plt.Axes) -> None:
    """Horizontal-only hairline grid, behind the data."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# Part 2 -- verification
# --------------------------------------------------------------------------


def fig_verification(report: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Layer-by-layer max absolute difference against the reference.

    A log y-axis, because the interesting structure is the *growth* of floating
    point error with depth, which spans orders of magnitude and is invisible on a
    linear scale. The asserted tolerance is drawn as the accent line: the story of
    the chart is that every measured point sits far below it.
    """
    use_style(1.15 if web else 1.0)
    layers = report["layers"]
    names = [row["name"] for row in layers]
    vals = [row["max_abs"] for row in layers]

    # Colour by sub-module kind so the reader can see that no one kind of
    # sub-module is systematically worse than the others.
    # Six distinct steps in lightness, so the legend is readable and the figure
    # survives greyscale. No two sub-modules share a shade.
    kinds = {
        "ln_1": "#C8CCD3",
        "attn": "#6E747E",
        "resid_mid": "#A2A8B2",
        "ln_2": "#8A9099",
        "mlp": "#3B404A",
        "resid_out": INK,
    }

    def kind_of(n: str) -> str:
        return n.split(".")[-1] if n.startswith("h.") else n

    fig, ax = plt.subplots(figsize=(9.5, 3.6) if not web else (7.2, 3.4))
    x = np.arange(len(names))
    colors = [kinds.get(kind_of(n), MUTED) for n in names]
    ax.bar(x, vals, color=colors, width=0.82, linewidth=0)

    tol = report.get("logit_tolerance", 1e-3)
    ax.axhline(tol, color=ACCENT, linewidth=1.4, zorder=5)
    ax.annotate(
        f"asserted tolerance  {tol:g}",
        xy=(len(names) * 0.985, tol),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=ACCENT,
        fontsize=8,
        fontweight="bold",
    )

    ax.set_yscale("log")
    ax.set_ylabel("max |ours - reference|")
    _hgrid(ax)

    # Label one tick per block rather than per sub-module: 74 rotated labels is
    # noise, 12 block numbers is information.
    block_ticks, block_labels = [], []
    for i, n in enumerate(names):
        if n.endswith(".resid_out"):
            block_ticks.append(i)
            block_labels.append(n.split(".")[1])
    ax.set_xticks(block_ticks)
    ax.set_xticklabels(block_labels)
    ax.set_xlabel("transformer block (ticks at each block's residual output)")
    ax.set_xlim(-0.8, len(names) - 0.2)

    worst = max(layers, key=lambda r: r["max_abs"])
    logit = report["logits"]["max_abs"]
    title = (
        f"Every activation matches HuggingFace GPT-2 to {worst['max_abs']:.1e} or better"
    )
    ax.set_title(title, loc="left", pad=14)
    ax.annotate(
        f"final logits: {logit:.2e}   ·   greedy generation: "
        f"{'token-exact' if report.get('all_generations_match') else 'MISMATCH'}",
        xy=(0, 1.005),
        xycoords="axes fraction",
        fontsize=8,
        color=MUTED,
    )
    if not web:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in kinds.values()]
        ax.legend(handles, list(kinds), ncol=6, loc="lower left", bbox_to_anchor=(0, -0.34))
    return save(fig, path)


# --------------------------------------------------------------------------
# Part 3 -- induction heads and ablations
# --------------------------------------------------------------------------


def fig_induction_heatmap(scores: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Per-head prefix-matching heatmap, with the named induction heads circled."""
    use_style(1.15 if web else 1.0)
    pm = np.array(scores["prefix_matching"])
    n_layer, n_head = pm.shape

    fig, ax = plt.subplots(figsize=(6.4, 4.6) if not web else (6.0, 4.4))
    im = ax.imshow(pm, cmap=HEAT, aspect="auto", vmin=0.0, vmax=max(0.5, pm.max()))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_head))
    ax.set_yticks(range(n_layer))
    ax.grid(False)
    ax.set_axisbelow(False)

    thresh = 0.3
    for i in range(n_layer):
        for j in range(n_head):
            if pm[i, j] >= thresh:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ACCENT, linewidth=2.0
                    )
                )
                if not web:
                    ax.text(
                        j,
                        i,
                        f"{pm[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=SURFACE if pm[i, j] > 0.6 else INK,
                        fontweight="bold",
                    )

    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("prefix-matching score", fontsize=8)
    cb.outline.set_visible(False)

    chance = scores.get("chance_level", 0.0)
    ax.set_title("Induction heads in GPT-2 small", loc="left", pad=12)
    ax.annotate(
        f"attention to the induction target on repeated random tokens · chance = {chance:.3f} · "
        f"{int((pm >= thresh).sum())} heads above {thresh}",
        xy=(0, 1.02),
        xycoords="axes fraction",
        fontsize=7.5,
        color=MUTED,
    )
    return save(fig, path)


def fig_ablations(summary: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Ablation deltas against baseline, with seed spread as the error bar."""
    use_style(1.15 if web else 1.0)
    rows = [r for r in summary["rows"] if r["arm"] != "baseline"]
    rows = sorted(rows, key=lambda r: r["delta_vs_baseline"])
    labels = [r["label"] for r in rows]
    deltas = [r["delta_vs_baseline"] for r in rows]
    errs = [r["pooled_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 3.8) if not web else (6.6, 3.9))
    y = np.arange(len(rows))
    # Accent only for the arms that are actually distinguishable from baseline;
    # everything indistinguishable is muted, so the chart's colour carries the
    # verdict rather than decorating it.
    colors = [ACCENT if r["verdict"] in ("worse", "better") else "#B9BEC6" for r in rows]
    ax.barh(y, deltas, xerr=errs, color=colors, height=0.68, linewidth=0,
            error_kw={"ecolor": MUTED, "elinewidth": 1.0, "capsize": 3})
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ final validation loss vs GPT-2 defaults (nats) — 3 seeds, mean ± pooled sd")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    for yi, r in zip(y, rows, strict=True):
        d = r["delta_vs_baseline"]
        off = 4 if d >= 0 else -4
        ax.annotate(
            f"{d:+.3f}" + ("" if r["verdict"] != "indistinguishable" else "  n.s."),
            xy=(d + (errs[yi] if d >= 0 else -errs[yi]), yi),
            xytext=(off, 0),
            textcoords="offset points",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=7.5,
            color=INK if r["verdict"] != "indistinguishable" else MUTED,
        )
    ax.set_title("What each GPT-2 design decision is worth", loc="left", pad=12)
    ax.annotate(
        "grey = not distinguishable from the seed-to-seed spread",
        xy=(0, 1.02), xycoords="axes fraction", fontsize=7.5, color=MUTED,
    )
    ax.margins(x=0.22)
    return save(fig, path)


# --------------------------------------------------------------------------
# Part 4 -- inference efficiency
# --------------------------------------------------------------------------


def fig_kv_latency(bench: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Cached vs uncached decoding latency across context length."""
    use_style(1.15 if web else 1.0)
    rows = bench["generation"]
    cached = sorted([r for r in rows if r["use_cache"]], key=lambda r: r["prompt_len"])
    uncached = sorted([r for r in rows if not r["use_cache"]], key=lambda r: r["prompt_len"])

    fig, ax = plt.subplots(figsize=(6.6, 3.8) if not web else (6.4, 3.8))
    if uncached:
        ax.plot(
            [r["prompt_len"] for r in uncached],
            [r["ms_per_token"] for r in uncached],
            marker="o", markersize=4, linewidth=1.8, color=INK, label="no cache  (O(T²))",
        )
    ax.plot(
        [r["prompt_len"] for r in cached],
        [r["ms_per_token"] for r in cached],
        marker="o", markersize=4, linewidth=2.2, color=ACCENT, label="KV cache  (O(T))",
    )
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("ms per generated token")
    _hgrid(ax)

    if uncached and cached:
        last_u, last_c = uncached[-1], cached[-1]
        speedup = last_u["ms_per_token"] / last_c["ms_per_token"]
        ax.annotate(
            f"{speedup:.1f}× faster\nat {last_c['prompt_len']} tokens",
            xy=(last_c["prompt_len"], last_c["ms_per_token"]),
            xytext=(-10, 34), textcoords="offset points",
            ha="right", fontsize=8, color=ACCENT, fontweight="bold",
        )
    ax.legend(loc="upper left")
    ax.set_title("The KV cache turns quadratic decoding into linear", loc="left", pad=12)
    ax.annotate(
        f"{bench['meta']['device']} · {bench['meta']['new_tokens']} new tokens · "
        f"median of {bench['meta']['repeats']} runs",
        xy=(0, 1.02), xycoords="axes fraction", fontsize=7.5, color=MUTED,
    )
    return save(fig, path)


def fig_kv_memory(bench: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Cache size vs context, against model weights, and the GQA/MQA reduction."""
    use_style(1.15 if web else 1.0)
    mem = bench["cache_memory"]
    variants = bench["attention_variants"]
    model_mb = bench["meta"]["model_bytes"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6) if not web else (8.6, 3.7))
    ax = axes[0]
    batches = sorted({r["batch_size"] for r in mem})
    shades = ["#C3C8D0", "#8C93A0", "#4A4F58", INK]
    for i, bs in enumerate(batches):
        pts = sorted([r for r in mem if r["batch_size"] == bs], key=lambda r: r["seq_len"])
        color = ACCENT if bs == max(batches) else shades[min(i, len(shades) - 1)]
        ax.plot([p["seq_len"] for p in pts], [p["cache_mb"] for p in pts],
                marker="o", markersize=3.5, linewidth=2.0 if bs == max(batches) else 1.4,
                color=color)
        ax.annotate(f"batch {bs}", xy=(pts[-1]["seq_len"], pts[-1]["cache_mb"]),
                    xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                    color=color, va="center", fontweight="bold" if bs == max(batches) else "normal")
    ax.axhline(model_mb, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    # Below the line, not above: the batch-32 series crosses it around 200 tokens
    # and an annotation above would sit on top of the data.
    ax.annotate(f"model weights  {model_mb:.0f} MB", xy=(0.30, model_mb),
                xycoords=("axes fraction", "data"), xytext=(0, -13),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV cache (MB, fp32)")
    _hgrid(ax)
    ax.margins(x=0.16)
    ax.set_title("The cache outgrows the model", loc="left", pad=10)

    ax2 = axes[1]
    names = [v["variant"] for v in variants]
    mbs = [v["cache_mb"] for v in variants]
    x = np.arange(len(variants))
    colors = [INK if v["variant"] == "MHA" else ACCENT for v in variants]
    ax2.bar(x, mbs, color=colors, width=0.62, linewidth=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("KV cache (MB)")
    _hgrid(ax2)
    for xi, v in zip(x, variants, strict=True):
        ax2.annotate(f"{v['cache_mb']:.0f} MB\n{v['reduction_vs_mha']:.0f}× smaller"
                     if v["reduction_vs_mha"] > 1 else f"{v['cache_mb']:.0f} MB",
                     xy=(xi, v["cache_mb"]), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK)
    ax2.margins(y=0.22)
    ax2.set_title(
        f"GQA / MQA at {variants[0]['seq_len']} tokens, batch {variants[0]['batch_size']}",
        loc="left", pad=10,
    )
    fig.tight_layout()
    return save(fig, path)


def fig_quantization(quant: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Perplexity cost per scheme, with compression printed alongside.

    Bars, not a scatter on (compression, quality): the two int8 schemes compress
    identically and differ only in quality, so on a scatter they land on top of
    each other and the one comparison the chart exists to make is the one it
    cannot show. Bars separate the schemes by construction and let compression be
    a printed value.

    The y-axis is logarithmic because int4 with a single per-tensor scale reaches
    perplexity ~2.3e3 against a baseline of ~18. Those three orders of magnitude
    are the finding, and a linear axis would flatten everything else to zero.
    """
    use_style(1.15 if web else 1.0)
    rows = quant["rows"]
    base_ppl = quant["baseline"]["ppl_of_mean_loss"]

    fig, ax = plt.subplots(figsize=(7.6, 4.0) if not web else (6.8, 4.2))
    x = np.arange(len(rows))
    labels = [r["label"].replace(" + embedding", "\n+ embedding") for r in rows]

    colors = []
    for r in rows:
        if r.get("highlight"):
            colors.append(ACCENT)
        elif r["granularity"] == "per_channel":
            colors.append("#4A4F58")
        else:
            colors.append("#B9BEC6")
    ax.bar(x, [r["quality"]["ppl_of_mean_loss"] for r in rows], color=colors,
           width=0.6, linewidth=0, zorder=3)
    ax.errorbar(x, [r["quality"]["ppl_of_mean_loss"] for r in rows],
                yerr=[r["quality"]["ppl_std"] for r in rows], fmt="none",
                ecolor=MUTED, elinewidth=1.0, capsize=3, zorder=4)

    ax.axhline(base_ppl, color=ACCENT if not any(r.get("highlight") for r in rows) else MUTED,
               linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"fp32 baseline  ppl {base_ppl:.2f}", xy=(-0.42, base_ppl),
                xytext=(0, -13), textcoords="offset points", ha="left",
                fontsize=7.5, color=MUTED)

    for xi, r in zip(x, rows, strict=True):
        ppl = r["quality"]["ppl_of_mean_loss"]
        delta = ppl - base_ppl
        ax.annotate(
            f"{ppl:,.1f}\n{delta:+,.1f} ppl\n{r['compression_ratio']:.2f}× smaller",
            xy=(xi, ppl), xytext=(0, 7), textcoords="offset points", ha="center",
            fontsize=7.2, linespacing=1.4, color=INK,
            fontweight="bold" if r.get("highlight") else "normal",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("held-out perplexity, log scale  (± sd over chunks)")
    _hgrid(ax)
    ax.set_ylim(base_ppl * 0.6, max(r["quality"]["ppl_of_mean_loss"] for r in rows) * 40)
    ax.set_title("Per-channel scales are what make low-bit quantization work",
                 loc="left", pad=14)
    ax.annotate(
        "one scale per output channel (dark) vs one scale for the whole tensor (light)",
        xy=(0, 1.02), xycoords="axes fraction", fontsize=7.5, color=MUTED,
    )
    return save(fig, path)


def fig_pruning_pareto(prune: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Quality against parameters removed, for heads and MLP neurons."""
    use_style(1.15 if web else 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 3.9) if not web else (6.4, 3.9))

    styles = {"heads": (ACCENT, "o", "attention heads"), "neurons": (INK, "s", "MLP neurons")}
    for kind, rows in prune["sweeps"].items():
        color, marker, label = styles.get(kind, (MUTED, "^", kind))
        xs = [100 * r["fraction_removed"] for r in rows]
        ys = [r["val_loss"] for r in rows]
        ax.plot(xs, ys, marker=marker, markersize=5, linewidth=1.8, color=color, label=label)

    ax.set_xlabel("parameters removed (%)")
    ax.set_ylabel("held-out loss (nats)")
    _hgrid(ax)
    ax.legend(loc="upper left")
    ax.set_title("Structured pruning: quality against size", loc="left", pad=12)
    ax.annotate("gradient-based importance (Michel et al.), normalised within each layer",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=7.5, color=MUTED)
    return save(fig, path)


def fig_distillation(dist: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """Distilled student against the same student trained from scratch."""
    use_style(1.15 if web else 1.0)
    fig, ax = plt.subplots(figsize=(6.8, 3.9) if not web else (6.5, 3.9))
    best = dist.get("best_distilled")
    shades = ["#C3C8D0", "#9AA1AC", "#6E747E", "#4A4F58"]
    n_dist = 0
    for arm in dist["arms"]:
        if arm["alpha"] == 0:
            color, width = INK, 2.4
        elif arm["label"] == best:
            color, width = ACCENT, 2.4
        else:
            color, width = shades[n_dist % len(shades)], 1.4
        if arm["alpha"] > 0:
            n_dist += 1
        hist = arm["mean_history"]
        ax.plot([h["step"] for h in hist], [h["val_loss"] for h in hist],
                marker="o", markersize=3.0, linewidth=width, color=color, label=arm["label"])
        if arm["alpha"] == 0 or arm["label"] == best:
            ax.annotate(f"{hist[-1]['val_loss']:.3f}",
                        xy=(hist[-1]["step"], hist[-1]["val_loss"]),
                        xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=color, va="center", fontweight="bold")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("held-out loss (nats)")
    _hgrid(ax)
    ax.legend(loc="upper right", ncol=2)
    ax.margins(x=0.16)
    ax.set_title("Distillation did not help at this budget", loc="left", pad=12)
    ax.annotate(dist["meta"]["caption"], xy=(0, 1.02), xycoords="axes fraction",
                fontsize=7.5, color=MUTED)
    return save(fig, path)


def fig_pareto(pareto: dict[str, Any], path: str | Path, web: bool = False) -> Path:
    """The summary chart: quality against size, every configuration as a point."""
    use_style(1.2 if web else 1.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.4) if not web else (6.6, 4.3))

    families = {
        "baseline": (ACCENT, "o", 9),
        "quantization": (INK, "o", 6),
        "pruning": ("#8C93A0", "s", 6),
        "attention": ("#4A4F58", "^", 6),
    }
    for p in pareto["points"]:
        color, marker, size = families.get(p["family"], (MUTED, "o", 5))
        ax.scatter(p["size_mb"], p["ppl"], s=size**2, marker=marker, color=color,
                   zorder=4 if p["family"] == "baseline" else 3,
                   edgecolor=SURFACE, linewidth=0.6)
        if p.get("label_it", True):
            ax.annotate(p["label"], xy=(p["size_mb"], p["ppl"]), xytext=(6, 3),
                        textcoords="offset points", fontsize=7,
                        color=color, fontweight="bold" if p["family"] == "baseline" else "normal")

    ax.set_xlabel("model size on disk (MB)")
    ax.set_ylabel("held-out perplexity")
    _hgrid(ax)
    ax.margins(x=0.20, y=0.16)
    ax.set_title("Quality against size, every configuration measured", loc="left", pad=12)
    ax.annotate(pareto["meta"]["caption"], xy=(0, 1.02), xycoords="axes fraction",
                fontsize=7.5, color=MUTED)
    return save(fig, path)
