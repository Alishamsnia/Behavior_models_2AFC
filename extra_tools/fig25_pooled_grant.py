#!/usr/bin/env python3
"""Single-page grant figure: pooled Figure 25 panels A (ranking) and B (all
pairwise comparisons, guide method).

Numbers are exactly those of page 1 of extra_tools/fig25_pooled.py: equal-session
mean held-out NLL over every mouse's sliding windows, paired moving-block
bootstrap stratified by mouse (blocks never cross mice), primary block length
L=5, percentile 95% CIs, uncentred two-minimum tail p-values, Holm across all
pairs. Only the presentation differs: pooled markers only, small markers,
thin dashed horizontal grid, square panels, and every count / test detail in legends anchored
outside the axes.

  python extra_tools/fig25_pooled_grant.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "extra_tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

from post_analysis.expanded_model_comparison import block_bootstrap_figures as bbf  # noqa: E402
from post_analysis.expanded_model_comparison.config import GROUP_COLOR, MODEL_BY_ID, validate_mouse_id  # noqa: E402
from fig25_pooled import DEFAULT_MICE, OUTPUT_DIR, load_pooled_scores  # noqa: E402

STEM = "fig25_pooled_grant_panels_AB"
GROUP_NAMES = {
    "wsls": "Win-stay/lose-shift only",
    "action": "Action history only",
    "m0": "Sensory only",
    "m1_history": "Sensory + choice-history kernel",
    "m1_bias": "Sensory + single-trial bias",
    "m2_switch": "Sensory + learned block inference",
    "m2_bias": "Sensory + block inference + single-trial bias",
}
GRID = {"color": "#bbbbbb", "linewidth": 0.3, "linestyle": (0, (3, 3))}


def _text_legend(figure, lines, title, anchor, fontsize=6.3):
    """A handle-free legend of plain text lines, anchored in page coordinates
    below the panels (outside every axes)."""
    handles = [plt.Line2D([], [], linestyle="", marker="") for _ in lines]
    legend = figure.legend(handles, lines, title=title, loc="upper left", bbox_to_anchor=anchor, fontsize=fontsize,
                       title_fontsize=fontsize + 0.6, handlelength=0, handletextpad=0, borderpad=0.7,
                       frameon=True, framealpha=1.0, edgecolor="#cccccc", alignment="left")
    legend.get_title().set_fontweight("bold")
    return legend


def _ranking_panel(ax, scores):
    means = scores.matrix.mean(axis=0)
    order = np.argsort(means)
    y = np.arange(len(order))
    for yi, index in zip(y, order):
        ax.scatter(means[index], yi, s=18, color=MODEL_BY_ID[scores.model_ids[index]].color, zorder=3, linewidths=0)
    ax.set_yticks(y, [bbf._model_label(scores.model_ids[index], 44) for index in order], fontsize=5.2)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", **GRID)
    ax.grid(axis="x", visible=False)
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean held-out NLL (nats/choice; lower is better)", fontsize=7)
    ax.set_title("A  Held-out ranking (pooled)", loc="left", fontweight="bold", fontsize=8.5)
    return scores.model_ids[order[0]], scores.model_ids[order[1]]


def _matrix_panel(ax, model_ids, results):
    lookup = bbf._result_lookup(results)
    n = len(model_ids)
    values = np.full((n, n), np.nan)
    for row in range(n):
        for col in range(row + 1, n):
            values[row, col] = bbf._oriented_result(lookup, model_ids[row], model_ids[col]).observed
    display = 1_000 * values
    maximum = max(float(np.nanmax(np.abs(display))), 0.01)
    image = ax.imshow(display, cmap="RdBu", norm=SymLogNorm(linthresh=2.0, vmin=-maximum, vmax=maximum),
                      interpolation="nearest", aspect="equal")
    for row in range(n):
        for col in range(row + 1, n):
            result = bbf._oriented_result(lookup, model_ids[row], model_ids[col])
            star = "*" if result.holm_p < 0.05 else ""
            ax.text(col, row, f"{1000 * result.observed:+.1f}{star}\n{bbf._format_p_short(result.raw_p)}",
                    ha="center", va="center", fontsize=3.6, linespacing=1.0,
                    color="white" if abs(result.observed) > 0.025 else "black")
    labels = [bbf._model_label(model_id, 44) for model_id in model_ids]
    ax.set_xticks(np.arange(n), labels, rotation=55, ha="right", fontsize=4.6, rotation_mode="anchor")
    ax.set_yticks(np.arange(n), labels, fontsize=4.6)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title("B  All pairwise comparisons (pooled)", loc="left", fontweight="bold", fontsize=8.5)
    return image


def generate(mice: tuple[str, ...], replicates: int) -> Path:
    pooled, per_mouse = load_pooled_scores(mice)
    length = bbf.DEFAULT_PRIMARY_LENGTH
    results, _ = bbf._document_pair_results(pooled.matrix, pooled.model_ids, length, replicates, pooled.strata)
    n_models = len(pooled.model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    n_sessions = len(pooled.matrix)
    session_parts = ", ".join(f"{mouse} {len(scores.matrix)}" for mouse, scores in per_mouse.items())

    style = {"font.family": "Liberation Sans", "pdf.fonttype": 42, "axes.spines.top": False,
             "axes.spines.right": False, "axes.linewidth": 0.6}
    with plt.rc_context(style):
        figure = plt.figure(figsize=(11, 8.5))
        # Both panels are 2.9 x 2.9 in squares on the 11 x 8.5 in page.
        side_w, side_h = 2.9 / 11, 2.9 / 8.5
        ranking_ax = figure.add_axes((2.2 / 11, 0.50, side_w, side_h))
        matrix_ax = figure.add_axes((6.75 / 11, 0.50, side_w, side_h))
        colorbar_ax = figure.add_axes((9.8 / 11, 0.55, 0.008, 0.24))

        leader, runner_up = _ranking_panel(ranking_ax, pooled)
        top_two = bbf._oriented_result(bbf._result_lookup(results), leader, runner_up)

        group_handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=4.4, markeredgewidth=0,
                       color=GROUP_COLOR[group], label=name)
            for group, name in GROUP_NAMES.items()
            if any(MODEL_BY_ID[model].group == group for model in pooled.model_ids)
        ]
        groups = figure.legend(handles=group_handles, title="Model class (marker color)", loc="upper left",
                               bbox_to_anchor=(0.03, 0.32), fontsize=6.3, title_fontsize=6.9,
                               frameon=True, edgecolor="#cccccc", alignment="left", handletextpad=0.3)
        groups.get_title().set_fontweight("bold")
        _text_legend(figure, [
            f"Mice: {len(per_mouse)} ({', '.join(per_mouse)})",
            f"Held-out test sessions: {n_sessions} ({session_parts})",
            "One sliding window per test session (9 train / 3 validation / 1 test)",
            f"Models: {n_models}; marker = equal-session mean held-out NLL",
            f"Leader vs runner-up: {1000 * top_two.observed:.1f}" + r" $\times10^{-3}$ nats/choice",
            f"   95% CI [{1000 * top_two.ci95_low:.1f}, {1000 * top_two.ci95_high:.1f}]" + r" $\times10^{-3}$",
            f"   Holm p = {top_two.holm_p:.3g} (raw p = {top_two.raw_p:.2g}; family = {n_pairs})",
        ], "Data and top-two test", anchor=(0.30, 0.32))

        image = _matrix_panel(matrix_ax, pooled.model_ids, results)
        colorbar = figure.colorbar(image, cax=colorbar_ax)
        colorbar.set_label(r"Row advantage $\times$1,000 (nats/choice)", fontsize=6.2)
        colorbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)
        colorbar.outline.set_linewidth(0.4)
        _text_legend(figure, [
            "Upper triangle; one cell per model pair",
            r"   top: row advantage = NLL(column) $-$ NLL(row), $\times$1,000; blue favors the row",
            "   bottom: raw two-sided bootstrap p (uncentred, 2x min tail)",
            f"   * Holm-adjusted p < 0.05 across all {n_pairs} pairs",
            f"Paired moving-block bootstrap: L = {length}, B = {replicates:,} resamples",
            "   blocks drawn within each mouse, never across mice",
            f"   {n_sessions} sessions equally weighted; percentile 95% CIs",
            "Inference conditional on these mice; exploratory",
        ], "Pairwise test (panel B)", anchor=(0.635, 0.32))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIR / f"{STEM}.pdf"
        figure.savefig(output, metadata={"Title": "Pooled held-out model comparison (Figure 25 A-B)"})
        figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300)
        plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--replicates", type=int, default=bbf.DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(tuple(validate_mouse_id(mouse) for mouse in args.mice), args.replicates))


if __name__ == "__main__":
    main()
