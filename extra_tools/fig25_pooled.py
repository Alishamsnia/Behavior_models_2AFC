#!/usr/bin/env python3
"""Figure 25 (two-page block-bootstrap model comparison) pooled across mice.

Same procedure as the per-mouse Figure 25 (extra_tools/fig25_for_mouse.py and
post_analysis/expanded_model_comparison/block_bootstrap_figures.py), applied to
the stacked per-window held-out NLL of every mouse:

* the estimand is the equal-session mean held-out NLL over all mice's windows;
* the moving-block bootstrap is stratified by mouse: each mouse's chronology is
  resampled with the identical non-circular moving-block scheme (same block
  lengths, same trimming) on its own independent random stream, and blocks
  never cross a mouse boundary -- chronology only exists within a mouse;
* percentile / basic CIs, uncentred two-minimum and null-centred tail p-values,
  Holm families, exact edge shifts, block-length sensitivity and the TOST
  equivalence step are the per-mouse functions, unchanged.

Only the two chronology panels (session trajectory, ACF) are drawn per mouse.
Inference is conditional on these mice (fixed effects): between-mouse
variability is not resampled, so this is not a population-level claim.

  python extra_tools/fig25_pooled.py                    # MC06 MC07 MC08
  python extra_tools/fig25_pooled.py --mice MC07 MC08
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
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from post_analysis.expanded_model_comparison import block_bootstrap_figures as bbf  # noqa: E402
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER, validate_mouse_id  # noqa: E402
from fig25_for_mouse import load_scores as load_mouse_scores  # noqa: E402

DEFAULT_MICE = ("MC06", "MC07", "MC08")
OUTPUT_DIR = bbf.ANALYSIS_ROOT / "pooled" / "model_comparison"
OUTPUT_NAME = "fig25_block_bootstrap_document_and_hybrid_pooled.pdf"
# Per-mouse styling for the chronology panels; neutral so it never competes
# with the model palette.
MOUSE_STYLES = (
    {"color": "#222222", "marker": "o", "linestyle": "-"},
    {"color": "#777777", "marker": "s", "linestyle": "--"},
    {"color": "#AAAAAA", "marker": "^", "linestyle": ":"},
)


def load_pooled_scores(mice: tuple[str, ...]) -> tuple[bbf.LoadedScores, dict[str, bbf.LoadedScores]]:
    per_mouse = {}
    for mouse in mice:
        scores, _ = load_mouse_scores(mouse)
        if scores.model_ids != tuple(MODEL_ORDER):
            missing = sorted(set(MODEL_ORDER) - set(scores.model_ids))
            raise ValueError(f"{mouse}: pooled comparison needs every configured model; missing {missing}")
        per_mouse[mouse] = scores
    pooled = bbf.LoadedScores(
        matrix=np.vstack([scores.matrix for scores in per_mouse.values()]),
        model_ids=tuple(MODEL_ORDER),
        test_ranks=np.concatenate([scores.test_ranks for scores in per_mouse.values()]),
        trial_counts=np.concatenate([scores.trial_counts for scores in per_mouse.values()]),
        strata=tuple(len(scores.matrix) for scores in per_mouse.values()),
    )
    return pooled, per_mouse


def _session_text(per_mouse: dict[str, bbf.LoadedScores]) -> str:
    total = sum(len(scores.matrix) for scores in per_mouse.values())
    parts = ", ".join(f"{mouse} {len(scores.matrix)}" for mouse, scores in per_mouse.items())
    return f"W={total} sessions ({parts})"


def _ranking_panel(ax, pooled, per_mouse) -> tuple[str, str]:
    """Per-mouse Figure 25 panel A, plus each mouse's own mean as a small grey
    marker on the same row so between-mouse consistency is visible."""
    leader, runner_up = bbf._ranking_panel(ax, pooled)
    order = np.argsort(pooled.matrix.mean(axis=0))
    handles = []
    for style, (mouse, scores) in zip(MOUSE_STYLES, per_mouse.items()):
        means = scores.matrix.mean(axis=0)[order]
        ax.scatter(means, np.arange(len(order)), s=14, marker=style["marker"], facecolors="none",
                   edgecolors=style["color"], linewidths=0.9, zorder=2)
        handles.append(plt.Line2D([], [], marker=style["marker"], linestyle="", markerfacecolor="none",
                                  markeredgecolor=style["color"], label=f"{mouse} mean"))
    handles.insert(0, plt.Line2D([], [], marker="o", linestyle="", color="0.3", label="Pooled mean"))
    ax.legend(handles=handles, fontsize=6, loc="center right", frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_title("A  Observed held-out ranking (pooled; grey = each mouse)", loc="left", fontweight="bold", fontsize=9)
    return leader, runner_up


def _trajectory_panel(ax, per_mouse, leader, runner_up) -> None:
    first, second = MODEL_ORDER.index(leader), MODEL_ORDER.index(runner_up)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for style, (mouse, scores) in zip(MOUSE_STYLES, per_mouse.items()):
        difference = scores.matrix[:, second] - scores.matrix[:, first]
        ax.plot(scores.test_ranks, difference, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], ms=3, lw=1, label=f"{mouse} (mean {difference.mean():+.4f})")
    ax.set_ylabel("NLL(runner-up) - NLL(leader)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e6e6e6", linewidth=0.5)
    ax.margins(y=0.3)
    ax.legend(fontsize=6, ncol=len(per_mouse), frameon=False, loc="upper left")
    ax.set_title("C  Top-two session difference, per mouse chronology", loc="left", fontweight="bold", fontsize=9)


def _acf_panel(ax, per_mouse, leader, runner_up) -> None:
    first, second = MODEL_ORDER.index(leader), MODEL_ORDER.index(runner_up)
    width = 0.8 / len(per_mouse)
    for offset, (style, (mouse, scores)) in enumerate(zip(MOUSE_STYLES, per_mouse.items())):
        acf = bbf._acf(scores.matrix[:, second] - scores.matrix[:, first], 7)
        positions = np.arange(1, 8) + (offset - (len(per_mouse) - 1) / 2) * width
        ax.bar(positions, acf[1:], width=width, color=style["color"], label=mouse)
    bound = 1.96 / np.sqrt(min(len(scores.matrix) for scores in per_mouse.values()))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(bound, color="#777777", lw=0.7, ls="--")
    ax.axhline(-bound, color="#777777", lw=0.7, ls="--")
    ax.set_xticks(np.arange(1, 8))
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Lag (test sessions, within mouse)", fontsize=7)
    ax.set_ylabel("ACF", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.legend(fontsize=5.5, ncol=len(per_mouse), frameon=False, loc="upper right")
    ax.set_title("ACF diagnostic per mouse; dashed = +/-1.96/sqrt(smallest W)", loc="left", fontsize=7.5)


def _title(mice, n_models) -> str:
    return f"Pooled held-out comparison, {' + '.join(mice)} ({n_models} models)"


def _document_page(pooled, per_mouse, replicates, primary_length):
    n_models = len(pooled.model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    primary_results, _ = bbf._document_pair_results(
        pooled.matrix, pooled.model_ids, primary_length, replicates, pooled.strata,
    )
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(5, 2, width_ratios=(0.86, 1.34), height_ratios=(1.0, 0.72, 0.42, 0.72, 0.13))
    ranking_ax = figure.add_subplot(grid[0, 0])
    leader, runner_up = _ranking_panel(ranking_ax, pooled, per_mouse)
    top_two = bbf._oriented_result(bbf._result_lookup(primary_results), leader, runner_up)
    verdict = "superiority supported" if top_two.holm_p < 0.05 else "superiority not established"
    ranking_ax.text(
        0.99, 0.96, f"Top two: Holm p={top_two.holm_p:.3g} (raw p={top_two.raw_p:.3g}); {verdict}",
        transform=ranking_ax.transAxes, ha="right", va="top", fontsize=6.3,
    )
    bbf._pair_matrix_panel(
        figure.add_subplot(grid[:4, 1]), pooled.model_ids, primary_results,
        f"B  Guide method: all {n_pairs} paired comparisons", label_fontsize=4.0,
    )
    trajectory_ax = figure.add_subplot(grid[1, 0])
    _trajectory_panel(trajectory_ax, per_mouse, leader, runner_up)
    trajectory_ax.set_xlabel("Chronological test rank within each mouse", fontsize=8)
    _acf_panel(figure.add_subplot(grid[2, 0]), per_mouse, leader, runner_up)
    bbf._document_sensitivity_panel(figure.add_subplot(grid[3, 0]), pooled, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[4, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"{_session_text(per_mouse)}; {n_models} models; B={replicates:,}; primary L={primary_length}; "
        f"equal-session weighting; moving blocks drawn within each mouse (never across mice); "
        f"pointwise percentile 95% CIs; uncentred 2x-min tails; Holm family={n_pairs}.\n"
        "Edge shift = exact expected bootstrap effect minus observed effect. Inference is conditional on these mice "
        "(between-mouse variability is not resampled); no model was retrained.",
        ha="center", va="center", fontsize=7.2, wrap=True,
    )
    figure.suptitle(_title(tuple(per_mouse), n_models) + " - paired moving-block bootstrap (guide)",
                    fontsize=12, fontweight="bold")
    return figure, primary_results


def _hybrid_page(pooled, per_mouse, replicates, primary_length):
    n_models = len(pooled.model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    order = np.argsort(pooled.matrix.mean(axis=0))
    leader, runner_up = pooled.model_ids[order[0]], pooled.model_ids[order[1]]
    results, _ = bbf._hybrid_pair_results(
        pooled.matrix, pooled.model_ids, primary_length, replicates, pooled.strata,
    )
    sampled = bbf._sampled_means(pooled.matrix, bbf.EQUIVALENCE_BLOCK_LENGTH, replicates, pooled.strata)
    equivalence = bbf._equivalence_rows(pooled, sampled, leader, replicates)
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(4, 2, width_ratios=(1.3, 0.7), height_ratios=(1.0, 0.92, 0.72, 0.12))
    bbf._pair_matrix_panel(
        figure.add_subplot(grid[:3, 0]), pooled.model_ids, results,
        f"A  Hybrid null-centred test: all {n_pairs} model pairs", label_fontsize=3.8,
    )
    bbf._reference_forest_panel(figure.add_subplot(grid[0, 1]), pooled, results, leader)
    bbf._equivalence_panel(figure.add_subplot(grid[1, 1]), equivalence, leader)
    bbf._hybrid_sensitivity_panel(figure.add_subplot(grid[2, 1]), pooled, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[3, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"{_session_text(per_mouse)}; equal-session weighting; {n_models} models; B={replicates:,}; "
        f"pairwise primary L={primary_length}; moving blocks drawn within each mouse; basic 95% CIs; "
        f"two-sided null-centred tails; Holm family={n_pairs}.\n"
        f"TOST L={bbf.EQUIVALENCE_BLOCK_LENGTH}, margin +/-{bbf.EQUIVALENCE_MARGIN}, 90% basic CIs, "
        f"Holm family={n_models - 1} post-selection leader contrasts. Conditional on these mice; exploratory.",
        ha="center", va="center", fontsize=7.0, wrap=True,
    )
    figure.suptitle(_title(tuple(per_mouse), n_models) + " - hybrid centred bootstrap and practical equivalence",
                    fontsize=12, fontweight="bold")
    return figure, results, equivalence


def _pair_table(results, method: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"method": method, "row_model": row.first, "column_model": row.second,
         "row_advantage_nll": row.observed, "ci95_low": row.ci95_low, "ci95_high": row.ci95_high,
         "raw_p": row.raw_p, "holm_p": row.holm_p, "edge_shift": row.edge_shift}
        for row in results
    ])


def generate(mice: tuple[str, ...], replicates: int) -> Path:
    pooled, per_mouse = load_pooled_scores(mice)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / OUTPUT_NAME
    pending = OUTPUT_DIR / f".{OUTPUT_NAME}.pending"
    style = {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    pending.unlink(missing_ok=True)
    try:
        with plt.rc_context(style), PdfPages(pending, metadata={"Title": f"Pooled block-bootstrap model comparison ({', '.join(mice)})"}) as pdf:
            document, document_results = _document_page(pooled, per_mouse, replicates, bbf.DEFAULT_PRIMARY_LENGTH)
            pdf.savefig(document)
            plt.close(document)
            hybrid, hybrid_results, equivalence = _hybrid_page(pooled, per_mouse, replicates, bbf.DEFAULT_PRIMARY_LENGTH)
            pdf.savefig(hybrid)
            plt.close(hybrid)
        pending.replace(output)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    pd.concat([_pair_table(document_results, "guide"), _pair_table(hybrid_results, "hybrid")]).to_csv(
        OUTPUT_DIR / "fig25_pooled_pairwise.csv", index=False)
    pd.DataFrame(equivalence).to_csv(OUTPUT_DIR / "fig25_pooled_equivalence.csv", index=False)
    means = pd.DataFrame(
        {mouse: scores.matrix.mean(axis=0) for mouse, scores in per_mouse.items()}, index=list(MODEL_ORDER),
    )
    means.insert(0, "pooled_equal_session", pooled.matrix.mean(axis=0))
    means.insert(0, "model_label", [MODEL_BY_ID[m].label for m in MODEL_ORDER])
    means.sort_values("pooled_equal_session").to_csv(OUTPUT_DIR / "fig25_pooled_mean_test_nll.csv", index_label="model_id")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--replicates", type=int, default=bbf.DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(tuple(validate_mouse_id(mouse) for mouse in args.mice), args.replicates))


if __name__ == "__main__":
    main()
