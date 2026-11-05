#!/usr/bin/env python3
"""Figure 25 (two-page block-bootstrap model comparison) for any mouse and any
set of eligible expanded models. A parametrized sibling of
post_analysis/expanded_model_comparison/block_bootstrap_figures.py, which is
hard-wired to the complete 13-model, 30-window MC08 comparison and is left
untouched; the statistics and panels are reused from it.

  python extra_tools/fig25_for_mouse.py --mouse MC07
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
from post_analysis.expanded_model_comparison.config import MODEL_ORDER  # noqa: E402

OUTPUT_NAME = "fig25_block_bootstrap_document_and_hybrid.pdf"


def load_scores(mouse: str) -> tuple[bbf.LoadedScores, int]:
    tables = bbf.ANALYSIS_ROOT / mouse / "tables"
    status = pd.read_csv(tables / "model_completion_status.csv")
    flag = status["eligible_for_comparison"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    eligible = tuple(m for m in MODEL_ORDER if m in set(status.loc[flag, "model_id"].astype(str)))
    if len(eligible) < 3:
        raise ValueError(f"{mouse}: fewer than 3 eligible models")
    metrics = pd.read_csv(tables / "sliding_window_metrics.csv")
    test = metrics[metrics["split_role"].eq("test") & metrics["model_id"].isin(eligible)].copy()
    if test.duplicated(["window_id", "model_id"]).any():
        raise ValueError("duplicate model/window test rows")
    wide = test.pivot(index="window_id", columns="model_id", values="mean_nll").reindex(columns=list(eligible))
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("incomplete score matrix across the eligible models")
    counts = test.groupby("window_id")["valid_trial_count"].nunique()
    if not counts.eq(1).all():
        raise ValueError("valid trial counts differ across models within a window")
    info = test.drop_duplicates("window_id").set_index("window_id").reindex(wide.index)
    scores = bbf.LoadedScores(
        matrix=wide.to_numpy(dtype=float),
        model_ids=eligible,
        test_ranks=info["test_rank"].to_numpy(dtype=int),
        trial_counts=info["valid_trial_count"].to_numpy(dtype=int),
    )
    return scores, len(MODEL_ORDER)


def _tag(mouse: str, n_models: int, n_all: int, n_windows: int) -> str:
    base = f"{mouse} held-out comparison ({n_models} models, {n_windows} windows)"
    return base + (f" - PROVISIONAL: {n_models} of {n_all} configured models" if n_models < n_all else "")


def _document_page(mouse, scores, n_all, replicates, primary_length):
    n_models, n_windows = len(scores.model_ids), len(scores.matrix)
    n_pairs = n_models * (n_models - 1) // 2
    primary_results, _ = bbf._document_pair_results(scores.matrix, scores.model_ids, primary_length, replicates)
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(5, 2, width_ratios=(0.86, 1.34), height_ratios=(1.0, 0.72, 0.42, 0.72, 0.13))
    ranking_ax = figure.add_subplot(grid[0, 0])
    leader, runner_up = bbf._ranking_panel(ranking_ax, scores)
    top_two = bbf._oriented_result(bbf._result_lookup(primary_results), leader, runner_up)
    ranking_ax.text(
        0.99, 0.96,
        f"Top two: Holm p={top_two.holm_p:.3g} (raw p={top_two.raw_p:.3g}); superiority not established",
        transform=ranking_ax.transAxes, ha="right", va="top", fontsize=6.3,
    )
    bbf._pair_matrix_panel(
        figure.add_subplot(grid[:4, 1]), scores.model_ids, primary_results,
        f"B  Guide method: all {n_pairs} paired comparisons", label_fontsize=4.0,
    )
    trajectory_ax = figure.add_subplot(grid[1, 0])
    bbf._trajectory_panel(trajectory_ax, scores, leader, runner_up)
    trajectory_ax.set_xlabel(
        f"Chronological test rank {scores.test_ranks.min()}-{scores.test_ranks.max()} ({n_windows} test sessions)",
        fontsize=8,
    )
    bbf._acf_panel(figure.add_subplot(grid[2, 0]), scores, leader, runner_up)
    bbf._document_sensitivity_panel(figure.add_subplot(grid[3, 0]), scores, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[4, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W={n_windows} sessions; {n_models} models; B={replicates:,}; primary L={primary_length}; "
        f"equal-session weighting; pointwise percentile 95% CIs; uncentred 2x-min tails; Holm family={n_pairs}.\n"
        "Edge shift = exact expected bootstrap effect minus observed effect. "
        "Within-mouse exploratory inference; no model was retrained.",
        ha="center", va="center", fontsize=7.2, wrap=True,
    )
    figure.suptitle(_tag(mouse, n_models, n_all, n_windows) + " - paired moving-block bootstrap (guide)",
                    fontsize=12, fontweight="bold")
    return figure


def _hybrid_page(mouse, scores, n_all, replicates, primary_length):
    n_models, n_windows = len(scores.model_ids), len(scores.matrix)
    n_pairs = n_models * (n_models - 1) // 2
    order = np.argsort(scores.matrix.mean(axis=0))
    leader, runner_up = scores.model_ids[order[0]], scores.model_ids[order[1]]
    results, _ = bbf._hybrid_pair_results(scores.matrix, scores.model_ids, primary_length, replicates)
    sampled = bbf._sampled_means(scores.matrix, bbf.EQUIVALENCE_BLOCK_LENGTH, replicates)
    equivalence = bbf._equivalence_rows(scores, sampled, leader, replicates)
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(4, 2, width_ratios=(1.3, 0.7), height_ratios=(1.0, 0.92, 0.72, 0.12))
    bbf._pair_matrix_panel(
        figure.add_subplot(grid[:3, 0]), scores.model_ids, results,
        f"A  Hybrid null-centred test: all {n_pairs} model pairs", label_fontsize=3.8,
    )
    bbf._reference_forest_panel(figure.add_subplot(grid[0, 1]), scores, results, leader)
    bbf._equivalence_panel(figure.add_subplot(grid[1, 1]), equivalence, leader)
    bbf._hybrid_sensitivity_panel(figure.add_subplot(grid[2, 1]), scores, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[3, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W={n_windows} sessions; equal-session weighting; {n_models} models; B={replicates:,}; "
        f"pairwise primary L={primary_length}; basic 95% CIs; two-sided null-centred tails; Holm family={n_pairs}.\n"
        f"TOST L={bbf.EQUIVALENCE_BLOCK_LENGTH}, margin +/-0.005, 90% basic CIs, "
        f"Holm family={n_models - 1} post-selection leader contrasts. Within-mouse exploratory only.",
        ha="center", va="center", fontsize=7.0, wrap=True,
    )
    figure.suptitle(_tag(mouse, n_models, n_all, n_windows) + " - hybrid centred bootstrap and practical equivalence",
                    fontsize=12, fontweight="bold")
    return figure


def generate(mouse: str, replicates: int) -> Path:
    from run_expanded_for_mouse import apply_mouse_patches
    apply_mouse_patches(mouse)
    scores, n_all = load_scores(mouse)
    figures_dir = bbf.ANALYSIS_ROOT / mouse / "figures"
    output = figures_dir / OUTPUT_NAME
    pending = figures_dir / f".{OUTPUT_NAME}.pending"
    style = {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    pending.unlink(missing_ok=True)
    try:
        with plt.rc_context(style), PdfPages(pending, metadata={"Title": f"{mouse} block-bootstrap model comparison"}) as pdf:
            for page in (
                _document_page(mouse, scores, n_all, replicates, bbf.DEFAULT_PRIMARY_LENGTH),
                _hybrid_page(mouse, scores, n_all, replicates, bbf.DEFAULT_PRIMARY_LENGTH),
            ):
                pdf.savefig(page)
                plt.close(page)
        pending.replace(output)
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", required=True)
    parser.add_argument("--replicates", type=int, default=bbf.DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(args.mouse, args.replicates))


if __name__ == "__main__":
    main()
