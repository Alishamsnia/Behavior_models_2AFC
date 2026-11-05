"""Supplementary MC08 block-bootstrap PDF: the 13 configured expanded models plus M3 and M4.

This is a *separate* output from ``block_bootstrap_figures`` (Figure 25) and
leaves that module, the frozen model registry and the existing PDF untouched.

M4 has no accepted fit for the final rolling window (window 30, test rank 42:
no start passed the convergence gate), so this comparison uses only the 29
windows (test ranks 13-41) that every one of the 15 models completed.  The 13
existing models take their per-window test NLL from the audited expanded
comparison table; M3 and M4 are read from their own ``results_sliding_window``
window folders (the ALL_SESSIONS test row, which is what the audited table
holds for the other models).  Nothing is fitted or re-fitted.

The statistics and panels are reused from ``block_bootstrap_figures``; only the
model registry (in-process, restored afterwards) and the hardcoded
13-model / 30-window annotations differ.

Run from the repository root:

    python -m post_analysis.expanded_model_comparison.block_bootstrap_figures_m3m4
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from post_analysis.common import RESULTS_SLIDING_ROOT
from post_analysis.expanded_model_comparison import block_bootstrap_figures as bbf
from post_analysis.expanded_model_comparison.config import (
    MODEL_ORDER, WORKFLOW_DIRECTORY, ExpandedModelSpec,
)

OUTPUT_NAME = "fig25_block_bootstrap_15models_29windows.pdf"
EXTRA_SPECS = (
    ExpandedModelSpec(
        "m3_hmm_delta", "M3_hmm_delta_v1",
        "sensory_deltaLearnRatio_learnStaticBlockSwitch_lapse",
        "sensory_deltaLearnRatio_learnStaticBlockSwitch_lapse", "M3", "m3",
        ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta"),
        "#EE7733",
    ),
    ExpandedModelSpec(
        "m4_hsmm_delta", "M4_hsmm_delta_v1",
        "sensory_deltaLearnRatio_learnDurationBlockSwitch_lapse",
        "sensory_deltaLearnRatio_learnDurationBlockSwitch_lapse", "M4", "m4",
        ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta", "bh"),
        "#009988",
    ),
)
ALL_IDS = tuple(MODEL_ORDER) + tuple(spec.model_id for spec in EXTRA_SPECS)
EXCLUDED_WINDOW = 30  # M4 has no accepted winner here


@contextmanager
def _extended_registry():
    """Temporarily let the reused panels resolve the M3/M4 specs."""
    added = {spec.model_id: spec for spec in EXTRA_SPECS}
    bbf.MODEL_BY_ID.update(added)
    try:
        yield
    finally:
        for key in added:
            bbf.MODEL_BY_ID.pop(key, None)


def _extra_model_tests(spec: ExpandedModelSpec, mouse_id: str) -> pd.DataFrame:
    rows = []
    root = RESULTS_SLIDING_ROOT / spec.directory / mouse_id / WORKFLOW_DIRECTORY
    for window_dir in sorted(root.glob("window_*")):
        window_id = int(window_dir.name.split("_")[1])
        manifest = json.loads((window_dir / "window_manifest.json").read_text(encoding="utf-8"))
        metrics_path = window_dir / "split_and_session_metrics.csv"
        if manifest.get("status") != "complete" or not metrics_path.is_file():
            if window_id == EXCLUDED_WINDOW and spec.model_id == "m4_hsmm_delta":
                continue
            raise ValueError(f"{spec.model_id} window {window_id} is not complete")
        metrics = pd.read_csv(metrics_path)
        test = metrics[(metrics["split_role"] == "test") & (metrics["session_id"] == "ALL_SESSIONS")]
        if len(test) != 1:
            raise ValueError(f"{spec.model_id} window {window_id}: expected one ALL_SESSIONS test row")
        rows.append({
            "model_id": spec.model_id, "window_id": window_id,
            "valid_trial_count": int(test["valid_trial_count"].iloc[0]),
            "mean_nll": float(test["mean_nll"].iloc[0]),
        })
    return pd.DataFrame(rows)


def load_scores(mouse_id: str) -> bbf.LoadedScores:
    # Reuse the strict audit of the 13 existing models (30 windows, snapshots, ...).
    base = bbf._load_scores(mouse_id)
    tables = bbf.ANALYSIS_ROOT / mouse_id / "tables" / "sliding_window_metrics.csv"
    metrics = pd.read_csv(tables)
    test = metrics[metrics["split_role"].eq("test") & metrics["model_id"].isin(MODEL_ORDER)]
    ranks = test.drop_duplicates("window_id").set_index("window_id")["test_rank"]

    wide = pd.DataFrame(base.matrix, index=range(1, 31), columns=base.model_ids)
    counts = pd.Series(base.trial_counts, index=range(1, 31))
    for spec in EXTRA_SPECS:
        extra = _extra_model_tests(spec, mouse_id).set_index("window_id")
        if (extra["valid_trial_count"] != counts.reindex(extra.index)).any():
            raise ValueError(f"{spec.model_id} trial counts differ from the existing models")
        wide[spec.model_id] = extra["mean_nll"].reindex(wide.index)

    wide = wide[list(ALL_IDS)].drop(index=EXCLUDED_WINDOW)
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("incomplete 15-model score matrix after excluding the common-window gap")
    windows = wide.index.to_numpy()
    if len(windows) != 29:
        raise ValueError("expected 29 common windows")
    return bbf.LoadedScores(
        matrix=wide.to_numpy(dtype=float),
        model_ids=ALL_IDS,
        test_ranks=ranks.reindex(windows).to_numpy(dtype=int),
        trial_counts=counts.reindex(windows).to_numpy(dtype=int),
    )


def _document_page(scores: bbf.LoadedScores, replicates: int, primary_length: int) -> plt.Figure:
    n_models, n_windows = len(scores.model_ids), len(scores.matrix)
    n_pairs = n_models * (n_models - 1) // 2
    primary_results, _ = bbf._document_pair_results(
        scores.matrix, scores.model_ids, primary_length, replicates,
    )
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(
        5, 2, width_ratios=(0.86, 1.34), height_ratios=(1.0, 0.72, 0.42, 0.72, 0.13),
    )
    ranking_ax = figure.add_subplot(grid[0, 0])
    leader, runner_up = bbf._ranking_panel(ranking_ax, scores)
    top_two = bbf._oriented_result(bbf._result_lookup(primary_results), leader, runner_up)
    ranking_ax.text(
        0.99, 0.96,
        f"Top two: Holm p={top_two.holm_p:.3g} (raw p={top_two.raw_p:.3g}); superiority not established",
        transform=ranking_ax.transAxes, ha="right", va="top", fontsize=6.3,
    )
    matrix_ax = figure.add_subplot(grid[:4, 1])
    bbf._pair_matrix_panel(
        matrix_ax, scores.model_ids, primary_results,
        f"B  Guide method: all {n_pairs} paired comparisons", label_fontsize=4.0,
    )
    trajectory_ax = figure.add_subplot(grid[1, 0])
    bbf._trajectory_panel(trajectory_ax, scores, leader, runner_up)
    trajectory_ax.set_xlabel(
        f"Chronological test rank {scores.test_ranks.min()}-{scores.test_ranks.max()} "
        f"({n_windows} test sessions)", fontsize=8,
    )
    acf_ax = figure.add_subplot(grid[2, 0])
    bbf._acf_panel(acf_ax, scores, leader, runner_up)
    sensitivity_ax = figure.add_subplot(grid[3, 0])
    bbf._document_sensitivity_panel(sensitivity_ax, scores, leader, runner_up, replicates)
    footer_ax = figure.add_subplot(grid[4, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W={n_windows} sessions (window 30 dropped: M4 has no accepted fit); {n_models} models; "
        f"B={replicates:,}; primary L={primary_length}; equal-session weighting; "
        f"pointwise percentile 95% CIs; uncentred 2x-min tails; Holm family={n_pairs}.\n"
        "Edge shift = exact expected bootstrap effect minus observed effect. "
        "Within-mouse exploratory inference; no model was retrained.",
        ha="center", va="center", fontsize=7.2, wrap=True,
    )
    figure.suptitle(
        "MC08 held-out comparison (15 models, 29 common windows) - paired moving-block bootstrap specified by the guide",
        fontsize=13, fontweight="bold",
    )
    return figure


def _hybrid_page(scores: bbf.LoadedScores, replicates: int, primary_length: int) -> plt.Figure:
    n_models, n_windows = len(scores.model_ids), len(scores.matrix)
    n_pairs = n_models * (n_models - 1) // 2
    order = np.argsort(scores.matrix.mean(axis=0))
    leader, runner_up = scores.model_ids[order[0]], scores.model_ids[order[1]]
    results, _ = bbf._hybrid_pair_results(
        scores.matrix, scores.model_ids, primary_length, replicates,
    )
    sampled = bbf._sampled_means(scores.matrix, bbf.EQUIVALENCE_BLOCK_LENGTH, replicates)
    equivalence = bbf._equivalence_rows(scores, sampled, leader, replicates)

    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(
        4, 2, width_ratios=(1.3, 0.7), height_ratios=(1.0, 0.92, 0.72, 0.12),
    )
    matrix_ax = figure.add_subplot(grid[:3, 0])
    bbf._pair_matrix_panel(
        matrix_ax, scores.model_ids, results,
        f"A  Hybrid null-centred test: all {n_pairs} model pairs", label_fontsize=3.8,
    )
    bbf._reference_forest_panel(figure.add_subplot(grid[0, 1]), scores, results, leader)
    bbf._equivalence_panel(figure.add_subplot(grid[1, 1]), equivalence, leader)
    bbf._hybrid_sensitivity_panel(
        figure.add_subplot(grid[2, 1]), scores, leader, runner_up, replicates,
    )
    footer_ax = figure.add_subplot(grid[3, :])
    footer_ax.axis("off")
    footer_ax.text(
        0.5, 0.5,
        f"W={n_windows} sessions (window 30 dropped: M4 has no accepted fit); equal-session weighting; "
        f"all {n_models} models including {sum(bbf.MODEL_BY_ID[m].oracle for m in scores.model_ids)} oracle benchmarks; B={replicates:,}; "
        f"pairwise primary L={primary_length}; basic 95% CIs; two-sided null-centred tails; Holm family={n_pairs}.\n"
        f"TOST L={bbf.EQUIVALENCE_BLOCK_LENGTH}, margin +/-0.005, 90% basic CIs, "
        f"Holm family={n_models - 1} post-selection leader contrasts. Within-mouse exploratory only.",
        ha="center", va="center", fontsize=7.0, wrap=True,
    )
    figure.suptitle(
        "MC08 held-out comparison (15 models, 29 common windows) - hybrid centred bootstrap and practical equivalence",
        fontsize=13, fontweight="bold",
    )
    return figure


def generate(mouse_id: str = "MC08", replicates: int = bbf.DEFAULT_REPLICATES) -> Path:
    if mouse_id != "MC08":
        raise ValueError("This figure is defined for the MC08 expanded comparison")
    if replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    scores = load_scores(mouse_id)
    figures_dir = bbf.ANALYSIS_ROOT / mouse_id / "figures"
    output = figures_dir / OUTPUT_NAME
    pending = figures_dir / f".{OUTPUT_NAME}.pending"
    style = {
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    }
    pending.unlink(missing_ok=True)
    try:
        with _extended_registry(), plt.rc_context(style), PdfPages(
            pending, metadata={
                "Title": "MC08 block-bootstrap model comparison: 15 models, 29 common windows",
                "Author": "Bayesian_model_v2 post-analysis",
            },
        ) as pdf:
            for page in (
                _document_page(scores, replicates, bbf.DEFAULT_PRIMARY_LENGTH),
                _hybrid_page(scores, replicates, bbf.DEFAULT_PRIMARY_LENGTH),
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
    parser.add_argument("--mouse", default="MC08")
    parser.add_argument("--replicates", type=int, default=bbf.DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(args.mouse, args.replicates))


if __name__ == "__main__":
    main()
