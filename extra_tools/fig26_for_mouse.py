#!/usr/bin/env python3
"""Figure 26 (event-conditioned pairwise bootstrap) for any mouse and any set
of eligible expanded models. Reuses post_analysis/expanded_model_comparison/
figure26_event_pairwise.py unchanged except that, in-process, the model set is
the mouse's eligible models and the page texts that hard-code "MC08", "30
windows", "13 models" and "78 pairs" are computed instead.

  python extra_tools/fig26_for_mouse.py --mouse MC07
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
import pandas as pd  # noqa: E402

from post_analysis.expanded_model_comparison import figure26_event_pairwise as f26  # noqa: E402
from post_analysis.expanded_model_comparison.config import MODEL_ORDER  # noqa: E402

CONTEXT: dict = {}


def _domain_page(domain, matrix, model_ids, event_counts, replicates, events):
    n_windows = len(matrix)
    window_ids = sorted(int(w) for w in matrix.index)
    if window_ids != list(range(window_ids[0], window_ids[0] + n_windows)):
        raise ValueError(f"Figure 26 {domain}: usable windows have gaps ({window_ids})")
    n_models = len(model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    block_length = min(f26.DEFAULT_PRIMARY_LENGTH, max(1, n_windows - 1))
    results, sampled = f26._document_pair_results(matrix.to_numpy(float), model_ids, block_length, replicates)
    figure = plt.figure(figsize=(19.5, 12.4), layout="constrained")
    grid = figure.add_gridspec(3, 2, width_ratios=(0.85, 1.35), height_ratios=(0.09, 1.0, 0.14))
    banner_ax = figure.add_subplot(grid[0, :])
    banner_ax.axis("off")
    if domain == "combined":
        banner_text, banner_color = "PRIMARY EVENT-CONDITIONED COMPARISON — pools all four event types below", "#1b5e20"
    else:
        banner_text = "DESCRIPTIVE BREAKDOWN of the combined (primary) page — not an independently confirmatory test"
        banner_color = "#8a6d00"
    banner_ax.text(0.5, 0.5, banner_text, ha="center", va="center", fontsize=11, fontweight="bold", color="white",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=banner_color, edgecolor="none"))
    f26._event_ranking_panel(figure.add_subplot(grid[1, 0]), model_ids, matrix, sampled)
    f26._pair_matrix_panel(figure.add_subplot(grid[1, 1]), model_ids, results,
                           f"B  {f26.DOMAIN_TITLES[domain]}: all {n_pairs} paired comparisons", label_fontsize=5.3)
    footer_ax = figure.add_subplot(grid[2, :])
    footer_ax.axis("off")
    lo, hi = int(event_counts.min()), int(event_counts.max())
    events_text = f"{lo}" if lo == hi else f"{lo}-{hi}"
    weighting_line = ""
    if domain == "combined":
        parts = {c: int(f26._domain_event_counts(events, c, model_ids, matrix.index).iloc[0])
                 for c in ("rare_short", "rare_long", "short_to_long", "long_to_short")}
        weighting_line = ("Component event counts feeding this pooled average: "
                          + ", ".join(f"{k}={v}" for k, v in parts.items()) + f" (of {events_text} total).")
    mouse, total = CONTEXT["mouse"], CONTEXT["total_windows"]
    prov = (f" PROVISIONAL: {n_models} of {CONTEXT['n_configured']} configured models." if n_models < CONTEXT["n_configured"] else "")
    footer_ax.text(
        0.5, 0.5,
        f"Single mouse ({mouse}).{prov} Event window: {f26.DOMAIN_WINDOW_DESCRIPTIONS[domain]}; neighbor trials "
        "included regardless of their own trial type. Each model's per-window score is an event-count-weighted mean "
        "NLL over all qualifying event instances in that window (n_events-weighted, held-out test split only).\n"
        f"n={n_windows} of {total} chronological sliding windows have a qualifying event for every one of the {n_models} "
        f"models; per-model total qualifying event count across those windows: {events_text} (identical across "
        f"models -- all {n_models} models are evaluated on the same held-out trials). " + weighting_line + "\n"
        f"Moving-block bootstrap as in Figure 25's guide method, resampling across these {n_windows} chronological "
        f"windows within this one mouse (block length {block_length}; {replicates:,} resamples): uncentred "
        f"two-minimum bootstrap-tail p-values, Holm-adjusted across the {n_pairs} pairs shown here. Exploratory, "
        "within-mouse result -- not an independent replication.",
        ha="center", va="center", fontsize=8.6, linespacing=1.4, wrap=True,
    )
    figure.suptitle(f"Figure 26 — Event-conditioned pairwise bootstrap comparison ({mouse})\n{f26.DOMAIN_TITLES[domain]}",
                    fontsize=14, fontweight="bold")
    return figure


def _insufficient_data_page(domain, n_windows):
    figure = plt.figure(figsize=(19.5, 12.4))
    figure.text(0.5, 0.5,
                f"Figure 26 — {f26.DOMAIN_TITLES[domain]}\n\nOnly {n_windows} of {CONTEXT['total_windows']} chronological "
                f"windows have a qualifying event for every one of the {CONTEXT['n_models']} models "
                f"(minimum required: {f26.MIN_USABLE_WINDOWS}).\nThe moving-block bootstrap is not reported for this "
                "event definition on this mouse.", ha="center", va="center", fontsize=12)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", required=True)
    parser.add_argument("--replicates", type=int, default=f26.DEFAULT_REPLICATES)
    args = parser.parse_args()
    from run_expanded_for_mouse import apply_mouse_patches
    apply_mouse_patches(args.mouse)
    tables = f26.ANALYSIS_ROOT / args.mouse / "tables"
    status = pd.read_csv(tables / "model_completion_status.csv")
    flag = status["eligible_for_comparison"].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    eligible = tuple(m for m in MODEL_ORDER if m in set(status.loc[flag, "model_id"].astype(str)))
    metrics = pd.read_csv(tables / "sliding_window_metrics.csv")
    CONTEXT.update(mouse=args.mouse, n_models=len(eligible), n_configured=len(MODEL_ORDER),
                   total_windows=int(metrics.loc[metrics["split_role"].eq("test"), "window_id"].nunique()))
    f26.MODEL_ORDER = eligible
    f26._domain_page = _domain_page
    f26._insufficient_data_page = _insufficient_data_page
    for path in f26.figure26_report(args.mouse, args.replicates):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
