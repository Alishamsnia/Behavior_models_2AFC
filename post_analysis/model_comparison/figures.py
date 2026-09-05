"""Paper-quality figures for the M0-M4 model-comparison analysis.

Reads the tidy CSV tables written by ``collect.py`` and writes PNG (300 dpi)
+ PDF (vector) figures. Every figure uses the same fixed per-model color
(``post_analysis.common.MODEL_SPECS``) so a color always means the same
model across the whole report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Callable, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from post_analysis import common
from post_analysis.common import MODEL_ORDER, MODEL_SPECS, SHARED_PARAM_NAMES
from post_analysis.model_comparison import stats as mc_stats
from post_analysis.model_comparison.collect import (
    common_metric_window_ids,
    common_session_ids,
    common_event_session_ids,
    common_event_window_ids,
    common_trial_subset_session_ids,
    common_trial_subset_window_ids,
    event_aligned_unit_scalar,
    BLOCK_TRANSITION_PERFORMANCE_COLUMNS,
    EVENT_ALIGNED_SPLIT_LABELS,
    EVENT_ALIGNED_SPLIT_ROLES,
    EVENT_RELATIVE_POSITIONS,
    PSYCHOMETRIC_BLOCK_LABELS,
    PSYCHOMETRIC_BLOCK_ORDER,
    PSYCHOMETRIC_SOURCE_LABELS,
    TRIAL_SUBSET_LABELS,
    TRIAL_SUBSET_ORDER,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10.5,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": False,
    "pdf.fonttype": 42,
})

MODEL_LABELS = {m: MODEL_SPECS[m].label for m in MODEL_ORDER}
MODEL_COLORS = {m: MODEL_SPECS[m].color for m in MODEL_ORDER}
MODEL_LINESTYLES = {"M0": "-", "M1": "-", "M2": "--", "M3": ":", "M4": "-."}
MODEL_MARKERS = {"M0": "o", "M1": "s", "M2": "^", "M3": "D", "M4": "v"}

# Model-specific (non-shared) parameters, in nesting order, with the models
# that actually fit each one. w/s0/ell_short/ell_long/kappa are shared by
# every model (SHARED_PARAM_NAMES); these are the coordinates each nested
# model adds on top.
NON_SHARED_PARAM_MODELS = {
    "ah": ["M1", "M2", "M3", "M4"],
    "delta0": ["M2", "M3", "M4"],
    "alpha_theta": ["M3", "M4"],
    "bh": ["M4"],
}

MODEL_SPECIFIC_PARAMETER_CARDS = {
    "ah": (
        "Switch hazard and learned block duration",
        "M1–M3: hB = sigmoid(ah); expected biased-block length = 1/hB trials.\n"
        "M4: hB(23) = sigmoid(ah). Because bh changes hazard with elapsed run length, "
        "M4 has no single constant mean block length.",
    ),
    "delta0": (
        "Initial subjective category-rate contrast",
        "At session start: θS = sigmoid(−delta0), θL = sigmoid(delta0).\n"
        "Larger delta0 means stronger expected short/long category bias in the corresponding block. "
        "M2 keeps it fixed; M3/M4 can update it within session.",
    ),
    "alpha_theta": (
        "Within-session belief-learning rate",
        "delta(t+1) = max[10⁻⁶, delta(t) + alpha_theta × feedback signal].\n"
        "Larger values produce faster updating of subjective category rates. A fitted value near "
        "zero is the practical no-learning boundary; positivity alone is not evidence of learning.",
    ),
    "bh": (
        "Run-length dependence of switch hazard (M4)",
        "hB(r) = sigmoid[ah + bh × (r−23)], with r capped at the R≥65 tail.\n"
        "Positive bh makes a switch progressively more likely as a block runs longer; bh=0 is the "
        "exact constant-hazard M3 boundary, so positivity alone is not evidence for M4.",
    ),
}


def _save(fig: plt.Figure, figures_dir, stem: str) -> None:
    """Stage both formats before replacing the current figure artifacts."""
    figures_dir = Path(figures_dir)
    temporary_paths: list[Path] = []
    try:
        for suffix in (".png", ".pdf"):
            with tempfile.NamedTemporaryFile(
                dir=figures_dir, prefix=f".{stem}.", suffix=suffix, delete=False,
            ) as temporary_file:
                temporary_paths.append(Path(temporary_file.name))

        fig.savefig(temporary_paths[0], bbox_inches="tight")
        fig.savefig(temporary_paths[1], bbox_inches="tight")
        os.replace(temporary_paths[0], figures_dir / f"{stem}.png")
        os.replace(temporary_paths[1], figures_dir / f"{stem}.pdf")
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        plt.close(fig)


def _save_multi_page(figs: list, figures_dir, stem: str, page_names: list[str]) -> None:
    """Save several already-built figures (one per page, in order) as a
    single multi-page PDF, one flat PNG per page (``{stem}_{page_names[i]}.png``),
    and a plain ``{stem}.png`` mirroring page 1 so anything already pointing
    at the original single-page filename keeps working unchanged.
    """
    figures_dir = Path(figures_dir)
    temporary_paths: list[Path] = []
    pending: list[tuple[Path, Path]] = []
    try:
        with tempfile.NamedTemporaryFile(
            dir=figures_dir, prefix=f".{stem}.", suffix=".pdf", delete=False,
        ) as temporary_file:
            pdf_temp_path = Path(temporary_file.name)
        temporary_paths.append(pdf_temp_path)
        with PdfPages(pdf_temp_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        pending.append((pdf_temp_path, figures_dir / f"{stem}.pdf"))

        for index, (fig, page_name) in enumerate(zip(figs, page_names)):
            with tempfile.NamedTemporaryFile(
                dir=figures_dir, prefix=f".{stem}.", suffix=".png", delete=False,
            ) as temporary_file:
                page_temp_path = Path(temporary_file.name)
            temporary_paths.append(page_temp_path)
            fig.savefig(page_temp_path, bbox_inches="tight")
            pending.append((page_temp_path, figures_dir / f"{stem}_{page_name}.png"))
            if index == 0:
                with tempfile.NamedTemporaryFile(
                    dir=figures_dir, prefix=f".{stem}.", suffix=".png", delete=False,
                ) as temporary_file:
                    plain_temp_path = Path(temporary_file.name)
                temporary_paths.append(plain_temp_path)
                fig.savefig(plain_temp_path, bbox_inches="tight")
                pending.append((plain_temp_path, figures_dir / f"{stem}.png"))

        for temporary_path, final_path in pending:
            os.replace(temporary_path, final_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        for fig in figs:
            plt.close(fig)


def _save_pdf_pages(figs: list, figures_dir, stem: str) -> None:
    """Save several already-built figures (one per page, in order) as a
    single multi-page PDF only -- no PNGs. For figures where a flat PNG
    per page isn't needed (see ``_save_multi_page`` for the PDF+PNGs form).
    """
    figures_dir = Path(figures_dir)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=figures_dir, prefix=f".{stem}.", suffix=".pdf", delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with PdfPages(temporary_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        os.replace(temporary_path, figures_dir / f"{stem}.pdf")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        for fig in figs:
            plt.close(fig)


def _model_x_positions() -> dict:
    return {m: i for i, m in enumerate(MODEL_ORDER)}


def _wrap(text: str, width: int = 40) -> str:
    """Pre-wrap caption text to a fixed character width and return it with
    explicit newlines. matplotlib's own ``ax.text(..., wrap=True)`` wraps to
    the FIGURE's width, not the local Axes' width, so on a multi-panel
    figure it lets one panel's caption bleed visually into its neighbor
    once panels get narrow (observed on fig01's 3-panel layout) -- wrapping
    the string ourselves before handing it to ``ax.text`` avoids that.
    """
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _format_adjusted_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def _paired_effect_summary(row: pd.Series) -> tuple[str, float]:
    """Return the descriptively lower-NLL model from the paired median effect."""
    difference = float(row["median_diff_a_minus_b"])
    if not np.isfinite(difference) or difference == 0:
        return "tie", difference
    lower_model = row["model_b"] if difference > 0 else row["model_a"]
    return str(lower_model), difference


def _draw_adjacent_significance(
    ax: plt.Axes, pairwise: pd.DataFrame, filters: dict[str, str],
    *, show_effect: bool = False, fontsize: float = 7.0,
) -> int:
    """Draw four adjacent nested-model brackets using Holm-adjusted p-values."""
    rows = pairwise.copy()
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    if rows.empty:
        return 0

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-9)
    base = ymax + 0.035 * span
    step = 0.105 * span
    height = 0.018 * span
    top = ymax
    xpos = _model_x_positions()
    for index, (model_a, model_b) in enumerate(mc_stats.ADJACENT_MODEL_PAIRS):
        match = rows[(rows["model_a"] == model_a) & (rows["model_b"] == model_b)]
        if match.empty:
            continue
        row = match.iloc[0]
        y = base + (index % 2) * step
        x1, x2 = xpos[model_a], xpos[model_b]
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
                color="0.2", linewidth=0.8, clip_on=False)
        p = float(row["wilcoxon_p_holm"])
        formatted_p = _format_adjusted_p(p)
        separator = "" if formatted_p.startswith("<") else "="
        lower_model, paired_effect = _paired_effect_summary(row)
        direction = "paired tie" if lower_model == "tie" else f"{lower_model}↓"
        label_lines = [f"{direction} {mc_stats.significance_stars(p)}"]
        if show_effect:
            label_lines.append(f"|Δmed|={abs(paired_effect):.3g}")
        label_lines.append(f"pH{separator}{formatted_p}")
        label = "\n".join(label_lines)
        ax.text((x1 + x2) / 2, y + height + 0.008 * span, label,
                ha="center", va="bottom", fontsize=fontsize, linespacing=0.9)
        top = max(top, y + height + 0.055 * span)
    ax.set_ylim(ymin, top)
    return int(rows["n_pairs"].iloc[0])


def _draw_adjacent_summary_table(
    ax: plt.Axes, pairwise: pd.DataFrame, filters: dict[str, str],
    *, fontsize: float = 8.8,
) -> int:
    """Place four compact paired-test rows below a narrow model panel.

    Figure 16 has four side-by-side summary panels, so bracket labels do not
    have enough horizontal room once paired effect sizes are included.
    """
    rows = pairwise.copy()
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    if rows.empty:
        return 0
    lines = ["pair: lower   |Δmed|   pH"]
    for model_a, model_b in mc_stats.ADJACENT_MODEL_PAIRS:
        match = rows[(rows["model_a"] == model_a) & (rows["model_b"] == model_b)]
        if match.empty:
            continue
        row = match.iloc[0]
        p = float(row["wilcoxon_p_holm"])
        formatted_p = _format_adjusted_p(p)
        separator = "" if formatted_p.startswith("<") else "="
        lower_model, paired_effect = _paired_effect_summary(row)
        lower = "tie" if lower_model == "tie" else f"{lower_model}↓"
        lines.append(
            f"{model_a}–{model_b}: {lower}   {abs(paired_effect):.3g}   "
            f"{separator}{formatted_p} {mc_stats.significance_stars(p)}"
        )
    ax.text(
        0.5, -0.29, "\n".join(lines), transform=ax.transAxes,
        ha="center", va="top", fontsize=fontsize, linespacing=1.15,
        family="monospace",
    )
    return int(rows["n_pairs"].iloc[0])


def _box_strip_diamond_panel(
    ax: plt.Axes, detailed: pd.DataFrame, value_col: str, diamond_row: pd.DataFrame, seed: int,
) -> dict:
    """Box+strip+diamond+n-label panel for one (row, column) cell of a
    model-comparison grid: per-model box+strip distribution over ``detailed``
    plus an optional trial-weighted pooled diamond marker from
    ``diamond_row``. Extracted from Figure 01's original single-panel body
    so every subplot in the expanded trial-subset grid gets the same
    treatment, not just "all trials". Returns ``{"own_n": {model_id: n}}``
    for the caller's headroom/caption logic.
    """
    xpos = _model_x_positions()
    box_data, positions, colors, own_n = [], [], [], {}
    for model_id in MODEL_ORDER:
        values = detailed.loc[detailed["model_id"] == model_id, value_col].dropna().to_numpy()
        if len(values) == 0:
            continue
        box_data.append(values)
        positions.append(xpos[model_id])
        colors.append(MODEL_COLORS[model_id])
        own_n[model_id] = len(values)
        jitter = np.random.default_rng(seed).normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), xpos[model_id]) + jitter, values,
            s=10, color=colors[-1], alpha=0.55, zorder=3, linewidths=0,
        )

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.5, showfliers=False,
        patch_artist=True, zorder=2,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps", "medians"):
        for line, color in zip(bp[element], np.repeat(colors, 2 if element in ("whiskers", "caps") else 1)):
            line.set_color(color)

    if diamond_row is not None:
        for _, row in diamond_row.iterrows():
            if row["model_id"] not in xpos or not np.isfinite(row["mean_nll"]):
                continue
            ax.scatter(
                xpos[row["model_id"]], row["mean_nll"], marker="D", s=45,
                color=MODEL_COLORS[row["model_id"]], edgecolor="black",
                linewidth=0.6, zorder=4,
            )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin))
    for model_id in MODEL_ORDER:
        if model_id not in own_n:
            continue
        top = detailed.loc[detailed["model_id"] == model_id, value_col].dropna().max()
        ax.text(
            xpos[model_id], top + 0.02 * (ymax - ymin), f"n={own_n[model_id]}",
            ha="center", va="bottom", fontsize=5.2, color=MODEL_COLORS[model_id],
        )
    ax.set_xticks([xpos[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER, fontsize=6.5)
    return {"own_n": own_n}


def fig_global_and_detailed_nll(
    sw_trial_subset: pd.DataFrame, sw_trial_subset_diamonds: pd.DataFrame,
    sw_trial_subset_paired: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Per-window (detailed) mean NLL distribution per model, split by
    train/validation/test (rows) and by trial subset (columns: all trials,
    short block, long block, common short, common long, rare short, rare
    long), with the trial-weighted pooled NLL overlaid as a diamond marker
    in every column.

    Every box, point, diamond, and paired test in one subplot uses the same
    complete set of rolling windows reached by all five models FOR THAT
    (split, subset) combination -- a window can have zero qualifying trials
    of a given subset in a given split (e.g. no rare-long trials in a short
    test session), so the common population is recomputed per subplot (see
    ``collect.common_trial_subset_window_ids``) rather than inherited
    wholesale from an "all trials" population. Four pre-specified adjacent
    nested-model Wilcoxon comparisons are Holm-corrected within each
    subplot. Their p-values remain exploratory because rolling windows reuse
    training history (and train/validation windows also repeat evaluated
    sessions).
    """
    splits = ["train", "validation", "test"]
    fig, axes = plt.subplots(3, 7, figsize=(27, 13.5), sharey=False)

    for row_idx, split_role in enumerate(splits):
        for col_idx, trial_subset in enumerate(TRIAL_SUBSET_ORDER):
            ax = axes[row_idx, col_idx]
            common_ids = common_trial_subset_window_ids(sw_trial_subset, mouse_id, split_role, trial_subset)
            detailed = sw_trial_subset[
                (sw_trial_subset["split_role"] == split_role)
                & (sw_trial_subset["trial_subset"] == trial_subset)
                & (sw_trial_subset["window_id"].isin(common_ids))
            ]
            diamond_row = sw_trial_subset_diamonds[
                (sw_trial_subset_diamonds["split_role"] == split_role)
                & (sw_trial_subset_diamonds["trial_subset"] == trial_subset)
            ]
            _box_strip_diamond_panel(ax, detailed, "mean_nll", diamond_row, seed=row_idx * 7 + col_idx)

            if row_idx == 0:
                ax.set_title(TRIAL_SUBSET_LABELS[trial_subset], fontsize=8.5)
            if col_idx == 0:
                ax.set_ylabel(f"{split_role.capitalize()}\nMean NLL/trial (nats)", fontsize=7.0)

            n_pairs = _draw_adjacent_significance(
                ax, sw_trial_subset_paired, {"split_role": split_role, "trial_subset": trial_subset},
            )
            ax.text(
                0.5, -0.34, f"N={n_pairs}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.0, style="italic",
            )

    handle_box = plt.Line2D([0], [0], color="gray", lw=6, alpha=0.3, label="per-window distribution (box+points)")
    handle_diamond = plt.Line2D(
        [0], [0], marker="D", color="gray", lw=0, markeredgecolor="black",
        label="◆ trial-weighted pooled NLL (boxes give each window equal weight)",
    )
    fig.legend(handles=[handle_box, handle_diamond], loc="upper center",
               bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=9)
    fig.suptitle(
        f"Sliding-window NLL by model and trial subset -- {mouse_id}", y=0.995, fontsize=13,
    )
    fig.text(
        0.5, 0.935,
        "Each subplot uses the identical window IDs across all 5 models for that (split, subset); "
        "four adjacent paired Wilcoxon tests, Holm-adjusted within each subplot. ↓ identifies the "
        "lower paired-median NLL model; all sliding-window p-values are exploratory "
        f"(rolling windows share data/history). Single mouse ({mouse_id}); within-subject comparison only.",
        ha="center", fontsize=8.5, style="italic", transform=fig.transFigure,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.55, hspace=0.75, top=0.86)
    _save(fig, figures_dir, "fig01_sliding_window_global_and_detailed_nll")


def fig_window_nll_trajectory(sw_split_session: pd.DataFrame, figures_dir) -> None:
    """Mean NLL across the 32 rolling windows (x = test-session chronological
    rank), one line per model, for train, validation, and test splits.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    for ax, split_role in zip(axes, ["train", "validation", "test"]):
        detailed = sw_split_session[
            (sw_split_session["split_role"] == split_role)
            & (sw_split_session["session_id"] == "ALL_SESSIONS")
        ]
        for model_id in MODEL_ORDER:
            subset = detailed[detailed["model_id"] == model_id].sort_values("test_rank")
            if subset.empty:
                continue
            ax.plot(
                subset["test_rank"], subset["mean_nll"], marker="o", markersize=3.5,
                linewidth=1.4, color=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id],
            )
        ax.set_ylabel(f"{split_role.capitalize()} mean NLL")
    axes[-1].set_xlabel("Rolling-window test session (chronological rank)")
    axes[0].legend(loc="upper right", ncol=5, bbox_to_anchor=(1.0, 1.35))
    fig.suptitle("Held-out NLL across the 32 rolling windows", y=1.0, fontsize=12.5)
    fig.tight_layout()
    _save(fig, figures_dir, "fig02_sliding_window_nll_trajectory")


def _box_strip_panel(
    ax, data: pd.DataFrame, value_col: str, seed: int,
    *, paired_unit_col: str | None = None, show_n: bool = True,
) -> None:
    xpos = _model_x_positions()
    if paired_unit_col is not None:
        paired = data.pivot(index=paired_unit_col, columns="model_id", values=value_col)
        paired = paired.reindex(columns=MODEL_ORDER).dropna()
        for _, values in paired.iterrows():
            ax.plot(
                [xpos[model_id] for model_id in MODEL_ORDER], values.to_numpy(float),
                color="0.65", alpha=0.28, linewidth=0.65, zorder=1,
            )
    label_targets = []
    for model_id in MODEL_ORDER:
        values = data.loc[data["model_id"] == model_id, value_col].dropna().to_numpy()
        if len(values) == 0:
            continue
        jitter = np.random.default_rng(seed).normal(0, 0.06, size=len(values))
        ax.scatter(
            np.full(len(values), xpos[model_id]) + jitter, values, s=20,
            color=MODEL_COLORS[model_id], alpha=0.6, linewidths=0, zorder=3,
        )
        bp = ax.boxplot(
            [values], positions=[xpos[model_id]], widths=0.5, showfliers=False,
            patch_artist=True, zorder=2,
        )
        bp["boxes"][0].set_facecolor(MODEL_COLORS[model_id])
        bp["boxes"][0].set_alpha(0.18)
        bp["boxes"][0].set_edgecolor(MODEL_COLORS[model_id])
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(MODEL_COLORS[model_id])
        label_targets.append((xpos[model_id], values.max(), len(values), MODEL_COLORS[model_id]))

    # Reserve headroom above the tallest point so optional "n=" labels never
    # get clipped by the axes border.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.06 * (ymax - ymin))
    if show_n:
        for x, y, n, color in label_targets:
            ax.text(x, y + 0.012 * (ymax - ymin), f"n={n}", ha="center", va="bottom",
                    fontsize=8.5, color=color)
    ax.set_xticks([xpos[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER)


def fig_single_session_heldout_nll(
    ss_trial_subset: pd.DataFrame, ss_trial_subset_paired: pd.DataFrame,
    mouse_id: str, figures_dir,
) -> None:
    """Training (fine-tune prefix) and held-out (final two blocks) NLL,
    split by trial subset (columns: all trials, short block, long block,
    common short, common long, rare short, rare long). Training NLL is what
    the optimizer directly minimized on that session's own prefix, so it is
    expected to sit lower than held-out NLL; the gap is a per-model,
    per-session generalization signal.

    Every subplot restricts to the session IDs every model reached for that
    (analysis_role, trial_subset) combination (see
    ``collect.common_trial_subset_session_ids``) -- a session can have zero
    qualifying trials of a given subset (e.g. very few rare trials in a
    2-block held-out portion), so the common population is recomputed per
    subplot rather than inherited from an "all trials" population. No
    diamond marker here (unlike Figure 01) -- single-session fine-tuning has
    no existing trial-weighted pooling concept to extend.
    """
    roles = [("fine_tune_prefix", "Training (fine-tune prefix)"), ("heldout", "Held-out (final two blocks)")]
    fig, axes = plt.subplots(2, 7, figsize=(27, 9.5), sharey=False)

    for row_idx, (analysis_role, row_label) in enumerate(roles):
        for col_idx, trial_subset in enumerate(TRIAL_SUBSET_ORDER):
            ax = axes[row_idx, col_idx]
            common_ids = common_trial_subset_session_ids(ss_trial_subset, analysis_role, trial_subset)
            detailed = ss_trial_subset[
                (ss_trial_subset["analysis_role"] == analysis_role)
                & (ss_trial_subset["trial_subset"] == trial_subset)
                & (ss_trial_subset["session_id"].isin(common_ids))
            ]
            _box_strip_panel(ax, detailed, "mean_nll", seed=row_idx * 7 + col_idx)

            if row_idx == 0:
                ax.set_title(TRIAL_SUBSET_LABELS[trial_subset], fontsize=8.5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nMean NLL/trial (nats)", fontsize=6.8)

            n_pairs = _draw_adjacent_significance(
                ax, ss_trial_subset_paired, {"analysis_role": analysis_role, "trial_subset": trial_subset},
            )
            ax.text(
                0.5, -0.32, f"N={n_pairs}", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.0, style="italic",
            )

    fig.suptitle(
        f"Single-session fine-tuning NLL by model and trial subset -- {mouse_id}", y=1.02, fontsize=13,
    )
    fig.text(
        0.5, 0.975,
        "Each subplot uses the identical session IDs across all 5 models for that (role, subset); "
        "four adjacent paired Wilcoxon tests, Holm-adjusted within each subplot. ↓ identifies the "
        "lower paired-median NLL model. Training-prefix values "
        "include completed fits even when their convergence flag failed. "
        f"Single mouse ({mouse_id}); within-subject comparison only.",
        ha="center", fontsize=8.5, style="italic", transform=fig.transFigure,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.55, hspace=0.6, top=0.88)
    _save(fig, figures_dir, "fig03_single_session_heldout_nll")


def _plot_win_count_bars(ax, win_summary: pd.DataFrame, won_col: str) -> None:
    xpos = _model_x_positions()
    for _, row in win_summary.iterrows():
        model_id = row["model_id"]
        ax.bar(
            xpos[model_id], row[won_col], width=0.6,
            color=MODEL_COLORS[model_id], edgecolor="black", linewidth=0.6,
        )
        ax.text(
            xpos[model_id], row[won_col] + 0.05, str(int(row[won_col])),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks([xpos[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER)


def _win_count_caveat_text(win_summary: pd.DataFrame) -> str:
    median_margin = win_summary["median_winning_margin_nll"].iloc[0] if len(win_summary) else float("nan")
    return (
        "Descriptive point-estimate argmin per unit -- no significance test. "
        f"Median margin over the runner-up: {median_margin:.4f} NLL "
        "(some \"wins\" are near-ties; see winning_margin_nll in the CSV)."
    )


def fig_single_session_win_counts(win_summary: pd.DataFrame, figures_dir) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    _plot_win_count_bars(ax, win_summary, "n_sessions_won")
    n_compared = int(win_summary["n_sessions_compared"].iloc[0]) if len(win_summary) else 0
    ax.set_ylabel("Sessions with the lowest held-out NLL")
    ax.set_title(
        "Single-session fine-tuning: which model wins per session?\n"
        f"(n = {n_compared} sessions with a valid fit from all 5 models)"
    )
    fig.text(0.5, -0.02, _win_count_caveat_text(win_summary), ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    _save(fig, figures_dir, "fig04_single_session_win_counts")


def fig_sliding_window_win_counts(win_summary: pd.DataFrame, figures_dir) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    _plot_win_count_bars(ax, win_summary, "n_windows_won")
    n_compared = int(win_summary["n_windows_compared"].iloc[0]) if len(win_summary) else 0
    ax.set_ylabel("Rolling windows with the lowest walk-forward test NLL")
    ax.set_title(
        "Sliding-window rolling evaluation: which model wins per window?\n"
        f"(n = {n_compared} windows with a winner from all 5 models)"
    )
    fig.text(0.5, -0.02, _win_count_caveat_text(win_summary), ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    _save(fig, figures_dir, "fig06_sliding_window_win_counts")


def fig_win_counts_combined(
    ss_win_summary: pd.DataFrame, sw_win_summary: pd.DataFrame, figures_dir,
) -> None:
    """Side-by-side comparison: for each model, how often is it the single
    best model at the single-session level vs. the sliding-window level?
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xpos = _model_x_positions()
    width = 0.34
    for model_id in MODEL_ORDER:
        x = xpos[model_id]
        ss_wins = int(ss_win_summary.loc[ss_win_summary["model_id"] == model_id, "n_sessions_won"].iloc[0])
        sw_wins = int(sw_win_summary.loc[sw_win_summary["model_id"] == model_id, "n_windows_won"].iloc[0])
        ax.bar(
            x - width / 2, ss_wins, width=width, color=MODEL_COLORS[model_id],
            alpha=0.5, edgecolor="black", linewidth=0.6, hatch="//",
        )
        ax.bar(
            x + width / 2, sw_wins, width=width, color=MODEL_COLORS[model_id],
            alpha=1.0, edgecolor="black", linewidth=0.6,
        )
        ax.text(x - width / 2, ss_wins + 0.08, str(ss_wins), ha="center", va="bottom", fontsize=9)
        ax.text(x + width / 2, sw_wins + 0.08, str(sw_wins), ha="center", va="bottom", fontsize=9)

    ax.set_xticks([xpos[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Units where the model has the lowest NLL")
    n_ss = int(ss_win_summary["n_sessions_compared"].iloc[0]) if len(ss_win_summary) else 0
    n_sw = int(sw_win_summary["n_windows_compared"].iloc[0]) if len(sw_win_summary) else 0
    handle_ss = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=0.5, hatch="//", edgecolor="black",
        label=f"Single-session held-out fine-tuning (n={n_ss} sessions)",
    )
    handle_sw = plt.Rectangle(
        (0, 0), 1, 1, facecolor="gray", alpha=1.0, edgecolor="black",
        label=f"Sliding-window walk-forward test (n={n_sw} windows)",
    )
    ax.legend(handles=[handle_ss, handle_sw], loc="upper right", bbox_to_anchor=(1.0, 1.22))
    ax.set_title("Which model is best: single-session vs. sliding-window evaluation?")
    ss_margin = ss_win_summary["median_winning_margin_nll"].iloc[0] if len(ss_win_summary) else float("nan")
    sw_margin = sw_win_summary["median_winning_margin_nll"].iloc[0] if len(sw_win_summary) else float("nan")
    fig.text(
        0.5, -0.02,
        "Descriptive point-estimate argmin per unit -- no significance test. Median margin "
        f"over the runner-up: {ss_margin:.4f} NLL (single-session), {sw_margin:.4f} NLL "
        "(sliding-window); see winning_margin_nll in the source CSVs.",
        ha="center", fontsize=7.5, style="italic",
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig07_win_counts_single_session_vs_sliding_window")


def fig_shared_parameter_trajectories(sw_params: pd.DataFrame, figures_dir) -> None:
    fig, axes = plt.subplots(len(SHARED_PARAM_NAMES), 1, figsize=(9.5, 12), sharex=True)
    for ax, param_name in zip(axes, SHARED_PARAM_NAMES):
        subset_all = sw_params[sw_params["param_name"] == param_name]
        for model_id in MODEL_ORDER:
            subset = subset_all[subset_all["model_id"] == model_id].sort_values("test_rank")
            if subset.empty:
                continue
            ax.plot(
                subset["test_rank"], subset["param_value"], marker="o", markersize=3,
                linewidth=1.2, color=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id],
            )
        ax.set_ylabel(param_name)
    axes[-1].set_xlabel("Rolling-window test session (chronological rank)")
    axes[0].legend(loc="upper right", ncol=5, bbox_to_anchor=(1.0, 1.45))
    fig.suptitle("Shared sensory/lapse parameters across rolling windows", y=1.0, fontsize=12.5)
    fig.tight_layout()
    _save(fig, figures_dir, "fig05_shared_parameter_trajectories")


def _sigmoid_series(values: pd.Series) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-values.to_numpy(dtype=float))), index=values.index)


def _param_series_by_model(sw_params: pd.DataFrame, param_name: str) -> dict[str, pd.Series]:
    """{model_id: Series(param_value indexed by test_rank)}, one series per
    model that actually fits ``param_name``."""
    subset = sw_params[sw_params["param_name"] == param_name]
    series = {}
    for model_id in NON_SHARED_PARAM_MODELS.get(param_name, []):
        model_subset = subset[subset["model_id"] == model_id].sort_values("test_rank")
        if model_subset.empty:
            continue
        series[model_id] = model_subset.set_index("test_rank")["param_value"]
    return series


def _derived_quantity_curves(
    param_name: str, sw_params: pd.DataFrame,
) -> list[tuple[str, str, dict[str, pd.Series]]]:
    """(curve_label, y_axis_label, {model_id: derived_series}) for the
    closed-form quantity each parameter card's explanation actually defines
    -- Figure 9's third column plots that derived quantity instead of
    repeating the raw fitted coordinate already shown in column 1.
    """
    if param_name == "ah":
        raw = _param_series_by_model(sw_params, "ah")
        derived = {m: 1.0 / _sigmoid_series(s) for m, s in raw.items()}
        return [("Expected block length = 1 / sigmoid(ah)", "Expected block length (trials)", derived)]
    if param_name == "delta0":
        raw = _param_series_by_model(sw_params, "delta0")
        theta_s = {m: _sigmoid_series(-s) for m, s in raw.items()}
        theta_l = {m: _sigmoid_series(s) for m, s in raw.items()}
        return [
            ("θS = sigmoid(−delta0), solid", "Subjective category rate", theta_s),
            ("θL = sigmoid(delta0), dashed", "Subjective category rate", theta_l),
        ]
    if param_name == "alpha_theta":
        raw = _param_series_by_model(sw_params, "alpha_theta")
        return [("alpha_theta itself (0 = no-learning boundary)", "alpha_theta", raw)]
    if param_name == "bh":
        ah_raw = _param_series_by_model(sw_params, "ah")
        bh_raw = _param_series_by_model(sw_params, "bh")
        derived = {}
        for model_id, bh_series in bh_raw.items():
            ah_series = ah_raw.get(model_id)
            if ah_series is None:
                continue
            joined = pd.concat({"ah": ah_series, "bh": bh_series}, axis=1).dropna()
            if joined.empty:
                continue
            hazard_23 = _sigmoid_series(joined["ah"])
            hazard_65 = _sigmoid_series(joined["ah"] + joined["bh"] * 42)
            derived[model_id] = hazard_65 - hazard_23
        return [("hB(r=65) − hB(r=23)", "Excess switch hazard by run length", derived)]
    raise ValueError(f"No derived-quantity definition registered for {param_name!r}")


def _linear_trend(series: pd.Series, all_ranks: list) -> pd.Series | None:
    """Straight-line OLS summary of ``series`` over chronological test rank,
    evaluated across the full displayed rank range -- a compact stand-in for
    "does this derived quantity drift up or down across sessions"."""
    valid = series.dropna()
    if len(valid) < 2:
        return None
    slope, intercept = np.polyfit(valid.index.to_numpy(dtype=float), valid.to_numpy(dtype=float), 1)
    ranks = np.asarray(all_ranks, dtype=float)
    return pd.Series(slope * ranks + intercept, index=all_ranks)


def fig_sliding_window_model_specific_parameters(
    sw_params: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Trajectories for the parameters each nested model adds beyond the
    shared five (ah, delta0, alpha_theta, bh) -- only plotted for the models
    that actually fit that coordinate.
    """
    param_names = list(NON_SHARED_PARAM_MODELS.keys())
    curve_linestyles = ["-", "--"]
    fig, axes = plt.subplots(
        len(param_names), 3, figsize=(21.5, 11.5), sharex="col",
        gridspec_kw={"width_ratios": [2.25, 1.0, 2.0]},
    )
    all_ranks = sorted(pd.to_numeric(sw_params["test_rank"], errors="coerce").dropna().unique())
    for row_index, param_name in enumerate(param_names):
        ax, meaning_ax, derived_ax = axes[row_index]
        subset_all = sw_params[sw_params["param_name"] == param_name]
        for model_id in NON_SHARED_PARAM_MODELS[param_name]:
            subset = subset_all[subset_all["model_id"] == model_id].sort_values("test_rank")
            if subset.empty:
                continue
            if subset["test_rank"].duplicated().any():
                raise ValueError(f"Duplicate test ranks for {model_id}/{param_name}")
            trajectory = pd.DataFrame({"test_rank": all_ranks}).merge(
                subset[["test_rank", "param_value"]], on="test_rank", how="left",
            )
            ax.plot(
                trajectory["test_rank"], trajectory["param_value"], markersize=3,
                linewidth=1.2, color=MODEL_COLORS[model_id],
                linestyle=MODEL_LINESTYLES[model_id], marker=MODEL_MARKERS[model_id],
                label=f"{MODEL_LABELS[model_id]} ({subset['param_value'].notna().sum()} fitted windows)",
            )
        if param_name in {"alpha_theta", "bh"}:
            ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_ylabel(param_name)
        ax.legend(loc="upper right", ncol=len(NON_SHARED_PARAM_MODELS[param_name]), fontsize=8)
        meaning_ax.axis("off")
        heading, explanation = MODEL_SPECIFIC_PARAMETER_CARDS[param_name]
        meaning_ax.text(
            0.02, 0.88, heading, transform=meaning_ax.transAxes,
            ha="left", va="top", fontsize=10.5, fontweight="bold",
        )
        meaning_ax.text(
            0.02, 0.70, _wrap(explanation, width=54), transform=meaning_ax.transAxes,
            ha="left", va="top", fontsize=9.0, linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "0.97", "edgecolor": "0.82"},
        )

        # Third column: the closed-form quantity the explanation actually
        # defines (e.g. expected block length, not raw ah), plus a per-model
        # straight-line trend summarizing its drift across chronological
        # sessions.
        curves = _derived_quantity_curves(param_name, sw_params)
        for curve_index, (curve_label, y_label, model_series) in enumerate(curves):
            style = curve_linestyles[curve_index % len(curve_linestyles)]
            for model_id, series in model_series.items():
                plotted = series.reindex(all_ranks)
                derived_ax.plot(
                    all_ranks, plotted.to_numpy(dtype=float), color=MODEL_COLORS[model_id],
                    linestyle=style, marker=MODEL_MARKERS[model_id], markersize=2.3,
                    linewidth=1.0, alpha=0.9, zorder=2,
                )
                trend = _linear_trend(series, all_ranks)
                if trend is None:
                    continue
                derived_ax.plot(
                    all_ranks, trend.to_numpy(dtype=float), color=MODEL_COLORS[model_id],
                    linestyle=style, linewidth=2.6, alpha=0.30, zorder=1,
                )
        if param_name in {"alpha_theta", "bh"}:
            derived_ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        derived_ax.set_ylabel(curves[0][1], fontsize=9.5)
        derived_ax.set_title(" | ".join(c[0] for c in curves), fontsize=8.3)
        derived_ax.tick_params(labelsize=8.5)
        if row_index == 0:
            legend_handles = [
                plt.Line2D([0], [0], color="0.25", linewidth=1.0, marker="o", markersize=3.5,
                           label="Derived value (per window)"),
                plt.Line2D([0], [0], color="0.25", linewidth=2.6, alpha=0.4,
                           label="Linear trend across sessions"),
            ]
            derived_ax.legend(handles=legend_handles, loc="best", fontsize=6.6, framealpha=0.85)
    axes[-1, 0].set_xlabel("Rolling-window test session (chronological rank)")
    axes[-1, 2].set_xlabel("Rolling-window test session (chronological rank)")
    fig.suptitle(
        f"Model-specific parameters across rolling windows -- {mouse_id}",
        y=0.995, fontsize=12.5,
    )
    fig.text(
        0.5, 0.008,
        "Single mouse; point estimates from overlapping rolling windows. Lines show chronological "
        "sequence, not independent replications. Gaps are unavailable fits; a model absent from a row "
        "does not contain that parameter. Column 3 plots the closed-form quantity named in column 2's "
        "explanation (derived from column 1's fitted values) with a per-model linear-trend overlay.",
        ha="center", fontsize=8.5, style="italic",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    _save(fig, figures_dir, "fig09_sliding_window_model_specific_parameters")


def fig_single_session_availability(ss_avail: pd.DataFrame, figures_dir) -> None:
    """How many of the 32 matched sessions each model produced a usable
    single-session fine-tune for. ``valid`` and ``never attempted`` are
    genuinely complementary (they sum to the expected count), so this is a
    safe stacked bar. The "flagged" diagnostics overlap ``valid`` (a session
    can be both valid and flagged not-converged) and so are shown as text
    annotations instead of additional stacked segments -- stacking them
    would double-count and misstate the total.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xpos = _model_x_positions()
    n_expected = int(ss_avail["n_matched_windows_expected"].iloc[0]) if len(ss_avail) else 0
    for _, row in ss_avail.iterrows():
        model_id = row["model_id"]
        x = xpos[model_id]
        valid = int(row["n_sessions_valid"])
        never = int(row["n_sessions_never_attempted"])
        ax.bar(x, valid, width=0.6, color=MODEL_COLORS[model_id], edgecolor="black", linewidth=0.6)
        ax.bar(x, never, width=0.6, bottom=valid, color="0.9", edgecolor="black", linewidth=0.6, hatch="//")
        ax.text(
            x, valid / 2 if valid else 0, str(valid), ha="center", va="center",
            fontsize=9, color="white" if valid > 4 else "black", fontweight="bold",
        )
        annotation_lines = [f"{int(row['n_sessions_flagged_not_converged'])} not converged"]
        if row["n_sessions_flagged_ineligible_by_pipeline_gate"] > 0:
            annotation_lines.append(f"{int(row['n_sessions_flagged_ineligible_by_pipeline_gate'])} gate-failed")
        ax.text(
            x, n_expected + 1.5, "\n".join(annotation_lines),
            ha="center", va="bottom", fontsize=7.5, style="italic",
        )

    ax.axhline(n_expected, color="0.4", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks([xpos[m] for m in MODEL_ORDER])
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Sessions (of 32 matched)")
    ax.set_ylim(0, n_expected + 8)
    valid_handle = plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="black", label="valid (usable fit)")
    never_handle = plt.Rectangle((0, 0), 1, 1, facecolor="0.9", hatch="//", edgecolor="black", label="never attempted")
    ax.legend(handles=[valid_handle, never_handle], loc="upper left", fontsize=8.5)
    ax.set_title(
        "Single-session fine-tuning: how many of the 32 matched sessions are usable?\n"
        "Italic annotations are diagnostic-only subsets of \"valid\", not additional sessions"
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig10_single_session_availability")


def _robust_ylim(values: np.ndarray, fence: float = 3.0, pad_frac: float = 0.08):
    """Tukey-fence range (median IQR +/- ``fence`` * IQR, clipped to the
    actual data range) with padding, or ``None`` if too few finite values to
    define a range.

    Percentile cutoffs (e.g. 1st/99th) are NOT robust enough here: with only
    ~20-30 points per panel, 1% is under one point, so a single wild outlier
    fit (e.g. a non-converged single-session optimization landing on an
    unconstrained parameter value in the hundreds of thousands) still sets
    the percentile itself. An IQR fence stays anchored to the bulk of the
    data regardless of how extreme a handful of outliers are.
    """
    finite = values[np.isfinite(values)]
    if len(finite) < 4:
        return None
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            return None
    else:
        lo = max(q1 - fence * iqr, float(finite.min()))
        hi = min(q3 + fence * iqr, float(finite.max()))
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _single_session_parameter_panel(ax, subset_all: pd.DataFrame, model_ids: list, param_name: str) -> None:
    """Plot one parameter's fitted value per model across chronological rank.

    The y-axis is clipped to a robust (Tukey-fence) range around the
    converged points, because a handful of non-converged single-session
    fits can land on unconstrained parameter values in the hundreds of
    thousands and would otherwise flatten the whole informative range.
    Clipping is NOT limited to non-converged points -- a converged fit can
    also land far outside the bulk of the other converged estimates -- so
    every point outside the drawn range, converged or not, is marked with a
    triangle at the axis boundary rather than silently cut off. The
    connecting line is drawn underneath at reduced opacity for continuity
    only; the triangle markers are the authoritative "this rank is
    off-scale" signal.
    """
    converged_all = subset_all["converged"].astype(bool)
    converged_values = subset_all.loc[converged_all, "fitted_value"].to_numpy(float)
    ylim = _robust_ylim(converged_values)

    for model_id in model_ids:
        subset = subset_all[subset_all["model_id"] == model_id].sort_values("chronological_rank")
        if subset.empty:
            continue
        converged = subset["converged"].astype(bool)
        ax.plot(
            subset["chronological_rank"], subset["fitted_value"], linewidth=1.0,
            color=MODEL_COLORS[model_id], alpha=0.3, zorder=1,
        )
        ax.scatter(
            subset.loc[converged, "chronological_rank"], subset.loc[converged, "fitted_value"],
            s=16, color=MODEL_COLORS[model_id], alpha=0.9, zorder=3, linewidths=0,
            label=MODEL_LABELS[model_id],
        )
        ax.scatter(
            subset.loc[~converged, "chronological_rank"], subset.loc[~converged, "fitted_value"],
            s=16, color=MODEL_COLORS[model_id], alpha=0.4, zorder=2, linewidths=0,
        )

        if ylim is not None:
            lo, hi = ylim
            inset = 0.03 * (hi - lo)
            for conv_flag, marker_alpha in [(True, 0.95), (False, 0.5)]:
                flagged = subset[converged == conv_flag]
                above = flagged[flagged["fitted_value"] > hi]
                below = flagged[flagged["fitted_value"] < lo]
                if not above.empty:
                    ax.scatter(
                        above["chronological_rank"], np.full(len(above), hi - inset),
                        marker="^", s=45, color=MODEL_COLORS[model_id], alpha=marker_alpha,
                        edgecolors="black", linewidths=0.6, zorder=5,
                    )
                if not below.empty:
                    ax.scatter(
                        below["chronological_rank"], np.full(len(below), lo + inset),
                        marker="v", s=45, color=MODEL_COLORS[model_id], alpha=marker_alpha,
                        edgecolors="black", linewidths=0.6, zorder=5,
                    )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(param_name)


def _convergence_legend_handles() -> list:
    return [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25",
                   markeredgewidth=0, markersize=6, alpha=0.9, label="converged"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25",
                   markeredgewidth=0, markersize=6, alpha=0.4, label="not converged"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="0.25",
                   markeredgecolor="black", markeredgewidth=0.6, markersize=7,
                   label="off-scale (clipped; converged or not -- see CSV)"),
    ]


def fig_single_session_shared_parameter_trajectories(ss_params: pd.DataFrame, figures_dir) -> None:
    """Fine-tuned shared-parameter values across chronological rank."""
    fig, axes = plt.subplots(len(SHARED_PARAM_NAMES), 1, figsize=(9.5, 12.5), sharex=True)
    for ax, param_name in zip(axes, SHARED_PARAM_NAMES):
        subset_all = ss_params[ss_params["param_name"] == param_name]
        _single_session_parameter_panel(ax, subset_all, MODEL_ORDER, param_name)
    axes[-1].set_xlabel("Session (chronological rank)")
    model_handles, model_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=model_handles + _convergence_legend_handles(), loc="upper center",
        bbox_to_anchor=(0.5, 1.09), ncol=4, fontsize=8,
    )
    fig.suptitle(
        "Single-session fine-tuned shared parameters\n"
        "y-axis clipped to a robust range around the converged points (Tukey fence); "
        "ANY point outside it (converged or not) is marked with a triangle at the "
        "boundary rather than hidden -- exact values are in the source CSV",
        y=1.145, fontsize=10.5,
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig11_single_session_shared_parameter_trajectories")


def fig_single_session_model_specific_parameters(ss_params: pd.DataFrame, figures_dir) -> None:
    """Fine-tuned model-specific parameters (ah, delta0, alpha_theta, bh)
    across chronological rank, only for the models that fit each one."""
    param_names = list(NON_SHARED_PARAM_MODELS.keys())
    fig, axes = plt.subplots(len(param_names), 1, figsize=(9.5, 10.5), sharex=True)
    for ax, param_name in zip(axes, param_names):
        subset_all = ss_params[ss_params["param_name"] == param_name]
        model_ids = NON_SHARED_PARAM_MODELS[param_name]
        _single_session_parameter_panel(ax, subset_all, model_ids, param_name)
        ax.legend(loc="upper right", ncol=len(model_ids), fontsize=8)
    axes[-1].set_xlabel("Session (chronological rank)")
    fig.legend(
        handles=_convergence_legend_handles(), loc="upper center",
        bbox_to_anchor=(0.5, 1.11), ncol=3, fontsize=8,
    )
    fig.suptitle(
        "Single-session fine-tuned model-specific parameters\n"
        "y-axis clipped to a robust range around the converged points (Tukey fence); "
        "ANY point outside it (converged or not) is marked with a triangle at the "
        "boundary rather than hidden -- exact values are in the source CSV",
        y=1.155, fontsize=10.5,
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig12_single_session_model_specific_parameters")


_MARGIN_ALPHA_LO_NLL = 1e-5
_MARGIN_ALPHA_HI_NLL = 1e-1
_MARGIN_ALPHA_FLOOR = 0.4
_MARGIN_ALPHA_CEIL = 1.0


def _margin_to_alpha(margin: float) -> float:
    """Map a winning margin (NLL, best minus runner-up) to a tile opacity on
    a log scale between ``_MARGIN_ALPHA_LO_NLL`` and ``_MARGIN_ALPHA_HI_NLL``,
    so both razor-thin and clearly decisive wins are visually distinguishable
    instead of all reading as equally "won". The opacity floor is kept well
    above 0 (0.4, not e.g. 0.1) so even a near-tie tile stays recognizably
    the winning model's hue rather than fading toward white/gray.
    """
    if margin is None or not np.isfinite(margin) or margin <= 0:
        return _MARGIN_ALPHA_FLOOR
    clipped = min(max(margin, _MARGIN_ALPHA_LO_NLL), _MARGIN_ALPHA_HI_NLL)
    frac = (
        (np.log10(clipped) - np.log10(_MARGIN_ALPHA_LO_NLL))
        / (np.log10(_MARGIN_ALPHA_HI_NLL) - np.log10(_MARGIN_ALPHA_LO_NLL))
    )
    return _MARGIN_ALPHA_FLOOR + frac * (_MARGIN_ALPHA_CEIL - _MARGIN_ALPHA_FLOOR)


def fig_chronological_winner_timeline(
    timeline: pd.DataFrame, figures_dir, *, stem: str, row_labels: dict,
    title: str, subtitle: str,
) -> None:
    """Which model wins at each chronological rank, for both analysis
    levels, covering every matched rank (13-44) -- ranks where fewer than
    five models produced a comparable fit are hatched, not dropped, since
    they are disproportionately early in this dataset.

    The two rows are NOT paired replicate estimates of the same thing (see
    ``row_labels``/``subtitle`` for what each one actually evaluates). A
    visible gap plus divider line separates the two rows so the shared
    x-axis is not mistaken for a rank-by-rank paired comparison.

    Tile opacity encodes the winning margin (NLL, nats; log scale) with an
    explicit swatch legend giving the numeric anchors, so "winner" cannot be
    read as a significance claim -- these are descriptive point-estimate
    argmins only.
    """
    ranks = sorted(timeline["chronological_rank"].unique())
    row_order = ["sliding_window", "single_session"]
    row_gap = 0.35
    row_height = 0.85
    row_bottoms = {"sliding_window": row_height + row_gap, "single_session": 0.0}

    fig, ax = plt.subplots(figsize=(13, 4.2))
    for level in row_order:
        y = row_bottoms[level]
        level_rows = timeline[timeline["analysis_level"] == level].set_index("chronological_rank")
        for rank in ranks:
            row = level_rows.loc[rank]
            if bool(row["comparable"]):
                ax.bar(
                    rank, row_height, bottom=y, width=0.95,
                    color=MODEL_COLORS[row["winning_model"]],
                    alpha=_margin_to_alpha(row["winning_margin_nll"]),
                    edgecolor="black", linewidth=0.4, zorder=2,
                )
            else:
                ax.bar(
                    rank, row_height, bottom=y, width=0.95, facecolor="0.92",
                    hatch="//", edgecolor="0.6", linewidth=0.4, zorder=2,
                )

    divider_y = row_height + row_gap / 2
    ax.axhline(divider_y, color="0.75", linewidth=0.8, zorder=1)
    ax.set_yticks([row_bottoms[level] + row_height / 2 for level in row_order])
    ax.set_yticklabels([row_labels[level] for level in row_order], fontsize=9)
    ax.set_xlabel("Chronological rank (session order)")
    ax.set_xlim(min(ranks) - 0.6, max(ranks) + 0.6)
    ax.set_xticks(ranks)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylim(-0.1, row_height + row_gap + row_height + 0.1)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], edgecolor="black",
                       linewidth=0.4, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    na_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor="0.92", hatch="//", edgecolor="0.6",
        label="not comparable (<5 models)",
    )
    margin_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.3", alpha=_MARGIN_ALPHA_FLOOR,
                      edgecolor="black", linewidth=0.4,
                      label=f"margin ≤{_MARGIN_ALPHA_LO_NLL:.0e} NLL (near-tie)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.3", alpha=(_MARGIN_ALPHA_FLOOR + _MARGIN_ALPHA_CEIL) / 2,
                      edgecolor="black", linewidth=0.4, label="margin 1e-3 NLL"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.3", alpha=_MARGIN_ALPHA_CEIL,
                      edgecolor="black", linewidth=0.4,
                      label=f"margin ≥{_MARGIN_ALPHA_HI_NLL:.0e} NLL (decisive)"),
    ]
    legend1 = ax.legend(handles=model_handles + [na_handle], loc="upper center",
                         bbox_to_anchor=(0.5, 1.40), ncol=6, fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=margin_handles, loc="upper center", bbox_to_anchor=(0.5, 1.30),
              ncol=3, fontsize=7.5, title="Tile opacity = winning margin (NLL, nats; log scale)",
              title_fontsize=7.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, pad=68)
    fig.tight_layout()
    _save(fig, figures_dir, stem)


def fig_chronological_winner_timeline_test(timeline: pd.DataFrame, figures_dir) -> None:
    """Held-out/generalization variant: sliding-window walk-forward test NLL
    and single-session fine-tune held-out NLL."""
    fig_chronological_winner_timeline(
        timeline, figures_dir,
        stem="fig08_chronological_winner_timeline",
        row_labels={
            "sliding_window": "Sliding-window\n(walk-forward test)",
            "single_session": "Single-session\n(fine-tune held-out)",
        },
        title="Which model wins, in chronological order? (held-out NLL)",
        subtitle=(
            "Descriptive point-estimate argmin per rank -- NOT a significance test. "
            "The two rows are different methodologies (see caption), not paired trials."
        ),
    )


def fig_chronological_winner_timeline_train(timeline: pd.DataFrame, figures_dir) -> None:
    """In-sample/training-fit variant: sliding-window pooled training NLL
    and single-session fine-tune training-prefix NLL. This is a fit-quality
    diagnostic, not a generalization one -- richer nested models (more free
    parameters) are expected to win here more often almost by construction.
    Compare against fig08 (held-out) or the train-vs-test agreement table to
    see where that extra flexibility helps generalization versus merely
    overfits.
    """
    fig_chronological_winner_timeline(
        timeline, figures_dir,
        stem="fig13_chronological_winner_timeline_train",
        row_labels={
            "sliding_window": "Sliding-window\n(pooled training NLL)",
            "single_session": "Single-session\n(fine-tune training NLL)",
        },
        title="Which model best FITS the training data, in chronological order?",
        subtitle=(
            "In-sample NLL only -- NOT a generalization measure. Richer models (more "
            "parameters) are expected to win here almost by construction; compare "
            "against fig08 (held-out) to see where that flexibility actually helps."
        ),
    )


def fig_train_vs_test_winner_agreement(agreement_summary: pd.DataFrame, figures_dir) -> None:
    """How often does the model with the best in-sample training fit also
    have the best held-out fit, per analysis level? Low agreement is the
    expected/interesting result: it means the ranking produced by training
    NLL alone (fig13) is a poor guide to which model actually generalizes
    best (fig08) -- i.e. richer models often win on fit but not on
    generalization, the classic overfitting signature.
    """
    level_labels = {
        "sliding_window": "Sliding-window\n(train vs. walk-forward test)",
        "single_session": "Single-session\n(fine-tune train vs. held-out)",
    }
    levels = [lvl for lvl in level_labels if lvl in agreement_summary["analysis_level"].to_numpy()]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    xpos = {lvl: i for i, lvl in enumerate(levels)}
    for _, row in agreement_summary.iterrows():
        level = row["analysis_level"]
        if level not in xpos:
            continue
        rate = float(row["agreement_rate"])
        ax.bar(xpos[level], rate, width=0.55, color="0.35", edgecolor="black", linewidth=0.6)
        ax.text(
            xpos[level], rate + 0.02,
            f"{rate:.0%}\n({int(row['n_agree'])}/{int(row['n_ranks_compared'])})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(1 / 5, color="0.6", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        len(levels) - 0.5, 1 / 5 + 0.015, "chance level (1/5 models)",
        ha="right", va="bottom", fontsize=7.5, color="0.5", style="italic",
    )
    ax.set_xticks([xpos[lvl] for lvl in levels])
    ax.set_xticklabels([level_labels[lvl] for lvl in levels])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate: training-fit winner == held-out winner")
    ax.set_title(
        "Does the best-fitting model on training data also generalize best?\n"
        "Low agreement = training-fit ranking is a poor guide to generalization (overfitting)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig14_train_vs_test_winner_agreement")


def fig_information_criteria(
    sw_ic: pd.DataFrame, ss_ic: pd.DataFrame,
    ic_paired_pairwise: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """AIC/BIC for common rolling windows and numerically sound sessions.

    The first row uses the 27 rolling-window training fits (restricted to
    ``common_window_ids``, windows every model reached with a converged
    multi-start winner). The second row is a sensitivity analysis restricted
    to sessions for which all five single-session fine-tunes are numerically
    sound (``collect.collect_single_session_information_criteria``) and use
    the same trial count. This is deliberately NOT each model's own binary
    ``converged`` flag: that flag's absolute gradient-norm/tail-occupancy
    checks were calibrated against the much larger pooled rolling-window
    training set and are systematically too strict for one session's far
    smaller trial count, which on MC06 collapsed the jointly-converged
    population from 27 sessions to 5 even though most of the excluded fits
    were only a few-fold over that tolerance rather than genuinely diverged.
    "Numerically sound" instead requires finite objective/parameters/
    gradients, a stable training-NLL trajectory, and no parameter-transform/
    probability-normalization/optimizer-runtime error -- i.e. it still
    excludes fits that hit a real numerical breakdown. AIC/BIC are penalized
    in-sample fit criteria, not held-out generalization measures.
    """
    xpos = _model_x_positions()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2), sharey=False)
    levels = [
        ("sliding_window", "Sliding-window training fits", sw_ic, "window_id", 5,
         "EXPLORATORY—overlapping rolling fits"),
        ("single_session", "Single-session fine-tuning", ss_ic, "session_id", 7,
         "REDUCED-N SENSITIVITY—numerically sound subset, not each model's own converged flag"),
    ]

    for row_index, (level, row_label, table, unit_col, seed, caveat) in enumerate(levels):
        n_units = table[unit_col].nunique() if not table.empty else 0
        mean_trials = int(round(table["n_trials"].mean())) if not table.empty else 0
        k_by_model = (
            table.groupby("model_id")["n_params"].first()
            if not table.empty else pd.Series(dtype=int)
        )
        for col_index, metric in enumerate(("aic", "bic")):
            ax = axes[row_index, col_index]
            _box_strip_panel(ax, table, metric, seed=seed + col_index)
            ax.set_ylabel(f"{metric.upper()} (training fit)")
            penalty = "2k" if metric == "aic" else "k·ln(n_trials)"
            primary = " -- primary criterion" if metric == "bic" else ""
            ax.set_title(f"{row_label}: {metric.upper()} ({penalty}){primary}")
            ax.set_xticklabels([
                f"{model_id}\nk={int(k_by_model[model_id])}" if model_id in k_by_model.index else model_id
                for model_id in MODEL_ORDER
            ])
            n_pairs = _draw_adjacent_significance(
                ax, ic_paired_pairwise,
                {"analysis_level": level, "metric": metric},
            )
            ax.text(
                0.5, -0.22,
                _wrap(
                    f"Adjacent paired Wilcoxon, Holm-adjusted (N={n_pairs}); {caveat}.",
                    width=58,
                ),
                transform=ax.transAxes, ha="center", va="top", fontsize=7.3, style="italic",
            )

        win_ax = axes[row_index, 2]
        counts = (
            table.loc[table["delta_bic"] == 0, "model_id"].value_counts()
            .reindex(MODEL_ORDER, fill_value=0)
        ) if not table.empty else pd.Series(0, index=MODEL_ORDER)
        proportions = counts / n_units if n_units else counts.astype(float)
        for model_id in MODEL_ORDER:
            x = xpos[model_id]
            value = float(proportions[model_id])
            win_ax.bar(
                x, value, width=0.6, color=MODEL_COLORS[model_id],
                edgecolor="black", linewidth=0.6,
                alpha=0.55 if level == "single_session" else 1.0,
                hatch="//" if level == "single_session" else None,
            )
            win_ax.text(x, value + 0.025, f"{int(counts[model_id])}/{n_units}",
                        ha="center", va="bottom", fontsize=8)
        win_ax.set_ylim(0, 1.12)
        win_ax.set_xticks([xpos[m] for m in MODEL_ORDER])
        win_ax.set_xticklabels(MODEL_ORDER)
        win_ax.set_ylabel("Fraction of units with lowest BIC")
        title_suffix = " (low-N)" if level == "single_session" else ""
        win_ax.set_title(f"{row_label}: best-BIC share (N={n_units}){title_suffix}", fontsize=11.5)
        winner_caveat = (
            "LOW-N descriptive winners; no independent-population inference."
            if level == "single_session"
            else "Overlapping windows; descriptive winners, not independent evidence."
        )
        win_ax.text(
            0.5, -0.15,
            f"~{mean_trials} valid training trials per unit; {winner_caveat}",
            transform=win_ax.transAxes, ha="center", va="top", fontsize=7.3, style="italic",
        )

    fig.suptitle(
        f"Model comparison by information criterion -- {mouse_id}", y=1.02, fontsize=13,
    )
    fig.text(
        0.5, 0.975,
        "All models use identical units within each row; lower is better. Penalized IN-SAMPLE "
        "fit only—not a substitute for held-out NLL in Figures 01/03. The single-session row "
        "contains only sessions where every model's fine-tune fit was numerically sound (finite, "
        "stable, error-free) -- a looser, reduced-N sensitivity analysis, not each model's own "
        "stricter converged flag. Panel-specific "
        "y-scales. Single mouse "
        f"({mouse_id}); within-subject comparison only.",
        ha="center", fontsize=8, style="italic", transform=fig.transFigure,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.52, wspace=0.30)
    _save(fig, figures_dir, "fig15_model_comparison_information_criteria")


def _draw_event_significance_text(
    ax: plt.Axes, pairwise: pd.DataFrame, filters: dict,
    *, scope_label: str | None = None, fontsize: float = 6.5,
    y_position: float = -0.24,
) -> int:
    """Compact text-block alternative to ``_draw_adjacent_significance`` for
    a trajectory subplot (Figures 16/17): its x-axis is continuous
    relative-trial-position, not discrete model x-positions, so the bracket
    geometry used elsewhere in this module does not apply. Renders the same
    four Holm-adjusted adjacent-pair results as a small text block below the
    subplot instead. Returns ``n_pairs`` (0 if no rows matched).
    """
    rows = pairwise.copy()
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    if rows.empty:
        ax.text(
            0.5, -0.24, "No adjacent-pair tests available for this event category.",
            transform=ax.transAxes, ha="center", va="top", fontsize=fontsize, style="italic",
        )
        return 0
    lines = []
    for model_a, model_b in mc_stats.ADJACENT_MODEL_PAIRS:
        match = rows[(rows["model_a"] == model_a) & (rows["model_b"] == model_b)]
        if match.empty:
            continue
        row = match.iloc[0]
        p = float(row["wilcoxon_p_holm"])
        formatted_p = _format_adjusted_p(p)
        separator = "" if formatted_p.startswith("<") else "="
        lower_model, paired_effect = _paired_effect_summary(row)
        direction = "tie" if lower_model == "tie" else f"{lower_model}↓"
        lines.append(
            f"{model_a}-{model_b}: {direction}, |Δmed|={abs(paired_effect):.3g}, "
            f"pH{separator}{formatted_p} "
            f"{mc_stats.significance_stars(p)}"
        )
    text_lines = []
    if scope_label:
        text_lines.append(scope_label)
    text_lines.extend(["   ".join(lines[:2]), "   ".join(lines[2:])])
    ax.text(
        0.5, y_position, "\n".join(line for line in text_lines if line), transform=ax.transAxes,
        ha="center", va="top", fontsize=fontsize, linespacing=1.15,
    )
    return int(rows["n_pairs"].iloc[0])


def _plot_event_trajectory_panel(
    ax: plt.Axes, event_aligned: pd.DataFrame, event_category: str, common_ids: set, unit_col: str,
) -> dict:
    """Mean-NLL-vs-relative-trial-position line per model for one
    event_category, restricted to the common window/session population for
    that category (see ``collect.common_event_window_ids``/
    ``common_event_session_ids``). A dashed vertical line marks the event
    anchor (relative_position == 0: first trial of the new block for
    transitions, the rare trial itself for rare-trial events). Returns
    ``{"n_units": ..., "n_events": ...}`` for the caller's title.
    """
    subset = event_aligned[
        (event_aligned["event_category"] == event_category)
        & (event_aligned[unit_col].isin(common_ids))
    ]
    for model_id in MODEL_ORDER:
        model_subset = subset[subset["model_id"] == model_id]
        trajectory = model_subset.groupby("relative_position")["mean_nll"].mean().sort_index()
        if trajectory.empty:
            continue
        ax.plot(
            trajectory.index, trajectory.to_numpy(), markersize=3.5,
            linewidth=1.6, color=MODEL_COLORS[model_id], label=MODEL_LABELS[model_id],
            linestyle=MODEL_LINESTYLES[model_id], marker=MODEL_MARKERS[model_id],
        )
    ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Relative trial position")
    ax.set_ylabel("Mean NLL per trial (nats)")

    # n_events_in_unit is identical across models for one matched unit (the
    # trial set is shared -- see collect.py's module docstring), so any one
    # model's value per unit is authoritative; take the first available.
    per_unit_events = subset.drop_duplicates([unit_col, "model_id"]).groupby(unit_col)["n_events_in_unit"].first()
    n_events = int(per_unit_events.reindex(sorted(common_ids)).fillna(0).sum())
    return {"n_units": len(common_ids), "n_events": n_events}


_EVENT_CATEGORY_LAYOUT = [
    ("block_transition_short_to_long", "Short→Long transition"),
    ("block_transition_long_to_short", "Long→Short transition"),
    ("rare_short", "Rare short (in a long block)"),
    ("rare_long", "Rare long (in a short block)"),
]


def _event_range_label(event_category: str) -> str:
    """``"−5..+5"``-style label for an event category's registered display
    range, read from ``collect.EVENT_RELATIVE_POSITIONS`` so it can never
    drift out of sync with the actual displayed positions."""
    positions = EVENT_RELATIVE_POSITIONS[event_category]
    return f"−{abs(positions[0])}..+{positions[-1]}"


def _build_sliding_window_event_aligned_page(
    sw_event_aligned: pd.DataFrame, sw_event_paired: pd.DataFrame,
    sw_event_trial0_paired: pd.DataFrame, mouse_id: str, split_role: str, page_number: int,
) -> plt.Figure:
    """Build one page of Figure 16, restricted to ``split_role``'s own rows
    (see ``collect.EVENT_ALIGNED_SPLIT_ROLES``) -- filtering the three input
    tables once here means every helper below sees exactly the single-page
    table shape it was written for, unchanged from before Figure 16 grew
    pages.

    Page layout, identical across all 4 pages: block transitions (top row,
    trial 0 = first trial of the new block) and rare trials (second row,
    trial 0 = the rare trial itself), one line per model. Multiple
    qualifying events within one window are averaged together BEFORE this
    figure is built (see ``collect._aggregate_event_aligned_nll``), so a
    window contributes one number per relative position regardless of how
    many events it contains -- trials are never treated as independent
    samples. The four adjacent-model paired tests per subplot use each
    window's mean NLL over the WHOLE displayed relative-position range (one
    set of 4 Holm-corrected tests per subplot, not per relative position --
    this tests which model has lower NLL averaged across the whole window,
    not whether the event-locked shape itself differs). Rolling-window
    inference is exploratory (overlapping training history). The third row
    adds paired-window distributions for the two transition directions at
    trial 0 and over the complete displayed range; the fourth row mirrors
    that same layout for the two rare-trial categories (rare-long-in-a-
    short-block and rare-short-in-a-long-block).
    """
    event_aligned = sw_event_aligned.loc[sw_event_aligned["split_role"] == split_role]
    event_paired = sw_event_paired.loc[sw_event_paired["split_role"] == split_role]
    event_trial0_paired = sw_event_trial0_paired.loc[sw_event_trial0_paired["split_role"] == split_role]

    fig = plt.figure(figsize=(18, 16.0))
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.05, 1.05])
    trajectory_axes = [
        fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:]),
        fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:]),
    ]
    for ax, (event_category, title) in zip(trajectory_axes, _EVENT_CATEGORY_LAYOUT):
        common_ids = common_event_window_ids(event_aligned, mouse_id, event_category)
        info = _plot_event_trajectory_panel(ax, event_aligned, event_category, common_ids, "window_id")
        _draw_event_significance_text(
            ax, event_paired, {"event_category": event_category},
            scope_label="Whole-range paired tests (mean over displayed positions):",
            fontsize=8.8, y_position=-0.31,
        )
        ax.set_title(
            f"{title}\nN={info['n_units']} matched windows (test units); "
            f"{info['n_events']} events aggregated",
            fontsize=11.0,
        )
        ax.tick_params(labelsize=9.5)
        ax.xaxis.label.set_size(10.5)
        ax.yaxis.label.set_size(10.5)
    trajectory_axes[0].legend(loc="upper right", fontsize=8.5)

    summary_specs = [
        ("block_transition_short_to_long", (0,), event_trial0_paired, "Trial 0: Short→Long"),
        ("block_transition_long_to_short", (0,), event_trial0_paired, "Trial 0: Long→Short"),
        (
            "block_transition_short_to_long", None, event_paired,
            f"Whole {_event_range_label('block_transition_short_to_long')}: Short→Long",
        ),
        (
            "block_transition_long_to_short", None, event_paired,
            f"Whole {_event_range_label('block_transition_long_to_short')}: Long→Short",
        ),
        ("rare_long", (0,), event_trial0_paired, "Trial 0: Rare long (in short block)"),
        ("rare_short", (0,), event_trial0_paired, "Trial 0: Rare short (in long block)"),
        (
            "rare_long", None, event_paired,
            f"Whole {_event_range_label('rare_long')}: Rare long (in short block)",
        ),
        (
            "rare_short", None, event_paired,
            f"Whole {_event_range_label('rare_short')}: Rare short (in long block)",
        ),
    ]
    for index, (event_category, positions, paired, title) in enumerate(summary_specs):
        row, col = divmod(index, 4)
        ax = fig.add_subplot(grid[2 + row, col])
        common_ids = common_event_window_ids(
            event_aligned, mouse_id, event_category, positions,
        )
        detailed = event_aligned_unit_scalar(
            event_aligned, "window_id", event_category, positions,
        )
        detailed = detailed[detailed["window_id"].isin(common_ids)]
        _box_strip_panel(
            ax, detailed, "mean_nll", seed=160 + index,
            paired_unit_col="window_id", show_n=False,
        )
        n_pairs = _draw_adjacent_summary_table(
            ax, paired, {"event_category": event_category}, fontsize=8.8,
        )
        ax.set_title(f"{title}\nN={n_pairs} matched windows", fontsize=10.5)
        ax.set_xlabel("Model", fontsize=10.5)
        ax.tick_params(labelsize=9.5)
        if col in (0, 2):
            ax.set_ylabel("Window-level mean NLL (nats)", fontsize=10.5)
    fig.suptitle(
        f"Sliding-window event-aligned NLL -- {mouse_id}\n"
        f"Page {page_number}/4: {EVENT_ALIGNED_SPLIT_LABELS[split_role]}\n"
        "Top: descriptive event trajectories (block transitions, then rare trials). Bottom: paired "
        "summaries at trial 0 and over the whole displayed range -- block transitions (row 3), rare "
        "trials (row 4).\n"
        "Events are aggregated within window first. Four adjacent two-sided Wilcoxon tests are "
        "Holm-adjusted within each panel.\n"
        "Gray lines connect matched windows; ↓ and |Δmed| use paired differences. Panel y-scales "
        "differ. Event counts are descriptive; the matched-window count is the test unit.\n"
        "Exploratory only: rolling windows overlap. Single mouse; within-subject comparison.",
        y=0.995, fontsize=11.5,
    )
    fig.subplots_adjust(hspace=1.05, wspace=0.42, top=0.81, bottom=0.10)
    return fig


def fig_sliding_window_event_aligned_nll(
    sw_event_aligned: pd.DataFrame, sw_event_paired: pd.DataFrame,
    sw_event_trial0_paired: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Figure 16, now 4 pages of the identical layout (see
    ``_build_sliding_window_event_aligned_page``), one per
    ``collect.EVENT_ALIGNED_SPLIT_ROLES``:

    1. **all** -- every session in each window pooled regardless of split
       (the original, unrestricted behavior).
    2. **train** -- only that window's 9 training sessions.
    3. **validation** -- only that window's 3 validation sessions.
    4. **test** -- only that window's 1 held-out test session.

    Saved as one multi-page PDF (page order above) plus one flat PNG per
    page; the plain ``fig16_sliding_window_event_aligned_nll.png`` mirrors
    page 1, matching the figure's behavior before it grew pages.
    """
    pages = [
        _build_sliding_window_event_aligned_page(
            sw_event_aligned, sw_event_paired, sw_event_trial0_paired,
            mouse_id, split_role, page_number,
        )
        for page_number, split_role in enumerate(EVENT_ALIGNED_SPLIT_ROLES, start=1)
    ]
    page_names = [f"page{page_number}_{split_role}" for page_number, split_role in enumerate(
        EVENT_ALIGNED_SPLIT_ROLES, start=1,
    )]
    _save_multi_page(pages, figures_dir, "fig16_sliding_window_event_aligned_nll", page_names)


def fig_single_session_event_aligned_nll(
    ss_event_aligned: pd.DataFrame, ss_event_paired: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Mirrors ``fig_sliding_window_event_aligned_nll`` for single-session
    fine-tuning: block transitions and rare trials within each session's own
    fitted trial sequence."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for ax, (event_category, title) in zip(axes.flat, _EVENT_CATEGORY_LAYOUT):
        common_ids = common_event_session_ids(ss_event_aligned, event_category)
        info = _plot_event_trajectory_panel(ax, ss_event_aligned, event_category, common_ids, "session_id")
        _draw_event_significance_text(ax, ss_event_paired, {"event_category": event_category})
        ax.set_title(f"{title}\nN={info['n_units']} matched sessions, {info['n_events']} events", fontsize=9.5)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Single-session event-aligned NLL -- {mouse_id}\n"
        "Events aggregated within each session first; 4 adjacent Holm-Wilcoxon tests per subplot use each "
        "session's mean NLL over the whole displayed range. Single mouse; within-subject comparison only.",
        y=1.01, fontsize=10.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.90)
    _save(fig, figures_dir, "fig17_single_session_event_aligned_nll")


_PSYCHOMETRIC_SOURCE_COLORS = {"sliding_window": "#1f77b4", "single_session": "#c62828"}


def fig_psychometric_comparison(summary: pd.DataFrame, mouse_id: str, figures_dir) -> None:
    """5 model rows x 4 block-type columns; each panel overlays the mouse's
    actual choices against two model predictions on the IDENTICAL trial
    population (every model, both sources -- see
    ``collect.collect_psychometric_comparison_trials``): the sliding-window
    model's zero-shot prediction (never trained on this session) versus the
    single-session model's fine-tuned prediction (trained directly on these
    trials, which are that session's blocks before the literal final two).
    Mouse choices are identical between the two sources by construction, so
    only one mouse curve is drawn per panel.
    """
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(PSYCHOMETRIC_BLOCK_ORDER), figsize=(17, 16.5), sharey=True,
    )
    for row_index, model_id in enumerate(MODEL_ORDER):
        model_summary = summary.loc[summary["model_id"] == model_id]
        n_sessions = (
            int(model_summary["n_matched_sessions"].iloc[0]) if not model_summary.empty else 0
        )
        for col_index, block_key in enumerate(PSYCHOMETRIC_BLOCK_ORDER):
            ax = axes[row_index, col_index]
            panel = model_summary.loc[model_summary["block_subset"] == block_key]
            if panel.empty:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center")
            else:
                mouse_source = panel.loc[panel["source"] == "single_session"]
                ax.errorbar(
                    mouse_source["bin_center_s"], mouse_source["mouse_probability_long"],
                    yerr=mouse_source["mouse_sem"], color="black", marker="o", markersize=4,
                    linewidth=1.3, label="Mouse", zorder=3,
                )
                for source in ("sliding_window", "single_session"):
                    curve = panel.loc[panel["source"] == source]
                    if curve.empty:
                        continue
                    ax.errorbar(
                        curve["bin_center_s"], curve["model_probability_long"],
                        yerr=curve["model_sem"], color=_PSYCHOMETRIC_SOURCE_COLORS[source],
                        marker="o", markersize=4, linewidth=1.3,
                        label=PSYCHOMETRIC_SOURCE_LABELS[source], zorder=2,
                    )
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, linestyle=":")
            ax.tick_params(labelsize=8.5)
            if row_index == 0:
                ax.set_title(PSYCHOMETRIC_BLOCK_LABELS[block_key], fontsize=11)
            if col_index == 0:
                ax.set_ylabel(
                    f"{MODEL_LABELS[model_id]}\nN={n_sessions} sessions\nP(long choice)", fontsize=9.5,
                )
            if row_index == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Stimulus duration (s)", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle(
        f"Sliding-window vs. single-session psychometrics -- {mouse_id}\n"
        "Same trial population for every model and both sources: each matched session's blocks before "
        "the literal final two, exact trial-index intersection across all 5 models -- N trials per "
        "panel is identical across every row. Mouse curve shown once per panel (identical between "
        "sources by construction). Four session-balanced quantile bins per trial type (short/long).",
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, figures_dir, "fig18_psychometric_comparison")


def fig_block_transition_performance_by_trial_type(
    block_transition_performance: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Figure 19: one page per model (5 pages; PDF only, no PNGs -- see
    ``_save_pdf_pages``). Each page is a plain 4x4 grid of Axes (rows =
    ``EVENT_ALIGNED_SPLIT_ROLES``, columns =
    ``BLOCK_TRANSITION_PERFORMANCE_COLUMNS``): the mouse's actual
    correctness (black) versus that model's expected correctness (color)
    across trials -10..+10 around a block transition, restricted to the
    column's own trial type (short/long stimulus) at each relative
    position. Mirrors the per-model "block transition performance" figure
    already generated under each model's own ``global_figures/`` (see
    ``src/m0_sensory_v1/analysis.py::_plot_event_grid``), extended with the
    train/validation/test breakdown and the trial-type split.

    Aggregated within window first (one number per window regardless of how
    many transitions/trials it contains -- see
    ``collect._aggregate_event_aligned_performance``), then averaged across
    whatever matched windows have data at each relative position -- some
    (relative_position, trial_type) combinations are sparser than others
    since block bias makes one trial type rarer near a transition. Panel
    y-axes are left to auto-scale (not fixed to 0-1) so that variation is
    visible regardless of a panel's own range.
    """
    pages = []
    n_rows, n_cols = len(EVENT_ALIGNED_SPLIT_ROLES), len(BLOCK_TRANSITION_PERFORMANCE_COLUMNS)
    for model_id in MODEL_ORDER:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20.0, 16.0))
        for row_index, split_role in enumerate(EVENT_ALIGNED_SPLIT_ROLES):
            for col_index, (event_category, trial_type, col_title) in enumerate(
                BLOCK_TRANSITION_PERFORMANCE_COLUMNS,
            ):
                ax = axes[row_index, col_index]
                panel = block_transition_performance[
                    (block_transition_performance["model_id"] == model_id)
                    & (block_transition_performance["split_role"] == split_role)
                    & (block_transition_performance["event_category"] == event_category)
                    & (block_transition_performance["trial_type"] == trial_type)
                ]
                trajectory = panel.groupby("relative_position").agg(
                    mouse_mean=("mean_mouse_correct", "mean"),
                    mouse_sem=("mean_mouse_correct", "sem"),
                    model_mean=("mean_model_probability_correct", "mean"),
                    model_sem=("mean_model_probability_correct", "sem"),
                ).sort_index()
                n_windows = int(panel["window_id"].nunique())
                if trajectory.empty:
                    ax.text(0.5, 0.5, "No matched windows", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.plot(
                        trajectory.index, trajectory["mouse_mean"], color="black", marker="o",
                        markersize=3, linewidth=1.4, label="Mouse",
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["mouse_mean"] - trajectory["mouse_sem"].fillna(0),
                        trajectory["mouse_mean"] + trajectory["mouse_sem"].fillna(0),
                        color="black", alpha=0.12,
                    )
                    ax.plot(
                        trajectory.index, trajectory["model_mean"], color=MODEL_COLORS[model_id],
                        marker="o", markersize=3, linewidth=1.4, label=MODEL_LABELS[model_id],
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["model_mean"] - trajectory["model_sem"].fillna(0),
                        trajectory["model_mean"] + trajectory["model_sem"].fillna(0),
                        color=MODEL_COLORS[model_id], alpha=0.18,
                    )
                ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
                if row_index == 0:
                    ax.set_title(f"{col_title}\nN={n_windows} matched windows", fontsize=10)
                else:
                    ax.set_title(f"N={n_windows} matched windows", fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"{EVENT_ALIGNED_SPLIT_LABELS[split_role]}\nP(correct)", fontsize=9)
                if row_index == n_rows - 1:
                    ax.set_xlabel("Relative trial position", fontsize=10)
                ax.tick_params(labelsize=8.5)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Block-transition performance by trial type -- {mouse_id} -- {MODEL_LABELS[model_id]}\n"
            "±10 trials around each block transition, restricted to that column's own trial type "
            "(short/long stimulus) at each relative position. Rows: all sessions pooled, then "
            "train/validation/test only. Mouse (black) vs model expected accuracy (color), aggregated "
            "within window first, then averaged across matched windows. Panel y-scales are not fixed.",
            fontsize=11.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pages.append(fig)
    _save_pdf_pages(pages, figures_dir, "fig19_sliding_window_block_transition_performance_by_trial_type")


@dataclass(frozen=True)
class FigureSpec:
    """One independently runnable post-analysis figure."""

    figure_id: str
    description: str
    render: Callable[[Callable[[str], pd.DataFrame], str, Path], None]


_FIGURE_SPECS = (
    FigureSpec(
        "fig01", "Sliding-window global and trial-subset NLL",
        lambda table, mouse, output: fig_global_and_detailed_nll(
            table("sliding_window_trial_subset_metrics.csv"),
            table("sliding_window_trial_subset_pooled_diamonds.csv"),
            table("sliding_window_trial_subset_paired_pairwise.csv"),
            mouse, output,
        ),
    ),
    FigureSpec(
        "fig02", "Sliding-window NLL trajectory",
        lambda table, mouse, output: fig_window_nll_trajectory(
            table("sliding_window_split_session_metrics.csv"), output,
        ),
    ),
    FigureSpec(
        "fig03", "Single-session training and held-out NLL",
        lambda table, mouse, output: fig_single_session_heldout_nll(
            table("single_session_trial_subset_metrics.csv"),
            table("single_session_trial_subset_paired_pairwise.csv"),
            mouse, output,
        ),
    ),
    FigureSpec(
        "fig04", "Single-session win counts",
        lambda table, mouse, output: fig_single_session_win_counts(
            table("single_session_win_counts_summary.csv"), output,
        ),
    ),
    FigureSpec(
        "fig05", "Shared-parameter sliding-window trajectories",
        lambda table, mouse, output: fig_shared_parameter_trajectories(
            table("sliding_window_parameters.csv"), output,
        ),
    ),
    FigureSpec(
        "fig06", "Sliding-window win counts",
        lambda table, mouse, output: fig_sliding_window_win_counts(
            table("sliding_window_win_counts_summary.csv"), output,
        ),
    ),
    FigureSpec(
        "fig07", "Single-session versus sliding-window win counts",
        lambda table, mouse, output: fig_win_counts_combined(
            table("single_session_win_counts_summary.csv"),
            table("sliding_window_win_counts_summary.csv"), output,
        ),
    ),
    FigureSpec(
        "fig08", "Held-out chronological winner timeline",
        lambda table, mouse, output: fig_chronological_winner_timeline_test(
            table("chronological_winner_timeline.csv"), output,
        ),
    ),
    FigureSpec(
        "fig09", "Model-specific sliding-window parameters",
        lambda table, mouse, output: fig_sliding_window_model_specific_parameters(
            table("sliding_window_parameters.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig10", "Single-session fit availability",
        lambda table, mouse, output: fig_single_session_availability(
            table("single_session_availability.csv"), output,
        ),
    ),
    FigureSpec(
        "fig11", "Shared single-session parameter trajectories",
        lambda table, mouse, output: fig_single_session_shared_parameter_trajectories(
            table("single_session_parameters.csv"), output,
        ),
    ),
    FigureSpec(
        "fig12", "Model-specific single-session parameters",
        lambda table, mouse, output: fig_single_session_model_specific_parameters(
            table("single_session_parameters.csv"), output,
        ),
    ),
    FigureSpec(
        "fig13", "Training-fit chronological winner timeline",
        lambda table, mouse, output: fig_chronological_winner_timeline_train(
            table("chronological_winner_timeline_train.csv"), output,
        ),
    ),
    FigureSpec(
        "fig14", "Training-versus-test winner agreement",
        lambda table, mouse, output: fig_train_vs_test_winner_agreement(
            table("chronological_train_vs_test_winner_agreement_summary.csv"), output,
        ),
    ),
    FigureSpec(
        "fig15", "AIC/BIC model comparison",
        lambda table, mouse, output: fig_information_criteria(
            table("information_criteria_sliding_window.csv"),
            table("information_criteria_single_session.csv"),
            table("information_criteria_paired_pairwise.csv"),
            mouse, output,
        ),
    ),
    FigureSpec(
        "fig16", "Sliding-window event-aligned NLL",
        lambda table, mouse, output: fig_sliding_window_event_aligned_nll(
            table("sliding_window_event_aligned_nll.csv"),
            table("sliding_window_event_aligned_paired_pairwise.csv"),
            table("sliding_window_event_aligned_trial0_paired_pairwise.csv"),
            mouse, output,
        ),
    ),
    FigureSpec(
        "fig17", "Single-session event-aligned NLL",
        lambda table, mouse, output: fig_single_session_event_aligned_nll(
            table("single_session_event_aligned_nll.csv"),
            table("single_session_event_aligned_paired_pairwise.csv"),
            mouse, output,
        ),
    ),
    FigureSpec(
        "fig18", "Sliding-window vs. single-session psychometrics",
        lambda table, mouse, output: fig_psychometric_comparison(
            table("psychometric_comparison_summary.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig19", "Block-transition performance by trial type (5 pages, PDF only)",
        lambda table, mouse, output: fig_block_transition_performance_by_trial_type(
            table("sliding_window_block_transition_performance.csv"), mouse, output,
        ),
    ),
)

_FIGURE_BY_ID = {spec.figure_id: spec for spec in _FIGURE_SPECS}
_FIGURE_SELECTOR = re.compile(r"^(?:fig)?(\d{1,2})$", re.IGNORECASE)


def list_figures() -> tuple[tuple[str, str], ...]:
    """Return the canonical figure IDs and their human-readable descriptions."""
    return tuple((spec.figure_id, spec.description) for spec in _FIGURE_SPECS)


def normalize_figure_selectors(
    figures: str | int | Iterable[str | int] | None,
) -> tuple[str, ...]:
    """Validate aliases, remove duplicates, and return figure IDs in numeric order.

    ``None`` or the selector ``all`` selects all figures. Individual selectors
    accept ``3``, ``03``, ``fig3``, or ``fig03``. An empty iterable is invalid.
    Every selector is validated before rendering begins.
    """
    if figures is None:
        return tuple(_FIGURE_BY_ID)
    selectors = [figures] if isinstance(figures, (str, int)) else list(figures)
    if not selectors:
        raise ValueError("No figures selected; omit --figure to render all figures")

    selected: set[str] = set()
    invalid = []
    for selector in selectors:
        if isinstance(selector, bool):
            invalid.append(repr(selector))
            continue
        text = str(selector).strip()
        if text.lower() == "all":
            selected.update(_FIGURE_BY_ID)
            continue
        match = _FIGURE_SELECTOR.fullmatch(text)
        figure_id = f"fig{int(match.group(1)):02d}" if match else None
        if figure_id not in _FIGURE_BY_ID:
            invalid.append(repr(selector))
        else:
            selected.add(figure_id)
    if invalid:
        valid = ", ".join(_FIGURE_BY_ID)
        raise ValueError(
            f"Unknown figure selector(s): {', '.join(invalid)}. "
            f"Use one of {valid}, a number from 1 to 19, or 'all'."
        )
    return tuple(spec.figure_id for spec in _FIGURE_SPECS if spec.figure_id in selected)


class _TableCache:
    """Load each selected figure table at most once during one render run."""

    def __init__(self, tables_dir: Path):
        self.tables_dir = tables_dir
        self._tables: dict[str, pd.DataFrame] = {}

    def __call__(self, filename: str) -> pd.DataFrame:
        if filename not in self._tables:
            self._tables[filename] = pd.read_csv(self.tables_dir / filename)
        return self._tables[filename]


def run(
    mouse_id: str, figures: str | int | Iterable[str | int] | None = None,
) -> tuple[str, ...]:
    """Render only the requested figures from existing post-analysis tables.

    The default remains all figures for backward compatibility. Selection
    controls which files are regenerated; older unselected outputs are left
    untouched in the figures directory.
    """
    selected_ids = normalize_figure_selectors(figures)
    tables_dir, figures_dir = common.ensure_output_dirs("model_comparison", mouse_id)
    table = _TableCache(tables_dir)

    for figure_id in selected_ids:
        figures_open_before = set(plt.get_fignums())
        try:
            _FIGURE_BY_ID[figure_id].render(table, mouse_id, figures_dir)
        finally:
            # Renderers normally close through _save(). Also clean up if a
            # plotting or schema error occurs before control reaches _save().
            for figure_number in set(plt.get_fignums()) - figures_open_before:
                plt.close(figure_number)
        print(f"wrote {figure_id} to {figures_dir}")
    return selected_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC06", help="Mouse/subject ID, e.g. MC06")
    parser.add_argument(
        "--figure", dest="figures", action="append", metavar="ID",
        help=(
            "Figure to render (repeatable): 3, 03, fig03, or all. "
            "Omit to render all 17 figures."
        ),
    )
    parser.add_argument(
        "--list-figures", action="store_true",
        help="List available figure IDs and exit without reading tables or writing outputs.",
    )
    args = parser.parse_args()
    if args.list_figures:
        if args.figures:
            parser.error("--list-figures cannot be combined with --figure")
        for figure_id, description in list_figures():
            print(f"{figure_id}: {description}")
        return
    try:
        selected_ids = normalize_figure_selectors(args.figures)
    except ValueError as exc:
        parser.error(str(exc))
    run(args.mouse, selected_ids)


if __name__ == "__main__":
    main()
