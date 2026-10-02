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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE,
    RARE_TRIAL_PERFORMANCE_COLUMNS,
    RARE_TRIAL_RELATIVE_RANGE,
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

PARAMETER_SUMMARY_ORDER = (*SHARED_PARAM_NAMES, *NON_SHARED_PARAM_MODELS)

MODEL_SPECIFIC_PARAMETER_CARDS = {
    "ah": (
        "Switch hazard and learned block duration",
        "M1–M3: hB = sigmoid(ah); expected biased-block length = 1/hB trials.\n"
        "M4: hB(23) = sigmoid(ah), but bh changes hazard with elapsed run length, "
        "so M4 has no single constant hazard. Its curve at right instead plots the "
        "exact mean of its full duration law (ah and bh together).",
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
    "bh_half_life": (
        "Run length at half-maximum switch hazard (M4)",
        "r* solves hB(r*) = 0.5, i.e. r* = 23 − ah/bh.\n"
        "Plotted only where that crossing actually falls inside M4's representable run-length range "
        "[1, 65]; outside it the fitted curve never reaches 0.5 there, so the point is left blank (NaN) "
        "rather than showing a meaningless extrapolated run length.",
    ),
    "bh_hazard_curves": (
        "M4's fitted hazard curve, one line per rolling window",
        "hB(r) = sigmoid(ah + bh × (r−23)) plotted directly (not a derived scalar) for every M4 window.\n"
        "Color runs from a pale tint (earliest window) to a dark shade (latest) of M4's report color, so "
        "drift in the curve's shape across the sliding-window fits is visible at a glance. No neutral "
        "hazard, no r=23 marker -- see fig22 for the full-size standalone version of this same plot.",
    ),
    "bh_hazard_at_r20": (
        "Switch hazard probability at r=20 (M4)",
        "hB(20) = sigmoid[ah + bh × (20−23)], evaluated at run length r=20 rather than the registered "
        "center r=23.\nM1-M3 have no run-length dependence (hB is constant in r), so this value only "
        "carries information beyond raw ah for M4 -- shown here for M4 only.",
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


def _window_nll_trajectory_data(
    sw_split_session: pd.DataFrame,
    sw_trial_subset: pd.DataFrame,
    split_role: str,
    model_id: str,
) -> pd.DataFrame:
    """Join each existing pooled mean to its within-window trial-level SEM."""
    required_uncertainty_columns = {"mean_nll", "nll_sem", "n_trials"}
    if missing := required_uncertainty_columns.difference(sw_trial_subset.columns):
        raise ValueError(
            "sliding_window_trial_subset_metrics.csv is missing Figure 02 "
            f"uncertainty column(s) {sorted(missing)}; rerun "
            "model_comparison.collect before rendering fig02"
        )
    keys = ["model_id", "window_id", "split_role"]
    centers = sw_split_session[
        (sw_split_session["split_role"] == split_role)
        & (sw_split_session["session_id"] == "ALL_SESSIONS")
        & (sw_split_session["model_id"] == model_id)
    ].copy()
    uncertainty = sw_trial_subset[
        (sw_trial_subset["split_role"] == split_role)
        & (sw_trial_subset["trial_subset"] == "all")
        & (sw_trial_subset["model_id"] == model_id)
    ][keys + ["mean_nll", "nll_sem", "n_trials"]].copy()
    uncertainty = uncertainty.rename(columns={"mean_nll": "trial_mean_nll"})
    result = centers.merge(
        uncertainty, on=keys, how="left", validate="one_to_one", indicator=True,
    )
    if not result["_merge"].eq("both").all():
        missing_windows = result.loc[result["_merge"] != "both", "window_id"].tolist()
        raise ValueError(
            f"Missing trial-level SEM rows for {model_id}/{split_role} window(s) "
            f"{missing_windows}; rerun model_comparison.collect"
        )
    center = pd.to_numeric(result["mean_nll"], errors="coerce")
    trial_center = pd.to_numeric(result["trial_mean_nll"], errors="coerce")
    centers_match = (
        np.isfinite(center)
        & np.isfinite(trial_center)
        & np.isclose(center, trial_center, rtol=1e-10, atol=1e-12)
    )
    if not centers_match.all():
        bad_windows = result.loc[~centers_match, "window_id"].tolist()
        raise ValueError(
            f"Pooled and trial-level mean NLL disagree for {model_id}/{split_role} "
            f"window(s) {bad_windows}; rerun collection from consistent inputs"
        )
    result["nll_sem"] = pd.to_numeric(result["nll_sem"], errors="coerce")
    result["n_trials"] = pd.to_numeric(result["n_trials"], errors="coerce")
    invalid_count = (
        ~np.isfinite(result["n_trials"])
        | (result["n_trials"] < 1)
        | (result["n_trials"] % 1 != 0)
    )
    invalid_sem = (
        invalid_count
        | (
            (result["n_trials"] >= 2)
            & (~np.isfinite(result["nll_sem"]) | (result["nll_sem"] < 0))
        )
    )
    if invalid_sem.any():
        bad_windows = result.loc[invalid_sem, "window_id"].tolist()
        raise ValueError(
            f"Invalid trial-level SEM values for {model_id}/{split_role} window(s) "
            f"{bad_windows}; rerun model_comparison.collect"
        )
    result.loc[result["n_trials"] == 1, "nll_sem"] = np.nan
    return result.drop(columns=["_merge"]).sort_values("test_rank")


def fig_window_nll_trajectory(
    sw_split_session: pd.DataFrame,
    sw_trial_subset: pd.DataFrame,
    figures_dir,
) -> None:
    """Pooled mean NLL +/- within-window individual-trial SEM by split."""
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    for ax, split_role in zip(axes, ["train", "validation", "test"]):
        for model_id in MODEL_ORDER:
            subset = _window_nll_trajectory_data(
                sw_split_session, sw_trial_subset, split_role, model_id,
            )
            if subset.empty:
                continue
            ax.errorbar(
                subset["test_rank"], subset["mean_nll"], yerr=subset["nll_sem"],
                fmt="-o", markersize=3.5, linewidth=1.4, elinewidth=0.9,
                capsize=2.0, capthick=0.9, color=MODEL_COLORS[model_id],
                label=MODEL_LABELS[model_id],
            )
        ax.set_ylabel("Mean NLL")
        if split_role == "test":
            ax.set_title(
                "Test mean NLL ± SEM across individual-trial NLL values "
                "(within each held-out session)"
            )
        else:
            ax.set_title(
                f"{split_role.capitalize()} mean NLL ± SEM across individual-trial "
                "NLL values (within each window)"
            )
    axes[-1].set_xlabel("Rolling-window test session (chronological rank)")
    axes[0].legend(loc="upper right", ncol=5, bbox_to_anchor=(1.0, 1.35))
    fig.suptitle("NLL across the 32 rolling windows", y=1.0, fontsize=12.5)
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


def _parameter_summary_models(param_name: str) -> list[str]:
    """Models that fit one constrained parameter, in nesting order."""
    return MODEL_ORDER if param_name in SHARED_PARAM_NAMES else NON_SHARED_PARAM_MODELS[param_name]


def summarize_sliding_window_parameters(sw_params: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD of each fitted parameter across rolling windows.

    All expected model/parameter combinations are returned, including groups
    with no finite values (``n_windows=0``), so Figure 21 always preserves its
    required 5/4/3/2/1 model layout.
    """
    required = {"model_id", "window_id", "param_name", "param_value"}
    if missing := required.difference(sw_params.columns):
        raise ValueError(
            f"sliding_window_parameters.csv missing column(s) {sorted(missing)}"
        )

    unknown_models = sorted(set(sw_params["model_id"].dropna()) - set(MODEL_ORDER))
    if sw_params["model_id"].isna().any() or unknown_models:
        raise ValueError(f"Unexpected sliding-window parameter models: {unknown_models}")
    if sw_params["window_id"].isna().any():
        raise ValueError("Sliding-window parameter rows must have a window_id")
    placeholder = sw_params["param_name"].isna()
    invalid_placeholder = placeholder & sw_params["param_value"].notna()
    if invalid_placeholder.any():
        raise ValueError("Null parameter names are only valid for empty-fit placeholder rows")

    data = sw_params.loc[~placeholder].copy()
    unknown_params = sorted(set(data["param_name"].dropna()) - set(PARAMETER_SUMMARY_ORDER))
    if unknown_params:
        raise ValueError(f"Unexpected sliding-window parameters: {unknown_params}")
    if data.duplicated(["model_id", "window_id", "param_name"]).any():
        raise ValueError("Duplicate (model_id, window_id, param_name) parameter rows")

    expected_pairs = {
        (param_name, model_id)
        for param_name in PARAMETER_SUMMARY_ORDER
        for model_id in _parameter_summary_models(param_name)
    }
    observed_pairs = set(zip(data["param_name"], data["model_id"]))
    if invalid_pairs := sorted(observed_pairs - expected_pairs):
        raise ValueError(f"Model does not fit the supplied parameter: {invalid_pairs}")

    numeric_values = pd.to_numeric(data["param_value"], errors="coerce")
    invalid_numeric = data["param_value"].notna() & numeric_values.isna()
    if invalid_numeric.any():
        raise ValueError("Sliding-window parameter values must be numeric")
    data["param_value"] = numeric_values
    data = data[np.isfinite(data["param_value"])]

    rows = []
    for param_name in PARAMETER_SUMMARY_ORDER:
        for model_id in _parameter_summary_models(param_name):
            subset = data[
                (data["param_name"] == param_name) & (data["model_id"] == model_id)
            ]
            values = subset["param_value"]
            n_windows = int(subset["window_id"].nunique())
            rows.append({
                "param_name": param_name,
                "model_id": model_id,
                "mean": float(values.mean()) if n_windows else float("nan"),
                "sd": float(values.std(ddof=1)) if n_windows >= 2 else float("nan"),
                "n_windows": n_windows,
            })
    return pd.DataFrame(rows)


def fig_sliding_window_parameter_mean_sd(
    sw_params: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Nine-panel mean ± SD summary of constrained rolling-window parameters."""
    summary = summarize_sliding_window_parameters(sw_params)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 11.0))
    for ax, param_name in zip(axes.flat, PARAMETER_SUMMARY_ORDER):
        model_ids = _parameter_summary_models(param_name)
        panel = summary[summary["param_name"] == param_name].set_index("model_id")
        for x, model_id in enumerate(model_ids):
            row = panel.loc[model_id]
            if not np.isfinite(row["mean"]):
                continue
            if row["n_windows"] >= 2 and np.isfinite(row["sd"]):
                ax.errorbar(
                    x, row["mean"], yerr=row["sd"], fmt="o", markersize=7,
                    capsize=5, capthick=1.3, elinewidth=1.3,
                    color=MODEL_COLORS[model_id], markeredgecolor="black",
                    markeredgewidth=0.6,
                )
            else:
                ax.plot(
                    x, row["mean"], marker="o", markersize=7, linestyle="none",
                    color=MODEL_COLORS[model_id], markeredgecolor="black",
                    markeredgewidth=0.6,
                )
        if param_name in {"alpha_theta", "bh"}:
            ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xticks(range(len(model_ids)))
        ax.set_xticklabels([
            f"{model_id}\n(n={int(panel.loc[model_id, 'n_windows'])})"
            for model_id in model_ids
        ])
        ax.set_xlim(-0.6, max(len(model_ids) - 0.4, 0.6))
        ax.set_title(param_name)
        ax.set_ylabel("Constrained fitted value")

    fig.suptitle(
        f"Sliding-window fitted parameters: mean ± SD — {mouse_id}",
        y=0.995, fontsize=13,
    )
    fig.text(
        0.5, 0.012,
        "Dots are means and whiskers are ±1 sample SD across each model's finite fitted rolling-window "
        "values (n shown below each model). Models can have different available windows; rolling windows "
        "overlap, so variability is descriptive rather than independent-sample uncertainty. Symmetric SD "
        "whiskers are not parameter bounds and may extend outside a parameter's valid domain.",
        ha="center", fontsize=8.5, style="italic", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.965], h_pad=2.0, w_pad=2.2)
    _save(fig, figures_dir, "fig21_sliding_window_parameter_mean_sd")


def _sigmoid_series(values: pd.Series) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-values.to_numpy(dtype=float))), index=values.index)


# Fixed (unfitted) M4 registry constants from config/m4_hsmm_delta_v1.json:
# the hazard run-length center and the represented-run-length cap beyond
# which the registered approximation freezes the hazard. These are
# architectural constants of the frozen M4 spec, not fitted per window.
_M4_HAZARD_CENTER = 23
_M4_MAXIMUM_REPRESENTED_RUN_LENGTH = 65

# Pale tint -> M4's report color -> dark shade, used to order M4's per-window
# hazard curves chronologically (fig09's bh_hazard_curves row and fig22).
_M4_HAZARD_CURVE_COLORMAP = LinearSegmentedColormap.from_list(
    "m4_hazard_curve_order", ["#F6E1EE", MODEL_COLORS["M4"], "#3B0A24"],
)


def _m4_hazard_curve(ah: float, bh: float, run_lengths: np.ndarray) -> np.ndarray:
    """hB(r) = sigmoid(ah + bh*(r-23)) for one fitted (ah, bh), evaluated at
    ``run_lengths`` (see ``m4_hsmm_delta_v1.parameters.M4DecodedParameters.
    hazard``, which this duplicates in closed form -- no lapse term)."""
    return 1.0 / (1.0 + np.exp(-(ah + bh * (run_lengths - _M4_HAZARD_CENTER))))


def _plot_m4_hazard_curve_family(ax: plt.Axes, sw_params: pd.DataFrame, *, colorbar: bool = True) -> None:
    """M4's fitted biased hazard hB(r) for every rolling window, one line
    each, ordered chronologically (test_rank) from a pale tint (earliest) to
    a dark shade (latest) of M4's report color. No neutral hazard, no r=23
    reference line -- shared by fig09's bh_hazard_curves row and fig22 so
    the two can't drift apart.
    """
    ah_series = _param_series_by_model(sw_params, "ah").get("M4")
    bh_series = _param_series_by_model(sw_params, "bh").get("M4")
    if ah_series is None or bh_series is None:
        ax.text(0.5, 0.5, "No M4 windows available", ha="center", va="center", transform=ax.transAxes)
        return
    joined = pd.concat({"ah": ah_series, "bh": bh_series}, axis=1).dropna().sort_index()
    if joined.empty:
        ax.text(0.5, 0.5, "No M4 windows available", ha="center", va="center", transform=ax.transAxes)
        return
    run_lengths = np.arange(1, _M4_MAXIMUM_REPRESENTED_RUN_LENGTH + 1, dtype=float)
    colors = _M4_HAZARD_CURVE_COLORMAP(np.linspace(0.08, 0.95, len(joined)))
    for color, (_, row) in zip(colors, joined.iterrows()):
        ax.plot(
            run_lengths, _m4_hazard_curve(row["ah"], row["bh"], run_lengths),
            color=color, linewidth=1.1, alpha=0.9, zorder=2,
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(1, _M4_MAXIMUM_REPRESENTED_RUN_LENGTH)
    ax.set_ylabel("hB(r)")
    if colorbar:
        scalar_mappable = plt.cm.ScalarMappable(
            cmap=_M4_HAZARD_CURVE_COLORMAP,
            norm=plt.Normalize(vmin=joined.index.min(), vmax=joined.index.max()),
        )
        scalar_mappable.set_array([])
        cbar = ax.figure.colorbar(scalar_mappable, ax=ax, pad=0.014, fraction=0.035)
        cbar.set_label("Test rank\n(chronological)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)


def _m4_expected_block_length(ah: float, bh: float) -> float:
    """Exact mean of M4's biased-block duration law for one fitted (ah, bh).

    M4's hazard is h(r) = sigmoid(ah + bh*(r-23)) for r < 65, frozen at
    h(65) for r >= 65 (the registered constant-tail approximation; see
    ``m4_hsmm_delta_v1.diagnostics.build_m4_implied_duration_distribution``).
    Unlike M1-M3, this hazard is NOT constant in r, so the expected block
    length is not `1/hB` for any single hB -- it is the mean of the full
    discrete duration law, summed here in closed form (individual terms for
    r=1..64, then an exact geometric-tail sum for r>=65, which is finite
    because the tail hazard is frozen rather than simulated step by step).
    This intentionally duplicates (rather than imports) the model-side
    computation: this analysis package has no dependency on the torch model
    code, but the two are numerically verified to agree (see
    ``tests/test_post_analysis_model_comparison.py``).
    """
    if not np.isfinite(ah) or not np.isfinite(bh):
        return float("nan")
    center = _M4_HAZARD_CENTER
    cap = _M4_MAXIMUM_REPRESENTED_RUN_LENGTH
    run_lengths = np.arange(1, cap, dtype=float)  # 1 .. cap-1
    hazards = 1.0 / (1.0 + np.exp(-(ah + bh * (run_lengths - center))))
    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - hazards)[:-1]))
    pmf = survival_before * hazards
    mean_before_tail = float(np.sum(run_lengths * pmf))
    survival_at_tail_start = float(np.prod(1.0 - hazards))
    tail_hazard = 1.0 / (1.0 + np.exp(-(ah + bh * (cap - center))))
    tail_mean_contribution = survival_at_tail_start * ((cap - 1) + 1.0 / tail_hazard)
    return mean_before_tail + tail_mean_contribution


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
        bh_raw = _param_series_by_model(sw_params, "bh")
        derived = {}
        for model_id, ah_series in raw.items():
            if model_id == "M4":
                # M4's hazard varies with run length (sigmoid(ah + bh*(r-23))),
                # so 1/sigmoid(ah) is only the reciprocal hazard AT r=23, not
                # the model's actual expected block length -- use the closed-
                # form mean of the full duration law instead (see
                # _m4_expected_block_length).
                bh_series = bh_raw.get(model_id)
                if bh_series is None:
                    continue
                joined = pd.concat({"ah": ah_series, "bh": bh_series}, axis=1).dropna()
                derived[model_id] = pd.Series(
                    {
                        rank: _m4_expected_block_length(row["ah"], row["bh"])
                        for rank, row in joined.iterrows()
                    }
                )
            else:
                derived[model_id] = 1.0 / _sigmoid_series(ah_series)
        return [
            (
                "Expected block length = 1/sigmoid(ah) (M1-M3); "
                "exact duration-law mean (M4, hazard varies with run length)",
                "Expected block length (trials)",
                derived,
            )
        ]
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
    if param_name == "bh_half_life":
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
            with np.errstate(divide="ignore", invalid="ignore"):
                r_half = _M4_HAZARD_CENTER - joined["ah"] / joined["bh"]
            in_range = r_half.between(1, _M4_MAXIMUM_REPRESENTED_RUN_LENGTH)
            derived[model_id] = r_half.where(in_range)
        return [(
            "r where hB(r) = 0.5 (blank if the crossing falls outside [1, 65])",
            "Represented run length r (trials)",
            derived,
        )]
    if param_name == "bh_hazard_at_r20":
        fixed_run_length = 20
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
            derived[model_id] = _sigmoid_series(
                joined["ah"] + joined["bh"] * (fixed_run_length - _M4_HAZARD_CENTER)
            )
        return [(
            "hB(r=20) = sigmoid[ah + bh×(20−23)]",
            "Switch hazard probability at r=20",
            derived,
        )]
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



# Row order for fig09. The first four are real fitted coordinates (one raw
# trajectory each, column 1); the last three are pure derived views of
# ah/bh with no fitted coordinate of their own, so column 1 is blank for
# them (``_MODEL_SPECIFIC_ROW_RAW_PARAM[row] is None``) except
# "bh_hazard_curves", which uses column 1 for the hazard-curve family plot
# instead (a different x-axis -- represented run length, not chronological
# window rank -- so it is deliberately excluded from that column's shared x).
_MODEL_SPECIFIC_ROWS = (
    "ah", "delta0", "alpha_theta", "bh",
    "bh_half_life", "bh_hazard_curves", "bh_hazard_at_r20",
)
_MODEL_SPECIFIC_ROW_RAW_PARAM = {
    "ah": "ah", "delta0": "delta0", "alpha_theta": "alpha_theta", "bh": "bh",
    "bh_half_life": None, "bh_hazard_curves": None, "bh_hazard_at_r20": None,
}


def fig_sliding_window_model_specific_parameters(
    sw_params: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Trajectories for the parameters each nested model adds beyond the
    shared five (ah, delta0, alpha_theta, bh) -- only plotted for the models
    that actually fit that coordinate -- plus three pure-derived M4 hazard
    views (see ``_MODEL_SPECIFIC_ROWS``).
    """
    curve_linestyles = ["-", "--"]
    n_rows = len(_MODEL_SPECIFIC_ROWS)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(21.5, 20.0),
        gridspec_kw={"width_ratios": [2.25, 1.0, 2.0]},
    )
    all_ranks = sorted(pd.to_numeric(sw_params["test_rank"], errors="coerce").dropna().unique())
    hazard_curve_row_index = _MODEL_SPECIFIC_ROWS.index("bh_hazard_curves")

    # Column 1 and column 3 share an x-axis (chronological test rank) across
    # every row EXCEPT the hazard-curve-family row, whose x-axis is
    # represented run length r -- sharing it with the rest would force a
    # nonsensical common range on every other row.
    for col in (0, 2):
        shared_rows = [i for i in range(n_rows) if i != hazard_curve_row_index]
        reference_ax = axes[shared_rows[0], col]
        for row_index in shared_rows[1:]:
            axes[row_index, col].sharex(reference_ax)
        for row_index in shared_rows[:-1]:
            plt.setp(axes[row_index, col].get_xticklabels(), visible=False)
    # Column 1's last row with real (non-blank) content is "bh" (row 3), not
    # the grid's actual last row -- rows 4 and 6 turn column 1 off entirely
    # (pure-derived rows), so re-show the tick labels the loop above hid.
    plt.setp(axes[3, 0].get_xticklabels(), visible=True)

    for row_index, row_key in enumerate(_MODEL_SPECIFIC_ROWS):
        ax, meaning_ax, derived_ax = axes[row_index]
        raw_param_name = _MODEL_SPECIFIC_ROW_RAW_PARAM[row_key]

        if row_key == "bh_hazard_curves":
            _plot_m4_hazard_curve_family(ax, sw_params)
            ax.set_xlabel("Represented run length r (trials)")
        elif raw_param_name is not None:
            subset_all = sw_params[sw_params["param_name"] == raw_param_name]
            for model_id in NON_SHARED_PARAM_MODELS[raw_param_name]:
                subset = subset_all[subset_all["model_id"] == model_id].sort_values("test_rank")
                if subset.empty:
                    continue
                if subset["test_rank"].duplicated().any():
                    raise ValueError(f"Duplicate test ranks for {model_id}/{raw_param_name}")
                trajectory = pd.DataFrame({"test_rank": all_ranks}).merge(
                    subset[["test_rank", "param_value"]], on="test_rank", how="left",
                )
                ax.plot(
                    trajectory["test_rank"], trajectory["param_value"], markersize=3,
                    linewidth=1.2, color=MODEL_COLORS[model_id],
                    linestyle=MODEL_LINESTYLES[model_id], marker=MODEL_MARKERS[model_id],
                    label=f"{MODEL_LABELS[model_id]} ({subset['param_value'].notna().sum()} fitted windows)",
                )
            if raw_param_name in {"alpha_theta", "bh"}:
                ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_ylabel(raw_param_name)
            ax.legend(loc="upper right", ncol=len(NON_SHARED_PARAM_MODELS[raw_param_name]), fontsize=8)
        else:
            ax.axis("off")
            ax.text(
                0.5, 0.5, "derived from ah & bh\n(rows 1, 4 above)", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="0.5", style="italic",
            )

        meaning_ax.axis("off")
        heading, explanation = MODEL_SPECIFIC_PARAMETER_CARDS[row_key]
        meaning_ax.text(
            0.02, 0.88, heading, transform=meaning_ax.transAxes,
            ha="left", va="top", fontsize=10.5, fontweight="bold",
        )
        meaning_ax.text(
            0.02, 0.70, _wrap(explanation, width=54), transform=meaning_ax.transAxes,
            ha="left", va="top", fontsize=9.0, linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "0.97", "edgecolor": "0.82"},
        )

        if row_key == "bh_hazard_curves":
            derived_ax.axis("off")
            continue

        # Third column: the closed-form quantity the explanation actually
        # defines (e.g. expected block length, not raw ah), plus a per-model
        # straight-line trend summarizing its drift across chronological
        # sessions.
        curves = _derived_quantity_curves(row_key, sw_params)
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
        if row_key in {"alpha_theta", "bh"}:
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
    axes[3, 0].set_xlabel("Rolling-window test session (chronological rank)")
    axes[-1, 2].set_xlabel("Rolling-window test session (chronological rank)")
    fig.suptitle(
        f"Model-specific parameters across rolling windows -- {mouse_id}",
        y=0.995, fontsize=12.5,
    )
    fig.text(
        0.5, 0.005,
        "Single mouse; point estimates from overlapping rolling windows. Lines show chronological "
        "sequence, not independent replications. Gaps are unavailable fits; a model absent from a row "
        "does not contain that parameter. Column 3 plots the closed-form quantity named in column 2's "
        "explanation (derived from column 1's fitted values) with a per-model linear-trend overlay, "
        "except the hazard-curve-family row, which plots hB(r) directly in column 1 against represented "
        "run length r (not chronological rank -- that row's x-axis is intentionally not shared with the "
        "others).",
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


_FIGURE19_RELATIVE_POSITIONS = pd.Index(
    tuple(BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE), name="relative_position",
)


def _window_count_range_label(
    counts: pd.Series, support: str, positions: pd.Index = _FIGURE19_RELATIVE_POSITIONS,
) -> str:
    """Compact pointwise-window-count label for a Figure 19/23-style panel."""
    counts = pd.to_numeric(counts, errors="coerce").reindex(
        positions, fill_value=0,
    ).fillna(0)
    lo, hi = int(counts.min()), int(counts.max())
    count = str(lo) if lo == hi else f"{lo}–{hi}"
    return f"N={count} {support} windows/position"


def _matched_block_transition_overlay_summary(
    panel: pd.DataFrame, *, label: str = "Figure 19",
) -> pd.DataFrame:
    """Summarize one Figure 19/23-style overlay panel on pointwise common support.

    At each relative position, only window IDs represented exactly once for
    every model are retained. The five copies of the observed mouse statistic
    and trial count must agree before one copy is used, preventing the mouse's
    uncertainty from being spuriously reduced by treating model copies as
    independent observations.
    """
    columns = ["series_id", "relative_position", "mean", "sem", "n_windows"]
    if panel.empty:
        return pd.DataFrame(columns=columns)

    required_columns = {
        "model_id", "window_id", "relative_position", "mean_mouse_correct",
        "mean_model_probability_correct", "n_trials", "support_signature",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            f"{label} table is stale or malformed; rerun the collection stage. "
            f"Missing columns: {sorted(missing_columns)}"
        )
    work = panel.loc[panel["model_id"].isin(MODEL_ORDER)].copy()
    key_columns = ["model_id", "window_id", "relative_position"]
    duplicated = work.duplicated(key_columns, keep=False)
    if duplicated.any():
        examples = work.loc[duplicated, key_columns].head().to_dict("records")
        raise ValueError(f"Duplicate {label} overlay rows: {examples}")

    numeric_columns = [
        "mean_mouse_correct", "mean_model_probability_correct", "n_trials",
    ]
    work[numeric_columns] = work[numeric_columns].apply(pd.to_numeric, errors="coerce")
    summaries = []
    for relative_position, position_rows in work.groupby("relative_position", sort=True):
        windows_by_model = [
            set(position_rows.loc[position_rows["model_id"] == model_id, "window_id"].dropna())
            for model_id in MODEL_ORDER
        ]
        common_windows = set.intersection(*windows_by_model) if all(windows_by_model) else set()
        validated_units = []
        for window_id in common_windows:
            unit = position_rows.loc[position_rows["window_id"] == window_id]
            if len(unit) != len(MODEL_ORDER) or set(unit["model_id"]) != set(MODEL_ORDER):
                raise ValueError(
                    f"Malformed {label} overlay rows for "
                    f"window={window_id}, relative_position={relative_position}"
                )
            if not np.isfinite(unit[numeric_columns].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Non-finite {label} overlay values for "
                    f"window={window_id}, relative_position={relative_position}"
                )
            mouse_values = unit["mean_mouse_correct"].to_numpy(dtype=float)
            trial_counts = unit["n_trials"].to_numpy(dtype=float)
            support_signatures = unit["support_signature"]
            if (
                support_signatures.isna().any()
                or support_signatures.astype(str).str.strip().eq("").any()
            ):
                raise ValueError(
                    f"Missing trial support signature for {label} overlay at "
                    f"window={window_id}, relative_position={relative_position}"
                )
            support_signatures = support_signatures.astype(str)
            if not np.allclose(mouse_values, mouse_values[0], rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"Mouse correctness differs across models for {label} overlay at "
                    f"window={window_id}, relative_position={relative_position}"
                )
            if not np.all(trial_counts == trial_counts[0]):
                raise ValueError(
                    f"Trial counts differ across models for {label} overlay at "
                    f"window={window_id}, relative_position={relative_position}"
                )
            if support_signatures.nunique(dropna=False) != 1:
                raise ValueError(
                    f"Trial support differs across models for {label} overlay at "
                    f"window={window_id}, relative_position={relative_position}"
                )
            validated_units.append(unit)

        if not validated_units:
            continue
        matched = pd.concat(validated_units, ignore_index=True)
        n_windows = len(validated_units)
        mouse_values = matched.loc[
            matched["model_id"] == MODEL_ORDER[0], "mean_mouse_correct",
        ]
        summaries.append({
            "series_id": "Mouse", "relative_position": int(relative_position),
            "mean": float(mouse_values.mean()), "sem": float(mouse_values.sem()),
            "n_windows": n_windows,
        })
        for model_id in MODEL_ORDER:
            model_values = matched.loc[
                matched["model_id"] == model_id, "mean_model_probability_correct",
            ]
            summaries.append({
                "series_id": model_id, "relative_position": int(relative_position),
                "mean": float(model_values.mean()), "sem": float(model_values.sem()),
                "n_windows": n_windows,
            })
    return pd.DataFrame(summaries, columns=columns)


def fig_block_transition_performance_by_trial_type(
    block_transition_performance: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Figure 19: one page per model plus an all-model overlay (6 pages;
    PDF only, no PNGs -- see ``_save_pdf_pages``). Each page is a plain 4x4
    grid of Axes (rows =
    ``EVENT_ALIGNED_SPLIT_ROLES``, columns =
    ``BLOCK_TRANSITION_PERFORMANCE_COLUMNS``): the mouse's actual
    correctness (black) versus that model's expected correctness (color)
    across trials -15..+20 around a block transition, restricted to the
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
    n_rows, n_cols = len(EVENT_ALIGNED_SPLIT_ROLES), len(BLOCK_TRANSITION_PERFORMANCE_COLUMNS)
    overlay_summaries = {}
    for split_role in EVENT_ALIGNED_SPLIT_ROLES:
        for event_category, trial_type, _ in BLOCK_TRANSITION_PERFORMANCE_COLUMNS:
            panel = block_transition_performance[
                (block_transition_performance["split_role"] == split_role)
                & (block_transition_performance["event_category"] == event_category)
                & (block_transition_performance["trial_type"] == trial_type)
            ]
            overlay_summaries[(split_role, event_category, trial_type)] = (
                _matched_block_transition_overlay_summary(panel)
            )

    pages = []
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
                ).reindex(_FIGURE19_RELATIVE_POSITIONS)
                windows_per_position = panel.groupby("relative_position")["window_id"].nunique()
                n_label = _window_count_range_label(windows_per_position, "available")
                if panel.empty:
                    ax.text(0.5, 0.5, "No available windows", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.plot(
                        trajectory.index, trajectory["mouse_mean"], color="black", marker="o",
                        markersize=3, linewidth=1.4, label="Mouse",
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["mouse_mean"] - trajectory["mouse_sem"],
                        trajectory["mouse_mean"] + trajectory["mouse_sem"],
                        color="black", alpha=0.12,
                    )
                    ax.plot(
                        trajectory.index, trajectory["model_mean"], color=MODEL_COLORS[model_id],
                        marker="o", markersize=3, linewidth=1.4, label=MODEL_LABELS[model_id],
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["model_mean"] - trajectory["model_sem"],
                        trajectory["model_mean"] + trajectory["model_sem"],
                        color=MODEL_COLORS[model_id], alpha=0.18,
                    )
                ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
                if row_index == 0:
                    ax.set_title(f"{col_title}\n{n_label}", fontsize=10)
                else:
                    ax.set_title(n_label, fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"{EVENT_ALIGNED_SPLIT_LABELS[split_role]}\nP(correct)", fontsize=9)
                if row_index == n_rows - 1:
                    ax.set_xlabel("Relative trial position", fontsize=10)
                ax.set_xlim(
                    BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE.start,
                    BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE.stop - 1,
                )
                ax.tick_params(labelsize=8.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, loc="best", fontsize=8)
        fig.suptitle(
            f"Block-transition performance by trial type -- {mouse_id} -- {MODEL_LABELS[model_id]}\n"
            "Trials -15 through +20 around each block transition, restricted to that column's own trial type "
            "(short/long stimulus) at each relative position. Rows: all sessions pooled, then "
            "train/validation/test only. Mouse (black) vs model expected accuracy (color), aggregated "
            "within window first, then averaged across that model's available windows. N may vary by "
            "position because trial types are sparse. Panel y-scales are not fixed.",
            fontsize=11.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pages.append(fig)

    overlay_fig, overlay_axes = plt.subplots(n_rows, n_cols, figsize=(20.0, 16.0))
    for row_index, split_role in enumerate(EVENT_ALIGNED_SPLIT_ROLES):
        for col_index, (event_category, trial_type, col_title) in enumerate(
            BLOCK_TRANSITION_PERFORMANCE_COLUMNS,
        ):
            ax = overlay_axes[row_index, col_index]
            summary = overlay_summaries[(split_role, event_category, trial_type)]
            n_by_position = (
                summary.loc[summary["series_id"] == "Mouse"]
                .set_index("relative_position")["n_windows"]
            )
            n_label = _window_count_range_label(n_by_position, "matched")
            if summary.empty:
                ax.text(0.5, 0.5, "No matched windows", ha="center", va="center", transform=ax.transAxes)
            else:
                for series_id in ("Mouse", *MODEL_ORDER):
                    trajectory = (
                        summary.loc[summary["series_id"] == series_id]
                        .set_index("relative_position")
                        .reindex(_FIGURE19_RELATIVE_POSITIONS)
                    )
                    is_mouse = series_id == "Mouse"
                    color = "black" if is_mouse else MODEL_COLORS[series_id]
                    ax.plot(
                        trajectory.index, trajectory["mean"],
                        color=color,
                        linestyle="-" if is_mouse else MODEL_LINESTYLES[series_id],
                        marker="o" if is_mouse else MODEL_MARKERS[series_id],
                        markevery=3, markersize=3 if is_mouse else 2.5,
                        linewidth=1.8 if is_mouse else 1.25,
                        label="Mouse" if is_mouse else MODEL_LABELS[series_id],
                        zorder=4 if is_mouse else 2,
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["mean"] - trajectory["sem"],
                        trajectory["mean"] + trajectory["sem"],
                        color=color, alpha=0.10 if is_mouse else 0.06,
                        zorder=3 if is_mouse else 1,
                    )
            ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
            if row_index == 0:
                ax.set_title(f"{col_title}\n{n_label}", fontsize=10)
            else:
                ax.set_title(n_label, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{EVENT_ALIGNED_SPLIT_LABELS[split_role]}\nP(correct)", fontsize=9)
            if row_index == n_rows - 1:
                ax.set_xlabel("Relative trial position", fontsize=10)
            ax.set_xlim(
                BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE.start,
                BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE.stop - 1,
            )
            ax.tick_params(labelsize=8.5)
    handles, labels = overlay_axes[0, 0].get_legend_handles_labels()
    if handles:
        overlay_fig.legend(
            handles, labels, loc="upper center", ncol=len(handles),
            bbox_to_anchor=(0.5, 0.955), fontsize=8.5,
        )
    overlay_fig.suptitle(
        f"Block-transition performance by trial type -- {mouse_id} -- all models\n"
        "Trials -15 through +20 around each block transition. Mouse and model curves use the same "
        "all-model matched windows at each relative position; N can vary by position. Rows: all "
        "sessions pooled, then train/validation/test only. Panel y-scales are not fixed.",
        fontsize=11.5,
    )
    overlay_fig.tight_layout(rect=[0, 0, 1, 0.91])
    pages.append(overlay_fig)
    _save_pdf_pages(pages, figures_dir, "fig19_sliding_window_block_transition_performance_by_trial_type")


_FIGURE23_RELATIVE_POSITIONS = pd.Index(
    tuple(RARE_TRIAL_RELATIVE_RANGE), name="relative_position",
)


def fig_rare_trial_performance_by_type(
    rare_trial_performance: pd.DataFrame, mouse_id: str, figures_dir,
) -> None:
    """Figure 23: mirrors Figure 19's page/row structure -- one page per
    model plus an all-model overlay (6 pages; PDF only, no PNGs), rows =
    ``EVENT_ALIGNED_SPLIT_ROLES`` -- but aligned on the rare trial itself
    (trials -2..+2 around it, ``RARE_TRIAL_RELATIVE_RANGE``) rather than a
    block transition, and with 2 columns (``RARE_TRIAL_PERFORMANCE_COLUMNS``)
    instead of Figure 19's 4: a rare-trial event's own stimulus category is
    already fixed by its event_category (rare short in a long block, rare
    long in a short block), so there is no separate trial_type split.

    Aggregated within window first (one number per window regardless of how
    many rare trials it contains -- see
    ``collect._aggregate_rare_trial_performance``), then averaged across
    whatever matched windows have data at each relative position. Panel
    y-axes are left to auto-scale (not fixed to 0-1).
    """
    n_rows, n_cols = len(EVENT_ALIGNED_SPLIT_ROLES), len(RARE_TRIAL_PERFORMANCE_COLUMNS)
    overlay_summaries = {}
    for split_role in EVENT_ALIGNED_SPLIT_ROLES:
        for event_category, _ in RARE_TRIAL_PERFORMANCE_COLUMNS:
            panel = rare_trial_performance[
                (rare_trial_performance["split_role"] == split_role)
                & (rare_trial_performance["event_category"] == event_category)
            ]
            overlay_summaries[(split_role, event_category)] = (
                _matched_block_transition_overlay_summary(panel, label="Figure 23")
            )

    pages = []
    for model_id in MODEL_ORDER:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 16.0))
        for row_index, split_role in enumerate(EVENT_ALIGNED_SPLIT_ROLES):
            for col_index, (event_category, col_title) in enumerate(RARE_TRIAL_PERFORMANCE_COLUMNS):
                ax = axes[row_index, col_index]
                panel = rare_trial_performance[
                    (rare_trial_performance["model_id"] == model_id)
                    & (rare_trial_performance["split_role"] == split_role)
                    & (rare_trial_performance["event_category"] == event_category)
                ]
                trajectory = panel.groupby("relative_position").agg(
                    mouse_mean=("mean_mouse_correct", "mean"),
                    mouse_sem=("mean_mouse_correct", "sem"),
                    model_mean=("mean_model_probability_correct", "mean"),
                    model_sem=("mean_model_probability_correct", "sem"),
                ).reindex(_FIGURE23_RELATIVE_POSITIONS)
                windows_per_position = panel.groupby("relative_position")["window_id"].nunique()
                n_label = _window_count_range_label(
                    windows_per_position, "available", positions=_FIGURE23_RELATIVE_POSITIONS,
                )
                if panel.empty:
                    ax.text(0.5, 0.5, "No available windows", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.plot(
                        trajectory.index, trajectory["mouse_mean"], color="black", marker="o",
                        markersize=3, linewidth=1.4, label="Mouse",
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["mouse_mean"] - trajectory["mouse_sem"],
                        trajectory["mouse_mean"] + trajectory["mouse_sem"],
                        color="black", alpha=0.12,
                    )
                    ax.plot(
                        trajectory.index, trajectory["model_mean"], color=MODEL_COLORS[model_id],
                        marker="o", markersize=3, linewidth=1.4, label=MODEL_LABELS[model_id],
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["model_mean"] - trajectory["model_sem"],
                        trajectory["model_mean"] + trajectory["model_sem"],
                        color=MODEL_COLORS[model_id], alpha=0.18,
                    )
                ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
                if row_index == 0:
                    ax.set_title(f"{col_title}\n{n_label}", fontsize=10)
                else:
                    ax.set_title(n_label, fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"{EVENT_ALIGNED_SPLIT_LABELS[split_role]}\nP(correct)", fontsize=9)
                if row_index == n_rows - 1:
                    ax.set_xlabel("Relative trial position", fontsize=10)
                ax.set_xlim(RARE_TRIAL_RELATIVE_RANGE.start, RARE_TRIAL_RELATIVE_RANGE.stop - 1)
                ax.set_xticks(list(RARE_TRIAL_RELATIVE_RANGE))
                ax.tick_params(labelsize=8.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, loc="best", fontsize=8)
        fig.suptitle(
            f"Rare-trial performance -- {mouse_id} -- {MODEL_LABELS[model_id]}\n"
            "Trials -2 through +2 around each rare trial (trial 0 = the rare trial itself). Rows: all "
            "sessions pooled, then train/validation/test only. Mouse (black) vs model expected accuracy "
            "(color), aggregated within window first, then averaged across that model's available "
            "windows. Panel y-scales are not fixed.",
            fontsize=11.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pages.append(fig)

    overlay_fig, overlay_axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 16.0))
    for row_index, split_role in enumerate(EVENT_ALIGNED_SPLIT_ROLES):
        for col_index, (event_category, col_title) in enumerate(RARE_TRIAL_PERFORMANCE_COLUMNS):
            ax = overlay_axes[row_index, col_index]
            summary = overlay_summaries[(split_role, event_category)]
            n_by_position = (
                summary.loc[summary["series_id"] == "Mouse"]
                .set_index("relative_position")["n_windows"]
            )
            n_label = _window_count_range_label(
                n_by_position, "matched", positions=_FIGURE23_RELATIVE_POSITIONS,
            )
            if summary.empty:
                ax.text(0.5, 0.5, "No matched windows", ha="center", va="center", transform=ax.transAxes)
            else:
                for series_id in ("Mouse", *MODEL_ORDER):
                    trajectory = (
                        summary.loc[summary["series_id"] == series_id]
                        .set_index("relative_position")
                        .reindex(_FIGURE23_RELATIVE_POSITIONS)
                    )
                    is_mouse = series_id == "Mouse"
                    color = "black" if is_mouse else MODEL_COLORS[series_id]
                    ax.plot(
                        trajectory.index, trajectory["mean"],
                        color=color,
                        linestyle="-" if is_mouse else MODEL_LINESTYLES[series_id],
                        marker="o" if is_mouse else MODEL_MARKERS[series_id],
                        markersize=3 if is_mouse else 2.5,
                        linewidth=1.8 if is_mouse else 1.25,
                        label="Mouse" if is_mouse else MODEL_LABELS[series_id],
                        zorder=4 if is_mouse else 2,
                    )
                    ax.fill_between(
                        trajectory.index,
                        trajectory["mean"] - trajectory["sem"],
                        trajectory["mean"] + trajectory["sem"],
                        color=color, alpha=0.10 if is_mouse else 0.06,
                        zorder=3 if is_mouse else 1,
                    )
            ax.axvline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
            if row_index == 0:
                ax.set_title(f"{col_title}\n{n_label}", fontsize=10)
            else:
                ax.set_title(n_label, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{EVENT_ALIGNED_SPLIT_LABELS[split_role]}\nP(correct)", fontsize=9)
            if row_index == n_rows - 1:
                ax.set_xlabel("Relative trial position", fontsize=10)
            ax.set_xlim(RARE_TRIAL_RELATIVE_RANGE.start, RARE_TRIAL_RELATIVE_RANGE.stop - 1)
            ax.set_xticks(list(RARE_TRIAL_RELATIVE_RANGE))
            ax.tick_params(labelsize=8.5)
    handles, labels = overlay_axes[0, 0].get_legend_handles_labels()
    if handles:
        overlay_fig.legend(
            handles, labels, loc="upper center", ncol=len(handles),
            bbox_to_anchor=(0.5, 0.955), fontsize=8.5,
        )
    overlay_fig.suptitle(
        f"Rare-trial performance -- {mouse_id} -- all models\n"
        "Trials -2 through +2 around each rare trial. Mouse and model curves use the same all-model "
        "matched windows at each relative position. Rows: all sessions pooled, then "
        "train/validation/test only. Panel y-scales are not fixed.",
        fontsize=11.5,
    )
    overlay_fig.tight_layout(rect=[0, 0, 1, 0.91])
    pages.append(overlay_fig)
    _save_pdf_pages(pages, figures_dir, "fig23_sliding_window_rare_trial_performance_by_type")


_CAUSAL_TRAJECTORY_SPLIT_ROLES = ("train", "validation", "test")
_CAUSAL_TRAJECTORY_SPLIT_LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}
_CAUSAL_TRAJECTORY_SAMPLES_PER_SPLIT = 5
# Pale, model-color-independent shades for the ground-truth block schedule
# (kept far from MODEL_COLORS so the shading never reads as a model line).
_CAUSAL_TRAJECTORY_BLOCK_COLORS = {"N": "#BFBFBF", "S": "#F4A6A6", "L": "#A6C8F4"}
_CAUSAL_TRAJECTORY_BLOCK_LABELS = {
    "N": "Neutral block (true)", "S": "Short block (true)", "L": "Long block (true)",
}


def _shade_true_blocks(ax: plt.Axes, trial_index: pd.Series, true_block: pd.Series) -> None:
    """Shade contiguous runs of the ground-truth block label behind the
    prior trajectory, so the panel shows at a glance whether the model's
    prior tracks the true block schedule."""
    order = np.argsort(trial_index.to_numpy())
    trial_sorted = trial_index.to_numpy()[order]
    block_sorted = true_block.to_numpy()[order]
    if len(block_sorted) == 0:
        return
    run_id = np.r_[0, np.cumsum(block_sorted[1:] != block_sorted[:-1])]
    for run in np.unique(run_id):
        mask = run_id == run
        color = _CAUSAL_TRAJECTORY_BLOCK_COLORS.get(block_sorted[mask][0], "#FFFFFF")
        ax.axvspan(
            trial_sorted[mask].min() - 0.5, trial_sorted[mask].max() + 0.5,
            color=color, alpha=0.4, linewidth=0, zorder=0,
        )


def _causal_trajectory_session_label(trajectory: pd.DataFrame) -> str:
    return str(trajectory["session_id"].iloc[0]).split("_")[-1]


def fig_prior_trajectory_samples(samples: pd.DataFrame, mouse_id: str, figures_dir) -> None:
    """5 pages: one per model (M1-M4), then a final cross-model overlay page.

    Each page is a 5-sample x train/validation/test grid of the long-block
    prior (``model_prior_long``) over trials for a fixed random sample of
    sessions (see ``collect.collect_prior_trajectory_samples``) -- the SAME
    5 sessions per split on every page -- with the true block schedule
    shaded behind each trajectory. This is the same first-row panel as e.g.
    ``m2_dynamic_block_length_causal_trajectory``, replicated across sampled
    sessions and split roles, with ground-truth shading added; the last page
    superimposes all four models' priors on those same sessions. M0 has no
    block prior and is excluded.
    """
    n_rows = _CAUSAL_TRAJECTORY_SAMPLES_PER_SPLIT
    n_cols = len(_CAUSAL_TRAJECTORY_SPLIT_ROLES)
    block_legend_handles = [
        Patch(facecolor=color, alpha=0.4, edgecolor="none", label=_CAUSAL_TRAJECTORY_BLOCK_LABELS[key])
        for key, color in _CAUSAL_TRAJECTORY_BLOCK_COLORS.items()
    ]

    pages = []
    for model_id in ("M1", "M2", "M3", "M4"):
        model_samples = samples.loc[samples["model_id"] == model_id]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.0, 13.0), sharey=True)
        for col_index, split_role in enumerate(_CAUSAL_TRAJECTORY_SPLIT_ROLES):
            split_samples = model_samples.loc[model_samples["split_role"] == split_role]
            sample_ranks = sorted(split_samples["sample_rank"].unique())
            for row_index in range(n_rows):
                ax = axes[row_index, col_index]
                if row_index >= len(sample_ranks):
                    ax.axis("off")
                    continue
                trajectory = split_samples.loc[
                    split_samples["sample_rank"] == sample_ranks[row_index]
                ].sort_values("trial_index")
                _shade_true_blocks(ax, trajectory["trial_index"], trajectory["true_block"])
                ax.plot(
                    trajectory["trial_index"], trajectory["model_prior_long"],
                    color=MODEL_COLORS[model_id], linewidth=1.5, zorder=3,
                )
                window_id = trajectory["window_id"].iloc[0]
                panel_title = f"{_causal_trajectory_session_label(trajectory)} (win {int(window_id)})"
                if row_index == 0:
                    ax.set_title(
                        f"{_CAUSAL_TRAJECTORY_SPLIT_LABELS[split_role]}\n{panel_title}", fontsize=10,
                    )
                else:
                    ax.set_title(panel_title, fontsize=8.5)
                ax.set_ylim(0, 1)
                ax.set_xlim(trajectory["trial_index"].min(), trajectory["trial_index"].max())
                ax.tick_params(labelsize=7.5)
                if row_index == n_rows - 1:
                    ax.set_xlabel("Trial index", fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"Sample {row_index + 1}\nP(long) prior", fontsize=9)
        fig.legend(
            handles=block_legend_handles, loc="upper center", ncol=len(block_legend_handles),
            bbox_to_anchor=(0.5, 0.965), fontsize=9, frameon=False,
        )
        fig.suptitle(
            f"Long-block prior causal trajectory -- {mouse_id} -- {MODEL_LABELS[model_id]}\n"
            "5 random sessions per split (seeded, reproducible; the same sessions on every fig20 page), "
            "each paired with this model's own randomly chosen fitted rolling window (train/validation "
            "sessions recur across windows; test has exactly one window per session). Line: model's "
            "P(long) prior before each trial. Shading: ground-truth block type.",
            fontsize=11, y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        pages.append(fig)

    # Final page: the same sampled sessions with all four models superimposed
    # (each still from its own randomly chosen fitted window for that session).
    reference_model = "M1"
    model_legend_handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], linewidth=1.8, label=MODEL_LABELS[m])
        for m in ("M1", "M2", "M3", "M4")
    ]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.0, 13.0), sharey=True)
    for col_index, split_role in enumerate(_CAUSAL_TRAJECTORY_SPLIT_ROLES):
        split_samples = samples.loc[samples["split_role"] == split_role]
        sample_ranks = sorted(split_samples["sample_rank"].unique())
        for row_index in range(n_rows):
            ax = axes[row_index, col_index]
            if row_index >= len(sample_ranks):
                ax.axis("off")
                continue
            rank_samples = split_samples.loc[split_samples["sample_rank"] == sample_ranks[row_index]]
            reference = rank_samples.loc[rank_samples["model_id"] == reference_model].sort_values("trial_index")
            _shade_true_blocks(ax, reference["trial_index"], reference["true_block"])
            for model_id in ("M1", "M2", "M3", "M4"):
                trajectory = rank_samples.loc[rank_samples["model_id"] == model_id].sort_values("trial_index")
                ax.plot(
                    trajectory["trial_index"], trajectory["model_prior_long"],
                    color=MODEL_COLORS[model_id], linewidth=1.3, zorder=3,
                )
            session_label = _causal_trajectory_session_label(reference)
            if row_index == 0:
                ax.set_title(f"{_CAUSAL_TRAJECTORY_SPLIT_LABELS[split_role]}\n{session_label}", fontsize=10)
            else:
                ax.set_title(session_label, fontsize=8.5)
            ax.set_ylim(0, 1)
            ax.set_xlim(reference["trial_index"].min(), reference["trial_index"].max())
            ax.tick_params(labelsize=7.5)
            if row_index == n_rows - 1:
                ax.set_xlabel("Trial index", fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"Sample {row_index + 1}\nP(long) prior", fontsize=9)
    fig.legend(
        handles=block_legend_handles + model_legend_handles,
        loc="upper center", ncol=len(block_legend_handles) + len(model_legend_handles),
        bbox_to_anchor=(0.5, 0.965), fontsize=9, frameon=False,
    )
    fig.suptitle(
        f"Long-block prior causal trajectory -- {mouse_id} -- all models superimposed\n"
        "Same 5 random sessions per split as the per-model pages, each model plotted from its own "
        "randomly chosen fitted rolling window for that session. Shading: ground-truth block type.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    pages.append(fig)

    _save_pdf_pages(pages, figures_dir, "fig20_sliding_window_long_block_prior_causal_trajectory_samples")


def fig_m4_hazard_curves_across_windows(sw_params: pd.DataFrame, mouse_id: str, figures_dir) -> None:
    """M4's fitted biased hazard hB(r) for every rolling window, overlaid in
    chronological order from a pale tint (earliest window) to a dark shade
    (latest) of M4's report color -- shows how the fitted run-length
    dependence of the switch hazard drifts across the sliding-window fits.
    No neutral hazard, no r=23 reference line (see fig09's bh row for the
    excess-hazard summary and the bh_half_life row for the r=0.5 crossing).
    Shares its plotting logic with fig09's bh_hazard_curves row via
    ``_plot_m4_hazard_curve_family`` so the two can't drift apart.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    _plot_m4_hazard_curve_family(ax, sw_params, colorbar=True)
    ax.set_xlabel("Represented run length r (trials)")
    ax.set_ylabel("Biased switch hazard hB(r)")
    ax.set_title(f"M4 fitted hazard curve across sliding windows -- {mouse_id}", fontsize=12)
    fig.tight_layout()
    _save(fig, figures_dir, "fig22_sliding_window_m4_hazard_curves_across_windows")


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
            table("sliding_window_split_session_metrics.csv"),
            table("sliding_window_trial_subset_metrics.csv"),
            output,
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
        "fig19", "Block-transition performance by trial type (6 pages, PDF only)",
        lambda table, mouse, output: fig_block_transition_performance_by_trial_type(
            table("sliding_window_block_transition_performance.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig20", "Long-block prior causal trajectory samples, M1-M4 + overlay (5 pages, PDF only)",
        lambda table, mouse, output: fig_prior_trajectory_samples(
            table("sliding_window_prior_trajectory_samples.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig21", "Sliding-window parameter means and standard deviations",
        lambda table, mouse, output: fig_sliding_window_parameter_mean_sd(
            table("sliding_window_parameters.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig22", "M4 fitted hazard curve across sliding windows (dim-to-dark)",
        lambda table, mouse, output: fig_m4_hazard_curves_across_windows(
            table("sliding_window_parameters.csv"), mouse, output,
        ),
    ),
    FigureSpec(
        "fig23", "Rare-trial performance by type (6 pages, PDF only)",
        lambda table, mouse, output: fig_rare_trial_performance_by_type(
            table("sliding_window_rare_trial_performance.csv"), mouse, output,
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
        highest_figure_number = max(int(figure_id[3:]) for figure_id in _FIGURE_BY_ID)
        raise ValueError(
            f"Unknown figure selector(s): {', '.join(invalid)}. "
            f"Use one of {valid}, a number from 1 to {highest_figure_number}, or 'all'."
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
            "Omit to render all figures."
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
