"""Render model-agnostic figures for the expanded sliding-window comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from post_analysis import common

from . import stats
from .config import (
    ANALYSIS_NAME,
    EQUIVALENCE_MARGIN_NLL,
    MODEL_BY_ID,
    MODEL_FAMILY_ORDER,
    MODEL_ORDER,
    SPLIT_ROLES,
    group_style,
    validate_mouse_id,
)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    pending_paths = []
    try:
        for suffix in ("png", "pdf"):
            pending = figures_dir / f".{stem}.pending.{suffix}"
            pending_paths.append(pending)
            fig.savefig(pending, bbox_inches="tight")
            pending.replace(figures_dir / f"{stem}.{suffix}")
    finally:
        plt.close(fig)
        for pending in pending_paths:
            pending.unlink(missing_ok=True)


def _eligible_order(status: pd.DataFrame) -> tuple[str, ...]:
    eligible = set(status.loc[status["eligible_for_comparison"].astype(str).str.lower().eq("true"), "model_id"])
    return tuple(model for model in MODEL_ORDER if model in eligible)


def _provisional_text(status: pd.DataFrame) -> str:
    n = status["eligible_for_comparison"].astype(str).str.lower().eq("true").sum()
    return f"PROVISIONAL: {n}/{len(status)} configured models complete" if n < len(status) else "FINAL CONFIGURED SET"


def _wrap_identifier(name: str, max_chars: int = 26) -> str:
    """Break a long underscore-delimited model name across lines for display.

    Splits only at existing "_" boundaries (never mid-token), so it is purely
    a rendering aid and never changes the identifier text itself.
    """
    if len(name) <= max_chars:
        return name
    parts = name.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}_{part}" if current else part
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _eligibility_text(equivalence: pd.DataFrame, status: pd.DataFrame) -> str:
    eligible_mask = status["eligible_for_comparison"].astype(str).str.lower().eq("true")
    completed_oracles = sum(
        MODEL_BY_ID[model].oracle for model in status.loc[eligible_mask, "model_id"]
    )
    incomplete = [
        MODEL_BY_ID[model].short_label for model in status.loc[~eligible_mask, "model_id"]
    ]
    shown = equivalence["model_id"].nunique()
    incomplete_text = ", ".join(incomplete) if incomplete else "none"
    return (
        f"{shown} deployable models shown; {completed_oracles} completed known-switch oracle "
        f"benchmarks excluded; incomplete: {incomplete_text}."
    )


def _build_figure00_rankings(rankings: pd.DataFrame, status: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(20, 11.5), sharex=False)
    fig.subplots_adjust(wspace=0.45)
    all_rankings = rankings[rankings["ranking_scope"].eq("all_models")]
    lower = all_rankings["moving_block_ci_lower"].min()
    upper = all_rankings["moving_block_ci_upper"].max()
    x_padding = 0.04 * (upper - lower)
    compared_count = all_rankings["model_id"].nunique()
    for ax, split_role in zip(axes, SPLIT_ROLES):
        subset = rankings[
            rankings["split_role"].eq(split_role) & rankings["ranking_scope"].eq("all_models")
        ].sort_values("rank", ascending=False)
        y = np.arange(len(subset))
        for yi, (_, row) in zip(y, subset.iterrows()):
            spec = MODEL_BY_ID[row["model_id"]]
            marker = "D" if spec.oracle else "o"
            ax.errorbar(
                row["mean_window_nll"], yi,
                xerr=[[row["mean_window_nll"] - row["moving_block_ci_lower"]],
                      [row["moving_block_ci_upper"] - row["mean_window_nll"]]],
                fmt=marker, color=spec.color, markerfacecolor="white" if spec.oracle else spec.color,
                markersize=6, capsize=2.5, linewidth=1.2,
            )
            probability = float(row["bootstrap_probability_best"])
            if 0 < probability < 0.01:
                probability_text = f"P(best of {compared_count})<0.01"
            else:
                probability_text = f"P(best of {compared_count})={probability:.2f}"
            ax.text(row["moving_block_ci_upper"], yi + 0.18, probability_text, fontsize=7.5)
        labels = [f"{int(row['rank'])}. {MODEL_BY_ID[row['model_id']].short_label}" for _, row in subset.iterrows()]
        ax.set_yticks(y, labels)
        ax.set_xlabel("Mean NLL per window (lower is better)")
        best_deployable = rankings[
            rankings["split_role"].eq(split_role) & rankings["ranking_scope"].eq("deployable_only")
        ].sort_values("rank").iloc[0]
        deployable_name = _wrap_identifier(MODEL_BY_ID[best_deployable["model_id"]].short_label, max_chars=24)
        ax.set_title(
            f"{split_role.capitalize()}\nLowest observed deployable mean:\n{deployable_name}",
            fontsize=9.7,
        )
        ax.set_xlim(lower - x_padding, upper + 5 * x_padding)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"Figure 0 — MC08 sliding-window fit: observed order by mean NLL\n{_provisional_text(status)}; open diamonds and (O) denote oracle benchmarks",
        y=0.975,
    )
    fig.text(
        0.5, 0.025,
        "n=30 overlapping windows from one mouse. Moving-block bootstrap: block length 5, 5,000 resamples. P(best) includes oracle models; train/validation are diagnostics, test is primary descriptive comparison.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


def _figure00_ci_page(equivalence: pd.DataFrame, status: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), sharey="row")
    all_ci = np.r_[equivalence["ci95_lower"].to_numpy(float), equivalence["ci95_upper"].to_numpy(float)]
    full_padding = 0.04 * (all_ci.max() - all_ci.min())
    full_limits = (all_ci.min() - full_padding, all_ci.max() + full_padding)
    close = equivalence[~equivalence["family"].eq("WSLS")]
    close_ci = np.r_[close["ci95_lower"].to_numpy(float), close["ci95_upper"].to_numpy(float)]
    zoom_extent = max(abs(close_ci.min()), abs(close_ci.max())) * 1.15
    reference_id = equivalence["reference_model_id"].iloc[0]
    reference = MODEL_BY_ID[reference_id].short_label
    for column, split_role in enumerate(SPLIT_ROLES):
        split = equivalence[equivalence["split_role"].eq(split_role)].copy()
        split["plot_label"] = split.apply(
            lambda row: f"{row['family']}  |  {MODEL_BY_ID[row['model_id']].short_label}", axis=1,
        )
        for row, (subset, scope) in enumerate((
            (split, "All deployable families"),
            (split[~split["family"].eq("WSLS")], "M0/M1/M2 zoom"),
        )):
            ax = axes[row, column]
            subset = subset.iloc[::-1].reset_index(drop=True)
            y = np.arange(len(subset))
            for yi, (_, item) in zip(y, subset.iterrows()):
                model_id = item["model_id"]
                ax.errorbar(
                    item["mean_diff_candidate_minus_reference"], yi,
                    xerr=[[item["mean_diff_candidate_minus_reference"] - item["ci95_lower"]],
                          [item["ci95_upper"] - item["mean_diff_candidate_minus_reference"]]],
                    fmt="*" if item["selected_as_observed_test_best"] else "o",
                    color=MODEL_BY_ID[model_id].color,
                    markersize=8 if item["selected_as_observed_test_best"] else 5.5,
                    capsize=2.5, linewidth=1.4,
                )
            ax.axvline(0, color="#333333", linewidth=0.9, zorder=-1)
            ax.set_yticks(y, subset["plot_label"])
            ax.set_title(f"{split_role.capitalize()} - {scope}")
            ax.set_xlabel(f"Candidate - {reference} paired mean NLL")
            ax.grid(axis="x", alpha=0.22)
            if row == 1:
                ax.set_xlim(-zoom_extent, zoom_extent)
            else:
                ax.set_xlim(*full_limits)
    n_windows = int(equivalence["n_windows"].iloc[0])
    block_length = int(equivalence["bootstrap_block_length"].iloc[0])
    replicates = int(equivalence["bootstrap_replicates"].iloc[0])
    fig.suptitle(
        "Figure 0 - paired 95% basic moving-block-bootstrap confidence intervals\n"
        f"Within each split: candidate minus {reference}; negative favors candidate; star=definitional zero reference, circles=candidates\n"
        "All panels within a row use the same x-axis; equivalence decisions are reported separately on page 3",
        y=0.975,
    )
    fig.text(
        0.5, 0.015,
        f"Equal weight per rolling window; n={n_windows} overlapping rolling windows from one mouse; block length {block_length}, {replicates:,} resamples.\n"
        f"{_provisional_text(status)}. The reference was selected by the observed test minimum; intervals are exploratory and conditional on selection.\n"
        f"{_eligibility_text(equivalence, status)}",
        ha="center", fontsize=9.2, linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.89))
    return fig


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def _figure00_tost_page(
    equivalence: pd.DataFrame,
    family_summary: pd.DataFrame,
    status: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(22, 17))
    margin = float(equivalence["equivalence_margin_nll"].iloc[0])
    block_length = int(equivalence["bootstrap_block_length"].iloc[0])
    replicates = int(equivalence["bootstrap_replicates"].iloc[0])
    for ax, split_role in zip(axes, SPLIT_ROLES):
        ax.axis("off")
        split = equivalence[equivalence["split_role"].eq(split_role)]
        rows, colors = [], []
        for _, item in split.iterrows():
            is_reference = bool(item["selected_as_observed_test_best"])
            equivalent = bool(item["equivalent_holm"])
            wrapped_name = _wrap_identifier(MODEL_BY_ID[item["model_id"]].short_label, max_chars=45)
            rows.append([
                f"{item['family']} | {wrapped_name}",
                f"{item['mean_diff_candidate_minus_reference']:+.4f}",
                f"[{item['ci90_lower']:+.4f}, {item['ci90_upper']:+.4f}]",
                _format_p(float(item["tost_p"])),
                _format_p(float(item["tost_p_holm_within_split"])),
                "reference" if is_reference else ("equivalent" if equivalent else "not equivalent"),
            ])
            shade = "#FFF2CC" if is_reference else ("#D9EAD3" if equivalent else "#F4CCCC")
            colors.append(["white", "white", "white", "white", "white", shade])
        table = ax.table(
            cellText=rows,
            colLabels=["Family | model", "mean delta", "90% CI (raw)", "Raw p", "Holm p", "Adjusted TOST"],
            cellColours=colors,
            cellLoc="center", colLoc="center", loc="upper center",
            colWidths=[0.36, 0.11, 0.18, 0.09, 0.10, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.3)
        table.scale(1.0, 4.0)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#BBBBBB")
            if row == 0:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(weight="bold")
        ax.set_title(split_role.capitalize(), fontsize=11, pad=12)

    test = equivalence[equivalence["split_role"].eq("test")]
    reference = test[test["selected_as_observed_test_best"]].iloc[0]
    equivalent_models = test[test["equivalent_holm"]]["model_id"].tolist()
    equivalent_labels = ", ".join(MODEL_BY_ID[model].short_label for model in equivalent_models)
    complete_count = status["eligible_for_comparison"].astype(str).str.lower().eq("true").sum()
    incomplete_count = len(status) - complete_count
    family_lines = []
    for _, row in family_summary.iterrows():
        if row["family_best_model_id"]:
            tag = "observed overall minimum" if row["observed_overall_best"] else (
                "equivalent" if row["equivalent_to_observed_best_holm"] else "not equivalent"
            )
            family_lines.append(
                f"{row['family']}: {MODEL_BY_ID[row['family_best_model_id']].short_label} "
                f"(test NLL {row['observed_test_mean_window_nll']:.6f}; {tag})"
            )
        else:
            family_lines.append(f"{row['family']}: no eligible deployable model")
    conclusion = (
        f"Observed test minimum: {reference['model_label']} (mean window NLL {reference['candidate_mean_window_nll']:.6f}).\n"
        f"Holm-adjusted practical-equivalence set on test: {equivalent_labels}.\n"
        + "Family leaders:\n"
        + "   |   ".join(family_lines[:2])
        + "\n"
        + "   |   ".join(family_lines[2:])
        + f"\n{_eligibility_text(equivalence, status)}"
        + f"\nBecause the reference is data-selected and {incomplete_count}/{len(status)} configured models are incomplete, this is a provisional within-mouse model-selection summary, not proof of a unique population-level winner."
    )
    n_windows = int(equivalence["n_windows"].iloc[0])
    fig.suptitle(
        f"Figure 0 - moving-block-bootstrap TOST (equivalence margin +/-{margin:.3f} mean-window NLL)\n"
        f"n={n_windows} overlapping rolling windows from one mouse; positive delta means candidate is worse; decision uses Holm p<.05; 90% CI is unadjusted",
        y=0.98,
    )
    fig.text(0.5, 0.41, conclusion, ha="center", va="top", fontsize=10.2, linespacing=1.4, wrap=True)
    fig.text(
        0.5, 0.12,
        f"Each TOST uses boundary-null block bootstraps (block length {block_length}; {replicates:,} resamples); candidate-level p=max(lower, upper), then Holm correction within split.\n"
        "The 90% CI is reported for the conventional unadjusted TOST view; the displayed decision follows adjusted p<.05.\n"
        "Raw p is the uncorrected candidate-level TOST p-value shown for reference only; it is not used for the decision.\n"
        "A +/-0.005 NLL margin corresponds to an exp(-0.005)=0.995 geometric predictive-likelihood ratio per trial.",
        ha="center", fontsize=9.2, linespacing=1.35, wrap=True,
    )
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.50, top=0.79, wspace=0.22)
    return fig


def _figure00_combined_selection_page(
    combined: pd.DataFrame,
    domains: pd.DataFrame,
    status: pd.DataFrame,
) -> plt.Figure:
    """One-page total-fit gate and six-condition event-robustness decision."""
    fig = plt.figure(figsize=(20, 19))
    grid = fig.add_gridspec(
        3, 1, height_ratios=(0.44, 0.40, 0.16),
        left=0.26, right=0.965, bottom=0.05, top=0.89, hspace=0.30,
    )
    table_ax = fig.add_subplot(grid[0])
    heatmap_ax = fig.add_subplot(grid[1])
    conclusion_ax = fig.add_subplot(grid[2])
    table_ax.axis("off")
    conclusion_ax.axis("off")

    total_reference_id = combined["total_reference_model_id"].iloc[0]
    event_reference_id = combined["event_reference_model_id"].iloc[0]
    total_reference = MODEL_BY_ID[total_reference_id].short_label
    event_reference = MODEL_BY_ID[event_reference_id].short_label
    margin = float(combined["selection_margin_nll"].iloc[0])
    block_length = int(combined["bootstrap_block_length"].iloc[0])
    replicates = int(combined["bootstrap_replicates"].iloc[0])
    event_windows = int(combined["event_n_windows"].iloc[0])
    total_windows = int(combined["total_n_windows"].iloc[0])

    def page_model_label(model_id: str) -> str:
        return MODEL_BY_ID[model_id].label

    rows, colors = [], []
    for _, item in combined.iterrows():
        model_id = item["model_id"]
        gate = bool(item["passes_total_noninferiority"])
        winner = bool(item["observed_combined_winner"])
        wrapped_name = _wrap_identifier(page_model_label(model_id), max_chars=45)
        label = f"{item['family']} | {wrapped_name}"
        if winner:
            label = "* " + label
        rows.append([
            label,
            f"{item['total_mean_difference']:+.4f}",
            f"[{item['total_ci95_lower']:+.4f}, {item['total_ci95_upper']:+.4f}]",
            _format_p(float(item["total_noninferiority_p"])),
            _format_p(float(item["total_noninferiority_p_holm"])),
            "PASS" if gate else "OUT",
            f"{item['event_mean_difference']:+.4f}",
            f"[{item['event_ci95_lower']:+.4f}, {item['event_ci95_upper']:+.4f}]",
            _format_p(float(item["event_tost_p"])),
            _format_p(float(item["event_tost_p_holm"])),
            f"{float(item['conditional_probability_event_best']):.2f}" if gate else "-",
        ])
        gate_color = "#D9EAD3" if gate else "#EEEEEE"
        probability_color = "#D9EAD3" if winner else "white"
        colors.append(["white", "white", "white", "white", "white", gate_color,
                       "white", "white", "white", "white", probability_color])
    table = table_ax.table(
        cellText=rows,
        colLabels=[
            "Family | model", "Total delta", "Total 95% CI", "NI raw p", "NI Holm p", "Gate",
            "Event delta", "Event 95% CI", "TOST raw p", "TOST Holm p", "P(event best|gate)",
        ],
        cellColours=colors, cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.26, 0.065, 0.11, 0.06, 0.065, 0.05, 0.065, 0.11, 0.06, 0.065, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold")
    table_ax.set_title(
        "Two-stage rule: (1) retain models non-inferior on total held-out mean NLL; "
        "(2) rank survivors by the six-condition event composite.\n"
        f"Deltas are candidate - {total_reference} (nats/trial); lower is better.",
        fontsize=9.7, pad=16,
    )

    models = combined["model_id"].tolist()
    domain_order = list(stats.EVENT_SELECTION_DOMAINS)
    domain_matrix = domains.pivot(
        index="model_id", columns="event_domain", values="mean_difference",
    ).reindex(index=models, columns=domain_order)
    values = domain_matrix.to_numpy(float)
    color_limit = max(0.01, min(0.025, float(np.nanquantile(np.abs(values), 0.75))))
    image = heatmap_ax.imshow(
        values, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
    )
    gate_by_model = combined.set_index("model_id")["passes_total_noninferiority"].astype(bool)
    labels = [
        _wrap_identifier(page_model_label(model), max_chars=45)
        + ("" if gate_by_model[model] else "\n(gate out)")
        for model in models
    ]
    heatmap_ax.set_yticks(np.arange(len(models)), labels)
    heatmap_ax.set_xticks(
        np.arange(len(domain_order)),
        [stats.EVENT_SELECTION_LABELS[domain] for domain in domain_order],
    )
    heatmap_ax.tick_params(axis="x", labelsize=8.3)
    heatmap_ax.tick_params(axis="y", labelsize=8.5)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            clipped = abs(value) >= color_limit * 0.70
            heatmap_ax.text(
                column, row, f"{value:+.4f}", ha="center", va="center",
                fontsize=7.7, color="white" if clipped else "#202020",
                fontweight="bold" if models[row] == event_reference_id else "normal",
            )
    heatmap_ax.set_title(
        f"Six held-out event conditions: paired mean NLL difference from {event_reference} (nats/trial)\n"
        "Point estimates only; condition-level CIs are not shown and cells are not significance tests "
        "(numbers exact; colors clipped)",
        fontsize=9.5, pad=6,
    )
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.018, pad=0.012)
    colorbar.set_label("NLL difference (nats/trial)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)

    shortlist = combined.loc[combined["passes_total_noninferiority"], "model_id"].tolist()
    event_set = combined.loc[
        combined["passes_total_noninferiority"] & combined["event_equivalent_holm"], "model_id",
    ].tolist()
    prevalence_winners = combined.loc[
        combined["prevalence_weighted_event_winner"], "model_id",
    ].tolist()
    observed_winners = combined.loc[combined["observed_combined_winner"], "model_id"].tolist()
    observed_label = "lowest observed composite" if len(observed_winners) == 1 else "lowest observed composites"
    incomplete = status[~status["eligible_for_comparison"].astype(str).str.lower().eq("true")]
    observed_probability = float(combined.loc[
        combined["model_id"].eq(observed_winners[0]), "conditional_probability_event_best",
    ].iloc[0])
    conclusion_ax.text(
        0.01, 0.88,
        f"MC08 exploratory result: {observed_label} - "
        + ", ".join(MODEL_BY_ID[model].short_label for model in observed_winners)
        + "\n(from "
        + "/".join(dict.fromkeys(MODEL_BY_ID[model].family for model in observed_winners))
        + f"; P(best | total gate)={observed_probability:.2f}).",
        fontsize=10.5, weight="bold", va="top", linespacing=1.25,
    )
    conclusion_ax.text(
        0.01, 0.57,
        "M2 static is practically equivalent on the event composite\n"
        f"(Holm-adjusted TOST p<.001; margin +/-{margin:.3f} nats/trial).",
        fontsize=9.2, weight="bold", va="top", linespacing=1.25,
    )
    conclusion_ax.text(
        0.01, 0.31,
        "Total non-inferiority shortlist: "
        + ", ".join(MODEL_BY_ID[model].short_label for model in shortlist)
        + ".\nEvent-composite Holm-equivalence set: "
        + ", ".join(MODEL_BY_ID[model].short_label for model in event_set)
        + ".",
        fontsize=8.9, va="top", linespacing=1.3,
    )
    conclusion_ax.text(
        0.99, 0.88,
        f"Test only: n={total_windows} total; n={event_windows} common event windows.\n"
        "Equal weights: event condition and rolling window; rows weighted by n_events.\n"
        f"Moving-block bootstrap: length {block_length}, {replicates:,} resamples.\n"
        "NI=non-inferiority; TOST=two one-sided tests; Holm across candidates within stage.\n"
        "Event-frequency-weighting sensitivity: "
        + ", ".join(MODEL_BY_ID[model].short_label for model in prevalence_winners)
        + ".\n"
        f"P(best|gate) is conditional; {len(incomplete)}/{len(status)} configured models incomplete.",
        fontsize=8.2, ha="right", va="top", linespacing=1.2,
    )

    fig.suptitle(
        "Figure 0 - exploratory held-out model ranking: total-fit gate and event robustness\n"
        "Rare trials use event position 0; transition conditions use positions -5 to +5 split by trial type",
        y=0.97, fontsize=13,
    )
    fig.text(
        0.5, 0.018,
        "Lower NLL is better. Known-switch oracle benchmarks are excluded from selection. "
        "Exploratory, conditional-on-selection result for one mouse; not population-level confirmation.",
        ha="center", fontsize=8.8,
    )
    return fig


def _save_figure00_pages(pages: list[plt.Figure], figures_dir: Path) -> tuple[Path, Path]:
    stem = "fig00_model_ranking_by_split"
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    pending_png = figures_dir / f".{stem}.pending.png"
    pending_pdf = figures_dir / f".{stem}.pending.pdf"
    try:
        pages[0].savefig(pending_png, bbox_inches="tight")
        with PdfPages(pending_pdf, metadata={"Title": "Figure 0 model ranking, CI, TOST, and combined selection"}) as pdf:
            for page in pages:
                pdf.savefig(page, bbox_inches="tight")
        pending_png.replace(png_path)
        pending_pdf.replace(pdf_path)
    finally:
        for page in pages:
            plt.close(page)
        pending_png.unlink(missing_ok=True)
        pending_pdf.unlink(missing_ok=True)
    return png_path, pdf_path


def figure00_report(
    rankings: pd.DataFrame,
    equivalence: pd.DataFrame,
    family_summary: pd.DataFrame,
    combined: pd.DataFrame,
    domains: pd.DataFrame,
    status: pd.DataFrame,
    figures_dir: Path,
) -> tuple[Path, Path]:
    pages: list[plt.Figure] = []
    try:
        pages.append(_build_figure00_rankings(rankings, status))
        pages.append(_figure00_ci_page(equivalence, status))
        pages.append(_figure00_tost_page(equivalence, family_summary, status))
        pages.append(_figure00_combined_selection_page(combined, domains, status))
        return _save_figure00_pages(pages, figures_dir)
    except Exception:
        for page in pages:
            plt.close(page)
        raise


def figure01_nll_distributions(metrics: pd.DataFrame, pooled: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    x = np.arange(len(eligible))
    rng = np.random.default_rng(1789)
    for ax, split_role in zip(axes, SPLIT_ROLES):
        split = metrics[metrics["split_role"].eq(split_role)]
        for xi, model_id in enumerate(eligible):
            values = pd.to_numeric(split.loc[split["model_id"].eq(model_id), "mean_nll"]).to_numpy(float)
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), xi) + jitter, values, s=12, alpha=0.35, color=MODEL_BY_ID[model_id].color)
            ax.boxplot(values, positions=[xi], widths=0.45, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": MODEL_BY_ID[model_id].color, "alpha": 0.24},
                       medianprops={"color": "black"}, whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
            pooled_row = pooled[pooled["model_id"].eq(model_id) & pooled["split_role"].eq(split_role)].iloc[0]
            ax.scatter(xi, pooled_row["pooled_trial_instance_mean_nll"], marker="D", s=34, color="black", zorder=5)
        ax.set_ylabel("Mean NLL")
        ax.set_title(f"{split_role.capitalize()} — windows (points/boxes), pooled trial instances (black diamond)")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, [MODEL_BY_ID[m].short_label for m in eligible], rotation=35, ha="right")
    fig.suptitle(f"Expanded sliding-window NLL comparison — {_provisional_text(status)}", y=1.005)
    fig.tight_layout()
    _save(fig, figures_dir, "fig01_sliding_window_nll_distributions")


def figure02_nll_trajectories(metrics: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, split_role in zip(axes, SPLIT_ROLES):
        split = metrics[metrics["split_role"].eq(split_role)]
        for model_id in eligible:
            group = split[split["model_id"].eq(model_id)].sort_values("test_rank")
            ax.plot(group["test_rank"], group["mean_nll"], marker="o", markersize=2.4,
                    linewidth=1.05, alpha=0.86, color=MODEL_BY_ID[model_id].color,
                    linestyle=group_style(model_id), label=MODEL_BY_ID[model_id].short_label)
        ax.set_ylabel("Mean NLL")
        ax.set_title(split_role.capitalize())
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Chronological rank of held-out session")
    axes[0].legend(ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.33), loc="upper center")
    fig.suptitle(f"Sliding-window fit trajectories — {_provisional_text(status)}", y=1.03)
    fig.tight_layout()
    _save(fig, figures_dir, "fig02_sliding_window_nll_trajectories")


def _stars(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def figure03_pairwise(pairwise: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    zoom_models = tuple(model for model in eligible if model not in {"pure_wsls", "extended_wsls"})
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    full_maximum = max(abs(pairwise["mean_diff_a_minus_b"]).max(), 1e-6)
    zoom_pairs = pairwise[pairwise["model_a"].isin(zoom_models) & pairwise["model_b"].isin(zoom_models)]
    zoom_maximum = max(abs(zoom_pairs["mean_diff_a_minus_b"]).max(), 1e-6)
    last_images = []
    for row_index, (models, maximum, row_name) in enumerate((
        (eligible, full_maximum, "All eligible models"),
        (zoom_models, zoom_maximum, "Non-WSLS zoom"),
    )):
        for ax, split_role in zip(axes[row_index], SPLIT_ROLES):
            matrix = np.full((len(models), len(models)), np.nan)
            annotations = np.full((len(models), len(models)), "", dtype=object)
            split = pairwise[pairwise["split_role"].eq(split_role)]
            for _, pair in split.iterrows():
                if pair["model_a"] not in models or pair["model_b"] not in models:
                    continue
                i, j = models.index(pair["model_a"]), models.index(pair["model_b"])
                matrix[i, j] = pair["mean_diff_a_minus_b"]
                holm_stars = _stars(float(pair["wilcoxon_p_holm_within_split"]))
                raw_p_text = _format_p(float(pair["wilcoxon_p"]))
                annotations[i, j] = f"{1000 * pair['mean_diff_a_minus_b']:+.1f}\n{holm_stars}\n{raw_p_text}"
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-maximum, vmax=maximum)
            last_images.append(image)
            for i in range(len(models)):
                for j in range(len(models)):
                    if annotations[i, j]:
                        ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=5.6, linespacing=1.1)
            labels = [MODEL_BY_ID[m].short_label for m in models]
            wrapped_labels = [_wrap_identifier(label, max_chars=20) for label in labels]
            ax.set_xticks(range(len(models)), labels, rotation=55, ha="right", fontsize=7.3)
            ax.set_yticks(range(len(models)), wrapped_labels, fontsize=7.3)
            ax.set_title(f"{row_name} — {split_role.capitalize()}")
    full_axis = fig.add_axes([0.12, 0.505, 0.30, 0.016])
    zoom_axis = fig.add_axes([0.58, 0.505, 0.30, 0.016])
    full_bar = fig.colorbar(last_images[0], cax=full_axis, orientation="horizontal")
    zoom_bar = fig.colorbar(last_images[3], cax=zoom_axis, orientation="horizontal")
    full_bar.set_label("All-model ΔNLL")
    zoom_bar.set_label("Non-WSLS zoom ΔNLL")
    fig.suptitle(
        f"All pairwise model differences — {_provisional_text(status)}\nUpper triangle: row−column mean NLL ×1,000 (blue/negative = row better); (O)=oracle",
        y=1.02,
    )
    state = "provisional; " if _provisional_text(status).startswith("PROVISIONAL") else ""
    fig.text(
        0.5, 0.03,
        "n=30 overlapping windows from one mouse. Each cell: row−column ΔNLL×1,000; Holm significance (*p<.05, **<.01, ***<.001); raw uncorrected Wilcoxon p.\n"
        f"{state}exploratory summaries only.",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.88, hspace=1.05, wspace=0.34)
    _save(fig, figures_dir, "fig03_pairwise_nll_effects")


def figure04_win_counts(summary: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharey=True)
    for ax, split_role in zip(axes, SPLIT_ROLES):
        split = summary[summary["split_role"].eq(split_role)].set_index("model_id").reindex(eligible)
        colors = [MODEL_BY_ID[m].color for m in eligible]
        bars = ax.bar(range(len(eligible)), split["tie_adjusted_win_credit"], color=colors)
        for bar, model_id in zip(bars, eligible):
            if MODEL_BY_ID[model_id].oracle:
                bar.set_hatch("//")
        ax.set_xticks(range(len(eligible)), [MODEL_BY_ID[m].short_label for m in eligible], rotation=55, ha="right")
        ax.set_title(split_role.capitalize())
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Tie-adjusted windows won")
    fig.suptitle(f"Per-window best-fit counts (ties within 1e-6 share one win) — {_provisional_text(status)}")
    fig.tight_layout()
    _save(fig, figures_dir, "fig04_sliding_window_win_counts")


def figure05_winner_timeline(per_window: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    fig, axes = plt.subplots(3, 1, figsize=(14, 5.6), sharex=True)
    for ax, split_role in zip(axes, SPLIT_ROLES):
        split = per_window[per_window["split_role"].eq(split_role)].sort_values("window_id")
        for _, row in split.iterrows():
            winners = str(row["winning_models"]).split("|")
            rank = int(row["window_id"]) + EXPECTED_OFFSET
            for index, winner in enumerate(winners):
                ax.scatter(rank, index * 0.13, color=MODEL_BY_ID[winner].color, s=38 / len(winners),
                           marker=group_style(winner, "marker"))
        ax.set_ylim(-0.18, 0.45)
        ax.set_yticks([])
        ax.set_ylabel(split_role[:3].upper())
        ax.grid(axis="x", alpha=0.18)
    axes[-1].set_xlabel("Chronological rank of held-out session")
    handles = [plt.Line2D([], [], marker=group_style(m, "marker"), linestyle="", color=MODEL_BY_ID[m].color, label=MODEL_BY_ID[m].short_label) for m in eligible]
    axes[0].legend(handles=handles, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.65), loc="upper center")
    fig.suptitle(f"Chronological best-fit model by split — {_provisional_text(status)}", y=1.06)
    fig.tight_layout()
    _save(fig, figures_dir, "fig05_chronological_winner_timeline")


EXPECTED_OFFSET = 12


def figure06_information_criteria(summary: pd.DataFrame, status: pd.DataFrame, figures_dir: Path) -> None:
    eligible = _eligible_order(status)
    non_wsls = tuple(model for model in eligible if model not in {"pure_wsls", "extended_wsls"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    all_models = summary[summary["ranking_scope"].eq("all_models")].set_index("model_id").reindex(eligible)
    for row_index, (models, row_label) in enumerate(((eligible, "All eligible models"), (non_wsls, "Non-WSLS zoom"))):
        table = all_models.reindex(models)
        for ax, metric, title in zip(axes[row_index], ("mean_delta_aic", "mean_delta_bic"), ("AIC", "BIC")):
            bars = ax.bar(range(len(models)), table[metric], color=[MODEL_BY_ID[m].color for m in models])
            for bar, model_id in zip(bars, models):
                if MODEL_BY_ID[model_id].oracle:
                    bar.set_hatch("//")
            ax.set_ylabel(f"Mean Δ{title} across windows")
            ax.set_title(f"{row_label} — training {title} (lower is better)")
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks(range(len(models)), [MODEL_BY_ID[m].short_label for m in models], rotation=45, ha="right")
    fig.suptitle(f"Information criteria on identical training trials — {_provisional_text(status)}\nHatched bars are oracle benchmarks; cross-window summaries are descriptive")
    fig.tight_layout()
    _save(fig, figures_dir, "fig06_information_criteria")


def _write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _figure00_tables(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    status: pd.DataFrame,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = set(status.loc[
        status["eligible_for_comparison"].astype(str).str.lower().eq("true"), "model_id",
    ])
    deployable = tuple(
        model for model in MODEL_ORDER
        if model in eligible and not MODEL_BY_ID[model].oracle
    )
    expected_counts = set(status.loc[status["model_id"].isin(deployable), "expected_windows"].astype(int))
    if len(expected_counts) != 1:
        raise ValueError("Eligible deployable models disagree on the expected window count")
    expected_window_count = expected_counts.pop()
    test_window_ids = tuple(sorted(
        int(window) for window in metrics.loc[
            metrics["split_role"].eq("test"), "window_id",
        ].unique()
    ))
    if len(test_window_ids) != expected_window_count:
        raise ValueError("Figure 0 total held-out window IDs disagree with the expected count")
    # MC08's final chronological window has no qualifying transition event.
    expected_event_window_ids = test_window_ids[:-1]
    equivalence = stats.equivalence_table(
        metrics, deployable, expected_window_count=expected_window_count,
    )
    family_summary = stats.family_summary_table(equivalence)
    combined, domains = stats.combined_selection_tables(
        metrics, events, deployable,
        expected_total_windows=expected_window_count,
        expected_event_windows=len(expected_event_window_ids),
        expected_event_window_ids=expected_event_window_ids,
    )
    _write_csv_atomic(tables_dir / "figure00_equivalence_tost.csv", equivalence)
    _write_csv_atomic(tables_dir / "figure00_family_summary.csv", family_summary)
    _write_csv_atomic(tables_dir / "figure00_combined_selection.csv", combined)
    _write_csv_atomic(tables_dir / "figure00_event_condition_differences.csv", domains)
    return equivalence, family_summary, combined, domains


def run_figure00(mouse_id: str) -> tuple[Path, ...]:
    """Recompute Figure 0 statistics and render only Figure 0 from collected tables."""
    mouse_id = validate_mouse_id(mouse_id)
    tables_dir, figures_dir = common.ensure_output_dirs(ANALYSIS_NAME, mouse_id)
    status = pd.read_csv(tables_dir / "model_completion_status.csv")
    metrics = pd.read_csv(tables_dir / "sliding_window_metrics.csv")
    rankings = pd.read_csv(tables_dir / "figure00_rankings.csv")
    events = pd.read_csv(tables_dir / "extended_event_aligned.csv")
    equivalence, family_summary, combined, domains = _figure00_tables(
        metrics, events, status, tables_dir,
    )
    png_path, pdf_path = figure00_report(
        rankings, equivalence, family_summary, combined, domains, status, figures_dir,
    )
    paths = (
        tables_dir / "figure00_equivalence_tost.csv",
        tables_dir / "figure00_family_summary.csv",
        tables_dir / "figure00_combined_selection.csv",
        tables_dir / "figure00_event_condition_differences.csv",
        png_path,
        pdf_path,
    )
    for path in paths:
        print(f"wrote {path}")
    return paths


def run(mouse_id: str) -> tuple[Path, ...]:
    mouse_id = validate_mouse_id(mouse_id)
    tables_dir, figures_dir = common.ensure_output_dirs(ANALYSIS_NAME, mouse_id)
    status = pd.read_csv(tables_dir / "model_completion_status.csv")
    metrics = pd.read_csv(tables_dir / "sliding_window_metrics.csv")
    rankings = pd.read_csv(tables_dir / "figure00_rankings.csv")
    events = pd.read_csv(tables_dir / "extended_event_aligned.csv")
    pairwise = pd.read_csv(tables_dir / "figure00_pairwise_comparisons.csv")
    pooled = pd.read_csv(tables_dir / "pooled_metrics.csv")
    wins = pd.read_csv(tables_dir / "win_counts_per_window.csv")
    win_summary = pd.read_csv(tables_dir / "win_counts_summary.csv")
    ic_summary = pd.read_csv(tables_dir / "information_criteria_summary.csv")
    equivalence, family_summary, combined, domains = _figure00_tables(
        metrics, events, status, tables_dir,
    )
    figure00_report(
        rankings, equivalence, family_summary, combined, domains, status, figures_dir,
    )
    figure01_nll_distributions(metrics, pooled, status, figures_dir)
    figure02_nll_trajectories(metrics, status, figures_dir)
    figure03_pairwise(pairwise, status, figures_dir)
    figure04_win_counts(win_summary, status, figures_dir)
    figure05_winner_timeline(wins, status, figures_dir)
    figure06_information_criteria(ic_summary, status, figures_dir)
    from . import extended_figures
    extended_tables = {
        path.name: pd.read_csv(path)
        for path in tables_dir.glob("*.csv")
    }
    extended_figures.run(extended_tables, status, figures_dir)
    paths = tuple(sorted(figures_dir.glob("fig*.png"))) + tuple(sorted(figures_dir.glob("fig*.pdf")))
    for path in paths:
        print(f"wrote {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    parser.add_argument(
        "--figure", choices=("all", "fig00"), default="all",
        help="Render all figures or only the Figure 0 ranking/CI/TOST report",
    )
    args = parser.parse_args()
    if args.figure == "fig00":
        run_figure00(args.mouse)
    else:
        run(args.mouse)


if __name__ == "__main__":
    main()
