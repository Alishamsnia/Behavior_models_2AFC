"""Render the 5-page identifiability report PDF for one mouse.

Covers the same 13-model registry as the expanded full-session-matched
model comparison (``post_analysis.expanded_model_comparison.config``),
reusing its model names/colors, rather than the older M0-M4 chain.

Page 1: multistart parameter spread (real; zero new model evaluations).
Page 2: Fisher information / Hessian curvature with Laplace-approximated SEs
        (real; re-evaluation only -- see collect.py).
Page 3: the same Hessians' eigenspectra ("sloppy model" analysis; real).
Page 4: profile likelihood -- placeholder (too slow for this report's
        per-mouse budget; see the page for the cost estimate and what
        partial machinery already exists).
Page 5: simulation-based parameter/model recovery -- placeholder (same
        reason).
Page 6: fitted parameter VALUES across all 30 windows, chronologically
        (real; zero new model evaluations).
Page 7: per-window "restart-instability" rank -- NOT a Fisher-info/curvature
        rank -- across all 30 windows (real; zero new model evaluations; see
        the page's own caption for why this deliberately isn't a per-window
        Hessian sweep, and for the small benchmarked cross-check against real
        Hessians that validates it).
Page 8: cross-model parameter-identifiability comparison -- two independent
        panels (restart instability aggregated across all windows; curvature
        anisotropy from the same single-session Hessian as pages 2-3), not
        collapsed into one score.

Run with ``python -m post_analysis.identifiability.figures --mouse MC08``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from post_analysis import common
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER, group_style
from post_analysis.identifiability.collect import (
    FisherInformationResult,
    collect_model_identifiability_summary,
    collect_multistart_spread,
    collect_parameter_trajectories,
    collect_restart_instability_rank,
    compute_fisher_information,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10.5,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

# Same model set, names, and colors as the expanded full-session-matched
# model comparison (post_analysis.expanded_model_comparison.config), so this
# report's figures read consistently against those.
MODEL_LABELS = {m: MODEL_BY_ID[m].short_label for m in MODEL_ORDER}
MODEL_COLORS = {m: MODEL_BY_ID[m].color for m in MODEL_ORDER}


def _save_pdf_pages(figs: list, figures_dir, stem: str) -> None:
    figures_dir = Path(figures_dir)
    temporary_path = None
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


def _wrap_width_chars(fig_width_inches: float, fontsize: float, margin: float = 0.92) -> int:
    """Character count that fills ``fig_width_inches`` at ``fontsize`` (pt),
    for wrapping caption text so it never overflows the figure's own width."""
    fig_width_pt = fig_width_inches * 72.0
    average_char_width_pt = 0.52 * fontsize
    return max(20, int(fig_width_pt * margin / average_char_width_pt))


def _add_wrapped_header(
    fig: plt.Figure, title: str, body: str, *,
    title_fontsize: float = 12.5, body_fontsize: float = 9.0,
    body_color: str = "0.25", top_margin: float = 0.05,
) -> float:
    """Draw a bold title plus a word-wrapped caption at the top of ``fig``,
    wrapped to the figure's OWN width in inches.

    An unwrapped long caption is wider than the figure's nominal size, and
    ``savefig(..., bbox_inches="tight")`` then expands the saved page to fit
    that overflowing text -- shrinking the actual plot to a fraction of the
    page and making the title look wider than the plot. Wrapping here to a
    character count sized from ``fig.get_size_inches()`` keeps every header
    line within the figure's real bounds, so the saved page stays the
    figure's actual size.

    Returns the top fraction to pass to
    ``fig.tight_layout(rect=[0, 0, 1, top])`` (or
    ``fig.subplots_adjust(top=top)``) so the axes below don't collide with
    the header; the header's own height is fixed in inches (not a fixed
    fraction of the page), so it stays proportionate whether the page is
    short (page 3) or tall (many stacked per-model rows, pages 1-2).
    """
    fig_width_in, fig_height_in = fig.get_size_inches()
    wrap_width = _wrap_width_chars(fig_width_in, body_fontsize)
    wrapped_body = "\n".join(textwrap.wrap(body, width=wrap_width))
    body_line_count = wrapped_body.count("\n") + 1

    title_line_in = title_fontsize / 72.0 * 1.3
    body_line_in = body_fontsize / 72.0 * 1.55
    title_body_gap_in = 0.07
    header_in = top_margin + title_line_in + title_body_gap_in + body_line_count * body_line_in + 0.15

    fig.text(
        0.5, 1.0 - top_margin / fig_height_in, title, ha="center", va="top",
        fontsize=title_fontsize, fontweight="bold",
    )
    body_top = 1.0 - (top_margin + title_line_in + title_body_gap_in) / fig_height_in
    fig.text(
        0.5, body_top, wrapped_body, ha="center", va="top",
        fontsize=body_fontsize, color=body_color, linespacing=1.45,
    )

    top_frac = 1.0 - header_in / fig_height_in
    return max(0.5, min(0.97, top_frac))


def _placeholder_page(title: str, body_lines: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")
    for spine_alpha, pad in ((0.35, 0.06),):
        ax.add_patch(plt.Rectangle(
            (pad, pad), 1 - 2 * pad, 1 - 2 * pad, transform=ax.transAxes,
            fill=False, edgecolor="0.6", linestyle="--", linewidth=1.4,
        ))
    ax.text(0.5, 0.86, "PLACEHOLDER", transform=ax.transAxes, ha="center", va="center",
            fontsize=15, fontweight="bold", color="0.55")
    ax.text(0.5, 0.78, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=13.5, fontweight="bold")
    ax.text(
        0.5, 0.42, "\n".join(body_lines), transform=ax.transAxes,
        ha="center", va="center", fontsize=10.5, linespacing=1.7, wrap=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1: multistart parameter spread
# ---------------------------------------------------------------------------

def _page_multistart_spread(spread: pd.DataFrame, mouse_id: str) -> plt.Figure:
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(12.0, 2.1 * len(MODEL_ORDER)))
    for row_index, model_id in enumerate(MODEL_ORDER):
        ax = axes[row_index]
        param_names = MODEL_BY_ID[model_id].parameter_names
        subset = spread.loc[spread["model_id"] == model_id]
        data = [
            subset.loc[subset["param_name"] == name, "spread_std"].to_numpy()
            for name in param_names
        ]
        n_windows = subset["window_id"].nunique() if not subset.empty else 0
        box = ax.boxplot(
            data, tick_labels=param_names, showfliers=True, patch_artist=True,
            widths=0.55, flierprops={"markersize": 3, "alpha": 0.5},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(MODEL_COLORS[model_id])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODEL_COLORS[model_id])
        for median in box["medians"]:
            median.set_color(MODEL_COLORS[model_id])
            median.set_linewidth(1.8)
        ax.set_ylabel("SD across 8 starts", fontsize=9.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.set_title(MODEL_LABELS[model_id], fontsize=9.5, loc="left", color=MODEL_COLORS[model_id], fontweight="bold")
        ax.set_title(f"N={n_windows} rolling windows", fontsize=9, loc="right", color="0.4")
        ax.tick_params(labelsize=9)
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    axes[-1].set_xlabel("Constrained parameter")
    top = _add_wrapped_header(
        fig, f"Multistart parameter spread -- {mouse_id}",
        "Per rolling window, SD of each parameter's fitted (constrained-space) value across the 8 "
        "independently-initialized restarts; boxplots summarize that SD's distribution across every "
        "window. Large spread = the 8 restarts disagree on that parameter even though they were scored "
        "on the same data -- a practical-identifiability red flag. Zero new model evaluations: reads "
        "only the multistart summaries already written during fitting.",
    )
    fig.tight_layout(rect=[0, 0, 1, top], h_pad=1.6)
    return fig


# ---------------------------------------------------------------------------
# Page 2: Fisher information / Hessian curvature (Laplace-approximated SEs)
# ---------------------------------------------------------------------------

def _page_fisher_information(fisher: dict[str, FisherInformationResult], mouse_id: str) -> plt.Figure:
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(12.0, 2.1 * len(MODEL_ORDER)))
    for row_index, model_id in enumerate(MODEL_ORDER):
        ax = axes[row_index]
        result = fisher.get(model_id)
        if result is None:
            ax.text(0.5, 0.5, "No converged session available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(MODEL_LABELS[model_id], fontsize=9.5, loc="left", color=MODEL_COLORS[model_id], fontweight="bold")
            ax.axis("off")
            continue
        x = np.arange(len(result.param_names))
        ax.errorbar(
            x, result.raw_point, yerr=2 * result.raw_se, fmt="none",
            ecolor=MODEL_COLORS[model_id], elinewidth=1.2, capsize=4, alpha=0.35, zorder=1,
        )
        ax.errorbar(
            x, result.raw_point, yerr=result.raw_se, fmt="o", markersize=6,
            color=MODEL_COLORS[model_id], ecolor=MODEL_COLORS[model_id],
            elinewidth=2.2, capsize=4, zorder=2,
        )
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, symmetric=True))
        ax.set_xticks(x)
        ax.set_xticklabels(result.param_names, fontsize=9)
        ax.set_ylabel("raw value\n(symlog)", fontsize=9.5)
        ax.set_title(MODEL_LABELS[model_id], fontsize=9.5, loc="left", color=MODEL_COLORS[model_id], fontweight="bold")
        ax.set_title(
            f"session {result.session_id.split('_')[1]} -- n={result.n_trials} train trials -- "
            f"Hessian in {result.hessian_seconds:.1f}s",
            fontsize=9, loc="right", color="0.4",
        )
        ax.tick_params(labelsize=8.5)
    axes[-1].set_xlabel("Raw (pre-sigmoid/pre-transform) parameter")
    top = _add_wrapped_header(
        fig, f"Fisher information / Laplace-approximated parameter uncertainty -- {mouse_id}",
        "Point = fitted raw parameter at one representative converged single-session fit (first "
        "chronologically, per model). Dark whiskers = ±1 SE, pale whiskers = ±2 SE, from "
        "sqrt(diag) of the pseudo-inverse Hessian of the summed training NLL (Laplace approximation to "
        "the posterior covariance). Y-axis is symlog (linear within ±1, log beyond) because sloppy "
        "directions' SEs are orders of magnitude larger than stiff ones -- see page 3. SEs are in "
        "RAW/logit units, not the constrained parameter's own units -- a delta-method transform to "
        "constrained-space SEs is a natural next step, not done here. Near-zero/negative eigenvalues "
        "are floored before inverting, so a huge SE here means a genuinely sloppy direction, not a "
        "numerical artifact.",
    )
    fig.tight_layout(rect=[0, 0, 1, top], h_pad=1.6)
    return fig


# ---------------------------------------------------------------------------
# Page 3: sloppy-model eigenspectrum
# ---------------------------------------------------------------------------

def _dominant_loadings_text(result: FisherInformationResult, top_n: int = 2) -> str:
    sloppiest_vector = result.eigenvectors[:, 0]
    order = np.argsort(-np.abs(sloppiest_vector))[:top_n]
    terms = [f"{result.param_names[i]} ({sloppiest_vector[i]:+.2f})" for i in order]
    return f"{MODEL_LABELS[result.model_id]}: " + ", ".join(terms)


def _page_sloppy_eigenspectrum(fisher: dict[str, FisherInformationResult], mouse_id: str) -> plt.Figure:
    fig = plt.figure(figsize=(12.0, 11.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.6], hspace=0.38)
    ax = fig.add_subplot(grid[0])
    text_ax = fig.add_subplot(grid[1])
    text_ax.axis("off")

    floor = 1e-12
    for model_id in MODEL_ORDER:
        result = fisher.get(model_id)
        if result is None:
            continue
        eigenvalues = np.clip(result.eigenvalues, floor, None)
        n = len(eigenvalues)
        rank = np.arange(n) / max(n - 1, 1)
        ax.plot(
            rank, np.log10(eigenvalues), color=MODEL_COLORS[model_id],
            linestyle=group_style(model_id, "line"), marker=group_style(model_id, "marker"),
            markersize=4, linewidth=1.6,
            label=f"{MODEL_LABELS[model_id]} ({n} params)",
        )
    ax.set_xlabel(
        "Eigenvalue rank, one marker per parameter combination\n"
        "(0 = sloppiest, 1 = stiffest; rescaled 0-1 per model so models with different parameter counts line up)"
    )
    ax.set_ylabel("log10(Hessian eigenvalue)\n(curvature of the training NLL along that direction)")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, title="model (parameter count)", title_fontsize=7.5)
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)

    how_to_read = textwrap.fill(
        "How to read the plot above: each Hessian (one per model, from page 2) has as many eigenvalues "
        "as that model has parameters -- one per independent direction/combination in parameter space, "
        "not one per named parameter. Each MARKER on a model's line is one such eigenvalue, sorted from "
        "smallest (sloppiest = least constrained by the data) to largest (stiffest = most tightly "
        "constrained), then placed at its RESCALED RANK on the X AXIS (0-1) so models with different "
        "parameter counts can be compared on the same axis. The Y AXIS is log10 of that eigenvalue: "
        "higher means the training NLL rises steeply if that parameter combination is perturbed away "
        "from the fit (well identified); lower/flatter means the NLL barely changes (poorly identified, "
        "even if every individual raw parameter's own point estimate in page 2 looked fine). A model "
        "whose line spans many orders of magnitude top-to-bottom is \"sloppy\": most combinations are "
        "well constrained but a few are nearly free to vary.",
        width=118,
    )
    table_header = textwrap.fill(
        "Table below, per model: the SLOPPIEST direction's dominant parameter loadings -- the (up to) 2 "
        "raw parameters that make up that flattest/least-identifiable combination, and how strongly (the "
        "signed eigenvector component in parentheses; larger magnitude = that parameter dominates the "
        "combination) -- followed by \"decades\": log10(largest eigenvalue) - log10(smallest eigenvalue), "
        f"the number of orders of magnitude spanning stiffest-to-sloppiest, from the same floor "
        f"({floor:.0e}) used in the plot (the true sloppiest eigenvalue is often indistinguishable from "
        "zero at double precision, so this is a lower bound on the span, not an exact figure):",
        width=118,
    )
    lines = [how_to_read, "", table_header, ""]
    for model_id in MODEL_ORDER:
        result = fisher.get(model_id)
        if result is None:
            continue
        floored_min = max(float(result.eigenvalues.min()), floor)
        decades = np.log10(float(result.eigenvalues.max())) - np.log10(floored_min)
        lines.append(f"{_dominant_loadings_text(result)}  |  ≥{decades:.1f} decades")
    text_ax.text(
        0.0, 1.0, "\n".join(lines), transform=text_ax.transAxes, ha="left", va="top",
        fontsize=7.8, linespacing=1.55, family="monospace",
    )

    top = _add_wrapped_header(
        fig, f"Sloppy-model eigenspectrum -- {mouse_id}",
        "Eigenvalues of the same per-model Hessians as page 2 (raw-parameter space), sorted ascending. "
        "A wide vertical spread (many orders of magnitude) is the classic \"sloppy model\" signature: "
        "most parameter combinations (stiff directions, high eigenvalue) are tightly constrained by the "
        "data, while a few combinations (sloppy directions, low eigenvalue) are barely constrained at "
        "all even though every individual raw parameter looks well-fit. See the \"how to read\" note "
        "below the plot for what each marker/axis/table entry means.",
    )
    fig.subplots_adjust(top=top, bottom=0.06, left=0.08, right=0.97)
    return fig


# ---------------------------------------------------------------------------
# Pages 4-5: placeholders
# ---------------------------------------------------------------------------

def _page_profile_likelihood_placeholder(mouse_id: str) -> plt.Figure:
    return _placeholder_page(
        f"Profile likelihood -- {mouse_id}",
        [
            "Not computed here: at each grid point, profile likelihood re-optimizes every OTHER",
            "parameter (unlike Fisher info, which needs no re-optimization at all).",
            "",
            "Partial machinery already exists (profile_*_dynamic_surface in src/{m1_dynamic_block_length_v1,",
            "m2_dynamic_block_length_v1,m2_dynamic_previous_wsls_bias_v1,m2_dynamic_previous_action_bias_v1,",
            "m3_hmm_delta_v1,m4_hsmm_delta_v1}/identifiability.py) but only covers each model's \"dynamic\"",
            "parameters (ah, bh, delta0, alpha_theta, ...), not the shared sensory/lapse parameters every",
            "model in this report's 13-model registry also has, and is opt-in only, never run automatically.",
            "",
            "Cost estimate at that machinery's own defaults (15x15 grid, 2 parameter pairs, 40-iteration",
            "capped L-BFGS per grid point = 450 re-optimizations): on the order of 1-3 hours for ONE",
            "model on ONE window -- far past this report's per-mouse budget. A full 1D profile per",
            "individual parameter (not just pairs), across all 13 models, would be substantially more.",
            "",
            "To fill in this page: pick a reduced scope (e.g. one parameter, one model, a coarser grid)",
            "before running the existing profile_* functions, or budget a background/overnight job.",
        ],
    )


def _page_parameter_recovery_placeholder(mouse_id: str) -> plt.Figure:
    return _placeholder_page(
        f"Model / parameter recovery -- {mouse_id}",
        [
            "Not computed here: simulate trials from known ground-truth parameters, refit the model on",
            "the simulation, and compare recovered vs. true parameters -- the strongest empirical",
            "identifiability check (no quadratic/Laplace approximation error), but it needs MANY full",
            "multistart refits (one per simulated dataset), each on the order of minutes (M0) to",
            "~10-20 minutes (M4) per single-session 8-start fit -- see the real elapsed_seconds already",
            "logged in results_single_session/*/full_session_matched_v1/single_session_run_summary.json.",
            "",
            "run_m2_recovery.py and run_m4_recovery.py already exist and implement this end-to-end for",
            "M2 and M4; check whether they have already been run before starting fresh.",
            "",
            "To fill in this page: run (or locate existing output from) those recovery scripts for a",
            "small number of simulated datasets per model, then plot recovered-vs-true scatter per",
            "parameter here.",
        ],
    )


# ---------------------------------------------------------------------------
# Page 6: fitted parameter values across all windows, chronologically
# ---------------------------------------------------------------------------

# One qualitative color per parameter SLOT within a model row (distinguishes
# up to 9 parameters plotted together on one axis; unrelated to MODEL_COLORS,
# which distinguishes MODELS from each other on other pages).
_PARAM_PALETTE = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#994455", "#997700",
)


def _page_parameter_trajectories(trajectories: pd.DataFrame, mouse_id: str) -> plt.Figure:
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(12.0, 2.1 * len(MODEL_ORDER)))
    for row_index, model_id in enumerate(MODEL_ORDER):
        ax = axes[row_index]
        param_names = MODEL_BY_ID[model_id].parameter_names
        subset = trajectories.loc[trajectories["model_id"] == model_id]
        n_windows = subset["window_id"].nunique() if not subset.empty else 0
        for param_index, param_name in enumerate(param_names):
            series = subset.loc[subset["param_name"] == param_name].sort_values("window_id")
            ax.plot(
                series["window_id"], series["value"], marker="o", markersize=3, linewidth=1.3,
                color=_PARAM_PALETTE[param_index % len(_PARAM_PALETTE)], label=param_name,
            )
        ax.set_ylabel("constrained value\n(symlog)", fontsize=9.5)
        ax.set_title(MODEL_LABELS[model_id], fontsize=9.5, loc="left", color=MODEL_COLORS[model_id], fontweight="bold")
        ax.set_title(f"N={n_windows} windows", fontsize=9, loc="right", color="0.4")
        ax.tick_params(labelsize=9)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, symmetric=True))
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        ax.legend(
            fontsize=7, ncol=min(len(param_names), 5), loc="upper center",
            bbox_to_anchor=(0.5, 1.0), frameon=True, framealpha=0.85, edgecolor="none",
            handlelength=1.4, columnspacing=1.0,
        )
    axes[-1].set_xlabel("Sliding window (chronological order, window_id)")
    top = _add_wrapped_header(
        fig, f"Fitted parameter values across all windows -- {mouse_id}",
        "Per window, the WINNING multistart restart's fitted constrained parameter value (not a "
        "restart mean or spread -- the actual value this window's fit reports downstream elsewhere "
        "in the pipeline). One line per named parameter per model, in chronological window order. "
        "Y-axis is symlog (linear within ±1, log beyond) because some parameters stay in [0,1]-ish "
        "ranges while others (e.g. an unbounded history-weight logit) occasionally swing into the "
        "hundreds for one or two windows -- a purely linear axis would flatten every other parameter "
        "to an invisible line near zero whenever that happens. Zero new model evaluations: reads only "
        "each window's own already-written winning fit.",
    )
    fig.tight_layout(rect=[0, 0, 1, top], h_pad=1.6)
    return fig


# ---------------------------------------------------------------------------
# Page 7: restart-instability rank across all windows -- NOT a Fisher-info/
# curvature rank (see collect.py module note + caption below).
# ---------------------------------------------------------------------------

def _instability_extremes(rank: pd.DataFrame, model_id: str, param_names: tuple) -> pd.DataFrame:
    """Per parameter: how many windows it was THE sloppiest (largest
    spread_std that window) and THE stiffest (smallest). Ties broken by
    parameter order (pandas idxmax/idxmin's first-occurrence rule) -- with
    30 overlapping, non-independent windows (adjacent windows share 8/9
    training sessions), treat these as descriptive occupancy counts, not
    independent-trial statistics.
    """
    subset = rank.loc[rank["model_id"] == model_id]
    sloppiest_counts = {name: 0 for name in param_names}
    stiffest_counts = {name: 0 for name in param_names}
    for _, group in subset.groupby("window_id"):
        if len(group) < 2:
            continue
        sloppiest_counts[group.loc[group["spread_std"].idxmax(), "param_name"]] += 1
        stiffest_counts[group.loc[group["spread_std"].idxmin(), "param_name"]] += 1
    return pd.DataFrame({
        "param_name": list(param_names),
        "sloppiest_count": [sloppiest_counts[name] for name in param_names],
        "stiffest_count": [stiffest_counts[name] for name in param_names],
    })


def _page_restart_instability_rank(rank: pd.DataFrame, mouse_id: str, validation: Optional[pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(
        len(MODEL_ORDER), 2, figsize=(13.5, 2.3 * len(MODEL_ORDER)),
        gridspec_kw={"width_ratios": [5.6, 1.7]},
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    image = None
    for row_index, model_id in enumerate(MODEL_ORDER):
        heat_ax, bar_ax = axes[row_index, 0], axes[row_index, 1]
        param_names = MODEL_BY_ID[model_id].parameter_names
        subset = rank.loc[rank["model_id"] == model_id]
        n_windows = subset["window_id"].nunique() if not subset.empty else 0

        if subset.empty:
            heat_ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=heat_ax.transAxes)
            heat_ax.axis("off")
            bar_ax.axis("off")
        else:
            pivot = subset.pivot(index="param_name", columns="window_id", values="percentile_rank")
            pivot = pivot.reindex(index=param_names)
            full_range = range(int(subset["window_id"].min()), int(subset["window_id"].max()) + 1)
            pivot = pivot.reindex(columns=full_range)
            image = heat_ax.imshow(
                np.ma.masked_invalid(pivot.to_numpy()), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                extent=[full_range.start - 0.5, full_range.stop - 0.5, len(param_names) - 0.5, -0.5],
                interpolation="nearest",
            )
            heat_ax.set_yticks(range(len(param_names)))
            heat_ax.set_yticklabels(param_names, fontsize=8.5)
            heat_ax.tick_params(labelsize=8.5)

            extremes = _instability_extremes(rank, model_id, param_names)
            y_pos = np.arange(len(param_names))
            bar_ax.barh(y_pos - 0.2, extremes["sloppiest_count"], height=0.38, color="#CC6677", label="sloppiest")
            bar_ax.barh(y_pos + 0.2, extremes["stiffest_count"], height=0.38, color="#4477AA", label="stiffest")
            bar_ax.set_ylim(len(param_names) - 0.5, -0.5)
            bar_ax.set_yticks([])
            bar_ax.set_xlabel("# windows", fontsize=7.5)
            bar_ax.tick_params(labelsize=7.5)
            if row_index == 0:
                bar_ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.85, edgecolor="none")

        heat_ax.set_title(MODEL_LABELS[model_id], fontsize=9.5, loc="left", color=MODEL_COLORS[model_id], fontweight="bold")
        heat_ax.set_title(f"N={n_windows} windows", fontsize=9, loc="right", color="0.4")
    axes[-1, 0].set_xlabel("Sliding window (chronological order, window_id)")
    axes[-1, 1].set_xlabel("# windows (sloppiest / stiffest)", fontsize=8)

    validation_note = "Not yet benchmarked against a real Hessian on this run."
    if validation is not None and not validation.empty:
        non_trivial = validation.loc[validation["n_params"] > 2, "spearman_rho"]
        rho_min, rho_max = validation["spearman_rho"].min(), validation["spearman_rho"].max()
        non_trivial_mean = non_trivial.mean() if not non_trivial.empty else float("nan")
        validation_note = (
            f"CAUTION, benchmarked on only {len(validation)} (model, window) pairs against real "
            f"per-window Hessians (see collect.py's validate_restart_instability_against_hessian): "
            f"Spearman rank correlation between this restart-instability rank and the Hessian raw-SE "
            f"rank for the same window ranged {rho_min:+.2f} to {rho_max:+.2f} -- INCONSISTENT, and for "
            f"the {len(non_trivial)} pairs with >2 parameters (where the comparison is non-trivial) the "
            f"mean was only {non_trivial_mean:+.2f}, i.e. close to no relationship on average. Read this "
            "page's rank as its own restart-stability diagnostic, NOT as a cheap proxy for Fisher-info "
            "curvature -- the two frequently disagree, including sign-flipping between windows for the "
            "same model."
        )

    top = _add_wrapped_header(
        fig, f"Restart-instability rank across all windows -- {mouse_id}",
        "NOT a Fisher-information/curvature rank -- see page 2-3 for that. Per window, each model's "
        "named parameters are ranked by how much the near-best-NLL multistart restarts DISAGREE on "
        "that parameter's constrained value (collect_multistart_spread's spread_std), from 0 "
        "(largest disagreement = \"sloppiest\" in this restart-instability sense) to 1 (smallest "
        "disagreement = \"stiffest\"/most restart-stable), as a percentile so models with different "
        "parameter counts are comparable. Left heatmap: that rank across all windows, chronologically "
        "-- blank cells are windows without enough qualifying restarts. Right bars: how many windows "
        "each parameter was THE single sloppiest / stiffest (ties broken by parameter order). The 30 "
        "windows overlap (adjacent windows share 8 of 9 training sessions), so these are descriptive "
        f"occupancy counts over an autocorrelated sequence, not 30 independent replications. {validation_note}",
        body_fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0, 0.93, top], h_pad=1.8, w_pad=1.2)
    if image is not None:
        cbar_ax = fig.add_axes([0.945, 0.06, 0.014, top - 0.10])
        fig.colorbar(image, cax=cbar_ax, label="percentile rank (0=sloppiest, 1=stiffest)")
    return fig


# ---------------------------------------------------------------------------
# Page 8: cross-model parameter-identifiability comparison (two independent
# panels -- deliberately not collapsed into one "trustworthiness" score).
# ---------------------------------------------------------------------------

def _page_model_identifiability_summary(summary: pd.DataFrame, mouse_id: str) -> plt.Figure:
    fig, (ax_restart, ax_decades) = plt.subplots(1, 2, figsize=(14.0, 7.8))

    restart_order = summary.sort_values("median_relative_restart_instability", ascending=True)
    colors_restart = [MODEL_COLORS[m] for m in restart_order["model_id"]]
    ax_restart.barh(
        [MODEL_LABELS[m] for m in restart_order["model_id"]],
        restart_order["median_relative_restart_instability"], color=colors_restart,
    )
    ax_restart.set_xlabel("Median relative restart instability\n(spread_std / (|value| + 1e-3), across all windows & parameters)")
    ax_restart.set_title("(a) Restart-instability summary\n(all 30 windows, lower = more restart-stable)", fontsize=10.5)
    ax_restart.tick_params(labelsize=8.5)
    ax_restart.set_xscale("log")
    ax_restart.invert_yaxis()  # best (lowest instability) at top

    decades_order = summary.sort_values("decades_spanned", ascending=True)
    colors_decades = [MODEL_COLORS[m] for m in decades_order["model_id"]]
    ax_decades.barh(
        [MODEL_LABELS[m] for m in decades_order["model_id"]],
        decades_order["decades_spanned"], color=colors_decades,
    )
    ax_decades.set_xlabel("Decades spanned (curvature anisotropy)\nsame single-session Hessian as pages 2-3")
    ax_decades.set_title("(b) Curvature-anisotropy summary\n(one representative session, lower = less sloppy spectrum)", fontsize=10.5)
    ax_decades.tick_params(labelsize=8.5)
    ax_decades.invert_yaxis()  # best (lowest anisotropy) at top

    top = _add_wrapped_header(
        fig, f"Cross-model parameter-identifiability comparison -- {mouse_id}",
        "Two independent views, deliberately NOT collapsed into one \"trustworthiness\" score. (a) "
        "reuses page 7's restart-instability data, aggregated (median, epsilon-floored to avoid "
        "blow-up near zero-valued parameters) across every parameter and all 30 windows per model -- "
        "an approximate, scale-normalized summary, not an exact one. (b) reuses the SAME single "
        "representative-session Hessian as pages 2-3 (not an all-windows aggregate -- per-window "
        "Hessians for all 13 models x 30 windows were estimated at several hours of compute and ruled "
        "out); it measures curvature ANISOTROPY (condition-number-like), not absolute information, so "
        "a uniformly-sloppy model can score \"better\" here than a mostly-stiff model with one weak "
        "direction. The two panels can and do disagree -- that's expected, not a bug: neither alone is "
        "a full picture, and neither is a claim about predictive accuracy or model-selection quality.",
        body_fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0, 1, top], w_pad=3.0)
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(mouse_id: str) -> Path:
    tables_dir, figures_dir = common.ensure_output_dirs("identifiability", mouse_id)

    spread = collect_multistart_spread(mouse_id)
    fisher = compute_fisher_information(mouse_id)
    trajectories = collect_parameter_trajectories(mouse_id)
    restart_rank = collect_restart_instability_rank(mouse_id, spread)
    identifiability_summary = collect_model_identifiability_summary(restart_rank, trajectories, fisher)

    validation_path = Path(tables_dir) / "restart_instability_validation.csv"
    validation = pd.read_csv(validation_path) if validation_path.is_file() else None

    pages = [
        _page_multistart_spread(spread, mouse_id),
        _page_fisher_information(fisher, mouse_id),
        _page_sloppy_eigenspectrum(fisher, mouse_id),
        _page_profile_likelihood_placeholder(mouse_id),
        _page_parameter_recovery_placeholder(mouse_id),
        _page_parameter_trajectories(trajectories, mouse_id),
        _page_restart_instability_rank(restart_rank, mouse_id, validation),
        _page_model_identifiability_summary(identifiability_summary, mouse_id),
    ]
    _save_pdf_pages(pages, figures_dir, "identifiability_report")
    return Path(figures_dir) / "identifiability_report.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08", help="Mouse/subject ID, e.g. MC08")
    args = parser.parse_args()
    path = run(args.mouse)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
