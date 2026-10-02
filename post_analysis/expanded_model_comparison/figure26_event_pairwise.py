"""Figure 26: event-conditioned pairwise block-bootstrap model comparison.

Extends Figure 25's moving-block-bootstrap pairwise comparison (13 models, 78
pairs, Holm correction within each family) from whole-session held-out NLL to
four prespecified event-conditioned trial windows, plus one combined page
that pools all four:

  - rare_short:    a rare short trial in a long block, +/-1 trial
  - rare_long:     a rare long trial in a short block, +/-1 trial
  - short_to_long: a short->long block transition, 2 trials before to
                    4 trials after
  - long_to_short: a long->short block transition, 2 trials before to
                    4 trials after
  - combined:      all four event types pooled, event-count-weighted
                    (the primary page; the four separate pages are
                    descriptive breakdowns of it, not independent
                    confirmatory tests -- see each page's footer)

Neighbor trials are included regardless of their own trial type (a rare
short trial's +/-1 neighbors can themselves be short- or long-category
trials). Within a window, each event type's per-position NLL rows are
combined into one event-count-weighted mean per model (n_events-weighted,
matching the convention already used for Figure 0's six-domain composite).

The four event windows are narrower than what post_analysis/expanded_model_
comparison/extended_collect.py's shared ``extended_event_aligned.csv``
covers, and reusing that file by filtering relative_position would silently
under-count events (see collect_figure26_events's docstring), so this module
collects its own dedicated, correctly-windowed event table
(figure26_event_aligned.csv) once and caches it.

Run from the repository root:

    python -m post_analysis.expanded_model_comparison.figure26_event_pairwise
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from post_analysis.common import REPO_ROOT
from post_analysis.expanded_model_comparison.block_bootstrap_figures import (
    DEFAULT_REPLICATES,
    _document_pair_results,
    _model_label,
    _pair_matrix_panel,
)
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER
from post_analysis.expanded_model_comparison.extended_collect import collect_figure26_events

ANALYSIS_ROOT = REPO_ROOT / "results_post_analysis" / "expanded_model_comparison"
DEFAULT_PRIMARY_LENGTH = 5
MIN_USABLE_WINDOWS = 8  # below this, a moving-block bootstrap is not trustworthy

DOMAIN_ORDER = ("combined", "rare_short", "rare_long", "short_to_long", "long_to_short")
DOMAIN_TITLES = {
    "combined": "All four event types combined (event-count-weighted)",
    "rare_short": "Rare short trial in a long block (1 trial before/after)",
    "rare_long": "Rare long trial in a short block (1 trial before/after)",
    "short_to_long": "Short→long block transition (2 before, 4 after)",
    "long_to_short": "Long→short block transition (2 before, 4 after)",
}
DOMAIN_WINDOW_DESCRIPTIONS = {
    "combined": "all four event windows below, pooled",
    "rare_short": "positions -1, 0, +1 around the rare trial",
    "rare_long": "positions -1, 0, +1 around the rare trial",
    "short_to_long": "positions -2..+4 around the transition",
    "long_to_short": "positions -2..+4 around the transition",
}


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


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _eligible_fingerprints(status: pd.DataFrame) -> dict[str, str]:
    eligible = status[status["eligible_for_comparison"].astype(bool)]
    return dict(zip(eligible["model_id"].astype(str), eligible["source_snapshot_sha256"].astype(str)))


def _load_events(mouse_id: str, status: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Load the cached Figure 26 event table, (re)collecting it if missing or stale.

    A cache is trusted only if its provenance sidecar records exactly the
    same per-model source snapshot fingerprints as the current status table;
    otherwise the underlying predictions or eligible-model set have changed
    since the cache was written, and it is regenerated.
    """
    cache_path = tables_dir / "figure26_event_aligned.csv"
    provenance_path = tables_dir / "figure26_event_aligned.provenance.json"
    expected = _eligible_fingerprints(status)
    if cache_path.is_file() and provenance_path.is_file():
        recorded = json.loads(provenance_path.read_text())
        if recorded == expected:
            return pd.read_csv(cache_path)
    events = collect_figure26_events(mouse_id, status)
    _write_csv_atomic(cache_path, events)
    _write_json_atomic(provenance_path, expected)
    return events


def _domain_matrix(events: pd.DataFrame, domain: str, models: tuple[str, ...]) -> pd.DataFrame:
    """Event-count-weighted mean-NLL window x model matrix for one domain.

    Pools every relative position (and, for rare events, both trial types --
    a neighbor trial is included regardless of whether it is itself a short-
    or long-category trial) within the event's window, weighting each
    (position, trial_type) row's contribution by how many raw event
    instances it represents (n_events), then requires every one of the 13
    models to have at least one qualifying event in a window before that
    window is used (a strict complete-case matrix, matching the convention
    used for Figure 0's event-domain matrices).
    """
    held_out = events[events["split_role"].eq("test") & events["model_id"].isin(models)]
    rows = held_out if domain == "combined" else held_out[held_out["event"].eq(domain)]
    if rows.empty:
        return pd.DataFrame(columns=list(models))
    weighted = rows.assign(weighted_nll=rows["mean_nll"] * rows["n_events"])
    grouped = weighted.groupby(["model_id", "window_id"], as_index=False).agg(
        weighted_nll=("weighted_nll", "sum"), n_events=("n_events", "sum"),
    )
    grouped["mean_nll"] = grouped["weighted_nll"] / grouped["n_events"]
    matrix = grouped.pivot(index="window_id", columns="model_id", values="mean_nll").reindex(columns=models)
    return matrix.dropna()


def _domain_event_counts(
    events: pd.DataFrame, domain: str, models: tuple[str, ...], window_ids: pd.Index,
) -> pd.Series:
    """Per-model total event count, restricted to the complete-case windows actually used."""
    held_out = events[
        events["split_role"].eq("test")
        & events["model_id"].isin(models)
        & events["window_id"].isin(window_ids)
    ]
    rows = held_out if domain == "combined" else held_out[held_out["event"].eq(domain)]
    return rows.groupby("model_id")["n_events"].sum()


def _event_ranking_panel(
    ax: plt.Axes, model_ids: tuple[str, ...], matrix: pd.DataFrame, sampled: np.ndarray,
) -> None:
    means = matrix.mean(axis=0).to_numpy()
    order = np.argsort(means)
    lower = np.quantile(sampled, 0.025, axis=0)
    upper = np.quantile(sampled, 0.975, axis=0)
    y = np.arange(len(order))
    for yi, index in zip(y, order):
        model_id = model_ids[index]
        spec = MODEL_BY_ID[model_id]
        ax.errorbar(
            means[index], yi,
            xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
            fmt="o", color=spec.color, markerfacecolor="white" if spec.oracle else spec.color,
            markersize=6, capsize=3, linewidth=1.3,
        )
    ax.set_yticks(y, [_model_label(model_ids[index], 30) for index in order], fontsize=7.6)
    for tick, index in zip(ax.get_yticklabels(), order):
        if MODEL_BY_ID[model_ids[index]].oracle:
            tick.set_color("#9C0006")
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Event-conditioned mean NLL (nats/choice; lower is better)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_title(
        "A  Ordered by observed mean (95% CI)",
        loc="left", fontweight="bold", fontsize=9.5,
    )
    ax.text(
        0.99, 0.01,
        "Overlapping intervals are not distinguishable here -- see panel B for pairwise evidence.\n"
        "Hollow markers / dark-red names = oracle benchmarks.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, linespacing=1.4,
    )


def _domain_page(
    domain: str,
    matrix: pd.DataFrame,
    model_ids: tuple[str, ...],
    event_counts: pd.Series,
    replicates: int,
    events: pd.DataFrame,
) -> plt.Figure:
    n_windows = len(matrix)
    window_ids = sorted(int(window_id) for window_id in matrix.index)
    if window_ids != list(range(window_ids[0], window_ids[0] + n_windows)):
        raise ValueError(
            f"Figure 26 {domain}: usable windows have gaps ({window_ids}); the moving-block bootstrap "
            "assumes chronologically contiguous windows and would silently treat non-adjacent windows "
            "as neighbors."
        )
    block_length = min(DEFAULT_PRIMARY_LENGTH, max(1, n_windows - 1))
    results, sampled = _document_pair_results(matrix.to_numpy(float), model_ids, block_length, replicates)

    figure = plt.figure(figsize=(19.5, 12.4), layout="constrained")
    grid = figure.add_gridspec(3, 2, width_ratios=(0.85, 1.35), height_ratios=(0.09, 1.0, 0.14))
    banner_ax = figure.add_subplot(grid[0, :])
    banner_ax.axis("off")
    if domain == "combined":
        banner_text = "PRIMARY EVENT-CONDITIONED COMPARISON — pools all four event types below"
        banner_color = "#1b5e20"
    else:
        banner_text = "DESCRIPTIVE BREAKDOWN of the combined (primary) page — not an independently confirmatory test"
        banner_color = "#8a6d00"
    banner_ax.text(
        0.5, 0.5, banner_text, ha="center", va="center", fontsize=11, fontweight="bold",
        color="white", bbox=dict(boxstyle="round,pad=0.4", facecolor=banner_color, edgecolor="none"),
    )
    rank_ax = figure.add_subplot(grid[1, 0])
    _event_ranking_panel(rank_ax, model_ids, matrix, sampled)
    matrix_ax = figure.add_subplot(grid[1, 1])
    _pair_matrix_panel(
        matrix_ax, model_ids, results,
        f"B  {DOMAIN_TITLES[domain]}: all 78 paired comparisons",
        label_fontsize=5.3,
    )

    footer_ax = figure.add_subplot(grid[2, :])
    footer_ax.axis("off")
    events_min, events_max = int(event_counts.min()), int(event_counts.max())
    events_text = f"{events_min}" if events_min == events_max else f"{events_min}-{events_max}"
    if domain == "combined":
        component_counts = {
            component: int(_domain_event_counts(events, component, model_ids, matrix.index).iloc[0])
            for component in ("rare_short", "rare_long", "short_to_long", "long_to_short")
        }
        component_text = ", ".join(f"{name}={count}" for name, count in component_counts.items())
        weighting_line = f"Component event counts feeding this pooled average: {component_text} (of {events_text} total)."
    else:
        weighting_line = ""
    footer_ax.text(
        0.5, 0.5,
        f"Single mouse (MC08). Event window: {DOMAIN_WINDOW_DESCRIPTIONS[domain]}; neighbor trials included "
        "regardless of their own trial type. Each model's per-window score is an event-count-weighted mean NLL "
        "over all qualifying event instances in that window (n_events-weighted, held-out test split only).\n"
        f"n={n_windows} of 30 chronological sliding windows have a qualifying event for every one of the 13 "
        f"models; per-model total qualifying event count across those windows: {events_text} (identical across "
        "models -- all 13 models are evaluated on the same held-out trials). " + weighting_line + "\n"
        f"Moving-block bootstrap as in Figure 25's guide method, resampling across these {n_windows} chronological "
        f"windows within this one mouse (block length {block_length}; {replicates:,} resamples): uncentred "
        "two-minimum bootstrap-tail p-values, Holm-adjusted across the 78 pairs shown here. Exploratory, "
        "within-mouse result -- not an independent replication.",
        ha="center", va="center", fontsize=8.6, linespacing=1.4, wrap=True,
    )
    figure.suptitle(
        f"Figure 26 — Event-conditioned pairwise bootstrap comparison\n{DOMAIN_TITLES[domain]}",
        fontsize=14, fontweight="bold",
    )
    return figure


def _insufficient_data_page(domain: str, n_windows: int) -> plt.Figure:
    figure = plt.figure(figsize=(19.5, 12.4))
    figure.text(
        0.5, 0.5,
        f"Figure 26 — {DOMAIN_TITLES[domain]}\n\n"
        f"Only {n_windows} of 30 chronological windows have a qualifying event for every one of the 13 models "
        f"(minimum required: {MIN_USABLE_WINDOWS}).\nThe moving-block bootstrap is not reported for this "
        "event definition on this mouse.",
        ha="center", va="center", fontsize=12,
    )
    return figure


def figure26_report(mouse_id: str, replicates: int = DEFAULT_REPLICATES) -> tuple[Path, Path]:
    tables_dir = ANALYSIS_ROOT / mouse_id / "tables"
    figures_dir = ANALYSIS_ROOT / mouse_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(tables_dir / "model_completion_status.csv")
    events = _load_events(mouse_id, status, tables_dir)
    models = MODEL_ORDER

    pages: list[plt.Figure] = []
    try:
        for domain in DOMAIN_ORDER:
            matrix = _domain_matrix(events, domain, models)
            if len(matrix) < MIN_USABLE_WINDOWS:
                pages.append(_insufficient_data_page(domain, len(matrix)))
                continue
            event_counts = _domain_event_counts(events, domain, models, matrix.index)
            pages.append(_domain_page(domain, matrix, models, event_counts, replicates, events))

        output = figures_dir / "fig26_event_conditioned_pairwise.pdf"
        pending_pdf = figures_dir / ".fig26_event_conditioned_pairwise.pending.pdf"
        png_output = figures_dir / "fig26_event_conditioned_pairwise.png"
        pending_png = figures_dir / ".fig26_event_conditioned_pairwise.pending.png"
        pending_pdf.unlink(missing_ok=True)
        with PdfPages(pending_pdf, metadata={
            "Title": "Figure 26: event-conditioned pairwise bootstrap comparison",
            "Author": "Bayesian_model_v2 post-analysis",
        }) as pdf:
            for page in pages:
                pdf.savefig(page, bbox_inches="tight")
        pages[0].savefig(pending_png, bbox_inches="tight", dpi=200)
        pending_pdf.replace(output)
        pending_png.replace(png_output)
    finally:
        for page in pages:
            plt.close(page)
        for pending in (figures_dir / ".fig26_event_conditioned_pairwise.pending.pdf",
                        figures_dir / ".fig26_event_conditioned_pairwise.pending.png"):
            pending.unlink(missing_ok=True)
    return output, png_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args()
    for path in figure26_report(args.mouse, args.replicates):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
