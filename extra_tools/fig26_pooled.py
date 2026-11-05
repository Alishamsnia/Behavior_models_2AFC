#!/usr/bin/env python3
"""Figure 26 (event-conditioned pairwise bootstrap) pooled across mice.

Per-mouse Figure 26 logic, unchanged per mouse: each mouse's cached
figure26_event_aligned.csv gives, per event domain, an event-count-weighted
mean-NLL window x model matrix over the held-out test split, restricted to
complete-case windows that must be chronologically contiguous. The pooled page
stacks those per-mouse matrices and runs the same moving-block bootstrap
(_document_pair_results) stratified by mouse -- blocks are drawn within each
mouse on independent streams and never cross a mouse boundary -- with
equal-window weighting of the stacked rows. Inference is conditional on these
mice (between-mouse variability is not resampled).

Run after each mouse's Figure 26 (extra_tools/fig26_for_mouse.py or
figure26_event_pairwise.py) so the per-mouse event caches are current:

  python extra_tools/fig26_pooled.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from post_analysis.expanded_model_comparison import figure26_event_pairwise as f26  # noqa: E402
from post_analysis.expanded_model_comparison.config import MODEL_ORDER, validate_mouse_id  # noqa: E402

DEFAULT_MICE = ("MC06", "MC07", "MC08")
OUTPUT_DIR = f26.ANALYSIS_ROOT / "pooled" / "model_comparison"
STEM = "fig26_event_conditioned_pairwise_pooled"


def _cached_events(mouse: str) -> pd.DataFrame:
    """The mouse's own Figure 26 event cache, accepted only if it was written
    for exactly the models and source snapshots its current status table lists."""
    tables = f26.ANALYSIS_ROOT / mouse / "tables"
    status = pd.read_csv(tables / "model_completion_status.csv")
    expected = f26._eligible_fingerprints(status)
    if set(expected) != set(MODEL_ORDER):
        raise ValueError(f"{mouse}: pooled Figure 26 needs every configured model eligible")
    cache, provenance = tables / "figure26_event_aligned.csv", tables / "figure26_event_aligned.provenance.json"
    if not cache.is_file() or not provenance.is_file() or json.loads(provenance.read_text()) != expected:
        raise ValueError(f"{mouse}: Figure 26 event cache is missing or stale; rerun that mouse's Figure 26 first")
    return pd.read_csv(cache)


def _contiguous(matrix: pd.DataFrame) -> bool:
    ids = sorted(int(w) for w in matrix.index)
    return ids == list(range(ids[0], ids[0] + len(ids)))


def _domain_inputs(events_by_mouse: dict[str, pd.DataFrame], domain: str):
    """Per-mouse complete-case matrices stacked in mouse order; a mouse whose
    usable windows are too few or not contiguous is left out of this domain."""
    parts, strata, used, skipped, counts = [], [], [], [], []
    for mouse, events in events_by_mouse.items():
        matrix = f26._domain_matrix(events, domain, MODEL_ORDER)
        if len(matrix) < f26.MIN_USABLE_WINDOWS or not _contiguous(matrix):
            reason = "too few" if len(matrix) < f26.MIN_USABLE_WINDOWS else "gaps"
            skipped.append(f"{mouse} ({len(matrix)} windows, {reason})")
            continue
        parts.append(matrix.set_index(pd.Index([f"{mouse}:{w}" for w in matrix.index])))
        strata.append(len(matrix))
        used.append(mouse)
        counts.append(f26._domain_event_counts(events, domain, MODEL_ORDER, matrix.index))
    if not parts:
        return None
    event_counts = pd.concat(counts, axis=1).sum(axis=1)
    return pd.concat(parts), tuple(strata), used, skipped, event_counts


def _domain_page(domain, matrix, strata, used, skipped, event_counts, replicates):
    model_ids = tuple(MODEL_ORDER)
    n_models = len(model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    block_length = min(f26.DEFAULT_PRIMARY_LENGTH, max(1, min(strata) - 1))
    results, sampled = f26._document_pair_results(
        matrix.to_numpy(float), model_ids, block_length, replicates, strata,
    )
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
    windows_text = ", ".join(f"{mouse} {size}" for mouse, size in zip(used, strata))
    skipped_text = f" Left out of this domain (too few or non-contiguous usable windows): {', '.join(skipped)}." if skipped else ""
    footer_ax.text(
        0.5, 0.5,
        f"Pooled across mice ({windows_text} usable windows; {sum(strata)} total).{skipped_text} "
        f"Event window: {f26.DOMAIN_WINDOW_DESCRIPTIONS[domain]}; each model's per-window score is an "
        "event-count-weighted mean NLL over qualifying events (held-out test split only); windows are equally weighted.\n"
        f"Total qualifying event count per model: {events_text} (identical across models). Moving-block bootstrap as in "
        f"Figure 25's guide method, blocks drawn within each mouse and never across mice (block length {block_length}; "
        f"{replicates:,} resamples): uncentred two-minimum bootstrap-tail p-values, Holm-adjusted across the {n_pairs} pairs. "
        "Exploratory; conditional on these mice (between-mouse variability is not resampled).",
        ha="center", va="center", fontsize=8.6, linespacing=1.4, wrap=True,
    )
    figure.suptitle(f"Figure 26 — Event-conditioned pairwise bootstrap comparison (pooled: {' + '.join(used)})\n"
                    f"{f26.DOMAIN_TITLES[domain]}", fontsize=14, fontweight="bold")
    return figure


def _insufficient_page(domain, skipped):
    figure = plt.figure(figsize=(19.5, 12.4))
    figure.text(0.5, 0.5, f"Figure 26 (pooled) — {f26.DOMAIN_TITLES[domain]}\n\nNo mouse has at least "
                f"{f26.MIN_USABLE_WINDOWS} contiguous usable windows for this event definition: {', '.join(skipped)}.",
                ha="center", va="center", fontsize=12)
    return figure


def generate(mice: tuple[str, ...], replicates: int) -> Path:
    events_by_mouse = {mouse: _cached_events(mouse) for mouse in mice}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    try:
        for domain in f26.DOMAIN_ORDER:
            inputs = _domain_inputs(events_by_mouse, domain)
            if inputs is None:
                pages.append(_insufficient_page(domain, list(mice)))
                continue
            pages.append(_domain_page(domain, *inputs, replicates))
        output, pending = OUTPUT_DIR / f"{STEM}.pdf", OUTPUT_DIR / f".{STEM}.pending.pdf"
        with PdfPages(pending, metadata={"Title": "Figure 26 pooled: event-conditioned pairwise bootstrap comparison"}) as pdf:
            for page in pages:
                pdf.savefig(page, bbox_inches="tight")
        pages[0].savefig(OUTPUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=200)
        pending.replace(output)
    finally:
        for page in pages:
            plt.close(page)
        (OUTPUT_DIR / f".{STEM}.pending.pdf").unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--replicates", type=int, default=f26.DEFAULT_REPLICATES)
    args = parser.parse_args()
    print(generate(tuple(validate_mouse_id(mouse) for mouse in args.mice), args.replicates))


if __name__ == "__main__":
    main()
