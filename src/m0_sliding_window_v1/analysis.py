"""Experiment-specific figures for M0 sliding-window outputs.

The registered M0 analysis remains unchanged.  This module relaxes only the
psychometric plotting contract: each trial type uses at most four quantile
bins, and sparse panels use one bin per available trial when fewer than four
trials are present.  The plotted points are empirical/model bin averages; no
logistic curve is fitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m0_sensory_v1.analysis import (
    _locked_panel,
    _plot_event_grid,
    _role_subset,
    _save_figure,
    plot_nll_history,
    plot_parameter_history,
)


FIGURE_ANALYSIS_SCHEMA_VERSION = "m0_sliding_experiment_analysis_v2"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m0_sliding_global_analysis_v1"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def figure_ready(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize plotting columns to floating dtype, including nullable data."""

    result = predictions.copy()
    for column in (
        "duration_s",
        "choice_long",
        "category_long",
        "model_probability_long",
        "mouse_correct",
        "model_probability_correct",
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def adaptive_psychometric_summary(
    data: pd.DataFrame,
    *,
    maximum_bins: int = 4,
) -> pd.DataFrame:
    """Return session-balanced binned psychometrics with sparse-panel support."""

    if maximum_bins <= 0:
        raise ValueError("maximum_bins must be positive")
    work = data.loc[
        data["model_evaluated"].astype(bool)
        & data["trial_type"].isin(["short", "long"])
    ].copy()
    work["duration_s"] = pd.to_numeric(work["duration_s"], errors="coerce")
    work = work.loc[work["duration_s"].notna()].sort_values(
        ["session_date", "session_id", "trial_index"]
    )
    parts: list[pd.DataFrame] = []
    for offset, trial_type in ((0, "short"), (maximum_bins, "long")):
        subset = work.loc[work["trial_type"].eq(trial_type)].copy()
        if subset.empty:
            continue
        requested_bins = min(maximum_bins, len(subset))
        jitter = np.random.RandomState(42).normal(0.0, 1e-6, len(subset))
        labels = pd.qcut(
            subset["duration_s"].to_numpy(float) + jitter,
            q=requested_bins,
            labels=False,
            duplicates="drop",
        )
        subset["within_type_bin"] = np.asarray(labels, dtype=int)
        actual_bins = int(subset["within_type_bin"].nunique())
        subset["bin_order"] = subset["within_type_bin"] + offset
        subset["n_bins_for_trial_type"] = actual_bins
        parts.append(subset)
    if not parts:
        return pd.DataFrame()

    binned = pd.concat(parts, ignore_index=True)
    metadata = binned.groupby("bin_order", observed=False).agg(
        bin_center_s=("duration_s", "mean"),
        bin_min_s=("duration_s", "min"),
        bin_max_s=("duration_s", "max"),
        n_trials=("duration_s", "size"),
        n_bins_for_trial_type=("n_bins_for_trial_type", "first"),
    ).reset_index()
    per_session = binned.groupby(
        ["bin_order", "session_id"], observed=False
    ).agg(
        mouse_probability_long=("choice_long", "mean"),
        model_probability_long=("model_probability_long", "mean"),
    ).reset_index()
    summary = per_session.groupby("bin_order", observed=False).agg(
        mouse_probability_long=("mouse_probability_long", "mean"),
        mouse_sem=("mouse_probability_long", "sem"),
        model_probability_long=("model_probability_long", "mean"),
        model_sem=("model_probability_long", "sem"),
        n_sessions=("session_id", "nunique"),
    ).reset_index()
    summary = summary.merge(metadata, on="bin_order", validate="one_to_one")
    summary["trial_type"] = np.where(
        summary["bin_order"] < maximum_bins, "short", "long"
    )
    return summary.sort_values("bin_order").reset_index(drop=True)


def plot_adaptive_psychometrics(
    predictions: pd.DataFrame,
    output: Path,
    *,
    model_label: str = "M0",
) -> list[str]:
    """Plot binned psychometrics, reducing bins only for sparse panels."""

    all_label = (
        "All data"
        if predictions["split_role"].eq("test").any()
        else "All available (test locked)"
    )
    rows = (
        ("all", all_label),
        ("train", "Training"),
        ("validation", "Validation"),
        ("test", "Test"),
    )
    columns = (
        ("all", "Short + long blocks"),
        ("short_block", "Short block"),
        ("long_block", "Long block"),
        ("neutral", "Neutral block"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(17, 13), sharey=True)
    summaries: list[pd.DataFrame] = []
    for row_index, (role, row_label) in enumerate(rows):
        role_data = _role_subset(predictions, role)
        for column_index, (block, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            if role == "test" and role_data.empty:
                _locked_panel(ax)
                continue
            if block == "all":
                panel = role_data.loc[
                    role_data["block_type"].isin(["short_block", "long_block"])
                ]
            else:
                panel = role_data.loc[role_data["block_type"].eq(block)]
            summary = adaptive_psychometric_summary(panel)
            if not summary.empty:
                summary["data_subset"] = role
                summary["block_subset"] = block
                summaries.append(summary)
                ax.errorbar(
                    summary["bin_center_s"],
                    summary["mouse_probability_long"],
                    yerr=summary["mouse_sem"],
                    color="black",
                    marker="o",
                    label="Mouse",
                )
                ax.errorbar(
                    summary["bin_center_s"],
                    summary["model_probability_long"],
                    yerr=summary["model_sem"],
                    color="#c62828",
                    marker="o",
                    label=model_label,
                )
            else:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, linestyle=":")
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\nP(long choice)")
            if row_index == len(rows) - 1:
                ax.set_xlabel("Stimulus duration (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            output / "psychometric_summary.csv", index=False
        )
    fig.suptitle(
        f"{model_label} psychometrics: up to four within-type quantile bins "
        "(no logistic fit)"
    )
    return _save_figure(fig, output / "psychometrics_mouse_vs_model")


def _prediction_figures(predictions: pd.DataFrame, output: Path) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_adaptive_psychometrics(predictions, output)
    files["rare_trials"] = _plot_event_grid(
        predictions,
        output,
        events=(
            ("rare_long_in_short", "Rare long in short block"),
            ("rare_short_in_long", "Rare short in long block"),
        ),
        window=3,
        stem="rare_trial_performance",
        title="Performance around rare trials (±3 retained decisions)",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions,
        output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10,
        stem="block_transition_performance",
        title="Performance around block transitions (±10 retained decisions)",
    )
    return files


def generate_experiment_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    output_directory: str | Path,
) -> dict:
    """Generate per-window/session M0-style figures without changing base M0."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    files["parameters"] = plot_parameter_history(fit_summary, output)
    files["nll"] = plot_nll_history(fit_summary, output)
    manifest = {
        "analysis_schema_version": FIGURE_ANALYSIS_SCHEMA_VERSION,
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": True,
        "psychometric_binning": (
            "up to four quantile bins separately for short and long trials; "
            "fewer bins when fewer than four trials are available; deterministic "
            "N(0,1e-6) jitter, seed 42; session-balanced; no logistic fit"
        ),
        "psychometric_panels": {
            "rows": ["all", "train", "validation", "test"],
            "columns": [
                "short_and_long_blocks",
                "short_block",
                "long_block",
                "neutral_block",
            ],
        },
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "files": files,
    }
    _atomic_json(manifest, output / "figure_manifest.json")
    return manifest


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    """Return whether a completed run already has the experiment figure schema."""

    path = Path(output_directory) / "figure_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("analysis_schema_version") == FIGURE_ANALYSIS_SCHEMA_VERSION


def pool_prediction_tables(
    sources: Iterable[tuple[str, str | Path]],
) -> pd.DataFrame:
    """Pool run-specific predictions while keeping repeated sessions distinct."""

    parts: list[pd.DataFrame] = []
    for source_run_id, source_path in sources:
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction table: {path}")
        frame = pd.read_csv(path)
        frame["source_run_id"] = str(source_run_id)
        frame["original_session_id"] = frame["session_id"].astype(str)
        frame["analysis_session_id"] = (
            frame["source_run_id"] + "::" + frame["original_session_id"]
        )
        parts.append(frame)
    if not parts:
        raise ValueError("At least one prediction table is required")
    return pd.concat(parts, ignore_index=True)


def generate_global_prediction_figures(
    sources: Iterable[tuple[str, str | Path]],
    output_directory: str | Path,
    *,
    source_kind: str,
) -> dict:
    """Pool all source predictions and generate split-specific global figures."""

    source_list = list(sources)
    pooled = pool_prediction_tables(source_list)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "pooled_trial_predictions.csv", index=False)
    analysis = figure_ready(pooled)
    analysis["session_id"] = analysis["analysis_session_id"]
    files = _prediction_figures(analysis, output)
    role_counts = {
        role: int(pooled["split_role"].eq(role).sum())
        for role in ("train", "validation", "test")
    }
    role_source_units = {
        role: int(
            pooled.loc[
                pooled["split_role"].eq(role), "analysis_session_id"
            ].nunique()
        )
        for role in ("train", "validation", "test")
    }
    manifest = {
        "analysis_schema_version": GLOBAL_ANALYSIS_SCHEMA_VERSION,
        "source_kind": source_kind,
        "source_count": len(source_list),
        "source_run_ids": [source_id for source_id, _ in source_list],
        "split_trial_counts": role_counts,
        "split_prediction_unit_counts": role_source_units,
        "pooling_policy": (
            "all predictions retained within each split; repeated physical sessions "
            "from different rolling windows are distinct window-session analysis units"
        ),
        "psychometric_binning": (
            "up to four within-type quantile bins; sparse panels use fewer bins; "
            "session-balanced across prediction units; no logistic fit"
        ),
        "files": {
            "pooled_predictions": ["pooled_trial_predictions.csv"],
            **files,
        },
    }
    _atomic_json(manifest, output / "global_figure_manifest.json")
    return manifest
