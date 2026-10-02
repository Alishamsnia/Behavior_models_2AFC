"""Experiment-only figures and pooled exports for M2_known_switch sliding-window fits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from m0_sensory_v1.analysis import _plot_event_grid
from known_switch_analysis import fit_summary_has_trajectory_data, generate_known_switch_fit_figures
from m0_sliding_window_v1.analysis import (
    figure_ready,
    plot_adaptive_psychometrics,
    pool_prediction_tables,
)

from .experiment import M2_KNOWN_SWITCH_PARAMETER_NAMES


# Bumped to _v2 when the parameter-trajectory figure was added to
# generate_known_switch_fit_figures(): a manifest saved under _v1 is missing
# it, so figure_outputs_are_current() must treat it as stale and regenerate,
# not resume_skip, on --resume.
FIGURE_ANALYSIS_SCHEMA_VERSION = "m2_known_switch_sliding_experiment_analysis_v2"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m2_known_switch_sliding_global_analysis_v2"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_figures(
    predictions: pd.DataFrame,
    output: Path,
) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_adaptive_psychometrics(
        predictions, output, model_label="M2 known switch"
    )
    files["rare_trials"] = _plot_event_grid(
        predictions,
        output,
        events=(
            ("rare_long_in_short", "Rare long in short block"),
            ("rare_short_in_long", "Rare short in long block"),
        ),
        window=3,
        stem="rare_trial_performance",
        title="M2 performance around rare trials (±3 retained decisions)",
        model_label="M2 known switch",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions,
        output,
        events=(
            ("short_to_long", "Short → long"),
            ("long_to_short", "Long → short"),
        ),
        window=10,
        stem="block_transition_performance",
        title="M2 performance around block transitions (±10 retained decisions)",
        model_label="M2 known switch",
    )
    return files


def generate_m2_known_switch_experiment_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    m0_style_output_directory: str | Path,
    diagnostic_output_directory: str | Path,
) -> dict:
    """Generate adaptive M0-style and native static-prototype M2 diagnostics."""

    if fit_summary.get("model_id") != "M2_known_switch_v1":
        raise ValueError("M2_known_switch experiment figures require an M2 fit summary")
    output = Path(m0_style_output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    files.update(generate_known_switch_fit_figures(
        fit_summary, output, parameter_names=M2_KNOWN_SWITCH_PARAMETER_NAMES,
        model_label="M2 known switch",
    ))
    manifest = {
        "analysis_schema_version": FIGURE_ANALYSIS_SCHEMA_VERSION,
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": True,
        "psychometric_binning": (
            "up to four quantile bins separately for short and long trials; "
            "sparse panels use fewer bins; deterministic N(0,1e-6) jitter, "
            "seed 42; session-balanced; no logistic fit"
        ),
        "switch_timing": "oracle switch indicator; no fitted hazard",
        "m2_prototype_parameterization": (
            "theta_N=0.5; theta_S=sigmoid(-delta0); theta_L=sigmoid(delta0)"
        ),
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "parameter_trajectory_populated": fit_summary_has_trajectory_data(fit_summary),
        "files": files,
    }
    _atomic_json(manifest, output / "m2_sliding_figure_manifest.json")
    return manifest


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    """Return whether a run already has the current M2_known_switch experiment figures.

    Requires both the schema-version string and ``parameter_trajectory_populated``:
    a fit result saved by a pre-upgrade ``fit.py`` (e.g. reused across
    ``--resume`` from before per-step parameter recording existed) can still
    produce a manifest stamped with the current schema version, but its
    trajectory figure would be empty -- that must not be reported as current.
    """

    path = Path(output_directory) / "m2_sliding_figure_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("analysis_schema_version") == FIGURE_ANALYSIS_SCHEMA_VERSION
        and manifest.get("parameter_trajectory_populated") is True
    )


def generate_global_prediction_figures(
    sources: Iterable[tuple[str, str | Path]],
    output_directory: str | Path,
    *,
    source_kind: str,
) -> dict:
    """Pool all M2 prediction instances and plot each split separately."""

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
        "model_id": "M2_known_switch_v1",
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


__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION",
    "figure_outputs_are_current",
    "figure_ready",
    "generate_global_prediction_figures",
    "generate_m2_known_switch_experiment_figures",
]
