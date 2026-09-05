"""Experiment-only figures and pooled exports for M2 sliding-window fits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from m0_sensory_v1.analysis import (
    _plot_event_grid,
    plot_nll_history,
    plot_parameter_history,
)
from m0_sensory_v1.m2_figures import (
    generate_m2_diagnostic_figures,
    generate_m2_identifiability_figure,
)
from m0_sensory_v1.model import evaluate_m2_hazard_contrast_surface
from m0_sliding_window_v1.analysis import (
    figure_ready,
    plot_adaptive_psychometrics,
    pool_prediction_tables,
)

from .experiment import M2_PARAMETER_NAMES


FIGURE_ANALYSIS_SCHEMA_VERSION = "m2_sliding_experiment_analysis_v1"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m2_sliding_global_analysis_v1"


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
        predictions, output, model_label="M2"
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
        model_label="M2",
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
        model_label="M2",
    )
    return files


def generate_m2_experiment_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    m0_style_output_directory: str | Path,
    diagnostic_output_directory: str | Path,
) -> dict:
    """Generate adaptive M0-style and native static-prototype M2 diagnostics."""

    if fit_summary.get("model_id") != "M2_hmm_static_v1":
        raise ValueError("M2 experiment figures require an M2 fit summary")
    output = Path(m0_style_output_directory).expanduser().resolve()
    diagnostic_output = Path(diagnostic_output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    files["parameters"] = plot_parameter_history(
        fit_summary,
        output,
        parameter_names=M2_PARAMETER_NAMES,
        model_label="M2",
    )
    files["nll"] = plot_nll_history(fit_summary, output)
    diagnostic_manifest = generate_m2_diagnostic_figures(
        ready, diagnostic_output
    )
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
        "m2_transition_parameterization": "h_B=sigmoid(ah); h_N=h_B/2",
        "m2_prototype_parameterization": (
            "theta_N=0.5; theta_S=sigmoid(-delta0); theta_L=sigmoid(delta0)"
        ),
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "files": files,
        "native_m2_diagnostics": {
            "directory": str(diagnostic_output),
            "manifest": diagnostic_manifest,
        },
    }
    _atomic_json(manifest, output / "m2_sliding_figure_manifest.json")
    return manifest


def generate_identifiability_outputs(
    sessions,
    best_raw_parameters,
    registry,
    output_directory: str | Path,
    *,
    grid_points: int,
) -> dict:
    """Generate the optional fixed-other M2 hazard/contrast objective slice."""

    if grid_points < 3:
        raise ValueError("M2 identifiability grid requires at least three points")
    surface = evaluate_m2_hazard_contrast_surface(
        sessions,
        torch.as_tensor(best_raw_parameters, dtype=torch.float64),
        registry,
        grid_points=grid_points,
    )
    manifest = generate_m2_identifiability_figure(surface, output_directory)
    manifest["grid_points_per_axis"] = grid_points
    manifest["total_grid_evaluations"] = grid_points * grid_points
    _atomic_json(
        manifest, Path(output_directory) / "m2_identifiability_manifest.json"
    )
    return manifest


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    """Return whether a run already has the current M2 experiment figures."""

    path = Path(output_directory) / "m2_sliding_figure_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("analysis_schema_version") == FIGURE_ANALYSIS_SCHEMA_VERSION


def identifiability_outputs_are_current(
    output_directory: str | Path,
    *,
    grid_points: int,
) -> bool:
    """Return whether the requested optional M2 surface is already complete."""

    if grid_points == 0:
        return True
    path = Path(output_directory) / "m2_identifiability_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("grid_points_per_axis") == grid_points


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
        "model_id": "M2_hmm_static_v1",
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
    "generate_identifiability_outputs",
    "generate_m2_experiment_figures",
    "identifiability_outputs_are_current",
]
