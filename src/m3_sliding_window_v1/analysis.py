"""Experiment-only figures and pooled exports for M3 sliding-window fits."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from m0_sensory_v1.analysis import (
    _plot_event_grid,
    plot_nll_history,
    plot_parameter_history,
)
from m0_sliding_window_v1.analysis import (
    figure_ready as _base_figure_ready,
    plot_adaptive_psychometrics,
    pool_prediction_tables,
)
from m3_hmm_delta_v1.figures import (
    generate_m3_diagnostic_figures,
    generate_m3_identifiability_figure,
    generate_m3_learning_figures,
)
from m3_hmm_delta_v1.identifiability import (
    evaluate_m3_delta_alpha_surface,
    evaluate_m3_hazard_alpha_surface,
    evaluate_m3_hazard_delta_surface,
)

from .experiment import M3_PARAMETER_NAMES


FIGURE_ANALYSIS_SCHEMA_VERSION = "m3_sliding_experiment_analysis_v1"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m3_sliding_global_analysis_v1"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def figure_ready(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add the behavioral labels omitted by M3's native trial exporter."""

    result = _base_figure_ready(predictions)
    if "trial_type" not in result:
        category = pd.to_numeric(result["category_long"], errors="coerce")
        result["trial_type"] = np.select(
            [category.eq(0), category.eq(1)],
            ["short", "long"],
            default="undefined",
        )
    if "block_type" not in result:
        result["block_type"] = result["true_block"].map(
            {"N": "neutral", "S": "short_block", "L": "long_block"}
        ).fillna("undefined")
    return result


def _prediction_figures(
    predictions: pd.DataFrame,
    output: Path,
) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_adaptive_psychometrics(
        predictions, output, model_label="M3"
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
        title="M3 performance around rare trials (±3 retained decisions)",
        model_label="M3",
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
        title="M3 performance around block transitions (±10 retained decisions)",
        model_label="M3",
    )
    return files


def _normalized_fit_summary(fit_summary: dict) -> dict:
    """Adapt M3's step-history names to the shared parameter/NLL plotter."""

    normalized = copy.deepcopy(fit_summary)
    for run in normalized.get("runs", []):
        for row in run.get("history", []):
            row.setdefault("epoch", row.get("step"))
            row.setdefault(
                "constrained_parameters_after_step",
                row.get("constrained_parameters"),
            )
    return normalized


def generate_m3_experiment_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    m0_style_output_directory: str | Path,
    diagnostic_output_directory: str | Path,
) -> dict:
    """Generate adaptive behavioral and native M3 learning diagnostics."""

    if fit_summary.get("model_id") != "M3_hmm_delta_v1":
        raise ValueError("M3 experiment figures require an M3 fit summary")
    output = Path(m0_style_output_directory).expanduser().resolve()
    diagnostic_output = Path(diagnostic_output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_output.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    normalized = _normalized_fit_summary(fit_summary)
    files["parameters"] = plot_parameter_history(
        normalized,
        output,
        parameter_names=M3_PARAMETER_NAMES,
        model_label="M3",
    )
    files["nll"] = plot_nll_history(normalized, output)
    diagnostic_manifest = generate_m3_diagnostic_figures(
        ready, diagnostic_output
    )
    learning_files = generate_m3_learning_figures(
        ready, diagnostic_output / "learning_trajectories"
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
        "m3_transition_parameterization": "h_B=sigmoid(ah); h_N=h_B/2",
        "m3_learning_parameterization": (
            "delta starts at delta0 and receives responsibility-weighted "
            "updates scaled by alpha_theta with the registered hard floor"
        ),
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "files": files,
        "native_m3_diagnostics": {
            "directory": str(diagnostic_output),
            "manifest": diagnostic_manifest,
            "learning_trajectory_files": [str(path) for path in learning_files],
        },
    }
    _atomic_json(manifest, output / "m3_sliding_figure_manifest.json")
    return manifest


def generate_identifiability_outputs(
    sessions,
    best_raw_parameters,
    registry,
    output_directory: str | Path,
    *,
    grid_points: int,
) -> dict:
    """Generate M3's three optional fixed-other-five pairwise loss slices."""

    if grid_points < 3:
        raise ValueError("M3 identifiability grids require at least three points")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = torch.as_tensor(best_raw_parameters, dtype=torch.float64)
    functions = {
        "delta_alpha": evaluate_m3_delta_alpha_surface,
        "hazard_alpha": evaluate_m3_hazard_alpha_surface,
        "hazard_delta": evaluate_m3_hazard_delta_surface,
    }
    files: dict[str, dict[str, str]] = {}
    for stem, function in functions.items():
        surface = function(sessions, raw, registry, grid_points=grid_points)
        csv_path = output / f"{stem}_identifiability_surface.csv"
        surface.to_csv(csv_path, index=False)
        figure_path = generate_m3_identifiability_figure(surface, output)
        files[stem] = {"surface": csv_path.name, "figure": figure_path.name}
    manifest = {
        "model_id": "M3_hmm_delta_v1",
        "grid_points_per_axis": grid_points,
        "surface_count": len(functions),
        "total_grid_evaluations": len(functions) * grid_points * grid_points,
        "full_three_dimensional_volume_generated": False,
        "files": files,
    }
    _atomic_json(manifest, output / "m3_identifiability_manifest.json")
    return manifest


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    path = Path(output_directory) / "m3_sliding_figure_manifest.json"
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
    if grid_points == 0:
        return True
    path = Path(output_directory) / "m3_identifiability_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("grid_points_per_axis") == grid_points
        and manifest.get("surface_count") == 3
    )


def generate_global_prediction_figures(
    sources: Iterable[tuple[str, str | Path]],
    output_directory: str | Path,
    *,
    source_kind: str,
) -> dict:
    """Pool all M3 prediction instances and plot each split separately."""

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
        "model_id": "M3_hmm_delta_v1",
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
        "files": {"pooled_predictions": ["pooled_trial_predictions.csv"], **files},
    }
    _atomic_json(manifest, output / "global_figure_manifest.json")
    return manifest


__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION",
    "figure_outputs_are_current",
    "figure_ready",
    "generate_global_prediction_figures",
    "generate_identifiability_outputs",
    "generate_m3_experiment_figures",
    "identifiability_outputs_are_current",
]
