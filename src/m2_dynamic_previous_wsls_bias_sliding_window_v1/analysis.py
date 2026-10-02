"""Experiment-only figures and pooled exports for
M2_dynamic_previous_wsls_bias sliding-window fits."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from m0_sensory_v1.analysis import _plot_event_grid, plot_nll_history, plot_parameter_history
from m0_sliding_window_v1.analysis import (
    figure_ready,
    plot_adaptive_psychometrics,
    pool_prediction_tables,
)
from m2_dynamic_previous_wsls_bias_v1.analysis import build_m2_dynamic_previous_wsls_bias_hazard_curve
from m2_dynamic_previous_wsls_bias_v1.diagnostics import (
    build_m2_dynamic_previous_wsls_bias_implied_duration_distribution,
    evaluate_m2_dynamic_previous_wsls_bias_truncation_sensitivity,
    summarize_m2_dynamic_previous_wsls_bias_hazard_duration,
    summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics,
)
from m2_dynamic_previous_wsls_bias_v1.figures import (
    generate_m2_dynamic_previous_wsls_bias_diagnostic_figures,
    generate_m2_dynamic_previous_wsls_bias_duration_figures,
)
from m2_dynamic_previous_wsls_bias_v1.identifiability import evaluate_m2_dynamic_previous_wsls_bias_dynamic_surface

from .experiment import M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES


FIGURE_ANALYSIS_SCHEMA_VERSION = "m2_dynamic_previous_wsls_bias_sliding_experiment_analysis_v1"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m2_dynamic_previous_wsls_bias_sliding_global_analysis_v1"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_figures(predictions: pd.DataFrame, output: Path) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_adaptive_psychometrics(predictions, output, model_label="M2_dynamic_previous_wsls_bias")
    files["rare_trials"] = _plot_event_grid(
        predictions, output,
        events=(("rare_long_in_short", "Rare long in short block"),
                ("rare_short_in_long", "Rare short in long block")),
        window=3, stem="rare_trial_performance",
        title="M2_dynamic_previous_wsls_bias performance around rare trials (±3 retained decisions)",
        model_label="M2_dynamic_previous_wsls_bias",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions, output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10, stem="block_transition_performance",
        title="M2_dynamic_previous_wsls_bias performance around block transitions (±10 retained decisions)",
        model_label="M2_dynamic_previous_wsls_bias",
    )
    return files


def _normalized_fit_summary(fit_summary: dict) -> dict:
    normalized = copy.deepcopy(fit_summary)
    for run in normalized.get("runs", []):
        for row in run.get("history", []):
            row.setdefault("epoch", row.get("step"))
            row.setdefault("constrained_parameters_after_step", row.get("constrained_parameters"))
    return normalized


def generate_m2_dynamic_previous_wsls_bias_experiment_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    best_raw_parameters,
    registry,
    m0_style_output_directory: str | Path,
    diagnostic_output_directory: str | Path,
    *,
    include_full_run_length_distributions: bool,
) -> dict:
    """Generate adaptive behavioral, duration, tail, previous-category-bias, and
    native M2_dynamic_previous_wsls_bias outputs."""

    if fit_summary.get("model_id") != "M2_dynamic_previous_wsls_bias_v1":
        raise ValueError("M2_dynamic_previous_wsls_bias experiment outputs require a matching fit summary")
    output = Path(m0_style_output_directory).expanduser().resolve()
    diagnostics = Path(diagnostic_output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    normalized = _normalized_fit_summary(fit_summary)
    files["parameters"] = plot_parameter_history(
        normalized, output, parameter_names=M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES,
        model_label="M2_dynamic_previous_wsls_bias",
    )
    files["nll"] = plot_nll_history(normalized, output)

    hazard = build_m2_dynamic_previous_wsls_bias_hazard_curve(best_raw_parameters, registry)
    hazard.to_csv(diagnostics / "hazard_curve.csv", index=False)
    duration_files = {}
    for state in ("biased", "neutral"):
        path = diagnostics / f"implied_duration_distribution_{state}.csv"
        build_m2_dynamic_previous_wsls_bias_implied_duration_distribution(
            best_raw_parameters, registry, block_state=state
        ).to_csv(path, index=False)
        duration_files[state] = path.name
    duration_summary = summarize_m2_dynamic_previous_wsls_bias_hazard_duration(best_raw_parameters, registry).as_dict()
    _atomic_json(duration_summary, diagnostics / "hazard_duration_summary.json")
    summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics(ready).to_csv(
        diagnostics / "prediction_diagnostics_by_session.csv", index=False
    )
    native_manifest = generate_m2_dynamic_previous_wsls_bias_diagnostic_figures(ready, hazard, diagnostics)
    trajectory_files = generate_m2_dynamic_previous_wsls_bias_duration_figures(
        ready, hazard, diagnostics / "duration_trajectories"
    )
    manifest = {
        "analysis_schema_version": FIGURE_ANALYSIS_SCHEMA_VERSION,
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": True,
        "full_run_length_distributions_included": include_full_run_length_distributions,
        "psychometric_binning": (
            "up to four quantile bins separately for short and long trials; "
            "sparse panels use fewer bins; deterministic N(0,1e-6) jitter, "
            "seed 42; session-balanced; no logistic fit"
        ),
        "hazard_parameterization": "h_B(r)=sigmoid(ah+bh*(min(r,65)-23)); h_N(r)=h_B(r)/2",
        "prototype_policy": "static fitted (delta0), no learning; unchanged from M2_dynamic_block_length_v1",
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t) -> lapse_mixture(m_t)",
        "files": files,
        "native_diagnostics": {
            "directory": str(diagnostics),
            "manifest": native_manifest,
            "duration_distributions": duration_files,
            "duration_summary": duration_summary,
            "duration_trajectory_files": [str(path) for path in trajectory_files],
        },
    }
    _atomic_json(manifest, output / "m2_dynamic_previous_wsls_bias_sliding_figure_manifest.json")
    return manifest


def generate_identifiability_outputs(
    sessions, best_raw_parameters, registry, output_directory: str | Path, *, grid_points: int
) -> dict:
    """Generate the registered fixed-other-eight M2_dynamic_previous_wsls_bias
    dynamic surface (bias-mixing weight vs. hazard slope)."""

    if grid_points < 3:
        raise ValueError("M2_dynamic_previous_wsls_bias identifiability grids require at least three points")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = torch.as_tensor(best_raw_parameters, dtype=torch.float64)
    files = {}
    for pair in (("eta_logit", "bh"),):
        surface = evaluate_m2_dynamic_previous_wsls_bias_dynamic_surface(
            sessions, raw, registry, pair=pair, grid_points=grid_points
        )
        name = f"identifiability_{pair[0]}_{pair[1]}.csv"
        surface.to_csv(output / name, index=False)
        files[f"{pair[0]}_by_{pair[1]}"] = name
    manifest = {
        "model_id": "M2_dynamic_previous_wsls_bias_v1",
        "grid_points_per_axis": grid_points,
        "surface_count": 1,
        "total_grid_evaluations": grid_points * grid_points,
        "profiled": False,
        "files": files,
    }
    _atomic_json(manifest, output / "m2_dynamic_previous_wsls_bias_identifiability_manifest.json")
    return manifest


def generate_truncation_outputs(
    sessions, best_raw_parameters, registry, output_directory: str | Path
) -> dict:
    """Run the opt-in fixed-parameter Rmax 65/75/100 sensitivity check."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = evaluate_m2_dynamic_previous_wsls_bias_truncation_sensitivity(
        sessions, best_raw_parameters, registry, maximum_run_lengths=(65, 75, 100)
    )
    table.to_csv(output / "truncation_sensitivity.csv", index=False)
    manifest = {
        "model_id": "M2_dynamic_previous_wsls_bias_v1",
        "maximum_run_lengths": [65, 75, 100],
        "refit": False,
        "file": "truncation_sensitivity.csv",
    }
    _atomic_json(manifest, output / "m2_dynamic_previous_wsls_bias_truncation_manifest.json")
    return manifest


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    path = Path(output_directory) / "m2_dynamic_previous_wsls_bias_sliding_figure_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("analysis_schema_version") == FIGURE_ANALYSIS_SCHEMA_VERSION


def identifiability_outputs_are_current(output_directory: str | Path, *, grid_points: int) -> bool:
    if grid_points == 0:
        return True
    path = Path(output_directory) / "m2_dynamic_previous_wsls_bias_identifiability_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("grid_points_per_axis") == grid_points and manifest.get("surface_count") == 1


def truncation_outputs_are_current(output_directory: str | Path, *, enabled: bool) -> bool:
    if not enabled:
        return True
    path = Path(output_directory) / "m2_dynamic_previous_wsls_bias_truncation_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("maximum_run_lengths") == [65, 75, 100]


def generate_global_prediction_figures(
    sources: Iterable[tuple[str, str | Path]], output_directory: str | Path, *, source_kind: str
) -> dict:
    """Pool all M2_dynamic_previous_wsls_bias prediction instances and retain
    duration/tail/previous-category-bias diagnostics."""

    source_list = list(sources)
    pooled = pool_prediction_tables(source_list)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "pooled_trial_predictions.csv", index=False)
    analysis = figure_ready(pooled)
    analysis["session_id"] = analysis["analysis_session_id"]
    files = _prediction_figures(analysis, output)
    summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics(analysis).to_csv(
        output / "pooled_prediction_diagnostics.csv", index=False
    )
    manifest = {
        "analysis_schema_version": GLOBAL_ANALYSIS_SCHEMA_VERSION,
        "model_id": "M2_dynamic_previous_wsls_bias_v1",
        "source_kind": source_kind,
        "source_count": len(source_list),
        "source_run_ids": [source_id for source_id, _ in source_list],
        "split_trial_counts": {
            role: int(pooled["split_role"].eq(role).sum()) for role in ("train", "validation", "test")
        },
        "split_prediction_unit_counts": {
            role: int(pooled.loc[pooled["split_role"].eq(role), "analysis_session_id"].nunique())
            for role in ("train", "validation", "test")
        },
        "pooling_policy": (
            "all predictions retained within each split; repeated physical sessions "
            "from different rolling windows are distinct window-session analysis units"
        ),
        "files": {
            "pooled_predictions": ["pooled_trial_predictions.csv"],
            "pooled_duration_tail_diagnostics": ["pooled_prediction_diagnostics.csv"],
            **files,
        },
    }
    _atomic_json(manifest, output / "global_figure_manifest.json")
    return manifest


__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION", "figure_outputs_are_current", "figure_ready",
    "generate_global_prediction_figures", "generate_identifiability_outputs",
    "generate_m2_dynamic_previous_wsls_bias_experiment_outputs", "generate_truncation_outputs",
    "identifiability_outputs_are_current", "truncation_outputs_are_current",
]
