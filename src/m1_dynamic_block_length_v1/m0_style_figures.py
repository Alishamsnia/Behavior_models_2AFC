"""M0-equivalent behavioral and optimization figures for fitted
M1_dynamic_block_length results."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from m0_sensory_v1.analysis import (
    _atomic_json,
    _plot_event_grid,
    plot_nll_history,
    plot_parameter_history,
    plot_psychometrics,
)

from .parameters import M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER


def _m0_compatible_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add the derived behavioral columns expected by the shared M0 plots."""

    required = {
        "session_id", "trial_index", "split_role", "duration_s", "choice_long",
        "category_long", "true_block", "model_evaluated", "model_probability_long",
    }
    if missing := sorted(required.difference(predictions.columns)):
        raise KeyError(f"M0-style M1_dynamic_block_length predictions are missing columns: {missing}")

    result = predictions.copy()
    category = pd.to_numeric(result["category_long"], errors="coerce")
    choice = pd.to_numeric(result["choice_long"], errors="coerce")
    probability = pd.to_numeric(result["model_probability_long"], errors="coerce")
    if "trial_type" not in result:
        result["trial_type"] = np.select([category.eq(0), category.eq(1)], ["short", "long"], default="undefined")
    if "block_type" not in result:
        result["block_type"] = result["true_block"].map(
            {"N": "neutral", "S": "short_block", "L": "long_block"}
        ).fillna("undefined")
    if "mouse_correct" not in result:
        result["mouse_correct"] = np.where(
            choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan,
        )
    if "model_probability_correct" not in result:
        result["model_probability_correct"] = np.where(
            category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan),
        )
    return result


def _normalized_fit_summary(fit_summary: dict) -> dict:
    """Translate step records to the shared M0 history field names."""

    if fit_summary.get("model_id") != "M1_dynamic_block_length_v1":
        raise ValueError("M0-style M1_dynamic_block_length figures require a matching fit summary")
    normalized = copy.deepcopy(fit_summary)
    if "runs" not in normalized:
        if "start_id" not in normalized or "history" not in normalized:
            raise KeyError("M1_dynamic_block_length fit summary must contain runs or one complete run")
        run = copy.deepcopy(normalized)
        normalized = {
            "model_id": run["model_id"], "mouse_id": run.get("mouse_id"),
            "winner_start_id": run["start_id"], "runs": [run],
        }
    if not normalized.get("runs"):
        raise ValueError("M1_dynamic_block_length fit summary contains no runs")
    for run in normalized["runs"]:
        for row in run.get("history", []):
            row.setdefault("epoch", row.get("step"))
            row.setdefault("constrained_parameters_after_step", row.get("constrained_parameters"))
    return normalized


def generate_m1_dynamic_block_length_m0_style_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    output_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Generate all five established M0 figure families using M1_dynamic_block_length outputs."""

    roles = set(predictions["split_role"].astype(str))
    if test_unlocked != ("test" in roles):
        raise ValueError("test_unlocked disagrees with M1_dynamic_block_length prediction split roles")
    data = _m0_compatible_predictions(predictions)
    normalized_summary = _normalized_fit_summary(fit_summary)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_psychometrics(data, output, model_label="M1_dynamic")
    files["rare_trials"] = _plot_event_grid(
        data, output,
        events=(("rare_long_in_short", "Rare long in short block"), ("rare_short_in_long", "Rare short in long block")),
        window=3, stem="rare_trial_performance",
        title="M1_dynamic_block_length performance around rare trials (±3 retained decisions)",
        model_label="M1_dynamic",
    )
    files["block_transitions"] = _plot_event_grid(
        data, output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10, stem="block_transition_performance",
        title="M1_dynamic_block_length performance around block transitions (±10 retained decisions)",
        model_label="M1_dynamic",
    )
    files["parameters"] = plot_parameter_history(
        normalized_summary, output, parameter_names=M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER,
        model_label="M1_dynamic",
    )
    files["nll"] = plot_nll_history(normalized_summary, output)

    manifest = {
        "analysis_schema_version": "m1_dynamic_block_length_m0_style_analysis_v1",
        "model_id": normalized_summary.get("model_id"),
        "mouse_id": normalized_summary.get("mouse_id"),
        "winner_start_id": normalized_summary.get("winner_start_id"),
        "test_unlocked": test_unlocked,
        "all_data_definition": "train+validation+test" if test_unlocked else "train+validation; test remains locked",
        "m0_figure_families_reproduced": [
            "psychometrics_mouse_vs_model", "rare_trial_performance", "block_transition_performance",
            "parameter_trajectories", "training_validation_nll",
        ],
        "model_adaptation": (
            "Behavioral panels use M1_dynamic_block_length probabilities; parameter trajectories include "
            "w, s0, ell_short, ell_long, kappa, ah, and bh (fixed true-rate prototypes, no fitted "
            "prototype parameter)."
        ),
        "psychometric_binning": (
            "four quantile bins separately for short and long trials; deterministic "
            "N(0,1e-6) jitter, seed 42; session-balanced"
        ),
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "files": files,
    }
    _atomic_json(manifest, output / "m1_dynamic_block_length_m0_style_figure_manifest.json")
    return manifest
