"""M0-equivalent behavioral and optimization figures for fitted M1 results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import (
    _atomic_json,
    _plot_event_grid,
    plot_nll_history,
    plot_parameter_history,
    plot_psychometrics,
)
from .m1_analysis import M1_PARAMETER_NAMES


def generate_m1_m0_style_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    output_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Generate every M0 figure family using M1 predictions and fit history."""

    if fit_summary.get("model_id") != "M1_hmm_true_v1":
        raise ValueError("M0-style M1 figures require an M1 fit summary")
    roles = set(predictions["split_role"].astype(str))
    if test_unlocked != ("test" in roles):
        raise ValueError("test_unlocked disagrees with M1 prediction split roles")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_psychometrics(
        predictions, output, model_label="M1"
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
        title="M1 performance around rare trials (±3 retained decisions)",
        model_label="M1",
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
        title="M1 performance around block transitions (±10 retained decisions)",
        model_label="M1",
    )
    files["parameters"] = plot_parameter_history(
        fit_summary,
        output,
        parameter_names=M1_PARAMETER_NAMES,
        model_label="M1",
    )
    files["nll"] = plot_nll_history(fit_summary, output)

    manifest = {
        "analysis_schema_version": "m1_m0_style_analysis_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": test_unlocked,
        "all_data_definition": (
            "train+validation+test"
            if test_unlocked
            else "train+validation; test remains locked"
        ),
        "m0_figure_families_reproduced": [
            "psychometrics_mouse_vs_model",
            "rare_trial_performance",
            "block_transition_performance",
            "parameter_trajectories",
            "training_validation_nll",
        ],
        "model_adaptation": (
            "Behavioral panels use M1 probabilities; parameter trajectories include "
            "M1's fifth fitted coordinate ah."
        ),
        "psychometric_binning": (
            "four quantile bins separately for short and long trials; deterministic "
            "N(0,1e-6) jitter, seed 42; session-balanced"
        ),
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "files": files,
    }
    _atomic_json(manifest, output / "m1_m0_style_figure_manifest.json")
    return manifest
