"""M0-equivalent behavioral and optimization figures for fitted M3 results."""

from __future__ import annotations

from pathlib import Path
import copy

import pandas as pd

from m0_sensory_v1.analysis import (
    _atomic_json,
    _plot_event_grid,
    plot_nll_history,
    plot_parameter_history,
    plot_psychometrics,
)

from .parameters import M3_PARAMETER_ORDER


def generate_m3_m0_style_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    output_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Generate every established M0 figure family using M3 outputs."""

    if fit_summary.get("model_id") != "M3_hmm_delta_v1":
        raise ValueError("M0-style M3 figures require an M3 fit summary")
    roles = set(predictions["split_role"].astype(str))
    if test_unlocked != ("test" in roles):
        raise ValueError("test_unlocked disagrees with M3 prediction split roles")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {}
    normalized_summary = copy.deepcopy(fit_summary)
    for run in normalized_summary.get("runs", []):
        for row in run.get("history", []):
            row.setdefault("epoch", row.get("step"))
            row.setdefault(
                "constrained_parameters_after_step", row.get("constrained_parameters")
            )
    files["psychometrics"] = plot_psychometrics(predictions, output, model_label="M3")
    files["rare_trials"] = _plot_event_grid(
        predictions, output,
        events=(("rare_long_in_short", "Rare long in short block"),
                ("rare_short_in_long", "Rare short in long block")),
        window=3, stem="rare_trial_performance",
        title="M3 performance around rare trials (±3 retained decisions)",
        model_label="M3",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions, output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10, stem="block_transition_performance",
        title="M3 performance around block transitions (±10 retained decisions)",
        model_label="M3",
    )
    files["parameters"] = plot_parameter_history(
        normalized_summary, output, parameter_names=M3_PARAMETER_ORDER, model_label="M3"
    )
    files["nll"] = plot_nll_history(normalized_summary, output)
    manifest = {
        "analysis_schema_version": "m3_m0_style_analysis_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": test_unlocked,
        "all_data_definition": "train+validation+test" if test_unlocked else "train+validation; test remains locked",
        "m0_figure_families_reproduced": [
            "psychometrics_mouse_vs_model", "rare_trial_performance",
            "block_transition_performance", "parameter_trajectories",
            "training_validation_nll",
        ],
        "model_adaptation": "Behavioral panels use M3 probabilities; parameter trajectories include ah, delta0, and alpha_theta.",
        "files": files,
    }
    _atomic_json(manifest, output / "m3_m0_style_figure_manifest.json")
    return manifest
