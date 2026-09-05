"""Auditable trial-level exports for the causal static-prototype M2 HMM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .model.m2_parameters import M2_PARAMETER_ORDER, decode_m2_raw_parameters
from .model.m2_session import evaluate_m2_session, m2_session_trials_from_frame


M2_PARAMETER_NAMES = M2_PARAMETER_ORDER


def build_m2_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Rebuild M2 independently per session and export its causal trajectory."""

    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "feedback_valid", "true_block", "outcome", "split_role",
    }
    missing = sorted(required.difference(canonical_trials.columns))
    if missing:
        raise KeyError(f"Canonical M2 analysis table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("M2 test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (7,):
        raise ValueError("M2 raw_parameters must contain exactly seven values")
    parameters = decode_m2_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    constrained = [float(value) for value in parameters.as_tensor().detach().cpu()]
    prototypes = [float(value) for value in parameters.prototypes().detach().cpu()]
    parts: list[pd.DataFrame] = []
    for (_, _), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        role = str(frame["split_role"].iloc[0])
        session = m2_session_trials_from_frame(frame, split_role=role)
        with torch.no_grad():
            evaluation = evaluate_m2_session(
                session,
                parameters,
                registry.task_moments,
                first_long_probability=registry.first_long_probability,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        part["model_probability_long"] = prediction.probability_long.cpu().numpy()
        part["model_non_lapse_probability_long"] = prediction.non_lapse_probability_long.cpu().numpy()
        part["model_prior_long"] = prediction.prior_long.cpu().numpy()
        part["model_decision_lower_boundary_s"] = prediction.decision_lower_boundary_s.cpu().numpy()
        part["model_decision_upper_boundary_s"] = prediction.decision_upper_boundary_s.cpu().numpy()
        part["model_decision_regime"] = prediction.decision_regime
        part["model_sensory_sd_s"] = prediction.sensory_sd_s.cpu().numpy()
        part["model_emission_normalizer"] = prediction.emission_normalizer.cpu().numpy()
        for index, state in enumerate(registry.state_order):
            part[f"model_pre_choice_probability_{state}"] = prediction.pre_choice_block_probability[:, index].cpu().numpy()
            part[f"model_post_feedback_probability_{state}"] = prediction.post_feedback_block_probability[:, index].cpu().numpy()
            part[f"model_subjective_prototype_{state}"] = prototypes[index]
        part["per_trial_negative_log_likelihood"] = evaluation.per_trial_negative_log_likelihood.cpu().numpy()
        part["model_evaluated"] = prediction.valid_choice_mask.cpu().numpy()
        part["model_feedback_updated"] = prediction.feedback_update_mask.cpu().numpy()
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    category = pd.to_numeric(result["category_long"], errors="coerce")
    choice = pd.to_numeric(result["choice_long"], errors="coerce")
    probability = result["model_probability_long"].astype(float)
    result["trial_type"] = np.select(
        [category.eq(0), category.eq(1)], ["short", "long"], default="undefined"
    )
    result["block_type"] = result["true_block"].map(
        {"N": "neutral", "S": "short_block", "L": "long_block"}
    ).fillna("undefined")
    result["mouse_correct"] = np.where(
        choice.isin([0, 1]) & category.isin([0, 1]),
        (choice == category).astype(float),
        np.nan,
    )
    result["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan)
    )
    result["model_choice_long"] = np.where(
        probability.notna(), (probability >= 0.5).astype(float), np.nan
    )
    result["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (result["model_choice_long"] == category).astype(float),
        np.nan,
    )
    result["model_probability_true_block"] = np.nan
    for state in registry.state_order:
        selected = result["true_block"].eq(state)
        result.loc[selected, "model_probability_true_block"] = result.loc[
            selected, f"model_pre_choice_probability_{state}"
        ]
    for name, value in zip(M2_PARAMETER_NAMES, constrained, strict=True):
        result[f"fitted_{name}"] = value
    hazard = float(parameters.hazard.detach().cpu())
    result["fitted_hazard"] = hazard
    result["fitted_biased_hazard"] = hazard
    result["derived_neutral_hazard"] = hazard / 2.0
    result["fitted_subjective_mean_block_duration_trials"] = 1.0 / hazard
    result["fitted_biased_mean_block_duration_trials"] = 1.0 / hazard
    result["derived_neutral_mean_block_duration_trials"] = 2.0 / hazard
    result["fitted_delta0_distance_from_log4"] = abs(
        float(parameters.delta0.detach().cpu()) - np.log(4.0)
    )
    sort_columns = [
        name for name in ("split_role", "session_date", "session_id", "trial_index")
        if name in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def save_m2_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Save M2 predictions, per-session performance, and model inventory."""

    output = Path(results_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "trial_predictions.csv", index=False)
    valid = predictions.loc[predictions["model_evaluated"].astype(bool)]
    performance = valid.groupby(["split_role", "session_id"], observed=False).agg(
        n_valid_decisions=("model_evaluated", "size"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
        model_mean_nll=("per_trial_negative_log_likelihood", "mean"),
        mean_prior_long=("model_prior_long", "mean"),
        mean_probability_true_block=("model_probability_true_block", "mean"),
    ).reset_index()
    performance.to_csv(output / "performance_summary.csv", index=False)
    winner_id = fit_summary.get("winner_start_id")
    winner = next(
        (run for run in fit_summary.get("runs", []) if run["start_id"] == winner_id),
        None,
    )
    fitted = None
    if winner is not None:
        fitted = dict(zip(M2_PARAMETER_NAMES, winner["best_constrained_parameters"]))
        fitted["hazard"] = float(1.0 / (1.0 + np.exp(-fitted["ah"])))
        fitted["subjective_mean_block_duration_trials"] = 1.0 / fitted["hazard"]
        fitted["biased_hazard"] = fitted["hazard"]
        fitted["neutral_hazard"] = fitted["hazard"] / 2.0
        fitted["biased_mean_block_duration_trials"] = 1.0 / fitted["hazard"]
        fitted["neutral_mean_block_duration_trials"] = 2.0 / fitted["hazard"]
        fitted["theta_N"] = 0.5
        fitted["theta_S"] = float(1.0 / (1.0 + np.exp(fitted["delta0"])))
        fitted["theta_L"] = 1.0 - fitted["theta_S"]
    summary = {
        "output_schema_version": "m2_hmm_static_output_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": winner_id,
        "fitted_parameters": fitted,
        "state_order": ["N", "S", "L"],
        "prototype_policy": "static symmetric prototypes used in prior and emissions",
        "state_reset": "xi_pre=(1,0,0) independently at every session",
        "feedback_order": "predict_choice_then_category_update_then_transition",
        "outer_objective": "choice_likelihood_only",
        "test_unlocked": test_unlocked,
        "saved_trial_count": int(len(predictions)),
        "saved_valid_decision_count": int(predictions["model_evaluated"].sum()),
    }
    destination = output / "model_output_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return summary
