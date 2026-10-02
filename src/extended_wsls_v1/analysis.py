"""Predictive-metric evaluation for Extended_WSLS_v1.

Reuses M0's fully generic, duck-typed ``calculate_evaluation_metrics``
directly (it only accesses ``.predictions.valid_choice_mask``,
``.predictions.probability_long``, ``.duration_s``, ``.choice_long``,
``.per_trial_negative_log_likelihood``, ``.mouse_id``, ``.session_id``,
``.split_role`` -- every one of which this model's own
``ExtendedWslsSessionEvaluation``/``WslsPredictions`` already expose under the
same names), exactly as ``m1_stimulus_kernel_v1``'s ``analysis.py`` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from m0_sensory_v1.model.metrics import EvaluationMetrics, calculate_evaluation_metrics

from .parameters import (
    EXTENDED_WSLS_PARAMETER_ORDER,
    decode_extended_wsls_raw_parameters,
)
from .registry import ExtendedWslsRegistry
from .session import evaluate_extended_wsls_session


def evaluate_extended_wsls_split_metrics(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    registry: ExtendedWslsRegistry,
    *,
    required_split_role: str,
) -> EvaluationMetrics:
    r"""Evaluate Extended_WSLS without gradients and calculate Part IX metrics.

    Purpose
    -------
    Join this model's own session evaluator to the frozen metric
    configuration for a training or validation split, exactly mirroring
    what ``evaluate_m1_stimulus_kernel_split_metrics`` does.

    Constraints
    -----------
    Sessions whose ``split_role`` does not equal ``required_split_role`` are
    silently excluded; ``required_split_role="test"`` is rejected outright.

    Related tests
    -------------
    ``test_extended_wsls_split_metrics_match_direct_evaluation``.
    """

    if required_split_role == "test":
        raise ValueError(
            "Locked test sessions cannot enter Extended_WSLS split-metric evaluation"
        )
    parameters = decode_extended_wsls_raw_parameters(
        raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    with torch.no_grad():
        evaluations = [
            evaluate_extended_wsls_session(
                session, parameters, registry.task_moments,
                initial_pi=registry.initial_pi,
                log_probability_epsilon=registry.log_probability_epsilon,
            )
            for session in sessions
            if session.split_role == required_split_role
        ]
    return calculate_evaluation_metrics(
        evaluations,
        model_id=registry.model_id,
        calibration_bins=registry.metrics.calibration_bins,
        psychometric_duration_decimals=registry.metrics.psychometric_duration_decimals,
        classification_threshold=registry.metrics.classification_threshold,
    )


def build_extended_wsls_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters,
    registry: ExtendedWslsRegistry,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    r"""Build a per-trial prediction table for the sliding-window figure/pooling engine.

    Purpose
    -------
    Feed the shared, model-agnostic ``m0_sliding_window_v1`` psychometric and
    pooling helpers, which are duck-typed against a fixed set of derived
    column names. No sensory threshold/SD/decision-regime columns are
    emitted -- this model has no sensory readout of any kind.

    Input units and shapes
    ----------------------
    ``canonical_trials`` must carry ``mouse_id``, ``session_id``,
    ``trial_index``, ``session_date``, ``duration_s``, ``choice_long``,
    ``category_long``, ``feedback_valid``, ``true_block``, ``split_role``.
    ``raw_parameters`` has shape ``(2,)``.

    Related tests
    -------------
    ``test_extended_wsls_trial_predictions_match_direct_evaluation``.
    """

    required = {
        "mouse_id", "session_id", "trial_index", "session_date", "duration_s",
        "choice_long", "category_long", "feedback_valid", "true_block", "split_role",
    }
    if missing := sorted(required.difference(canonical_trials.columns)):
        raise KeyError(f"Extended_WSLS analysis table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("Extended_WSLS test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (len(EXTENDED_WSLS_PARAMETER_ORDER),):
        raise ValueError(
            f"Expected {len(EXTENDED_WSLS_PARAMETER_ORDER)} Extended_WSLS raw parameters"
        )
    parameters = decode_extended_wsls_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    constrained = [float(value) for value in parameters.as_tensor().detach().cpu()]

    parts: list[pd.DataFrame] = []
    for (_, _), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        role = str(frame["split_role"].iloc[0])
        session = m1_session_trials_from_frame(frame, split_role=role)
        with torch.no_grad():
            evaluation = evaluate_extended_wsls_session(
                session, parameters, registry.task_moments,
                initial_pi=registry.initial_pi,
                log_probability_epsilon=registry.log_probability_epsilon,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        direct = {
            "model_probability_long": prediction.probability_long,
            "model_non_lapse_probability_long": prediction.non_lapse_probability_long,
            "model_prior_long": prediction.prior_long,
            "model_post_update_prior_long": prediction.post_update_prior_long,
        }
        for name, values in direct.items():
            part[name] = values.detach().cpu().numpy()
        part["per_trial_negative_log_likelihood"] = (
            evaluation.per_trial_negative_log_likelihood.detach().cpu().numpy()
        )
        part["model_evaluated"] = prediction.valid_choice_mask.detach().cpu().numpy()
        part["model_feedback_updated"] = prediction.feedback_update_mask.detach().cpu().numpy()
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
        choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan
    )
    result["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan)
    )
    result["model_choice_long"] = np.where(
        probability.notna(), (probability >= 0.5).astype(float), np.nan
    )
    result["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (result["model_choice_long"] == category).astype(float), np.nan,
    )
    for name, value in zip(EXTENDED_WSLS_PARAMETER_ORDER, constrained, strict=True):
        result[f"fitted_{name}"] = value
    result["fitted_alpha"] = float(parameters.alpha)
    sort_columns = [
        name for name in ("split_role", "session_date", "session_id", "trial_index")
        if name in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def save_extended_wsls_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    r"""Write the trial-predictions CSV, a per-session performance table, and a summary.

    Related tests
    -------------
    ``test_extended_wsls_save_analysis_outputs_writes_expected_files``.
    """

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
    ).reset_index()
    performance.to_csv(output / "performance_summary.csv", index=False)
    winner_id = fit_summary.get("winner_start_id")
    winner = next(
        (run for run in fit_summary.get("runs", []) if run.get("start_id") == winner_id), None
    )
    fitted = None if winner is None else dict(
        zip(EXTENDED_WSLS_PARAMETER_ORDER, winner["best_constrained_parameters"], strict=True)
    )
    summary = {
        "output_schema_version": "extended_wsls_output_v1",
        "model_id": fit_summary.get("model_id"), "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": winner_id, "fitted_parameters": fitted,
        "prior_update_rule": "pi_next = pi + sigmoid(a) * (category - pi); pi_1 = 0.5",
        "session_state_reset": "pi resets to 0.5 independently at every physical session start",
        "test_unlocked": test_unlocked, "saved_trial_count": int(len(predictions)),
        "saved_valid_decision_count": int(predictions["model_evaluated"].sum()),
    }
    destination = output / "model_output_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return summary
