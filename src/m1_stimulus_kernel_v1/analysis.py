"""Predictive-metric evaluation for M1_stimulus_kernel_v1.

Reuses M0's fully generic, duck-typed ``calculate_evaluation_metrics``
directly (it only accesses ``.predictions.valid_choice_mask``,
``.predictions.probability_long``, ``.duration_s``, ``.choice_long``,
``.per_trial_negative_log_likelihood``, ``.mouse_id``, ``.session_id``,
``.split_role`` -- every one of which this model's own
``M1StimulusKernelSessionEvaluation``/``M1StimulusKernelPredictions``
already expose under the same names), exactly as ``m1_known_switch_v1``'s
``analysis.py`` does. It is NOT reached through the shared
``evaluate_split_metrics``/``prepare_registered_fit_data`` entry points,
which gate on a closed model-ID allowlist that does not include this model
(would need editing those shared, currently-in-use files to extend).
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
    M1_STIMULUS_KERNEL_PARAMETER_ORDER,
    decode_m1_stimulus_kernel_raw_parameters,
)
from .registry import M1StimulusKernelRegistry
from .session import evaluate_m1_stimulus_kernel_session


def evaluate_m1_stimulus_kernel_split_metrics(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    registry: M1StimulusKernelRegistry,
    *,
    required_split_role: str,
) -> EvaluationMetrics:
    r"""Evaluate M1_stimulus_kernel without gradients and calculate Part IX metrics.

    Purpose
    -------
    Join this model's own session evaluator to the frozen metric
    configuration for a training or validation split, exactly mirroring
    what ``evaluate_split_metrics`` does for M0/M1/M2 -- reimplemented
    because that shared function's dispatch does not recognize this model.

    Mathematical definition
    -----------------------
    :func:`calculate_evaluation_metrics` computes ELPD, mean NLL, Brier,
    accuracy, calibration, and psychometric summaries from the per-trial
    predictions this model's evaluator already produces.

    Parameter interpretation
    ------------------------
    ``raw_parameters`` is one restored pooled checkpoint, shape ``(6,)``.

    Input units and shapes
    ----------------------
    Session durations use seconds; all tensors use ``torch.float64``.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`EvaluationMetrics` bundle for the requested split.

    Constraints
    -----------
    Sessions whose ``split_role`` does not equal ``required_split_role`` are
    silently excluded (this lets a caller pass a mixed canonical set and
    select one split); ``required_split_role="test"`` is rejected outright
    by this function, since the per-session evaluator it calls has no
    ``split_role`` gate of its own.

    Numerical stability
    -------------------
    Runs under ``torch.no_grad()``; no gradient state is created.

    Causal restrictions
    -------------------
    Evaluation only; no parameter is fitted here.

    Important edge cases
    --------------------
    An empty ``sessions`` iterable raises inside
    :func:`calculate_evaluation_metrics`, not here.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_split_metrics_match_direct_evaluation``.
    """

    if required_split_role == "test":
        raise ValueError(
            "Locked test sessions cannot enter M1_stimulus_kernel split-metric evaluation"
        )
    parameters = decode_m1_stimulus_kernel_raw_parameters(
        raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    with torch.no_grad():
        evaluations = [
            evaluate_m1_stimulus_kernel_session(
                session, parameters, registry.task_moments,
                initial_pi=registry.initial_pi,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
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


def build_m1_stimulus_kernel_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters,
    registry: M1StimulusKernelRegistry,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    r"""Build a per-trial prediction table for the sliding-window figure/pooling engine.

    Purpose
    -------
    Feed the shared, model-agnostic ``m0_sliding_window_v1`` psychometric and
    pooling helpers, which are duck-typed against a fixed set of derived
    column names. This is a native implementation, not a reuse of
    ``known_switch_analysis.build_known_switch_trial_predictions`` -- that
    shared helper hard-requires oracle-switch-specific prediction fields
    (biased log-odds, in-Neutral flag, per-state N/S/L probabilities) that
    this model does not have, since ``pi_t`` here is a free scalar with no
    discrete hidden state at all.

    Mathematical definition
    -----------------------
    Evaluates :func:`evaluate_m1_stimulus_kernel_session` per physical
    session under ``torch.no_grad()`` and attaches its predictions to the
    corresponding canonical trial rows.

    Parameter interpretation
    ------------------------
    ``block_type`` (derived from the canonical ``true_block`` column) is
    included purely as a diagnostic grouping for the shared psychometric
    figure grid -- this model's own prediction never depends on block
    identity or timing.

    Input units and shapes
    ----------------------
    ``canonical_trials`` must carry ``mouse_id``, ``session_id``,
    ``trial_index``, ``session_date``, ``duration_s``, ``choice_long``,
    ``category_long``, ``feedback_valid``, ``true_block``, ``split_role``.
    ``raw_parameters`` has shape ``(6,)``.

    Returned quantities and shapes
    ------------------------------
    One row per input trial, with added ``model_*``/``fitted_*`` columns.

    Constraints
    -----------
    ``allow_test=False`` (the default) rejects any ``test``-role input row.

    Numerical stability
    -------------------
    Runs entirely under ``torch.no_grad()``.

    Causal restrictions
    -------------------
    Evaluation only; no parameter is fitted here.

    Important edge cases
    --------------------
    A session with zero feedback-valid trials still returns a full row set
    with a constant ``model_prior_long`` of 0.5.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_trial_predictions_match_direct_evaluation``.
    """

    required = {
        "mouse_id", "session_id", "trial_index", "session_date", "duration_s",
        "choice_long", "category_long", "feedback_valid", "true_block", "split_role",
    }
    if missing := sorted(required.difference(canonical_trials.columns)):
        raise KeyError(f"M1_stimulus_kernel analysis table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("M1_stimulus_kernel test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (len(M1_STIMULUS_KERNEL_PARAMETER_ORDER),):
        raise ValueError(
            f"Expected {len(M1_STIMULUS_KERNEL_PARAMETER_ORDER)} M1_stimulus_kernel raw parameters"
        )
    parameters = decode_m1_stimulus_kernel_raw_parameters(
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
            evaluation = evaluate_m1_stimulus_kernel_session(
                session, parameters, registry.task_moments,
                initial_pi=registry.initial_pi,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        direct = {
            "model_probability_long": prediction.probability_long,
            "model_non_lapse_probability_long": prediction.non_lapse_probability_long,
            "model_prior_long": prediction.prior_long,
            "model_post_update_prior_long": prediction.post_update_prior_long,
            "model_decision_lower_boundary_s": prediction.decision_lower_boundary_s,
            "model_decision_upper_boundary_s": prediction.decision_upper_boundary_s,
            "model_sensory_sd_s": prediction.sensory_sd_s,
        }
        for name, values in direct.items():
            part[name] = values.detach().cpu().numpy()
        part["model_decision_regime"] = prediction.decision_regime
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
    for name, value in zip(M1_STIMULUS_KERNEL_PARAMETER_ORDER, constrained, strict=True):
        result[f"fitted_{name}"] = value
    result["fitted_alpha"] = float(parameters.alpha)
    sort_columns = [
        name for name in ("split_role", "session_date", "session_id", "trial_index")
        if name in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def save_m1_stimulus_kernel_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    r"""Write the trial-predictions CSV, a per-session performance table, and a summary.

    Purpose
    -------
    Native counterpart to ``known_switch_analysis.save_known_switch_analysis_outputs``,
    trimmed to the fields this model actually predicts (no block-belief or
    switch-timing diagnostics).

    Returned quantities and shapes
    ------------------------------
    Returns the JSON-serializable summary dict also written to
    ``model_output_summary.json``.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_save_analysis_outputs_writes_expected_files``.
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
        zip(M1_STIMULUS_KERNEL_PARAMETER_ORDER, winner["best_constrained_parameters"], strict=True)
    )
    summary = {
        "output_schema_version": "m1_stimulus_kernel_output_v1",
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
