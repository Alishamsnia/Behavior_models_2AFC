"""Predictive-metric evaluation for M2_known_switch_v1.

Mirrors ``m1_known_switch_v1/analysis.py`` exactly (reuses M0's fully
generic, duck-typed ``calculate_evaluation_metrics`` directly).
"""

from __future__ import annotations

from typing import Iterable

import torch

from known_switch_analysis import (
    build_known_switch_trial_predictions,
    save_known_switch_analysis_outputs,
)

from m0_sensory_v1.model.metrics import EvaluationMetrics, calculate_evaluation_metrics

from .parameters import M2_KNOWN_SWITCH_PARAMETER_ORDER, decode_m2_known_switch_raw_parameters
from .registry import M2KnownSwitchRegistry
from .session import (
    M2KnownSwitchSessionTrials,
    evaluate_m2_known_switch_session,
    m2_known_switch_session_trials_from_frame,
)


def evaluate_m2_known_switch_split_metrics(
    sessions: Iterable[M2KnownSwitchSessionTrials],
    raw_parameters: torch.Tensor,
    registry: M2KnownSwitchRegistry,
    *,
    required_split_role: str,
) -> EvaluationMetrics:
    r"""Evaluate M2_known_switch without gradients and calculate Part IX metrics.

    Purpose
    -------
    Identical strategy to ``m1_known_switch_v1``'s split-metrics evaluator.

    Mathematical definition
    -----------------------
    :func:`calculate_evaluation_metrics` computes ELPD, mean NLL, Brier,
    accuracy, calibration, and psychometric summaries.

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
    Every session must have the requested ``split_role``.

    Numerical stability
    -------------------
    Runs under ``torch.no_grad()``.

    Causal restrictions
    -------------------
    Evaluation only; no parameter is fitted here.

    Important edge cases
    --------------------
    An empty ``sessions`` iterable raises inside
    :func:`calculate_evaluation_metrics`, not here.

    Related tests
    -------------
    ``test_m2_known_switch_split_metrics_match_direct_evaluation``.
    """

    parameters = decode_m2_known_switch_raw_parameters(
        raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    with torch.no_grad():
        evaluations = [
            evaluate_m2_known_switch_session(
                session, parameters, registry.task_moments,
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


def build_m2_known_switch_trial_predictions(canonical_trials, raw_parameters, registry, *, allow_test=False):
    return build_known_switch_trial_predictions(
        canonical_trials, raw_parameters, registry,
        parameter_names=M2_KNOWN_SWITCH_PARAMETER_ORDER,
        decode=decode_m2_known_switch_raw_parameters,
        make_session=m2_known_switch_session_trials_from_frame,
        evaluate_session=evaluate_m2_known_switch_session,
        allow_test=allow_test,
    )


def save_m2_known_switch_analysis_outputs(predictions, fit_summary, results_directory, *, test_unlocked):
    return save_known_switch_analysis_outputs(
        predictions, fit_summary, results_directory,
        parameter_names=M2_KNOWN_SWITCH_PARAMETER_ORDER, test_unlocked=test_unlocked,
    )
