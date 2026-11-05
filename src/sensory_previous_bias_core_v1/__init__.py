"""Shared M0-sensory + previous-trial-bias evaluator core.

Used by both ``m1_previous_action_bias_v1`` (bias state = last valid choice)
and ``m1_previous_wsls_bias_v1`` (bias state = last feedback-valid category)
so the readout, eta-mixing, lapse, and masks can never silently diverge
between the two models. See ``evaluator.py`` for the full contract.
"""

from .evaluator import (
    FIXED_SENSORY_PRIOR_LONG,
    HistoryDriver,
    SensoryPreviousBiasPredictions,
    SensoryPreviousBiasSessionEvaluation,
    evaluate_sensory_previous_bias_session,
    previous_bias_state_trajectory,
)

__all__ = [
    "FIXED_SENSORY_PRIOR_LONG",
    "HistoryDriver",
    "SensoryPreviousBiasPredictions",
    "SensoryPreviousBiasSessionEvaluation",
    "evaluate_sensory_previous_bias_session",
    "previous_bias_state_trajectory",
]
