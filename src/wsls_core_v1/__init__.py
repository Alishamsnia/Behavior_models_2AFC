"""Shared no-sensory-evidence scalar-belief evaluator core.

Used by both ``pure_wsls_v1`` (alpha fixed at exactly 1.0) and
``extended_wsls_v1`` (alpha = sigmoid(a), fitted) so the pi-recurrence,
masking, and lapse mixture can never silently diverge between the two
models. See ``evaluator.py`` for the full mathematical contract.
"""

from .evaluator import (
    WslsLapseParameters,
    WslsPredictions,
    WslsSessionEvaluation,
    evaluate_wsls_session,
)

__all__ = [
    "WslsLapseParameters",
    "WslsPredictions",
    "WslsSessionEvaluation",
    "evaluate_wsls_session",
]
