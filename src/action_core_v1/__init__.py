"""Shared no-sensory-evidence scalar action-history evaluator core.

Used by both ``pure_action_v1`` (alpha fixed at exactly 1.0) and
``extended_action_v1`` (alpha = sigmoid(a), fitted) so the pi-recurrence,
masking, and lapse mixture can never silently diverge between the two
models. The action-driven sibling of ``wsls_core_v1``; see ``evaluator.py``
for the full mathematical contract.
"""

from .evaluator import (
    ActionLapseParameters,
    ActionPredictions,
    ActionSessionEvaluation,
    evaluate_action_session,
)

__all__ = [
    "ActionLapseParameters",
    "ActionPredictions",
    "ActionSessionEvaluation",
    "evaluate_action_session",
]
