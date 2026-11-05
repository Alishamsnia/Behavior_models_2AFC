"""Session and pooled-objective evaluation for Pure_action_v1.

A thin wrapper around the shared recurrence in ``action_core_v1.evaluator``:
this module supplies only the fixed ``alpha=1.0`` and the decoded lapse
rates, then delegates the entire predict/update/mask/likelihood
computation to :func:`action_core_v1.evaluate_action_session`. Reuses
``m0_sensory_v1.model.m1_session.M1SessionTrials``/
``m1_session_trials_from_frame`` unchanged as the session-tensor container
(its ``duration_s`` field is carried through for output/CSV parity but is
never read by the evaluator itself -- see ``action_core_v1`` for why that is
a structural guarantee, not merely an unused field).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from action_core_v1 import ActionPredictions, evaluate_action_session

from .parameters import (
    PureActionDecodedParameters,
    decode_pure_action_raw_parameters,
)


PureActionSplitRole = Literal["train", "validation", "test"]
PureActionObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

__all__ = [
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "PureActionSessionEvaluation",
    "PureActionMouseObjectiveResult",
    "evaluate_pure_action_session",
    "evaluate_pure_action_mouse_objective",
]


@dataclass(frozen=True)
class PureActionSessionEvaluation:
    """Differentiable Pure_action session choice NLL and prior trajectory."""

    mouse_id: str
    session_id: str
    split_role: PureActionSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: ActionPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor
    parameters: PureActionDecodedParameters


@dataclass(frozen=True)
class PureActionMouseObjectiveResult:
    """One pooled Pure_action objective built from independently reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: PureActionObjectiveWeighting
    session_evaluations: tuple[PureActionSessionEvaluation, ...]
    parameters: PureActionDecodedParameters


def evaluate_pure_action_session(
    session: M1SessionTrials,
    parameters: PureActionDecodedParameters,
    moments: TaskMoments,
    *,
    initial_pi: float = 0.5,
    log_probability_epsilon: float = 1e-12,
) -> PureActionSessionEvaluation:
    r"""Evaluate one Pure_action session via the shared no-sensory-evidence core.

    Purpose
    -------
    Delegate the entire predict/update/mask/likelihood computation to
    :func:`action_core_v1.evaluate_action_session` with ``alpha=1.0`` fixed.

    Parameter interpretation
    ------------------------
    ``moments`` is accepted for call-site parity with every sibling
    family's per-session evaluator signature but is unused: this model has
    no sensory readout of any kind.

    Related tests
    -------------
    ``test_pure_action_session_matches_last_valid_action_reference``.
    """

    core = evaluate_action_session(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        duration_s=session.duration_s,
        choice_long=session.choice_long,
        category_long=session.category_long,
        alpha=parameters.alpha,
        ell_short=parameters.ell_short,
        ell_long=parameters.ell_long,
        initial_pi=initial_pi,
        log_probability_epsilon=log_probability_epsilon,
    )
    return PureActionSessionEvaluation(
        mouse_id=core.mouse_id,
        session_id=core.session_id,
        split_role=core.split_role,
        negative_log_likelihood=core.negative_log_likelihood,
        valid_trial_count=core.valid_trial_count,
        per_trial_negative_log_likelihood=core.per_trial_negative_log_likelihood,
        predictions=core.predictions,
        duration_s=core.duration_s,
        choice_long=core.choice_long,
        parameters=parameters,
    )


def evaluate_pure_action_mouse_objective(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: PureActionObjectiveWeighting = "pooled_trials",
    required_split_role: PureActionSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    initial_pi: float = 0.5,
) -> PureActionMouseObjectiveResult:
    r"""Evaluate Pure_action's full-batch pooled choice objective.

    Purpose
    -------
    Decode one two-parameter mouse vector and independently rebuild the
    scalar prior trajectory for every supplied session before returning one
    loss.

    Mathematical definition
    -----------------------
    .. math::

       J=\frac{\sum_s J_s}{\sum_s N_s},

    with the same optional equal-session sensitivity objective as every
    other family in this repository.

    Constraints
    -----------
    Sessions belong to one mouse and the requested non-test split. Test
    input is always rejected.

    Related tests
    -------------
    ``test_pure_action_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one Pure_action session is required")
    if raw_parameters.shape != (2,):
        raise ValueError("Pure_action raw_parameters must have shape (2,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown Pure_action objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("A Pure_action pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter Pure_action objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every Pure_action session must have role {required_split_role!r}")

    parameters = decode_pure_action_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_pure_action_session(
            session, parameters, moments,
            initial_pi=initial_pi,
            log_probability_epsilon=log_probability_epsilon,
        )
        for session in records
    )
    session_nll = torch.stack([record.negative_log_likelihood for record in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(record.valid_trial_count for record in evaluations)
    if total_valid == 0:
        raise ValueError("Pure_action objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid Pure_action session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("Pure_action mouse objective is nonfinite")
    return PureActionMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
