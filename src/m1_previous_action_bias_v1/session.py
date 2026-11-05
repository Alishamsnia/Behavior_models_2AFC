"""Session and pooled-objective evaluation for M1_previous_action_bias_v1.

sensory_actionSingle_lapse: M0's unchanged Bayesian sensory readout (fixed
neutral prior 0.5), mixed before lapse with a previous-action bias state:

    r_t = BayesianReadout(duration_t; w, s0, kappa, prior=0.5)   [M0's readout]
    m_t = (1-eta)*r_t + eta*pi_t
    p_t = lapse_mixture(m_t, category_t; ell_short, ell_long)
    pi_1 = 0.5;  pi_{t+1} = A_t if choice t is valid else pi_t   (alpha == 1)

This is ``M2_dynamic_previous_action_bias_v1`` with the block/run-length
HSMM removed (its block prior replaced by M0's fixed 0.5), and the
fixed-alpha previous-action special case of choice history. It is NOT
``M1_action_kernel_v1`` with alpha=1: feeding pi_t in {0,1} into the
Bayesian prior saturates the readout (see
``sensory_previous_bias_core_v1.evaluator``), so the bias is mixed into the
output instead, exactly as in the M2 bias models.

A thin wrapper: the entire readout/mixture/mask/likelihood computation is
delegated to :func:`sensory_previous_bias_core_v1.evaluate_sensory_previous_bias_session`
with ``history_driver="action"`` (update gate: choice validity alone,
independent of ``feedback_valid`` and of duration validity). The sibling
``m1_previous_wsls_bias_v1`` calls the same core with
``history_driver="category"``. Reuses
``m0_sensory_v1.model.m1_session.M1SessionTrials`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from sensory_previous_bias_core_v1 import (
    SensoryPreviousBiasPredictions,
    evaluate_sensory_previous_bias_session,
)

from .parameters import (
    M1PreviousActionBiasDecodedParameters,
    decode_m1_previous_action_bias_raw_parameters,
)


HISTORY_DRIVER = "action"
M1PreviousActionBiasSplitRole = Literal["train", "validation", "test"]
M1PreviousActionBiasObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]
M1PreviousActionBiasPredictions = SensoryPreviousBiasPredictions

__all__ = [
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1PreviousActionBiasPredictions",
    "M1PreviousActionBiasSessionEvaluation",
    "M1PreviousActionBiasMouseObjectiveResult",
    "evaluate_m1_previous_action_bias_session",
    "evaluate_m1_previous_action_bias_mouse_objective",
]


@dataclass(frozen=True)
class M1PreviousActionBiasSessionEvaluation:
    """Differentiable M1_previous_action_bias session choice NLL and trajectories."""

    mouse_id: str
    session_id: str
    split_role: M1PreviousActionBiasSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M1PreviousActionBiasPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor
    parameters: M1PreviousActionBiasDecodedParameters


@dataclass(frozen=True)
class M1PreviousActionBiasMouseObjectiveResult:
    """One pooled M1_previous_action_bias objective built from independently reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M1PreviousActionBiasObjectiveWeighting
    session_evaluations: tuple[M1PreviousActionBiasSessionEvaluation, ...]
    parameters: M1PreviousActionBiasDecodedParameters


def evaluate_m1_previous_action_bias_session(
    session: M1SessionTrials,
    parameters: M1PreviousActionBiasDecodedParameters,
    moments: TaskMoments,
    *,
    initial_pi: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1PreviousActionBiasSessionEvaluation:
    r"""Evaluate one M1_previous_action_bias session in strict predict/update order.

    Mathematical definition
    -----------------------
    .. math::

       m_t=(1-\eta)r_t+\eta\pi_t,\qquad
       \pi_1=0.5,\qquad
       \pi_{t+1}=\begin{cases}A_t,&V_t\\ \pi_t,&\text{otherwise}\end{cases}

    where :math:`A_t` is the mouse's own choice and :math:`V_t` its validity.
    :math:`\pi_t` is therefore exactly the last valid choice before trial
    :math:`t` (0.5 before any valid choice); a valid choice on an incorrect
    trial still moves it (perseveration, not win-stay/lose-shift).

    Related tests
    -------------
    ``test_matches_last_valid_action_and_mixture``,
    ``test_eta_zero_reproduces_m0_exactly``,
    ``test_no_current_trial_leakage``.
    """

    core = evaluate_sensory_previous_bias_session(
        session,
        parameters.common(),
        parameters.eta,
        moments,
        HISTORY_DRIVER,
        initial_pi=initial_pi,
        prior_probability_epsilon=prior_probability_epsilon,
        log_probability_epsilon=log_probability_epsilon,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    )
    return M1PreviousActionBiasSessionEvaluation(
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


def evaluate_m1_previous_action_bias_mouse_objective(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M1PreviousActionBiasObjectiveWeighting = "pooled_trials",
    required_split_role: M1PreviousActionBiasSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    initial_pi: float = 0.5,
) -> M1PreviousActionBiasMouseObjectiveResult:
    r"""Evaluate M1_previous_action_bias's full-batch pooled choice objective.

    Purpose
    -------
    Decode one six-parameter mouse vector and independently rebuild the
    previous-action bias trajectory for every supplied session before
    returning one loss.

    Mathematical definition
    -----------------------
    The primary loss is

    .. math::

       J=\frac{\sum_d J_d}{\sum_d N_d},

    with the same optional equal-session sensitivity objective as M0/M1.

    Parameter interpretation
    ------------------------
    Stable parameters are pooled; the bias state resets to 0.5 for each
    session.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(6,)`` and session durations use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a scalar loss, total NLL/count, decoded parameters, and all
    session evaluations.

    Constraints
    -----------
    Sessions belong to one mouse and the requested non-test split. Test
    input is always rejected.

    Numerical stability
    -------------------
    Final loss is checked for finiteness in float64.

    Causal restrictions
    -------------------
    The objective includes choice likelihood only; the mouse's own action
    updates the future bias state (used again only as next-trial input) and is
    never added as an auxiliary fitting target beyond that.

    Important edge cases
    --------------------
    Pooled weighting permits an omission-only session when another session
    has valid choices; equal-session weighting does not.

    Related tests
    -------------
    ``test_m1_previous_action_bias_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M1_previous_action_bias session is required")
    if raw_parameters.shape != (6,):
        raise ValueError("M1_previous_action_bias raw_parameters must have shape (6,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M1_previous_action_bias objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M1_previous_action_bias pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M1_previous_action_bias objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M1_previous_action_bias session must have role {required_split_role!r}")

    parameters = decode_m1_previous_action_bias_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m1_previous_action_bias_session(
            session,
            parameters,
            moments,
            initial_pi=initial_pi,
            prior_probability_epsilon=prior_probability_epsilon,
            log_probability_epsilon=log_probability_epsilon,
            quadratic_tolerance=quadratic_tolerance,
            discriminant_tolerance=discriminant_tolerance,
        )
        for session in records
    )
    session_nll = torch.stack([record.negative_log_likelihood for record in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(record.valid_trial_count for record in evaluations)
    if total_valid == 0:
        raise ValueError("M1_previous_action_bias objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M1_previous_action_bias session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M1_previous_action_bias mouse objective is nonfinite")
    return M1PreviousActionBiasMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
