"""Causal static-subjective-prototype constant-hazard HMM evaluation for M2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from ..data.task_moments import TaskMoments
from .bayesian import lapse_mixture
from .m1_bayesian import m1_bayesian_choice_probability
from .m1_session import (
    M1SessionTrials,
    constant_hazard_transition_matrix,
    m1_session_trials_from_frame,
)
from .m2_parameters import M2DecodedParameters, decode_m2_raw_parameters


M2SessionTrials = M1SessionTrials
M2SplitRole = Literal["train", "validation", "test"]
M2ObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]


def m2_session_trials_from_frame(*args, **kwargs) -> M2SessionTrials:
    r"""Build M2's full causal timeline using the unchanged M1 data contract.

    Purpose
    -------
    Retain the complete task timeline, including omissions and feedback masks.

    Mathematical definition
    -----------------------
    .. math::

       \mathcal S_d=\{(s_{d,t},Y_{d,t},C_{d,t},F_{d,t})\}_{t=1}^{T_d}.

    Parameter interpretation
    ------------------------
    Category is post-choice feedback and ``feedback_valid`` controls HMM updates.

    Input units and shapes
    ----------------------
    Accepts the same frame, split role, and optional device as M1's adapter.

    Returned quantities and shapes
    ------------------------------
    Returns one-dimensional float64 tensors and a Boolean feedback mask.

    Constraints
    -----------
    One mouse/session is required and feedback-valid categories must be binary.

    Numerical stability
    -------------------
    Missing choices/categories remain IEEE NaN and are never coerced to zero.

    Causal restrictions
    -------------------
    True block and run length are excluded from the model input object.

    Important edge cases
    --------------------
    A feedback-valid omission can update future belief without a current NLL term.

    Related tests
    -------------
    M1 adapter regression tests and ``test_current_category_only_changes_future_belief``.
    """

    return m1_session_trials_from_frame(*args, **kwargs)


@dataclass(frozen=True)
class M2Predictions:
    """Full-timeline M2 predictions, beliefs, and static prototypes."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    emission_normalizer: torch.Tensor
    pre_choice_block_probability: torch.Tensor
    post_feedback_block_probability: torch.Tensor
    subjective_prototypes: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    parameters: M2DecodedParameters


@dataclass(frozen=True)
class M2SessionEvaluation:
    """Differentiable M2 session choice NLL and state trajectories."""

    mouse_id: str
    session_id: str
    split_role: M2SplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M2Predictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M2MouseObjectiveResult:
    """One pooled M2 objective built from independently reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M2ObjectiveWeighting
    session_evaluations: tuple[M2SessionEvaluation, ...]
    parameters: M2DecodedParameters


def evaluate_m2_session(
    session: M2SessionTrials,
    parameters: M2DecodedParameters,
    moments: TaskMoments,
    *,
    first_long_probability: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M2SessionEvaluation:
    r"""Evaluate M2 in strict predict, feedback-update, transition order.

    Purpose
    -------
    Evaluate one M2 session while rebuilding its differentiable causal HMM state.

    Mathematical definition
    -----------------------
    .. math::

       \pi_t=\xi_t^-\theta^\mathsf T,\qquad
       \xi_t^+(b)\propto\xi_t^-(b)\theta_b^{C_t}(1-\theta_b)^{1-C_t},
       \qquad \xi_{t+1}^-=\xi_t^+T(h_B,h_N),\quad h_N=h_B/2.

    Parameter interpretation
    ------------------------
    ``delta0`` defines one static prototype tensor; ``hazard`` governs the
    unchanged M1 transition matrix. There is no prototype learning state.

    Input units and shapes
    ----------------------
    Session tensors have length ``T`` in float64 and durations use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns scalar session choice NLL, trialwise probabilities, and ``(T,3)``
    pre/post block-belief trajectories.

    Constraints
    -----------
    Beliefs and emission normalizers must remain finite, positive, and normalized.

    Numerical stability
    -------------------
    Only probabilities entering logarithms are clamped; saved probabilities and
    the shared prototype tensor are not clipped.

    Causal restrictions
    -------------------
    The current category is used only after current choice prediction. The same
    ``theta`` tensor constructed once before the loop is used in prior and emission.

    Important edge cases
    --------------------
    Omissions add no Bernoulli term but can update and transition; direct
    ``delta0=0`` gives the functional M0 boundary.

    Related tests
    -------------
    Exact M1/M0 restriction, consistent-prototype, leakage, reset, and gradient tests
    in ``tests/test_m2_model.py``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    transition = constant_hazard_transition_matrix(
        parameters.hazard,
        first_long_probability=first_long_probability,
    )
    theta = parameters.prototypes()
    xi_pre = parameters.w.new_tensor([1.0, 0.0, 0.0])

    priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    lower_boundaries: list[torch.Tensor] = []
    upper_boundaries: list[torch.Tensor] = []
    sensory_sd: list[torch.Tensor] = []
    decision_regimes: list[str] = []
    emission_normalizers: list[torch.Tensor] = []
    pre_states: list[torch.Tensor] = []
    post_states: list[torch.Tensor] = []
    valid_nll: list[torch.Tensor] = []

    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
    )
    for index in range(session.duration_s.numel()):
        if not bool(
            torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0.0)).detach().cpu()
        ) or not bool(
            torch.isclose(
                xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0.0
            )
        ):
            raise FloatingPointError("M2 pre-choice block belief is invalid")
        pre_states.append(xi_pre)
        prior = torch.dot(xi_pre, theta)
        priors.append(prior)

        if bool(valid_mask[index].detach().cpu()):
            result = m1_bayesian_choice_probability(
                session.duration_s[index],
                parameters.common(),
                moments,
                prior,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
            probability = lapse_mixture(
                result.probability_long,
                session.category_long[index],
                parameters.common(),
            )
            safe = torch.clamp(
                probability,
                min=log_probability_epsilon,
                max=1.0 - log_probability_epsilon,
            )
            observed = session.choice_long[index]
            valid_nll.append(
                -(observed * torch.log(safe) + (1.0 - observed) * torch.log1p(-safe))
            )
            probabilities.append(probability)
            non_lapse.append(result.probability_long)
            lower_boundaries.append(result.decision_region.lower_boundary_s)
            upper_boundaries.append(result.decision_region.upper_boundary_s)
            thresholds.append(result.decision_region.upper_boundary_s)
            decision_regimes.append(result.decision_region.regime)
            sensory_sd.append(result.sensory_sd_s)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            probabilities.append(missing)
            non_lapse.append(missing.clone())
            thresholds.append(missing.clone())
            lower_boundaries.append(missing.clone())
            upper_boundaries.append(missing.clone())
            sensory_sd.append(missing.clone())
            decision_regimes.append("not_evaluated_invalid_choice")

        if bool(session.feedback_valid[index].detach().cpu()):
            category = session.category_long[index]
            emission = theta.pow(category) * (1.0 - theta).pow(1.0 - category)
            unnormalized = xi_pre * emission
            normalizer = unnormalized.sum()
            if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
                raise FloatingPointError("M2 feedback update has invalid normalizer")
            emission_normalizers.append(normalizer)
            xi_post = unnormalized / normalizer
        else:
            emission_normalizers.append(session.duration_s.new_tensor(torch.nan))
            xi_post = xi_pre
        if not bool(
            torch.all(torch.isfinite(xi_post) & (xi_post >= 0.0)).detach().cpu()
        ) or not bool(
            torch.isclose(
                xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0.0
            )
        ):
            raise FloatingPointError("M2 post-feedback block belief is invalid")
        post_states.append(xi_post)
        if index + 1 < session.duration_s.numel():
            xi_pre = xi_post @ transition
            xi_pre = xi_pre / xi_pre.sum()

    probability_tensor = torch.stack(probabilities)
    valid_count = int(valid_mask.sum().detach().cpu())
    if valid_nll:
        valid_nll_tensor = torch.stack(valid_nll)
        total_nll = valid_nll_tensor.sum()
        per_trial_nll = torch.full_like(session.duration_s, torch.nan).masked_scatter(
            valid_mask, valid_nll_tensor
        )
    else:
        total_nll = parameters.as_tensor().sum() * 0.0
        per_trial_nll = torch.full_like(session.duration_s, torch.nan)
    return M2SessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=M2Predictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=torch.stack(non_lapse),
            prior_long=torch.stack(priors),
            positive_threshold_s=torch.stack(thresholds),
            decision_lower_boundary_s=torch.stack(lower_boundaries),
            decision_upper_boundary_s=torch.stack(upper_boundaries),
            decision_regime=tuple(decision_regimes),
            sensory_sd_s=torch.stack(sensory_sd),
            emission_normalizer=torch.stack(emission_normalizers),
            pre_choice_block_probability=torch.stack(pre_states),
            post_feedback_block_probability=torch.stack(post_states),
            subjective_prototypes=theta,
            valid_choice_mask=valid_mask,
            feedback_update_mask=session.feedback_valid,
            parameters=parameters,
        ),
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )


def evaluate_m2_mouse_objective(
    sessions: Iterable[M2SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M2ObjectiveWeighting = "pooled_trials",
    required_split_role: M2SplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    first_long_probability: float = 0.5,
) -> M2MouseObjectiveResult:
    r"""Evaluate M2's full-batch choice-only objective across reset sessions.

    Purpose
    -------
    Pool valid choice likelihood over one mouse while resetting state per session.

    Mathematical definition
    -----------------------
    .. math::

       J(\widetilde\Theta)=
       \frac{\sum_d-\sum_{t\in\mathcal V_d}
       [Y_{dt}\log p_{dt}+(1-Y_{dt})\log(1-p_{dt})]}
       {\sum_d N_d}.

    Parameter interpretation
    ------------------------
    All six decoded parameters are stable and pooled; only HMM belief is dynamic.

    Input units and shapes
    ----------------------
    ``raw_parameters`` has shape ``(6,)`` and sessions share one non-test role.

    Returned quantities and shapes
    ------------------------------
    Returns scalar loss, total NLL/count, parameters, and every session evaluation.

    Constraints
    -----------
    Sessions must be nonempty, belong to one mouse, and contain a valid choice.

    Numerical stability
    -------------------
    Float64 session NLLs are summed before one normalization and checked finite.

    Causal restrictions
    -------------------
    Category emissions update future belief only; their likelihood is not fitted.
    Test sessions are rejected.

    Important edge cases
    --------------------
    Pooled weighting permits individual omission-only sessions; equal-session
    weighting does not.

    Related tests
    -------------
    Six-gradient, exact pooled-equivalence, session-reset, and session-order tests
    in ``tests/test_m2_model.py``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M2 session is required")
    if raw_parameters.shape != (7,):
        raise ValueError("M2 raw_parameters must have shape (7,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M2 objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M2 pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M2 objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M2 session must have role {required_split_role!r}")

    parameters = decode_m2_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m2_session(
            session,
            parameters,
            moments,
            first_long_probability=first_long_probability,
            prior_probability_epsilon=prior_probability_epsilon,
            log_probability_epsilon=log_probability_epsilon,
            quadratic_tolerance=quadratic_tolerance,
            discriminant_tolerance=discriminant_tolerance,
        )
        for session in records
    )
    session_nll = torch.stack([item.negative_log_likelihood for item in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(item.valid_trial_count for item in evaluations)
    if total_valid == 0:
        raise ValueError("M2 objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(item.valid_trial_count == 0 for item in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M2 session")
        counts = session_nll.new_tensor(
            [item.valid_trial_count for item in evaluations]
        )
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M2 mouse objective is nonfinite")
    return M2MouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
