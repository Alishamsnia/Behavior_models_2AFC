"""Causal constant-hazard HMM with responsibility-weighted delta learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import (
    M1SessionTrials,
    constant_hazard_transition_matrix,
    m1_session_trials_from_frame,
)

from .parameters import M3DecodedParameters, decode_m3_raw_parameters


M3SessionTrials = M1SessionTrials
M3SplitRole = Literal["train", "validation", "test"]
M3ObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]


def m3_session_trials_from_frame(*args, **kwargs) -> M3SessionTrials:
    """Use M2's unchanged full-timeline duration/choice/category contract."""

    return m1_session_trials_from_frame(*args, **kwargs)


def m3_delta_update_signal(
    category_long: torch.Tensor,
    responsibility: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar responsibility-weighted update in delta coordinates.

    Short-block evidence enters with the opposite sign from long-block evidence:
    ``gamma_L * (c-theta_L) - gamma_S * (c-theta_S)``.  The neutral-state
    responsibility has no direct learning term.  All inputs remain in the
    autograd graph. With one-hot responsibility, long-in-L and short-in-S are
    positive; short-in-L and long-in-S are negative. ``category_long`` is a
    scalar and the other inputs have shape ``(3,)``; the output is scalar.
    """

    if responsibility.shape != (3,) or prototypes.shape != (3,):
        raise ValueError("M3 update requires three-state responsibility and prototype vectors")
    return (
        responsibility[2] * (category_long - prototypes[2])
        - responsibility[1] * (category_long - prototypes[1])
    )


def m3_project_delta(
    proposal: torch.Tensor,
    *,
    delta_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M3's registered hard lower projection.

    ``proposal`` is a scalar tensor and the result is ``(projected, active)`` as
    scalar tensors. The operation is exactly ``max(proposal, delta_min)`` with
    registered ``delta_min=1e-6`` in M3. It is deliberately non-smooth at the
    boundary. The caller must record every active projection with its causal
    inputs; this helper never hides an activation.
    """

    if delta_min <= 0.0:
        raise ValueError("delta_min must be positive")
    floor = proposal.new_tensor(delta_min)
    activated = proposal < floor
    return torch.maximum(proposal, floor), activated


@dataclass(frozen=True)
class M3ProjectionEvent:
    """One auditable hard-floor activation, indexed from one within a session."""

    mouse_id: str
    session_id: str
    trial_index: int
    category_long: float
    delta_pre: float
    update_signal: float
    alpha_theta: float
    unprojected_proposal: float
    projected_delta: float
    responsibility_short: float
    responsibility_long: float
    prototype_short: float
    prototype_long: float


@dataclass(frozen=True)
class M3Predictions:
    """Full causal choice, belief, learning, and projection trajectory."""

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
    posterior_block_responsibility: torch.Tensor
    subjective_prototypes_pre: torch.Tensor
    delta_pre: torch.Tensor
    delta_update_signal: torch.Tensor
    delta_unprojected_proposal: torch.Tensor
    delta_post: torch.Tensor
    projection_activated: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    parameters: M3DecodedParameters


@dataclass(frozen=True)
class M3SessionEvaluation:
    """Differentiable choice NLL and all M3 trialwise states for one session."""

    mouse_id: str
    session_id: str
    split_role: M3SplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    feedback_trial_count: int
    projection_activation_count: int
    projection_locations: tuple[int, ...]
    projection_events: tuple[M3ProjectionEvent, ...]
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M3Predictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M3MouseObjectiveResult:
    """One pooled eight-parameter M3 objective across reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    total_feedback_trial_count: int
    total_projection_activation_count: int
    projection_fraction: float
    objective_weighting: M3ObjectiveWeighting
    session_evaluations: tuple[M3SessionEvaluation, ...]
    parameters: M3DecodedParameters


def evaluate_m3_session(
    session: M3SessionTrials,
    parameters: M3DecodedParameters,
    moments: TaskMoments,
    *,
    first_long_probability: float = 0.5,
    delta_min: float = 1e-6,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M3SessionEvaluation:
    """Evaluate one independently reset M3 session in exact causal order.

    State starts at ``xi_pre=(1,0,0)`` and ``delta=delta0``. Each trial (1)
    validates state, (2) constructs pre-update prototypes, (3) constructs the
    pre-choice prior, (4) predicts choice, (5) scores a valid choice, (6)
    receives valid category feedback, (7) computes emissions from the same
    pre-update prototypes, (8) normalizes the post-feedback belief, (9) sets
    responsibilities equal to that belief, (10) computes the update signal,
    (11) applies the projected delta update, and (12) applies the constant-hazard
    transition. Invalid feedback skips steps 7–11 but still transitions.
    Omissions contribute no NLL while valid feedback still updates state.
    Dynamic tensors remain connected across the session and are discarded at
    return; no state can cross a session boundary.
    """

    if delta_min <= 0.0:
        raise ValueError("delta_min must be positive")
    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    transition = constant_hazard_transition_matrix(
        parameters.hazard, first_long_probability=first_long_probability
    )
    xi_pre = parameters.w.new_tensor([1.0, 0.0, 0.0])
    delta = parameters.delta0
    delta_floor = parameters.w.new_tensor(delta_min)

    priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    lower_boundaries: list[torch.Tensor] = []
    upper_boundaries: list[torch.Tensor] = []
    sensory_sd: list[torch.Tensor] = []
    decision_regimes: list[str] = []
    normalizers: list[torch.Tensor] = []
    pre_states: list[torch.Tensor] = []
    post_states: list[torch.Tensor] = []
    responsibilities: list[torch.Tensor] = []
    prototypes_pre: list[torch.Tensor] = []
    delta_pre_values: list[torch.Tensor] = []
    update_signals: list[torch.Tensor] = []
    unprojected_proposals: list[torch.Tensor] = []
    delta_post_values: list[torch.Tensor] = []
    projection_flags: list[torch.Tensor] = []
    valid_nll: list[torch.Tensor] = []
    projection_locations: list[int] = []
    projection_events: list[M3ProjectionEvent] = []

    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0.0) | (session.choice_long == 1.0))
    )
    for index in range(session.duration_s.numel()):
        if not bool(torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0.0))):
            raise FloatingPointError("M3 pre-choice block belief is invalid")
        if not bool(
            torch.isclose(xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0.0)
        ):
            raise FloatingPointError("M3 pre-choice block belief is not normalized")
        if not bool(torch.isfinite(delta) & (delta >= delta_floor)):
            raise FloatingPointError("M3 pre-choice delta is invalid")

        theta = parameters.prototypes(delta)
        pre_states.append(xi_pre)
        prototypes_pre.append(theta)
        delta_pre_values.append(delta)
        prior = torch.dot(xi_pre, theta)
        priors.append(prior)

        if bool(valid_mask[index]):
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

        if bool(session.feedback_valid[index]):
            category = session.category_long[index]
            emission = theta.pow(category) * (1.0 - theta).pow(1.0 - category)
            unnormalized = xi_pre * emission
            normalizer = unnormalized.sum()
            if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
                raise FloatingPointError("M3 feedback update has invalid normalizer")
            xi_post = unnormalized / normalizer
            gamma = xi_post
            signal = m3_delta_update_signal(category, gamma, theta)
            proposal = delta + parameters.alpha_theta * signal
            delta_next, activated = m3_project_delta(proposal, delta_min=delta_min)
            normalizers.append(normalizer)
            responsibilities.append(gamma)
            update_signals.append(signal)
            unprojected_proposals.append(proposal)
            projection_flags.append(activated)
            if bool(activated.detach().cpu()):
                projection_locations.append(index + 1)
                projection_events.append(M3ProjectionEvent(
                    mouse_id=session.mouse_id,
                    session_id=session.session_id,
                    trial_index=index + 1,
                    category_long=float(category.detach().cpu()),
                    delta_pre=float(delta.detach().cpu()),
                    update_signal=float(signal.detach().cpu()),
                    alpha_theta=float(parameters.alpha_theta.detach().cpu()),
                    unprojected_proposal=float(proposal.detach().cpu()),
                    projected_delta=float(delta_next.detach().cpu()),
                    responsibility_short=float(gamma[1].detach().cpu()),
                    responsibility_long=float(gamma[2].detach().cpu()),
                    prototype_short=float(theta[1].detach().cpu()),
                    prototype_long=float(theta[2].detach().cpu()),
                ))
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            xi_post = xi_pre
            gamma = torch.full_like(xi_pre, torch.nan)
            delta_next = delta
            normalizers.append(missing)
            responsibilities.append(gamma)
            update_signals.append(missing.clone())
            unprojected_proposals.append(missing.clone())
            projection_flags.append(torch.zeros((), dtype=torch.bool, device=delta.device))

        if not bool(torch.all(torch.isfinite(xi_post) & (xi_post >= 0.0))):
            raise FloatingPointError("M3 post-feedback block belief is invalid")
        if not bool(
            torch.isclose(xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0.0)
        ):
            raise FloatingPointError("M3 post-feedback block belief is not normalized")
        post_states.append(xi_post)
        delta_post_values.append(delta_next)
        if index + 1 < session.duration_s.numel():
            xi_pre = xi_post @ transition
            xi_pre = xi_pre / xi_pre.sum()
            delta = delta_next

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

    prediction = M3Predictions(
        probability_long=torch.stack(probabilities),
        non_lapse_probability_long=torch.stack(non_lapse),
        prior_long=torch.stack(priors),
        positive_threshold_s=torch.stack(thresholds),
        decision_lower_boundary_s=torch.stack(lower_boundaries),
        decision_upper_boundary_s=torch.stack(upper_boundaries),
        decision_regime=tuple(decision_regimes),
        sensory_sd_s=torch.stack(sensory_sd),
        emission_normalizer=torch.stack(normalizers),
        pre_choice_block_probability=torch.stack(pre_states),
        post_feedback_block_probability=torch.stack(post_states),
        posterior_block_responsibility=torch.stack(responsibilities),
        subjective_prototypes_pre=torch.stack(prototypes_pre),
        delta_pre=torch.stack(delta_pre_values),
        delta_update_signal=torch.stack(update_signals),
        delta_unprojected_proposal=torch.stack(unprojected_proposals),
        delta_post=torch.stack(delta_post_values),
        projection_activated=torch.stack(projection_flags),
        valid_choice_mask=valid_mask,
        feedback_update_mask=session.feedback_valid,
        parameters=parameters,
    )
    return M3SessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        feedback_trial_count=int(session.feedback_valid.sum().detach().cpu()),
        projection_activation_count=len(projection_locations),
        projection_locations=tuple(projection_locations),
        projection_events=tuple(projection_events),
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=prediction,
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )


def evaluate_m3_mouse_objective(
    sessions: Iterable[M3SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M3ObjectiveWeighting = "pooled_trials",
    required_split_role: M3SplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    delta_min: float = 1e-6,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    first_long_probability: float = 0.5,
) -> M3MouseObjectiveResult:
    """Evaluate one full-batch, choice-only objective over reset sessions.

    One decoded eight-vector is shared across the mouse. Each session is rebuilt
    from neutral belief and ``delta0``; NLLs are summed and divided once by the
    total valid choices under pooled weighting. No category likelihood is added,
    no optimizer step occurs here, and locked-test sessions are rejected.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M3 session is required")
    if raw_parameters.shape != (8,):
        raise ValueError("M3 raw_parameters must have shape (8,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M3 objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M3 pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M3 objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M3 session must have role {required_split_role!r}")

    parameters = decode_m3_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m3_session(
            session,
            parameters,
            moments,
            first_long_probability=first_long_probability,
            delta_min=delta_min,
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
        raise ValueError("M3 objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(item.valid_trial_count == 0 for item in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M3 session")
        counts = session_nll.new_tensor([item.valid_trial_count for item in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M3 mouse objective is nonfinite")
    feedback_count = sum(item.feedback_trial_count for item in evaluations)
    projection_count = sum(item.projection_activation_count for item in evaluations)
    return M3MouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        total_feedback_trial_count=feedback_count,
        total_projection_activation_count=projection_count,
        projection_fraction=(projection_count / feedback_count if feedback_count else 0.0),
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
