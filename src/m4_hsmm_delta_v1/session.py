"""Causal capped-tail HSMM with M3's unchanged delta learning rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame
from m3_hmm_delta_v1.session import (
    M3ProjectionEvent,
    m3_delta_update_signal,
    m3_project_delta,
)

from .parameters import M4DecodedParameters, decode_m4_raw_parameters
from .prior_state import (
    CappedTailHSMMPriorStateProvider,
    capped_tail_hsmm_transition as _provider_capped_tail_hsmm_transition,
)


M4SessionTrials = M1SessionTrials
M4ProjectionEvent = M3ProjectionEvent
M4SplitRole = Literal["train", "validation", "test"]
M4ObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]


def m4_session_trials_from_frame(*args, **kwargs) -> M4SessionTrials:
    """Use the unchanged complete-timeline session contract from M1--M3."""

    return m1_session_trials_from_frame(*args, **kwargs)


def capped_tail_hsmm_transition(
    joint_posterior: torch.Tensor,
    hazard_by_run_length: torch.Tensor,
    *,
    first_long_probability: float = 0.5,
) -> torch.Tensor:
    """Compatibility entry point for the provider-owned M4 transition.

    A survival at represented run length ``Rmax`` remains in the final tail
    bin. The supplied vector is the biased-block hazard ``h_B(r)``; neutral
    uses the fixed tie ``h_N(r)=h_B(r)/2``. A switch resets run length to one
    and uses exactly M3's destination rule: neutral switches into short/long
    according to the fixed registered first-block probability while short and
    long switch deterministically to long and short, respectively. The sensory
    Weber parameter ``w`` does not enter the latent-state transition.
    """

    return _provider_capped_tail_hsmm_transition(
        joint_posterior,
        hazard_by_run_length,
        first_long_probability=first_long_probability,
    )


@dataclass(frozen=True)
class M4Predictions:
    """Causal choice, learning, block-belief, and duration-belief trajectory."""

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
    pre_choice_run_length_probability: torch.Tensor
    post_feedback_run_length_probability: torch.Tensor
    posterior_block_responsibility: torch.Tensor
    subjective_prototypes_pre: torch.Tensor
    delta_pre: torch.Tensor
    delta_update_signal: torch.Tensor
    delta_unprojected_proposal: torch.Tensor
    delta_post: torch.Tensor
    projection_activated: torch.Tensor
    expected_run_length_pre: torch.Tensor
    expected_run_length_post: torch.Tensor
    map_run_length_pre: torch.Tensor
    map_run_length_post: torch.Tensor
    run_length_entropy_pre: torch.Tensor
    run_length_entropy_post: torch.Tensor
    expected_hazard_pre: torch.Tensor
    expected_hazard_post: torch.Tensor
    hazard_at_map_run_length_pre: torch.Tensor
    total_switch_mass_post: torch.Tensor
    total_survival_mass_post: torch.Tensor
    block_specific_switch_mass_post: torch.Tensor
    tail_probability_pre: torch.Tensor
    tail_probability_post: torch.Tensor
    maximum_reachable_run_length: torch.Tensor
    pre_state_normalization_error: torch.Tensor
    post_state_normalization_error: torch.Tensor
    transition_normalization_error: torch.Tensor
    unreachable_support_mass_pre: torch.Tensor
    unreachable_support_mass_post: torch.Tensor
    hazard_by_run_length: torch.Tensor
    neutral_hazard_by_run_length: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    parameters: M4DecodedParameters


@dataclass(frozen=True)
class M4SessionEvaluation:
    mouse_id: str
    session_id: str
    split_role: M4SplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    feedback_trial_count: int
    timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    projection_activation_count: int
    projection_locations: tuple[int, ...]
    projection_events: tuple[M3ProjectionEvent, ...]
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M4Predictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M4MouseObjectiveResult:
    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    total_feedback_trial_count: int
    total_timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    total_projection_activation_count: int
    projection_fraction: float
    objective_weighting: M4ObjectiveWeighting
    session_evaluations: tuple[M4SessionEvaluation, ...]
    parameters: M4DecodedParameters


def evaluate_m4_session(
    session: M4SessionTrials,
    parameters: M4DecodedParameters,
    moments: TaskMoments,
    *,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    first_long_probability: float = 0.5,
    delta_min: float = 1e-6,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M4SessionEvaluation:
    r"""Evaluate one reset session using feedback before the HSMM transition.

    Purpose
    -------
    Produce choice likelihoods and causal latent trajectories for one session.
    Mathematical definition
    -----------------------
    .. math:: NLL_s=-\sum_{t\in V_s}\log p(y_t).

    The objective contribution is :math:`-\sum_{t\in V}\log p(y_t)` while
    belief follows emission conditioning, M3's delta update, then the HSMM kernel.
    Parameter interpretation
    ------------------------
    ``parameters`` contains M3's seven values plus nonnegative hazard slope ``bh``.
    Input units and shapes
    ----------------------
    Trial vectors have shape ``(T,)``; duration is seconds; the state is ``(3,Rmax)``.
    Returned quantities and shapes
    ------------------------------
    Returns scalar NLL and trialwise tensors whose leading dimension is ``T``.
    Constraints
    -----------
    ``delta`` stays above ``delta_min`` and all probability states normalize.
    Numerical stability
    -------------------
    Float64 checks guard transforms, emission normalizers, support, and probability mass.
    Causal restrictions
    -------------------
    Only pre-choice belief predicts a choice; true labels never enter prediction.
    Important edge cases
    --------------------
    Omission and feedback masks are independent; no transition is computed after trial ``T``.
    Related tests
    -------------
    ``test_exact_bh_zero_reproduces_m3_beyond_cap`` and all-gradient/support tests.
    """

    if run_length_center <= 0:
        raise ValueError("M4 run_length_center must be positive")
    if maximum_run_length <= 1:
        raise ValueError("M4 maximum_run_length must exceed one")
    if delta_min <= 0 or not 0 < log_probability_epsilon < 0.5:
        raise ValueError("Invalid M4 numerical tolerance")

    provider = CappedTailHSMMPriorStateProvider(
        parameters=parameters,
        run_length_center=run_length_center,
        maximum_run_length=maximum_run_length,
        first_long_probability=first_long_probability,
    )
    run_lengths = provider.run_lengths
    biased_hazard = provider.hazard_by_run_length
    state_hazard = provider.state_hazard_by_run_length
    xi_pre = provider.initialize(parameters.w)
    delta = parameters.delta0
    delta_floor = parameters.w.new_tensor(delta_min)

    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "prior", "probability", "non_lapse", "threshold", "lower", "upper",
            "sensory_sd", "normalizer", "pre_block", "post_block", "pre_run",
            "post_run", "responsibility", "theta", "delta_pre", "signal",
            "proposal", "delta_post", "projection", "expected_run", "expected_run_post",
            "map_run", "map_run_post", "entropy", "entropy_post", "expected_hazard",
            "expected_hazard_post", "hazard_at_map", "switch_mass", "survival_mass",
            "block_switch_mass", "tail", "tail_post", "max_reachable", "pre_norm_error",
            "post_norm_error", "transition_norm_error", "unreachable_pre", "unreachable_post",
        )
    }
    regimes: list[str] = []
    valid_nll: list[torch.Tensor] = []
    projection_locations: list[int] = []
    projection_events: list[M3ProjectionEvent] = []
    valid_mask = (
        torch.isfinite(session.duration_s) & (session.duration_s > 0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0) | (session.choice_long == 1))
    )

    for index in range(session.duration_s.numel()):
        if not bool(torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0))):
            raise FloatingPointError("M4 pre-choice joint belief is invalid")
        if not bool(torch.isclose(xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0)):
            raise FloatingPointError("M4 pre-choice joint belief is not normalized")
        if not bool(torch.isfinite(delta) & (delta >= delta_floor)):
            raise FloatingPointError("M4 pre-choice delta is invalid")

        block_pre = provider.block_marginal(xi_pre)
        run_pre = provider.run_length_marginal(xi_pre)
        theta = parameters.prototypes(delta)
        prior = torch.dot(block_pre, theta)
        values["pre_block"].append(block_pre)
        values["pre_run"].append(run_pre)
        values["theta"].append(theta)
        values["delta_pre"].append(delta)
        values["prior"].append(prior)
        values["expected_run"].append(torch.dot(run_pre, run_lengths))
        values["expected_hazard"].append(torch.sum(xi_pre * state_hazard))
        values["tail"].append(run_pre[-1])
        map_index_pre = torch.argmax(run_pre)
        values["map_run"].append(run_lengths[map_index_pre])
        values["entropy"].append(-torch.xlogy(run_pre, run_pre).sum())
        map_run_mass = run_pre[map_index_pre]
        values["hazard_at_map"].append(
            torch.sum(xi_pre[:, map_index_pre] * state_hazard[:, map_index_pre])
            / map_run_mass
        )
        reachable = min(index + 1, maximum_run_length)
        values["max_reachable"].append(run_lengths.new_tensor(float(reachable)))
        pre_norm_error = torch.abs(xi_pre.sum() - xi_pre.new_tensor(1.0))
        values["pre_norm_error"].append(pre_norm_error)
        unreachable_pre = xi_pre[:, reachable:].sum()
        values["unreachable_pre"].append(unreachable_pre)
        if not bool(unreachable_pre <= xi_pre.new_tensor(1e-12)):
            raise FloatingPointError("M4 pre-choice state has unreachable run-length mass")

        if bool(valid_mask[index]):
            result = m1_bayesian_choice_probability(
                session.duration_s[index], parameters.common(), moments, prior,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
            probability = lapse_mixture(
                result.probability_long,
                session.category_long[index],
                parameters.common(),
            )
            safe = torch.clamp(probability, min=log_probability_epsilon, max=1-log_probability_epsilon)
            observed = session.choice_long[index]
            valid_nll.append(-(observed * torch.log(safe) + (1-observed) * torch.log1p(-safe)))
            values["probability"].append(probability)
            values["non_lapse"].append(result.probability_long)
            values["lower"].append(result.decision_region.lower_boundary_s)
            values["upper"].append(result.decision_region.upper_boundary_s)
            values["threshold"].append(result.decision_region.upper_boundary_s)
            values["sensory_sd"].append(result.sensory_sd_s)
            regimes.append(result.decision_region.regime)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            for name in ("probability", "non_lapse", "lower", "upper", "threshold", "sensory_sd"):
                values[name].append(missing.clone())
            regimes.append("not_evaluated_invalid_choice")

        if bool(session.feedback_valid[index]):
            category = session.category_long[index]
            emission = theta.pow(category) * (1-theta).pow(1-category)
            xi_post, normalizer = provider.feedback_update(xi_pre, emission)
            gamma = provider.block_marginal(xi_post)
            signal = m3_delta_update_signal(category, gamma, theta)
            proposal = delta + parameters.alpha_theta * signal
            delta_next, activated = m3_project_delta(proposal, delta_min=delta_min)
            values["normalizer"].append(normalizer)
            values["responsibility"].append(gamma)
            values["signal"].append(signal)
            values["proposal"].append(proposal)
            values["projection"].append(activated)
            if bool(activated.detach().cpu()):
                projection_locations.append(index + 1)
                projection_events.append(M3ProjectionEvent(
                    mouse_id=session.mouse_id, session_id=session.session_id,
                    trial_index=index + 1, category_long=float(category.detach().cpu()),
                    delta_pre=float(delta.detach().cpu()), update_signal=float(signal.detach().cpu()),
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
            gamma = torch.full((3,), torch.nan, dtype=xi_pre.dtype, device=xi_pre.device)
            delta_next = delta
            values["normalizer"].append(missing)
            values["responsibility"].append(gamma)
            values["signal"].append(missing.clone())
            values["proposal"].append(missing.clone())
            values["projection"].append(torch.zeros((), dtype=torch.bool, device=delta.device))

        if not bool(torch.all(torch.isfinite(xi_post) & (xi_post >= 0))):
            raise FloatingPointError("M4 post-feedback joint belief is invalid")
        if not bool(torch.isclose(
            xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0.0
        )):
            raise FloatingPointError("M4 post-feedback joint belief is not normalized")
        values["post_block"].append(xi_post.sum(dim=1))
        run_post = provider.run_length_marginal(xi_post)
        values["post_run"].append(run_post)
        values["delta_post"].append(delta_next)
        values["expected_run_post"].append(torch.dot(run_post, run_lengths))
        map_index_post = torch.argmax(run_post)
        values["map_run_post"].append(run_lengths[map_index_post])
        values["entropy_post"].append(-torch.xlogy(run_post, run_post).sum())
        block_switch_mass = (xi_post * state_hazard).sum(dim=1)
        switch_mass = block_switch_mass.sum()
        values["block_switch_mass"].append(block_switch_mass)
        values["switch_mass"].append(switch_mass)
        values["survival_mass"].append(xi_post.sum() - switch_mass)
        values["expected_hazard_post"].append(switch_mass)
        values["tail_post"].append(run_post[-1])
        post_norm_error = torch.abs(xi_post.sum() - xi_post.new_tensor(1.0))
        values["post_norm_error"].append(post_norm_error)
        unreachable_post = xi_post[:, reachable:].sum()
        values["unreachable_post"].append(unreachable_post)
        if not bool(unreachable_post <= xi_post.new_tensor(1e-12)):
            raise FloatingPointError("M4 post-feedback state has unreachable run-length mass")
        if index + 1 < session.duration_s.numel():
            xi_pre = provider.transition(xi_post)
            values["transition_norm_error"].append(
                torch.abs(xi_pre.sum() - xi_pre.new_tensor(1.0))
            )
            xi_pre = xi_pre / xi_pre.sum()
            delta = delta_next
        else:
            values["transition_norm_error"].append(xi_post.new_tensor(torch.nan))

    valid_count = int(valid_mask.sum().detach().cpu())
    if valid_nll:
        valid_nll_tensor = torch.stack(valid_nll)
        total_nll = valid_nll_tensor.sum()
        per_trial_nll = torch.full_like(session.duration_s, torch.nan).masked_scatter(valid_mask, valid_nll_tensor)
    else:
        total_nll = parameters.as_tensor().sum() * 0
        per_trial_nll = torch.full_like(session.duration_s, torch.nan)
    prediction = M4Predictions(
        probability_long=torch.stack(values["probability"]),
        non_lapse_probability_long=torch.stack(values["non_lapse"]),
        prior_long=torch.stack(values["prior"]),
        positive_threshold_s=torch.stack(values["threshold"]),
        decision_lower_boundary_s=torch.stack(values["lower"]),
        decision_upper_boundary_s=torch.stack(values["upper"]),
        decision_regime=tuple(regimes), sensory_sd_s=torch.stack(values["sensory_sd"]),
        emission_normalizer=torch.stack(values["normalizer"]),
        pre_choice_block_probability=torch.stack(values["pre_block"]),
        post_feedback_block_probability=torch.stack(values["post_block"]),
        pre_choice_run_length_probability=torch.stack(values["pre_run"]),
        post_feedback_run_length_probability=torch.stack(values["post_run"]),
        posterior_block_responsibility=torch.stack(values["responsibility"]),
        subjective_prototypes_pre=torch.stack(values["theta"]),
        delta_pre=torch.stack(values["delta_pre"]),
        delta_update_signal=torch.stack(values["signal"]),
        delta_unprojected_proposal=torch.stack(values["proposal"]),
        delta_post=torch.stack(values["delta_post"]),
        projection_activated=torch.stack(values["projection"]),
        expected_run_length_pre=torch.stack(values["expected_run"]),
        expected_run_length_post=torch.stack(values["expected_run_post"]),
        map_run_length_pre=torch.stack(values["map_run"]),
        map_run_length_post=torch.stack(values["map_run_post"]),
        run_length_entropy_pre=torch.stack(values["entropy"]),
        run_length_entropy_post=torch.stack(values["entropy_post"]),
        expected_hazard_pre=torch.stack(values["expected_hazard"]),
        expected_hazard_post=torch.stack(values["expected_hazard_post"]),
        hazard_at_map_run_length_pre=torch.stack(values["hazard_at_map"]),
        total_switch_mass_post=torch.stack(values["switch_mass"]),
        total_survival_mass_post=torch.stack(values["survival_mass"]),
        block_specific_switch_mass_post=torch.stack(values["block_switch_mass"]),
        tail_probability_pre=torch.stack(values["tail"]),
        tail_probability_post=torch.stack(values["tail_post"]),
        maximum_reachable_run_length=torch.stack(values["max_reachable"]),
        pre_state_normalization_error=torch.stack(values["pre_norm_error"]),
        post_state_normalization_error=torch.stack(values["post_norm_error"]),
        transition_normalization_error=torch.stack(values["transition_norm_error"]),
        unreachable_support_mass_pre=torch.stack(values["unreachable_pre"]),
        unreachable_support_mass_post=torch.stack(values["unreachable_post"]),
        hazard_by_run_length=biased_hazard,
        neutral_hazard_by_run_length=biased_hazard / 2.0,
        valid_choice_mask=valid_mask,
        feedback_update_mask=session.feedback_valid, parameters=parameters,
    )
    tail_values = prediction.tail_probability_pre.detach()
    return M4SessionEvaluation(
        mouse_id=session.mouse_id, session_id=session.session_id, split_role=session.split_role,
        negative_log_likelihood=total_nll, valid_trial_count=valid_count,
        feedback_trial_count=int(session.feedback_valid.sum().detach().cpu()),
        timeline_trial_count=int(session.duration_s.numel()),
        tail_probability_sum=float(tail_values.sum().cpu()),
        mean_tail_probability_pre=float(tail_values.mean().cpu()),
        maximum_tail_probability_pre=float(tail_values.max().cpu()),
        projection_activation_count=len(projection_locations),
        projection_locations=tuple(projection_locations), projection_events=tuple(projection_events),
        per_trial_negative_log_likelihood=per_trial_nll, predictions=prediction,
        duration_s=session.duration_s, choice_long=session.choice_long,
    )


def evaluate_m4_mouse_objective(
    sessions: Iterable[M4SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M4ObjectiveWeighting = "pooled_trials",
    required_split_role: M4SplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    delta_min: float = 1e-6,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    first_long_probability: float = 0.5,
) -> M4MouseObjectiveResult:
    r"""Evaluate the pooled nine-parameter choice-only M4 objective.

    Purpose
    -------
    Aggregate independent reset sessions for one mouse and one non-test split.
    Mathematical definition
    -----------------------
    .. math:: L_{pooled}=\frac{\sum_s NLL_s}{\sum_s n_s}.

    Pooled weighting is :math:`L=\sum_s NLL_s/\sum_s n_s`; equal-session
    weighting is :math:`L=S^{-1}\sum_s NLL_s/n_s`.
    Parameter interpretation
    ------------------------
    Raw coordinates decode to ``w,s0,ell_short,ell_long,kappa,ah,delta0,alpha_theta,bh``.
    Input units and shapes
    ----------------------
    Sessions contain seconds and binary choices; raw parameters have shape ``(9,)``.
    Returned quantities and shapes
    ------------------------------
    Returns scalar loss, counts, tail/projection audits, and session evaluations.
    Constraints
    -----------
    Sessions share mouse and requested role; locked-test sessions are rejected.
    Numerical stability
    -------------------
    The result must be finite and inherits each session's float64 checks.
    Causal restrictions
    -------------------
    No validation value selects checkpoints and test input cannot enter this objective.
    Important edge cases
    --------------------
    Zero valid choices and zero-valid equal-weight sessions are rejected.
    Related tests
    -------------
    ``test_session_reset_and_order_invariant_loss_and_gradient`` and gradient tests.
    """

    records = tuple(sessions)
    if not records or raw_parameters.shape != (9,):
        raise ValueError("M4 requires sessions and raw_parameters with shape (9,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError("Unknown M4 objective weighting")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1 or any(record.split_role == "test" for record in records):
        raise ValueError("M4 objective requires one mouse and excludes locked test sessions")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M4 session must have role {required_split_role!r}")
    parameters = decode_m4_raw_parameters(raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon)
    evaluations = tuple(evaluate_m4_session(
        session, parameters, moments, run_length_center=run_length_center,
        maximum_run_length=maximum_run_length, first_long_probability=first_long_probability,
        delta_min=delta_min, prior_probability_epsilon=prior_probability_epsilon,
        log_probability_epsilon=log_probability_epsilon,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    ) for session in records)
    session_nll = torch.stack([item.negative_log_likelihood for item in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(item.valid_trial_count for item in evaluations)
    if total_valid == 0:
        raise ValueError("M4 objective contains no valid choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(item.valid_trial_count == 0 for item in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M4 session")
        loss = torch.mean(session_nll / session_nll.new_tensor([item.valid_trial_count for item in evaluations]))
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M4 mouse objective is nonfinite")
    feedback_count = sum(item.feedback_trial_count for item in evaluations)
    projection_count = sum(item.projection_activation_count for item in evaluations)
    timeline_count = sum(item.timeline_trial_count for item in evaluations)
    tail_sum = sum(item.tail_probability_sum for item in evaluations)
    return M4MouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)), loss=loss, total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid, total_feedback_trial_count=feedback_count,
        total_projection_activation_count=projection_count,
        total_timeline_trial_count=timeline_count,
        tail_probability_sum=tail_sum,
        mean_tail_probability_pre=tail_sum / timeline_count,
        maximum_tail_probability_pre=max(
            item.maximum_tail_probability_pre for item in evaluations
        ),
        projection_fraction=projection_count / feedback_count if feedback_count else 0.0,
        objective_weighting=objective_weighting, session_evaluations=evaluations,
        parameters=parameters,
    )
