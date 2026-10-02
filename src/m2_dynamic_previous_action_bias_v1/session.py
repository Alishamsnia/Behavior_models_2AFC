"""Causal static-prototype duration-dependent-hazard HSMM evaluation, with a
previous-action bias mixed in before lapse, for
M2_dynamic_previous_action_bias_v1.

Independent, freshly-authored per-trial loop -- this module does not call
into ``m2_dynamic_block_length_v1.session``,
``m2_dynamic_action_kernel_bias_v1.session``, or ``m1_action_kernel_v1.session``
at runtime. It reuses only the same genuinely shared, pre-existing
primitives ``M2_dynamic_block_length_v1`` itself reuses
(``m1_bayesian_choice_probability``, ``lapse_mixture``, the full
``capped_tail_hsmm`` module) plus M2_dynamic_block_length_v1's own parameter
decoder (this model's true parent).

Two independent pieces of causal state are carried per session: the joint
``(3,Rmax)`` block/run-length belief ``xi`` (driven only by the true
category, exactly as in M2_dynamic_block_length_v1 -- completely unchanged)
and a scalar action-history state ``pi_t`` (driven only by the mouse's own
previous action). Unlike the sibling
``M2_dynamic_action_kernel_bias_v1``, the action-history learning rate is
fixed to the registry constant ``alpha=1.0`` -- the exact one-step
"previous action" special case of the shared delta rule, at which
``pi_{t+1}`` collapses to ``A_t`` exactly. Neither piece of state ever
touches the other's update. They combine at exactly one point per trial:

    r_t   = BayesianReadout(duration_t, w, s0, prior_block,t)   [unchanged solver]
    m_t   = (1-eta)*r_t + eta*pi_t                                [NEW]
    p_t   = lapse_mixture(m_t, category_t, ell_short, ell_long)   [unchanged function,
                                                                     fed m_t instead of r_t]

Lapse is therefore the outermost operation, matching every other model in
this family's lapse semantics. ``ell_short``/``ell_long`` are completely
unchanged; they still see only the true current category.

The action-history recursion and its two-mask discipline are reused as a
pattern from M1_action_kernel_v1 (``valid_mask`` gates the likelihood,
``action_valid_mask`` -- choice validity alone, independent of duration
validity and of ``feedback_valid`` -- gates the ``pi_t`` update), not as a
runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.capped_tail_hsmm import (
    block_marginal,
    capped_tail_hsmm_transition,
    feedback_update,
    initialize_capped_tail_state,
    run_length_marginal,
)
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame

from .parameters import M2DynamicPreviousActionBiasParameters


M2DynamicPreviousActionBiasSessionTrials = M1SessionTrials
M2DynamicPreviousActionBiasSplitRole = Literal["train", "validation", "test"]
M2DynamicPreviousActionBiasObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

__all__ = [
    "M2DynamicPreviousActionBiasSessionTrials",
    "m2_dynamic_previous_action_bias_session_trials_from_frame",
    "M2DynamicPreviousActionBiasPredictions",
    "M2DynamicPreviousActionBiasSessionEvaluation",
    "M2DynamicPreviousActionBiasMouseObjectiveResult",
    "evaluate_m2_dynamic_previous_action_bias_session",
    "evaluate_m2_dynamic_previous_action_bias_mouse_objective",
]


def m2_dynamic_previous_action_bias_session_trials_from_frame(*args, **kwargs) -> M2DynamicPreviousActionBiasSessionTrials:
    """Use the unchanged complete-timeline session contract shared M1-M4 all reuse."""

    return m1_session_trials_from_frame(*args, **kwargs)


@dataclass(frozen=True)
class M2DynamicPreviousActionBiasPredictions:
    """Causal choice, block/run-length-belief, action-history, and
    duration-diagnostic trajectory."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    bias_mixed_probability_long: torch.Tensor
    non_bias_probability_long: torch.Tensor
    prior_long: torch.Tensor
    action_history_state: torch.Tensor
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
    subjective_prototypes: torch.Tensor
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
    action_valid_mask: torch.Tensor
    feedback_update_mask: torch.Tensor
    parameters: M2DynamicPreviousActionBiasParameters


@dataclass(frozen=True)
class M2DynamicPreviousActionBiasSessionEvaluation:
    mouse_id: str
    session_id: str
    split_role: M2DynamicPreviousActionBiasSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    feedback_trial_count: int
    timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M2DynamicPreviousActionBiasPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M2DynamicPreviousActionBiasMouseObjectiveResult:
    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    total_feedback_trial_count: int
    total_timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    objective_weighting: M2DynamicPreviousActionBiasObjectiveWeighting
    session_evaluations: tuple[M2DynamicPreviousActionBiasSessionEvaluation, ...]
    parameters: M2DynamicPreviousActionBiasParameters


def evaluate_m2_dynamic_previous_action_bias_session(
    session: M2DynamicPreviousActionBiasSessionTrials,
    parameters: M2DynamicPreviousActionBiasParameters,
    moments: TaskMoments,
    *,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    first_long_probability: float = 0.5,
    initial_pi: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M2DynamicPreviousActionBiasSessionEvaluation:
    r"""Evaluate one M2_dynamic_previous_action_bias session: static
    prototypes and HSMM transition unchanged from M2_dynamic_block_length_v1,
    with the mouse's last valid action mixed in between the sensory readout
    and lapse.

    Purpose
    -------
    Produce choice likelihoods and causal joint block/run-length
    trajectories exactly as M2_dynamic_block_length_v1 does, plus a
    previous-action trajectory, combined only at the final probability.

    Mathematical definition
    -----------------------
    .. math::

       r_t=\text{BayesianReadout}(\cdot),\quad
       \pi_{t+1}=A_t\ (\alpha\equiv1),\quad
       m_t=(1-\eta)r_t+\eta\pi_t,\quad
       p_t=\text{lapse\_mixture}(m_t,\cdot),\quad
       NLL_s=-\sum_{t\in V_s}\log p(y_t).

    Parameter interpretation
    ------------------------
    ``parameters`` contains M2_dynamic_block_length_v1's eight values plus
    the bias-mixing-weight logit ``eta_logit``; ``alpha`` is the fixed
    constant ``1.0``, not fitted. ``theta`` is computed once from
    ``delta0`` and never updated, exactly as in M2_dynamic_block_length_v1.

    Input units and shapes
    ----------------------
    Trial vectors have shape ``(T,)``; duration is seconds; the joint state
    is ``(3,Rmax)``; ``pi_t`` is a scalar per trial.

    Returned quantities and shapes
    ------------------------------
    Returns scalar NLL and trialwise tensors whose leading dimension is
    ``T``, including both the pure block-inference prior and the pure
    non-lapse readout (unchanged meaning) alongside the new bias-mixed and
    final probabilities.

    Constraints
    -----------
    All probability states normalize; ``theta`` stays fixed for the
    session; ``pi_t`` remains in ``[0,1]`` by construction.

    Numerical stability
    -------------------
    Float64 checks guard transforms, emission normalizers, support, and
    probability mass, exactly as in M2_dynamic_block_length_v1; ``pi_t`` is
    checked finite every trial exactly as in M1_action_kernel_v1.

    Causal restrictions
    -------------------
    Only pre-choice belief and pre-update ``pi_t`` predict a choice; true
    labels and the mouse's own current-trial action never enter prediction.

    Important edge cases
    --------------------
    Omission, feedback, and action-validity masks are independent; no
    transition is computed after trial ``T``; an invalid/omitted choice
    leaves ``pi_t`` unchanged (feedback-independent, exactly as in
    M1_action_kernel_v1) -- so ``pi_t`` is the mouse's *last valid* action,
    not necessarily trial ``t-1``'s literal action.

    Related tests
    -------------
    ``test_eta_zero_reproduces_m2_dynamic_block_length`` and
    ``test_m2_dynamic_action_kernel_bias_at_alpha_one_equals_previous_action_bias``.
    """

    if run_length_center <= 0:
        raise ValueError("run_length_center must be positive")
    if maximum_run_length <= 1:
        raise ValueError("maximum_run_length must exceed one")
    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")

    run_lengths = torch.arange(
        1, maximum_run_length + 1, dtype=parameters.w.dtype, device=parameters.w.device
    )
    biased_hazard = parameters.hazard(run_lengths, center=run_length_center)
    state_hazard = torch.stack((biased_hazard / 2.0, biased_hazard, biased_hazard), dim=0)
    xi_pre = initialize_capped_tail_state(parameters.w, maximum_run_length)
    theta = parameters.prototypes()
    alpha = parameters.alpha
    eta = parameters.eta
    pi_current = parameters.w.new_tensor(initial_pi)

    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "prior", "probability", "non_lapse", "bias_mixed", "non_bias", "action_history",
            "threshold", "lower", "upper", "sensory_sd", "normalizer", "pre_block", "post_block",
            "pre_run", "post_run", "expected_run", "expected_run_post", "map_run", "map_run_post",
            "entropy", "entropy_post", "expected_hazard", "expected_hazard_post", "hazard_at_map",
            "switch_mass", "survival_mass", "block_switch_mass", "tail", "tail_post", "max_reachable",
            "pre_norm_error", "post_norm_error", "transition_norm_error", "unreachable_pre",
            "unreachable_post",
        )
    }
    regimes: list[str] = []
    valid_nll: list[torch.Tensor] = []
    valid_mask = (
        torch.isfinite(session.duration_s) & (session.duration_s > 0)
        & torch.isfinite(session.choice_long)
        & ((session.choice_long == 0) | (session.choice_long == 1))
    )
    action_valid_mask = torch.isfinite(session.choice_long) & (
        (session.choice_long == 0.0) | (session.choice_long == 1.0)
    )

    for index in range(session.duration_s.numel()):
        if not bool(torch.isfinite(pi_current).detach().cpu()):
            raise FloatingPointError("M2_dynamic_previous_action_bias action-history state is non-finite")
        if not bool(torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0))):
            raise FloatingPointError("M2_dynamic_previous_action_bias pre-choice joint belief is invalid")
        if not bool(torch.isclose(xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0)):
            raise FloatingPointError("M2_dynamic_previous_action_bias pre-choice joint belief is not normalized")

        block_pre = block_marginal(xi_pre)
        run_pre = run_length_marginal(xi_pre)
        prior = torch.dot(block_pre, theta)
        pi_before = pi_current
        values["pre_block"].append(block_pre)
        values["pre_run"].append(run_pre)
        values["prior"].append(prior)
        values["action_history"].append(pi_before)
        values["expected_run"].append(torch.dot(run_pre, run_lengths))
        values["expected_hazard"].append(torch.sum(xi_pre * state_hazard))
        values["tail"].append(run_pre[-1])
        map_index_pre = torch.argmax(run_pre)
        values["map_run"].append(run_lengths[map_index_pre])
        values["entropy"].append(-torch.xlogy(run_pre, run_pre).sum())
        map_run_mass = run_pre[map_index_pre]
        values["hazard_at_map"].append(
            torch.sum(xi_pre[:, map_index_pre] * state_hazard[:, map_index_pre]) / map_run_mass
        )
        reachable = min(index + 1, maximum_run_length)
        values["max_reachable"].append(run_lengths.new_tensor(float(reachable)))
        values["pre_norm_error"].append(torch.abs(xi_pre.sum() - xi_pre.new_tensor(1.0)))
        unreachable_pre = xi_pre[:, reachable:].sum()
        values["unreachable_pre"].append(unreachable_pre)
        if not bool(unreachable_pre <= xi_pre.new_tensor(1e-12)):
            raise FloatingPointError("M2_dynamic_previous_action_bias pre-choice state has unreachable run-length mass")

        if bool(valid_mask[index]):
            result = m1_bayesian_choice_probability(
                session.duration_s[index], parameters.common(), moments, prior,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
            r_t = result.probability_long
            m_t = (1.0 - eta) * r_t + eta * pi_before
            probability = lapse_mixture(m_t, session.category_long[index], parameters.common())
            non_bias = lapse_mixture(r_t, session.category_long[index], parameters.common())
            safe = torch.clamp(probability, min=log_probability_epsilon, max=1 - log_probability_epsilon)
            observed = session.choice_long[index]
            valid_nll.append(-(observed * torch.log(safe) + (1 - observed) * torch.log1p(-safe)))
            values["probability"].append(probability)
            values["non_lapse"].append(r_t)
            values["bias_mixed"].append(m_t)
            values["non_bias"].append(non_bias)
            values["lower"].append(result.decision_region.lower_boundary_s)
            values["upper"].append(result.decision_region.upper_boundary_s)
            values["threshold"].append(result.decision_region.upper_boundary_s)
            values["sensory_sd"].append(result.sensory_sd_s)
            regimes.append(result.decision_region.regime)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            for name in ("probability", "non_lapse", "bias_mixed", "non_bias", "lower", "upper", "threshold", "sensory_sd"):
                values[name].append(missing.clone())
            regimes.append("not_evaluated_invalid_choice")

        if bool(action_valid_mask[index]):
            action = session.choice_long[index]
            pi_next = pi_before + alpha * (action - pi_before)
        else:
            pi_next = pi_before
        pi_current = pi_next

        if bool(session.feedback_valid[index]):
            category = session.category_long[index]
            emission = theta.pow(category) * (1 - theta).pow(1 - category)
            xi_post, normalizer = feedback_update(xi_pre, emission)
            values["normalizer"].append(normalizer)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            xi_post = xi_pre
            values["normalizer"].append(missing)

        if not bool(torch.all(torch.isfinite(xi_post) & (xi_post >= 0))):
            raise FloatingPointError("M2_dynamic_previous_action_bias post-feedback joint belief is invalid")
        if not bool(torch.isclose(xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0)):
            raise FloatingPointError("M2_dynamic_previous_action_bias post-feedback joint belief is not normalized")
        values["post_block"].append(block_marginal(xi_post))
        run_post = run_length_marginal(xi_post)
        values["post_run"].append(run_post)
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
        values["post_norm_error"].append(torch.abs(xi_post.sum() - xi_post.new_tensor(1.0)))
        unreachable_post = xi_post[:, reachable:].sum()
        values["unreachable_post"].append(unreachable_post)
        if not bool(unreachable_post <= xi_post.new_tensor(1e-12)):
            raise FloatingPointError("M2_dynamic_previous_action_bias post-feedback state has unreachable run-length mass")

        if index + 1 < session.duration_s.numel():
            xi_pre = capped_tail_hsmm_transition(
                xi_post, biased_hazard, first_long_probability=first_long_probability
            )
            values["transition_norm_error"].append(torch.abs(xi_pre.sum() - xi_pre.new_tensor(1.0)))
            xi_pre = xi_pre / xi_pre.sum()
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

    prediction = M2DynamicPreviousActionBiasPredictions(
        probability_long=torch.stack(values["probability"]),
        non_lapse_probability_long=torch.stack(values["non_lapse"]),
        bias_mixed_probability_long=torch.stack(values["bias_mixed"]),
        non_bias_probability_long=torch.stack(values["non_bias"]),
        prior_long=torch.stack(values["prior"]),
        action_history_state=torch.stack(values["action_history"]),
        positive_threshold_s=torch.stack(values["threshold"]),
        decision_lower_boundary_s=torch.stack(values["lower"]),
        decision_upper_boundary_s=torch.stack(values["upper"]),
        decision_regime=tuple(regimes), sensory_sd_s=torch.stack(values["sensory_sd"]),
        emission_normalizer=torch.stack(values["normalizer"]),
        pre_choice_block_probability=torch.stack(values["pre_block"]),
        post_feedback_block_probability=torch.stack(values["post_block"]),
        pre_choice_run_length_probability=torch.stack(values["pre_run"]),
        post_feedback_run_length_probability=torch.stack(values["post_run"]),
        subjective_prototypes=theta,
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
        action_valid_mask=action_valid_mask,
        feedback_update_mask=session.feedback_valid, parameters=parameters,
    )
    tail_values = prediction.tail_probability_pre.detach()
    return M2DynamicPreviousActionBiasSessionEvaluation(
        mouse_id=session.mouse_id, session_id=session.session_id, split_role=session.split_role,
        negative_log_likelihood=total_nll, valid_trial_count=valid_count,
        feedback_trial_count=int(session.feedback_valid.sum().detach().cpu()),
        timeline_trial_count=int(session.duration_s.numel()),
        tail_probability_sum=float(tail_values.sum().cpu()),
        mean_tail_probability_pre=float(tail_values.mean().cpu()),
        maximum_tail_probability_pre=float(tail_values.max().cpu()),
        per_trial_negative_log_likelihood=per_trial_nll, predictions=prediction,
        duration_s=session.duration_s, choice_long=session.choice_long,
    )


def evaluate_m2_dynamic_previous_action_bias_mouse_objective(
    sessions: Iterable[M2DynamicPreviousActionBiasSessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M2DynamicPreviousActionBiasObjectiveWeighting = "pooled_trials",
    required_split_role: M2DynamicPreviousActionBiasSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    first_long_probability: float = 0.5,
    initial_pi: float = 0.5,
) -> M2DynamicPreviousActionBiasMouseObjectiveResult:
    r"""Evaluate the pooled nine-parameter choice-only
    M2_dynamic_previous_action_bias objective.

    Purpose
    -------
    Aggregate independent reset sessions for one mouse and one non-test
    split, rebuilding both the joint belief and the scalar action-history
    state from scratch for each session.

    Mathematical definition
    -----------------------
    .. math:: L_{pooled}=\frac{\sum_s NLL_s}{\sum_s n_s}.

    Parameter interpretation
    ------------------------
    Raw coordinates decode to
    ``w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit``.

    Input units and shapes
    ----------------------
    Sessions contain seconds and binary choices; raw parameters have shape ``(9,)``.

    Returned quantities and shapes
    ------------------------------
    Returns scalar loss, counts, tail audits, and session evaluations.

    Constraints
    -----------
    Sessions share mouse and requested role; locked-test sessions are rejected.

    Numerical stability
    -------------------
    The result must be finite and inherits each session's float64 checks.

    Causal restrictions
    -------------------
    No validation value selects checkpoints and test input cannot enter this
    objective.

    Important edge cases
    --------------------
    Zero valid choices and zero-valid equal-weight sessions are rejected.

    Related tests
    -------------
    ``test_session_reset_and_order_invariant_loss_and_gradient``.
    """

    from .parameters import decode_m2_dynamic_previous_action_bias_raw_parameters

    records = tuple(sessions)
    if not records or raw_parameters.shape != (9,):
        raise ValueError("M2_dynamic_previous_action_bias requires sessions and raw_parameters with shape (9,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError("Unknown M2_dynamic_previous_action_bias objective weighting")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1 or any(record.split_role == "test" for record in records):
        raise ValueError("M2_dynamic_previous_action_bias objective requires one mouse and excludes locked test sessions")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M2_dynamic_previous_action_bias session must have role {required_split_role!r}")
    parameters = decode_m2_dynamic_previous_action_bias_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(evaluate_m2_dynamic_previous_action_bias_session(
        session, parameters, moments, run_length_center=run_length_center,
        maximum_run_length=maximum_run_length, first_long_probability=first_long_probability,
        initial_pi=initial_pi,
        prior_probability_epsilon=prior_probability_epsilon,
        log_probability_epsilon=log_probability_epsilon,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    ) for session in records)
    session_nll = torch.stack([item.negative_log_likelihood for item in evaluations])
    total_nll = session_nll.sum()
    total_valid = sum(item.valid_trial_count for item in evaluations)
    if total_valid == 0:
        raise ValueError("M2_dynamic_previous_action_bias objective contains no valid choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(item.valid_trial_count == 0 for item in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M2_dynamic_previous_action_bias session")
        loss = torch.mean(session_nll / session_nll.new_tensor([item.valid_trial_count for item in evaluations]))
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M2_dynamic_previous_action_bias mouse objective is nonfinite")
    feedback_count = sum(item.feedback_trial_count for item in evaluations)
    timeline_count = sum(item.timeline_trial_count for item in evaluations)
    tail_sum = sum(item.tail_probability_sum for item in evaluations)
    return M2DynamicPreviousActionBiasMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)), loss=loss, total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid, total_feedback_trial_count=feedback_count,
        total_timeline_trial_count=timeline_count, tail_probability_sum=tail_sum,
        mean_tail_probability_pre=tail_sum / timeline_count,
        maximum_tail_probability_pre=max(item.maximum_tail_probability_pre for item in evaluations),
        objective_weighting=objective_weighting, session_evaluations=evaluations,
        parameters=parameters,
    )
