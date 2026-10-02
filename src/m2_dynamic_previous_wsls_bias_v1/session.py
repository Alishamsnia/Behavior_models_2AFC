"""Causal static-prototype duration-dependent-hazard HSMM evaluation, with a
previous-category (win-stay/lose-shift) bias mixed in before lapse, for
M2_dynamic_previous_wsls_bias_v1.

Independent, freshly-authored per-trial loop -- this module does not call
into ``m2_dynamic_block_length_v1.session`` or ``wsls_core_v1.evaluator`` at
runtime. It reuses only the same genuinely shared, pre-existing primitives
``M2_dynamic_block_length_v1`` itself reuses (``m1_bayesian_choice_probability``,
``lapse_mixture``, the full ``capped_tail_hsmm`` module) plus
M2_dynamic_block_length_v1's own parameter decoder (this model's true parent).

Two independent pieces of causal state are carried per session: the joint
``(3,Rmax)`` block/run-length belief ``xi`` (driven only by the true
category, exactly as in M2_dynamic_block_length_v1 -- completely unchanged)
and a scalar previous-category bias state ``pi_t`` (driven only by the true
category, gated by ``feedback_valid`` -- never by the mouse's own choice).
The learning rate is fixed to the registry constant ``alpha=1.0`` -- the
exact one-step "previous category" special case of the shared delta rule,
at which ``pi_{t+1}`` collapses to ``C_t`` exactly. This is the same
``alpha=1`` boundary and the same feedback-gated, direct-assignment causal
contract ``wsls_core_v1.evaluator`` uses for ``pure_wsls_v1`` (see that
module for why direct assignment, not the algebraic ``pi+alpha*(c-pi)``
form, is required for bit-exact fixed-alpha semantics) -- reused here as a
pattern, not as a runtime dependency, because this model additionally
carries the sensory/HSMM state ``xi`` that ``wsls_core_v1`` has no
equivalent of. Neither piece of state ever touches the other's update
directly, though both are gated by the same ``feedback_valid`` signal, since
both represent what the animal can causally learn about the true category.
They combine at exactly one point per trial:

    r_t   = BayesianReadout(duration_t, w, s0, prior_block,t)   [unchanged solver]
    m_t   = (1-eta)*r_t + eta*pi_t                                [NEW]
    p_t   = lapse_mixture(m_t, category_t, ell_short, ell_long)   [unchanged function,
                                                                     fed m_t instead of r_t]

Lapse is therefore the outermost operation, matching every other model in
this family's lapse semantics. ``ell_short``/``ell_long`` are completely
unchanged; they still see only the true current category.

Unlike the sibling ``M2_dynamic_previous_action_bias_v1`` (whose ``pi_t`` is
driven by the mouse's own choice and therefore gated by choice validity
alone, independent of feedback), this model's ``pi_t`` is gated by
``feedback_valid`` alone, independent of choice validity: the true category
is only causally knowable to the animal once feedback is delivered, exactly
as ``wsls_core_v1`` treats it. Because the update driver (``category_long``)
and its gate (``feedback_valid``) are both exogenous, task-design quantities
-- never the mouse's own sampled/observed choice -- ``pi_t`` (like ``xi``)
is fully independent of ``choice_long`` end to end; see ``simulation.py``
for the vectorized consequence of this.
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

from .parameters import M2DynamicPreviousWslsBiasParameters


M2DynamicPreviousWslsBiasSessionTrials = M1SessionTrials
M2DynamicPreviousWslsBiasSplitRole = Literal["train", "validation", "test"]
M2DynamicPreviousWslsBiasObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

__all__ = [
    "M2DynamicPreviousWslsBiasSessionTrials",
    "m2_dynamic_previous_wsls_bias_session_trials_from_frame",
    "M2DynamicPreviousWslsBiasPredictions",
    "M2DynamicPreviousWslsBiasSessionEvaluation",
    "M2DynamicPreviousWslsBiasMouseObjectiveResult",
    "evaluate_m2_dynamic_previous_wsls_bias_session",
    "evaluate_m2_dynamic_previous_wsls_bias_mouse_objective",
]


def m2_dynamic_previous_wsls_bias_session_trials_from_frame(*args, **kwargs) -> M2DynamicPreviousWslsBiasSessionTrials:
    """Use the unchanged complete-timeline session contract shared M1-M4 all reuse."""

    return m1_session_trials_from_frame(*args, **kwargs)


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasPredictions:
    """Causal choice, block/run-length-belief, previous-category-bias, and
    duration-diagnostic trajectory."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    bias_mixed_probability_long: torch.Tensor
    non_bias_probability_long: torch.Tensor
    prior_long: torch.Tensor
    previous_wsls_state: torch.Tensor
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
    feedback_update_mask: torch.Tensor
    parameters: M2DynamicPreviousWslsBiasParameters


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasSessionEvaluation:
    mouse_id: str
    session_id: str
    split_role: M2DynamicPreviousWslsBiasSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    feedback_trial_count: int
    timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M2DynamicPreviousWslsBiasPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasMouseObjectiveResult:
    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    total_feedback_trial_count: int
    total_timeline_trial_count: int
    tail_probability_sum: float
    mean_tail_probability_pre: float
    maximum_tail_probability_pre: float
    objective_weighting: M2DynamicPreviousWslsBiasObjectiveWeighting
    session_evaluations: tuple[M2DynamicPreviousWslsBiasSessionEvaluation, ...]
    parameters: M2DynamicPreviousWslsBiasParameters


def evaluate_m2_dynamic_previous_wsls_bias_session(
    session: M2DynamicPreviousWslsBiasSessionTrials,
    parameters: M2DynamicPreviousWslsBiasParameters,
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
) -> M2DynamicPreviousWslsBiasSessionEvaluation:
    r"""Evaluate one M2_dynamic_previous_wsls_bias session: static
    prototypes and HSMM transition unchanged from M2_dynamic_block_length_v1,
    with the previous trial's true category mixed in between the sensory
    readout and lapse.

    Purpose
    -------
    Produce choice likelihoods and causal joint block/run-length
    trajectories exactly as M2_dynamic_block_length_v1 does, plus a
    previous-category (WSLS) bias trajectory, combined only at the final
    probability.

    Mathematical definition
    -----------------------
    .. math::

       r_t=\text{BayesianReadout}(\cdot),\quad
       \pi_{t+1}=\begin{cases}C_t,&F_t=\text{True}\\\pi_t,&F_t=\text{False}\end{cases}
       \ (\alpha\equiv1),\quad
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
    checked finite every trial.

    Causal restrictions
    -------------------
    Only pre-choice belief and pre-update ``pi_t`` predict a choice; true
    labels and the mouse's own current-trial choice never enter prediction.
    ``pi_t`` and ``xi`` never depend on ``choice_long`` at any trial --
    matching ``wsls_core_v1``'s "duration never enters" structural property,
    this model's bias state has the parallel property that the mouse's own
    choice never enters ``pi_t``.

    Important edge cases
    --------------------
    A trial contributes to the choice likelihood only when its duration and
    choice are valid AND its own category is a finite binary code (the
    shared session contract permits an invalid category on a
    feedback-invalid trial, and ``lapse_mixture`` requires a binary
    category). ``pi_t`` updates on every feedback-valid trial regardless of
    that trial's own choice validity (an omission with valid feedback still
    teaches the next trial), and holds otherwise; no transition is computed
    after trial ``T``.

    Related tests
    -------------
    ``test_eta_zero_reproduces_m2_dynamic_block_length_exactly``,
    ``test_pi_trajectory_is_bit_exact_previous_category_matching_wsls_core_contract``,
    ``test_feedback_invalid_trial_holds_pi_unchanged``, and
    ``test_omission_with_valid_feedback_still_teaches_next_trial``.
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
    eta = parameters.eta
    pi_current = parameters.w.new_tensor(initial_pi)

    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "prior", "probability", "non_lapse", "bias_mixed", "non_bias", "previous_wsls",
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
        & torch.isfinite(session.category_long)
        & ((session.category_long == 0) | (session.category_long == 1))
    )

    for index in range(session.duration_s.numel()):
        if not bool(torch.isfinite(pi_current).detach().cpu()):
            raise FloatingPointError("M2_dynamic_previous_wsls_bias previous-wsls-bias state is non-finite")
        if not bool(torch.all(torch.isfinite(xi_pre) & (xi_pre >= 0))):
            raise FloatingPointError("M2_dynamic_previous_wsls_bias pre-choice joint belief is invalid")
        if not bool(torch.isclose(xi_pre.sum(), xi_pre.new_tensor(1.0), atol=1e-12, rtol=0)):
            raise FloatingPointError("M2_dynamic_previous_wsls_bias pre-choice joint belief is not normalized")

        block_pre = block_marginal(xi_pre)
        run_pre = run_length_marginal(xi_pre)
        prior = torch.dot(block_pre, theta)
        pi_before = pi_current
        values["pre_block"].append(block_pre)
        values["pre_run"].append(run_pre)
        values["prior"].append(prior)
        values["previous_wsls"].append(pi_before)
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
            raise FloatingPointError("M2_dynamic_previous_wsls_bias pre-choice state has unreachable run-length mass")

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

        if bool(session.feedback_valid[index]):
            category = session.category_long[index]
            pi_next = category
            emission = theta.pow(category) * (1 - theta).pow(1 - category)
            xi_post, normalizer = feedback_update(xi_pre, emission)
            values["normalizer"].append(normalizer)
        else:
            missing = session.duration_s.new_tensor(torch.nan)
            pi_next = pi_before
            xi_post = xi_pre
            values["normalizer"].append(missing)
        pi_current = pi_next

        if not bool(torch.all(torch.isfinite(xi_post) & (xi_post >= 0))):
            raise FloatingPointError("M2_dynamic_previous_wsls_bias post-feedback joint belief is invalid")
        if not bool(torch.isclose(xi_post.sum(), xi_post.new_tensor(1.0), atol=1e-12, rtol=0)):
            raise FloatingPointError("M2_dynamic_previous_wsls_bias post-feedback joint belief is not normalized")
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
            raise FloatingPointError("M2_dynamic_previous_wsls_bias post-feedback state has unreachable run-length mass")

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

    prediction = M2DynamicPreviousWslsBiasPredictions(
        probability_long=torch.stack(values["probability"]),
        non_lapse_probability_long=torch.stack(values["non_lapse"]),
        bias_mixed_probability_long=torch.stack(values["bias_mixed"]),
        non_bias_probability_long=torch.stack(values["non_bias"]),
        prior_long=torch.stack(values["prior"]),
        previous_wsls_state=torch.stack(values["previous_wsls"]),
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
        feedback_update_mask=session.feedback_valid, parameters=parameters,
    )
    tail_values = prediction.tail_probability_pre.detach()
    return M2DynamicPreviousWslsBiasSessionEvaluation(
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


def evaluate_m2_dynamic_previous_wsls_bias_mouse_objective(
    sessions: Iterable[M2DynamicPreviousWslsBiasSessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M2DynamicPreviousWslsBiasObjectiveWeighting = "pooled_trials",
    required_split_role: M2DynamicPreviousWslsBiasSplitRole = "train",
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
) -> M2DynamicPreviousWslsBiasMouseObjectiveResult:
    r"""Evaluate the pooled nine-parameter choice-only
    M2_dynamic_previous_wsls_bias objective.

    Purpose
    -------
    Aggregate independent reset sessions for one mouse and one non-test
    split, rebuilding both the joint belief and the scalar previous-category
    bias state from scratch for each session.

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

    from .parameters import decode_m2_dynamic_previous_wsls_bias_raw_parameters

    records = tuple(sessions)
    if not records or raw_parameters.shape != (9,):
        raise ValueError("M2_dynamic_previous_wsls_bias requires sessions and raw_parameters with shape (9,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError("Unknown M2_dynamic_previous_wsls_bias objective weighting")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1 or any(record.split_role == "test" for record in records):
        raise ValueError("M2_dynamic_previous_wsls_bias objective requires one mouse and excludes locked test sessions")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M2_dynamic_previous_wsls_bias session must have role {required_split_role!r}")
    parameters = decode_m2_dynamic_previous_wsls_bias_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(evaluate_m2_dynamic_previous_wsls_bias_session(
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
        raise ValueError("M2_dynamic_previous_wsls_bias objective contains no valid choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(item.valid_trial_count == 0 for item in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M2_dynamic_previous_wsls_bias session")
        loss = torch.mean(session_nll / session_nll.new_tensor([item.valid_trial_count for item in evaluations]))
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M2_dynamic_previous_wsls_bias mouse objective is nonfinite")
    feedback_count = sum(item.feedback_trial_count for item in evaluations)
    timeline_count = sum(item.timeline_trial_count for item in evaluations)
    tail_sum = sum(item.tail_probability_sum for item in evaluations)
    return M2DynamicPreviousWslsBiasMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)), loss=loss, total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid, total_feedback_trial_count=feedback_count,
        total_timeline_trial_count=timeline_count, tail_probability_sum=tail_sum,
        mean_tail_probability_pre=tail_sum / timeline_count,
        maximum_tail_probability_pre=max(item.maximum_tail_probability_pre for item in evaluations),
        objective_weighting=objective_weighting, session_evaluations=evaluations,
        parameters=parameters,
    )
