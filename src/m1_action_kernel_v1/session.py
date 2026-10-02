"""Causal scalar delta-rule prior evaluation for M1_action_kernel_v1.

M1_action_kernel_v1 keeps M1's unchanged Bayesian sensory/lapse readout
(``m0_sensory_v1.model.m1_bayesian.m1_bayesian_choice_probability`` and
``m0_sensory_v1.model.bayesian.lapse_mixture``, byte-for-byte the same
functions M1 uses) but replaces M1's entire three-state hazard-driven block
belief with a single scalar prior ``pi_t`` updated by a Rescorla-Wagner-
style delta rule directly on the mouse's own previous long/short action
(``choice_long``), NOT the true stimulus category:

    alpha = sigmoid(a)                         (fitted logit -> rate)
    pi_{t+1} = pi_t + alpha * (A_t - pi_t)      pi_1 = 0.5

where ``A_t = choice_long[t] in {0,1}`` is the mouse's own observed action
on trial ``t`` (1 = chose the long-associated response, 0 = chose the
short-associated response). This is the only scientific difference from the
sibling model ``M1_stimulus_kernel_v1``, whose recursion instead uses the
true category ``C_t`` gated on ``feedback_valid``. Every other detail --
sensory readout, lapse mixture (still keyed on the *true* current category),
parameterization, causal ordering, and session-reset policy -- is identical
and copied unchanged; see ``docs/models/M1_action_kernel_v1.md`` for the
full scientific contract and its explicit comparison with the stimulus
kernel. Repeated application of this rule is an exponentially-weighted
moving average of past actions -- the "kernel" is a consequence of the
recursion, not a separately coded function. There is no hidden state,
hazard, transition matrix, or fixed category-rate prototype anywhere in
this model, and ``pi_t`` is a choice-history/perseveration state, not an
inferred belief about the environment.

This module reuses ``m0_sensory_v1.model.m1_session.M1SessionTrials`` and
``m1_session_trials_from_frame`` unchanged as the session-tensor adapter:
``choice_long`` (the field this model's history update reads) is already
carried by every M1 session, so there is no oracle signal to add, unlike
``m1_known_switch_v1``'s ``is_switch``. ``feedback_valid`` is still carried
on the tensor (M1SessionTrials validates it) but is deliberately NOT
consulted by this model's update gate -- see the module-level gate note
below. Trial order within each session is guaranteed by the shared M0
pipeline (every session's canonical trial frame is sorted by
``trial_index`` before this adapter ever sees it; see
``m0_sensory_v1/data/preparation.py`` and ``m0_sensory_v1/pipeline.py``),
exactly the same precondition M1's own evaluator relies on -- this module
does not re-validate it.

Two independent masks, not one
-------------------------------
Unlike the stimulus kernel (where a single ``valid_mask`` both gates the
choice likelihood and gates the category-feedback update, because
likelihood-eligibility and feedback-eligibility happen to use the same
duration+choice validity condition), this model needs two masks that are
NOT definitionally the same:

- ``valid_mask`` (duration finite and positive, AND choice finite/binary):
  gates the Bernoulli choice likelihood exactly as it does for every model
  in this family. A trial can fail this mask (e.g. a corrupted/missing
  duration) while still carrying a perfectly valid recorded action.
- ``action_valid_mask`` (choice finite/binary only, independent of
  duration): gates the action-history update. The mouse's action was
  observed whenever it made a valid choice, regardless of whether that
  trial's duration can support a likelihood term.

Collapsing these into one mask (reusing ``valid_mask`` for both purposes)
would incorrectly skip a history update on a valid-choice/invalid-duration
trial -- a bug that was caught in this model's design review specifically
because it does not affect the stimulus kernel (there, feedback validity
and likelihood validity are already independent concerns tracked by
different fields, ``feedback_valid`` vs. duration+choice) but would be a
real causal error here if introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import M1SessionTrials, m1_session_trials_from_frame

from .parameters import (
    M1ActionKernelDecodedParameters,
    decode_m1_action_kernel_raw_parameters,
)


M1ActionKernelSplitRole = Literal["train", "validation", "test"]
M1ActionKernelObjectiveWeighting = Literal["pooled_trials", "equal_sessions"]

__all__ = [
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1ActionKernelPredictions",
    "M1ActionKernelSessionEvaluation",
    "M1ActionKernelMouseObjectiveResult",
    "evaluate_m1_action_kernel_session",
    "evaluate_m1_action_kernel_mouse_objective",
]


@dataclass(frozen=True)
class M1ActionKernelPredictions:
    """Full-timeline M1_action_kernel predictions and scalar prior trajectory."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    post_update_prior_long: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    valid_choice_mask: torch.Tensor
    action_update_mask: torch.Tensor
    parameters: M1ActionKernelDecodedParameters


@dataclass(frozen=True)
class M1ActionKernelSessionEvaluation:
    """Differentiable M1_action_kernel session choice NLL and prior trajectory."""

    mouse_id: str
    session_id: str
    split_role: M1ActionKernelSplitRole
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: M1ActionKernelPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


@dataclass(frozen=True)
class M1ActionKernelMouseObjectiveResult:
    """One pooled M1_action_kernel objective built from independently reset sessions."""

    mouse_id: str
    loss: torch.Tensor
    total_negative_log_likelihood: torch.Tensor
    total_valid_trial_count: int
    objective_weighting: M1ActionKernelObjectiveWeighting
    session_evaluations: tuple[M1ActionKernelSessionEvaluation, ...]
    parameters: M1ActionKernelDecodedParameters


def evaluate_m1_action_kernel_session(
    session: M1SessionTrials,
    parameters: M1ActionKernelDecodedParameters,
    moments: TaskMoments,
    *,
    initial_pi: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1ActionKernelSessionEvaluation:
    r"""Evaluate one M1_action_kernel session in strict predict/update order.

    Purpose
    -------
    Produce the causal choice likelihood while rebuilding the scalar
    action-history prior trajectory locally on every call, from the mouse's
    own previous actions rather than the true stimulus history.

    Mathematical definition
    -----------------------
    With learning rate :math:`\alpha=\operatorname{sigmoid}(a)`:

    .. math::

       \pi_1=0.5,\qquad
       \pi_{t+1}=\begin{cases}
       \pi_t+\alpha(A_t-\pi_t), & V_t=\text{True},\\
       \pi_t, & V_t=\text{False},
       \end{cases}

    where :math:`A_t=\text{choice\_long}[t]\in\{0,1\}` is the mouse's own
    observed action on trial :math:`t` and :math:`V_t` is whether that
    action itself is valid (finite and binary) -- NOT ``feedback_valid``,
    which this model's update deliberately ignores (see module docstring).
    The prediction at trial :math:`t` always uses :math:`\pi_t`, never
    :math:`\pi_{t+1}`.

    Parameter interpretation
    ------------------------
    ``alpha`` is the fitted per-trial learning rate. There is no hazard,
    transition matrix, or fixed category-rate prototype: :math:`\pi_t` is
    itself the fitted scalar action-history/perseveration state, not a
    function of a discrete hidden state and not an inferred belief about
    the environment -- it tracks the mouse's own recent long/short response
    tendency regardless of whether those responses were correct or
    rewarded.

    Input units and shapes
    ----------------------
    Session tensors have length ``T``; durations and ``s0`` are seconds.

    Returned quantities and shapes
    ------------------------------
    Returns scalar choice NLL and full-timeline probabilities, including
    both the pre-update prior used for prediction and the post-update prior
    carried into the next trial. ``predictions.action_update_mask`` records
    :math:`V_t` for every trial (diagnostic; see module docstring on why it
    is a distinct mask from ``valid_choice_mask``).

    Constraints
    -----------
    :math:`\pi_t` must remain finite at every trial; a session that drives
    it outside these bounds is a hard error, not a silently clamped value
    (the sensory readout itself clamps only its own internal logit, never
    the exported trajectory -- see
    :func:`m0_sensory_v1.model.m1_bayesian.m1_bayesian_choice_probability`).

    Numerical stability
    -------------------
    :math:`\pi_t` is a convex combination of :math:`\pi_{t-1}\in[0,1]` and
    :math:`A_{t-1}\in\{0,1\}` with weight :math:`\alpha\in(0,1)` (open,
    since ``sigmoid`` never reaches its endpoints for finite input), so it
    remains in :math:`[0,1]` by construction; only probabilities entering
    logs are clamped, and saved predictions remain unclamped.

    Causal restrictions
    -------------------
    Trial :math:`t`'s own action and category cannot affect its own
    ``prior_long`` or ``probability_long`` -- only trials before :math:`t`
    can. True block, true run length, and hidden state are absent -- they
    do not exist in this model. The true category still drives the current
    trial's sensory likelihood and lapse mixture (unchanged from M1/the
    stimulus kernel); it plays no role at all in the ``pi`` update.

    Important edge cases
    --------------------
    An invalid/omitted choice (:math:`V_t=\text{False}`) adds no likelihood
    AND leaves ``pi`` unchanged -- the opposite of the stimulus kernel,
    where a feedback-valid omission still updates the prior. Conversely, a
    valid action updates ``pi`` even when ``feedback_valid`` is False for
    that trial (feedback independence) and even when the action was
    incorrect relative to the true category (pure perseveration, not
    win-stay/lose-switch). A valid action with an invalid/missing duration
    still updates ``pi`` even though it contributes no likelihood term. The
    post-update value is still computed and exported for the final task
    trial (diagnostic only; there is no following trial in this session to
    use it as a prediction input).

    Related tests
    -------------
    ``test_matches_reference_kernel_across_random_action_sequences``,
    ``test_no_current_trial_leakage``, and
    ``test_action_kernel_matches_stimulus_kernel_when_actions_equal_categories``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")

    alpha = parameters.alpha
    pi_current = parameters.w.new_tensor(initial_pi)

    priors: list[torch.Tensor] = []
    post_update_priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    lower_boundaries: list[torch.Tensor] = []
    upper_boundaries: list[torch.Tensor] = []
    sensory_sd: list[torch.Tensor] = []
    decision_regimes: list[str] = []
    valid_nll: list[torch.Tensor] = []

    action_valid_mask = torch.isfinite(session.choice_long) & (
        (session.choice_long == 0.0) | (session.choice_long == 1.0)
    )
    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & action_valid_mask
    )
    for index in range(session.duration_s.numel()):
        if not bool(torch.isfinite(pi_current).detach().cpu()):
            raise FloatingPointError("M1_action_kernel prior state is non-finite")
        prior = pi_current
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

        if bool(action_valid_mask[index].detach().cpu()):
            action = session.choice_long[index]
            pi_next = pi_current + alpha * (action - pi_current)
        else:
            pi_next = pi_current
        post_update_priors.append(pi_next)
        pi_current = pi_next

    probability_tensor = torch.stack(probabilities)
    non_lapse_tensor = torch.stack(non_lapse)
    threshold_tensor = torch.stack(thresholds)
    lower_boundary_tensor = torch.stack(lower_boundaries)
    upper_boundary_tensor = torch.stack(upper_boundaries)
    sensory_tensor = torch.stack(sensory_sd)
    prior_tensor = torch.stack(priors)
    post_update_tensor = torch.stack(post_update_priors)
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
    return M1ActionKernelSessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=M1ActionKernelPredictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=non_lapse_tensor,
            prior_long=prior_tensor,
            post_update_prior_long=post_update_tensor,
            positive_threshold_s=threshold_tensor,
            decision_lower_boundary_s=lower_boundary_tensor,
            decision_upper_boundary_s=upper_boundary_tensor,
            decision_regime=tuple(decision_regimes),
            sensory_sd_s=sensory_tensor,
            valid_choice_mask=valid_mask,
            action_update_mask=action_valid_mask,
            parameters=parameters,
        ),
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )


def evaluate_m1_action_kernel_mouse_objective(
    sessions: Iterable[M1SessionTrials],
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    objective_weighting: M1ActionKernelObjectiveWeighting = "pooled_trials",
    required_split_role: M1ActionKernelSplitRole = "train",
    lapse_max: float = 1.0,
    parameter_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
    initial_pi: float = 0.5,
) -> M1ActionKernelMouseObjectiveResult:
    r"""Evaluate M1_action_kernel's full-batch pooled choice objective.

    Purpose
    -------
    Decode one six-parameter mouse vector and independently rebuild the
    scalar prior trajectory for every supplied session before returning one
    loss.

    Mathematical definition
    -----------------------
    The primary loss is

    .. math::

       J=\frac{\sum_d J_d}{\sum_d N_d},

    with the same optional equal-session sensitivity objective as M0/M1.

    Parameter interpretation
    ------------------------
    Stable parameters are pooled; the scalar prior resets to 0.5 for each
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
    updates the future prior (used again only as next-trial input) and is
    never added as an auxiliary fitting target beyond that.

    Important edge cases
    --------------------
    Pooled weighting permits an omission-only session when another session
    has valid choices; equal-session weighting does not.

    Related tests
    -------------
    ``test_m1_action_kernel_pooled_objective_autograd_and_session_order_invariance``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("At least one M1_action_kernel session is required")
    if raw_parameters.shape != (6,):
        raise ValueError("M1_action_kernel raw_parameters must have shape (6,)")
    if objective_weighting not in {"pooled_trials", "equal_sessions"}:
        raise ValueError(f"Unknown M1_action_kernel objective weighting: {objective_weighting!r}")
    mouse_ids = {record.mouse_id for record in records}
    if len(mouse_ids) != 1:
        raise ValueError("An M1_action_kernel pooled objective may contain only one mouse")
    if any(record.split_role == "test" for record in records):
        raise ValueError("Locked test sessions cannot enter M1_action_kernel objective evaluation")
    if any(record.split_role != required_split_role for record in records):
        raise ValueError(f"Every M1_action_kernel session must have role {required_split_role!r}")

    parameters = decode_m1_action_kernel_raw_parameters(
        raw_parameters, lapse_max=lapse_max, epsilon=parameter_epsilon
    )
    evaluations = tuple(
        evaluate_m1_action_kernel_session(
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
        raise ValueError("M1_action_kernel objective contains no valid completed choices")
    if objective_weighting == "pooled_trials":
        loss = total_nll / total_valid
    else:
        if any(record.valid_trial_count == 0 for record in evaluations):
            raise ValueError("equal_sessions is undefined for a zero-valid M1_action_kernel session")
        counts = session_nll.new_tensor([record.valid_trial_count for record in evaluations])
        loss = torch.mean(session_nll / counts)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("M1_action_kernel mouse objective is nonfinite")
    return M1ActionKernelMouseObjectiveResult(
        mouse_id=next(iter(mouse_ids)),
        loss=loss,
        total_negative_log_likelihood=total_nll,
        total_valid_trial_count=total_valid,
        objective_weighting=objective_weighting,
        session_evaluations=evaluations,
        parameters=parameters,
    )
