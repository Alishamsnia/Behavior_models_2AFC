r"""Shared no-sensory-evidence scalar action-history evaluator core.

This module holds the ONE recurrence, masking, and lapse-mixing
implementation shared by ``pure_action_v1`` (alpha fixed at exactly 1.0, not
fitted) and ``extended_action_v1`` (alpha = sigmoid(a), fitted). It is the
action-driven sibling of ``wsls_core_v1`` (which it does not import and does
not modify): every line is identical in structure except the signal that
drives the ``pi`` recurrence and the gate on that update. ``wsls_core_v1``
updates from the true category ``C_t`` gated on ``feedback_valid``; this core
updates from the mouse's own choice ``A_t`` gated on whether that choice is
itself valid (finite and binary), exactly the gate ``M1_action_kernel_v1``
and ``M2_dynamic_previous_action_bias_v1`` use.

Mathematical definition
------------------------
.. math::

   \pi_1=0.5,\qquad q_t=\pi_t,\qquad
   \epsilon_t=\ell_{\mathrm{long}}\mathbf 1(C_t=1)+\ell_{\mathrm{short}}\mathbf 1(C_t=0),\\
   P(y_t=\mathrm{long})=(1-\epsilon_t)q_t+\epsilon_t/2,\qquad
   \pi_{t+1}=\begin{cases}\pi_t+\alpha(A_t-\pi_t),&V_t=\mathrm{True}\\\pi_t,&V_t=\mathrm{False}\end{cases}

where :math:`A_t=\text{choice\_long}[t]` and :math:`V_t` is choice validity.
There is no sensory readout anywhere in this module: stimulus duration is
not read by any computation (it is accepted only for timeline-shape checks
and output parity), so "duration never enters" is structural. The current
trial's true category :math:`C_t` enters only through the category-dependent
lapse rate, exactly as in ``wsls_core_v1`` and every other model in this
repository; it never enters the ``pi`` update.

Two masks, not one
-------------------
- Likelihood eligibility: valid choice AND a finite binary category (the
  lapse mixture needs a binary category). Identical to ``wsls_core_v1``.
- History update: valid choice alone -- independent of category validity,
  ``feedback_valid``, and duration validity. A valid action on a trial whose
  category is missing still updates ``pi`` (the mouse's action was observed)
  even though that trial contributes no likelihood term. ``feedback_valid``
  is not read at all.

Causal restrictions
--------------------
The current trial's choice can never affect its own ``prior_long`` or
``probability_long`` (predict-then-update order). An omitted/invalid choice
contributes no likelihood AND leaves ``pi`` exactly unchanged (the opposite
of ``wsls_core_v1``, where a feedback-valid omission still updates ``pi``).

Exact fixed-alpha behavior
---------------------------
When ``alpha`` is passed as the literal Python float ``1.0`` (never a
tensor), the update is the direct assignment ``pi_next = action_value``, not
the algebraic form ``pi + alpha*(action - pi)`` -- the same exact-assignment
contract ``wsls_core_v1`` uses, so the fixed model satisfies
:math:`\pi_t=A_{t'}` exactly, where :math:`t'` is the last valid choice
before :math:`t`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from m0_sensory_v1.model.bayesian import lapse_mixture


__all__ = [
    "ActionLapseParameters",
    "ActionPredictions",
    "ActionSessionEvaluation",
    "evaluate_action_session",
]


@dataclass(frozen=True)
class ActionLapseParameters:
    """Minimal duck-typed stand-in for M0's ``DecodedParameters``.

    ``lapse_mixture`` only ever reads ``.ell_long``/``.ell_short`` off its
    ``parameters`` argument -- the same reasoning as
    ``wsls_core_v1.WslsLapseParameters``.
    """

    ell_short: torch.Tensor
    ell_long: torch.Tensor


@dataclass(frozen=True)
class ActionPredictions:
    """Full-timeline action-history predictions and scalar state trajectory."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    post_update_prior_long: torch.Tensor
    valid_choice_mask: torch.Tensor
    action_update_mask: torch.Tensor


@dataclass(frozen=True)
class ActionSessionEvaluation:
    """Differentiable action-history session choice NLL and state trajectory."""

    mouse_id: str
    session_id: str
    split_role: str
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: ActionPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


def evaluate_action_session(
    *,
    mouse_id: str,
    session_id: str,
    split_role: str,
    duration_s: torch.Tensor,
    choice_long: torch.Tensor,
    category_long: torch.Tensor,
    alpha: torch.Tensor | float,
    ell_short: torch.Tensor,
    ell_long: torch.Tensor,
    initial_pi: float = 0.5,
    log_probability_epsilon: float = 1e-12,
) -> ActionSessionEvaluation:
    r"""Evaluate one no-sensory-evidence action-history session in predict/update order.

    Purpose
    -------
    Produce the causal choice likelihood while rebuilding the scalar
    action-history trajectory ``pi_t`` locally on every call, with no
    dependence on stimulus duration anywhere in the computation.

    Parameter interpretation
    ------------------------
    ``alpha`` is either the literal Python float ``1.0`` (pure_action; see
    module docstring for the exact-assignment special case) or a 0-dim
    tensor ``sigmoid(a)`` (extended_action). ``ell_short``/``ell_long`` are
    0-dim tensors decoded the same way M0 decodes its own lapse rates.

    Input units and shapes
    ----------------------
    All tensors are one-dimensional with length ``T``; ``duration_s`` is
    carried through unread for output/CSV parity with sibling models.

    Returned quantities and shapes
    ------------------------------
    Scalar choice NLL and full-timeline probabilities, including both the
    pre-update state used for prediction and the post-update state carried
    into the next trial. ``predictions.action_update_mask`` records choice
    validity (the update gate) for every trial.

    Constraints
    -----------
    :math:`\pi_t` must remain finite; it is a convex combination of
    :math:`\pi_{t-1}\in[0,1]` and :math:`A_{t-1}\in\{0,1\}`, so it remains in
    :math:`[0,1]` by construction. Only probabilities entering logs are
    clamped; saved predictions remain unclamped.

    Important edge cases
    --------------------
    A session with zero eligible trials returns a total NLL of exactly
    ``0.0`` built from the fitted parameters (gradients stay defined) and an
    all-NaN per-trial NLL row.

    Related tests
    -------------
    ``test_action_core_matches_reference_recurrence``,
    ``test_action_core_alpha_one_gives_last_valid_action_exactly``,
    ``test_action_core_no_current_trial_leakage``,
    ``test_action_core_masks_are_independent``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")
    shapes = {tuple(tensor.shape) for tensor in (duration_s, choice_long, category_long)}
    if len(shapes) != 1:
        raise ValueError("Action-history session tensors must share one timeline shape")
    if choice_long.ndim != 1 or choice_long.numel() == 0:
        raise ValueError("Action-history session tensors must be one-dimensional and nonempty")

    alpha_is_tensor = torch.is_tensor(alpha)
    if alpha_is_tensor:
        if alpha.ndim != 0:
            raise ValueError("Tensor alpha must be 0-dimensional")
        alpha_is_exactly_one = False
    else:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Scalar alpha must lie in [0,1]")
        alpha_is_exactly_one = alpha == 1.0

    lapse_parameters = ActionLapseParameters(ell_short=ell_short, ell_long=ell_long)
    pi_current = ell_short.new_tensor(initial_pi)

    action_valid_mask = torch.isfinite(choice_long) & (
        (choice_long == 0.0) | (choice_long == 1.0)
    )
    eligible_mask = (
        action_valid_mask
        & torch.isfinite(category_long)
        & ((category_long == 0.0) | (category_long == 1.0))
    )

    priors: list[torch.Tensor] = []
    post_update_priors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    non_lapse: list[torch.Tensor] = []
    valid_nll: list[torch.Tensor] = []

    for index in range(choice_long.numel()):
        if not bool(torch.isfinite(pi_current).detach().cpu()):
            raise FloatingPointError("Action-history state is non-finite")
        prior = pi_current
        priors.append(prior)

        if bool(eligible_mask[index].detach().cpu()):
            probability = lapse_mixture(prior, category_long[index], lapse_parameters)
            safe = torch.clamp(
                probability,
                min=log_probability_epsilon,
                max=1.0 - log_probability_epsilon,
            )
            observed = choice_long[index]
            valid_nll.append(
                -(observed * torch.log(safe) + (1.0 - observed) * torch.log1p(-safe))
            )
            probabilities.append(probability)
            non_lapse.append(prior.clone())
        else:
            missing = duration_s.new_tensor(torch.nan)
            probabilities.append(missing)
            non_lapse.append(missing.clone())

        if bool(action_valid_mask[index].detach().cpu()):
            action = choice_long[index]
            if not alpha_is_tensor and alpha_is_exactly_one:
                pi_next = action
            else:
                pi_next = pi_current + alpha * (action - pi_current)
        else:
            pi_next = pi_current
        post_update_priors.append(pi_next)
        pi_current = pi_next

    probability_tensor = torch.stack(probabilities)
    non_lapse_tensor = torch.stack(non_lapse)
    prior_tensor = torch.stack(priors)
    post_update_tensor = torch.stack(post_update_priors)
    valid_count = int(eligible_mask.sum().detach().cpu())
    if valid_nll:
        valid_nll_tensor = torch.stack(valid_nll)
        total_nll = valid_nll_tensor.sum()
        per_trial_nll = torch.full_like(duration_s, torch.nan).masked_scatter(
            eligible_mask, valid_nll_tensor
        )
    else:
        total_nll = ell_short * 0.0 + ell_long * 0.0
        if alpha_is_tensor:
            total_nll = total_nll + alpha * 0.0
        per_trial_nll = torch.full_like(duration_s, torch.nan)
    return ActionSessionEvaluation(
        mouse_id=mouse_id,
        session_id=session_id,
        split_role=split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=ActionPredictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=non_lapse_tensor,
            prior_long=prior_tensor,
            post_update_prior_long=post_update_tensor,
            valid_choice_mask=eligible_mask,
            action_update_mask=action_valid_mask,
        ),
        duration_s=duration_s,
        choice_long=choice_long,
    )
