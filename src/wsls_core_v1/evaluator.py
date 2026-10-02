r"""Shared no-sensory-evidence scalar-belief evaluator core.

This module holds the ONE recurrence, masking, and lapse-mixing
implementation shared by ``pure_wsls_v1`` (alpha fixed at exactly 1.0, not
fitted) and ``extended_wsls_v1`` (alpha = sigmoid(a), fitted). Every other
module in each of those two packages (parameters/initialization/registry/
pipeline/fit/analysis) is an intentionally independent, duplicated
orchestration layer -- matching this repository's own convention of one
fully independent top-level package per model family (see
``m1_stimulus_kernel_v1`` vs. ``m1_known_switch_v1``) -- but the recurrence
itself lives here exactly once so the two models can never silently drift
on causal timing, the lapse equation, or masking rules.

Mathematical definition
------------------------
.. math::

   \pi_1=0.5,\qquad q_t=\pi_t,\qquad
   \epsilon_t=\ell_{\mathrm{long}}\mathbf 1(C_t=1)+\ell_{\mathrm{short}}\mathbf 1(C_t=0),\\
   P(y_t=\mathrm{long})=(1-\epsilon_t)q_t+\epsilon_t/2,\qquad
   \pi_{t+1}=\begin{cases}\pi_t+\alpha(C_t-\pi_t),&F_t=\mathrm{True}\\\pi_t,&F_t=\mathrm{False}\end{cases}

There is no sensory readout anywhere in this module: stimulus duration is
not accepted as an argument at all, so "duration never enters" is a
structural property of the function signature, not merely an unused field.

Causal restrictions
--------------------
The current trial's category/choice can never affect its own ``prior_long``
or ``probability_long`` (predict-then-update order). A trial contributes to
the choice likelihood only when its choice is valid AND its own category is
a finite binary code; a trial may still update ``pi`` for the next trial
whenever its feedback is valid, independent of whether its own choice was
valid that trial (an omission with valid category feedback still teaches
the next trial). ``M1SessionTrials.__post_init__`` already guarantees every
feedback-valid trial's category is finite and binary, so the update branch
does not re-validate it.

Exact fixed-alpha behavior
---------------------------
When ``alpha`` is passed as the literal Python float ``1.0`` (never a
tensor), the update is computed as a direct assignment ``pi_next =
category_value``, not via the algebraic form ``pi + alpha*(category - pi)``
-- floating-point subtraction-then-addition is not guaranteed bit-exact to
``category_value``, and the scientific contract requires
:math:`\pi_t=C_{t-1}` exactly for the fixed model, not merely to machine
precision. Any other alpha (a Python float in (0,1) or, for the fitted
model, a 0-dim tensor carrying gradients) uses the ordinary algebraic
recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from m0_sensory_v1.model.bayesian import lapse_mixture


__all__ = [
    "WslsLapseParameters",
    "WslsPredictions",
    "WslsSessionEvaluation",
    "evaluate_wsls_session",
]


@dataclass(frozen=True)
class WslsLapseParameters:
    """Minimal duck-typed stand-in for M0's ``DecodedParameters``.

    ``lapse_mixture`` only ever reads ``.ell_long``/``.ell_short`` off its
    ``parameters`` argument (no ``isinstance`` check; confirmed by reading
    its body) -- so passing this two-field object reuses the real, tested
    lapse-mixing formula without carrying a single meaningless ``w``/``s0``/
    ``kappa`` coordinate, which do not exist in either WSLS model.
    """

    ell_short: torch.Tensor
    ell_long: torch.Tensor


@dataclass(frozen=True)
class WslsPredictions:
    """Full-timeline WSLS predictions and scalar prior trajectory."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    prior_long: torch.Tensor
    post_update_prior_long: torch.Tensor
    valid_choice_mask: torch.Tensor
    feedback_update_mask: torch.Tensor


@dataclass(frozen=True)
class WslsSessionEvaluation:
    """Differentiable WSLS session choice NLL and prior trajectory."""

    mouse_id: str
    session_id: str
    split_role: str
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: WslsPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


def evaluate_wsls_session(
    *,
    mouse_id: str,
    session_id: str,
    split_role: str,
    duration_s: torch.Tensor,
    choice_long: torch.Tensor,
    category_long: torch.Tensor,
    feedback_valid: torch.Tensor,
    alpha: torch.Tensor | float,
    ell_short: torch.Tensor,
    ell_long: torch.Tensor,
    initial_pi: float = 0.5,
    log_probability_epsilon: float = 1e-12,
) -> WslsSessionEvaluation:
    r"""Evaluate one no-sensory-evidence WSLS session in predict/update order.

    Purpose
    -------
    Produce the causal choice likelihood while rebuilding the scalar belief
    trajectory ``pi_t`` locally on every call, with no dependence on
    stimulus duration anywhere in the computation.

    Parameter interpretation
    ------------------------
    ``alpha`` is either the literal Python float ``1.0`` (pure_wsls; see
    module docstring for the exact-assignment special case) or a 0-dim
    tensor ``sigmoid(a)`` (extended_wsls). ``ell_short``/``ell_long`` are
    0-dim tensors decoded the same way M0 decodes its own lapse rates
    (``lapse_max * sigmoid(raw)``).

    Input units and shapes
    ----------------------
    All tensors are one-dimensional with length ``T`` (one physical
    session's trial count); ``duration_s`` is carried through unread for
    output/CSV parity with sibling models, never used in any computation
    here.

    Returned quantities and shapes
    ------------------------------
    Returns scalar choice NLL and full-timeline probabilities, including
    both the pre-update prior used for prediction and the post-update prior
    carried into the next trial.

    Constraints
    -----------
    :math:`\pi_t` must remain finite at every trial; a session that drives
    it outside real values is a hard error, not a silently clamped value.
    :math:`\pi_t` is a convex combination of :math:`\pi_{t-1}\in[0,1]` and
    :math:`C_{t-1}\in\{0,1\}` (weight :math:`\alpha\in[0,1]`), so it remains
    in :math:`[0,1]` by construction; only probabilities entering logs are
    clamped, and saved predictions remain unclamped.

    Causal restrictions
    -------------------
    See module docstring: current trial's category/choice cannot affect its
    own prediction; a feedback-valid omission still updates ``pi``.

    Important edge cases
    --------------------
    A session with zero eligible trials returns a total NLL of exactly
    ``0.0`` (as a differentiable zero built from the fitted parameters, so
    gradients remain well-defined) and an all-NaN per-trial NLL row.

    Related tests
    -------------
    ``test_wsls_core_matches_closed_form_exponential_kernel``,
    ``test_wsls_core_alpha_one_gives_previous_category_exactly``,
    ``test_wsls_core_no_current_trial_leakage``,
    ``test_wsls_core_duration_never_enters``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")
    shapes = {
        tuple(tensor.shape)
        for tensor in (duration_s, choice_long, category_long, feedback_valid)
    }
    if len(shapes) != 1:
        raise ValueError("WSLS session tensors must share one timeline shape")
    if choice_long.ndim != 1 or choice_long.numel() == 0:
        raise ValueError("WSLS session tensors must be one-dimensional and nonempty")

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

    lapse_parameters = WslsLapseParameters(ell_short=ell_short, ell_long=ell_long)
    pi_current = ell_short.new_tensor(initial_pi)

    eligible_mask = (
        torch.isfinite(choice_long)
        & ((choice_long == 0.0) | (choice_long == 1.0))
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
            raise FloatingPointError("WSLS prior state is non-finite")
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

        if bool(feedback_valid[index].detach().cpu()):
            category = category_long[index]
            if not alpha_is_tensor and alpha_is_exactly_one:
                pi_next = category
            else:
                pi_next = pi_current + alpha * (category - pi_current)
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
    return WslsSessionEvaluation(
        mouse_id=mouse_id,
        session_id=session_id,
        split_role=split_role,
        negative_log_likelihood=total_nll,
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=per_trial_nll,
        predictions=WslsPredictions(
            probability_long=probability_tensor,
            non_lapse_probability_long=non_lapse_tensor,
            prior_long=prior_tensor,
            post_update_prior_long=post_update_tensor,
            valid_choice_mask=eligible_mask,
            feedback_update_mask=feedback_valid,
        ),
        duration_s=duration_s,
        choice_long=choice_long,
    )
