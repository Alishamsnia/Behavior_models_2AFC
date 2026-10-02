"""M0-specific neutral prior and omission-safe trial prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.task_moments import TaskMoments
from .bayesian import bayesian_choice_probability, lapse_mixture
from .parameters import DecodedParameters, decode_raw_parameters


@dataclass(frozen=True)
class MaskedM0Predictions:
    """Full-timeline predictions with NaN on trials that were not evaluated."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    valid_choice_mask: torch.Tensor
    parameters: DecodedParameters


def fixed_neutral_prior(reference: torch.Tensor) -> torch.Tensor:
    r"""Return M0's exact, history-independent long-category prior.

    Purpose
    -------
    Provide the only prior allowed by M0.

    Mathematical definition
    -----------------------
    .. math::

       \pi_{d,\tau}=0.5,\qquad \operatorname{logit}(\pi_{d,\tau})=0.

    Parameter interpretation
    ------------------------
    The neutral prior has no fitted parameter and is not an approximate sigmoid
    transform.

    Input units and shapes
    ----------------------
    ``reference`` supplies only output shape, floating dtype, and device.

    Returned quantities and shapes
    ------------------------------
    Tensor exactly equal to 0.5 with ``reference.shape``.

    Constraints
    -----------
    ``reference`` must be a floating Torch tensor.

    Numerical stability
    -------------------
    ``0.5`` is exactly representable in binary floating-point.

    Causal restrictions
    -------------------
    This function cannot receive block, category, choice, reward, history,
    trial position, session identity, or session-specific parameters.

    Important edge cases
    --------------------
    Empty valid-trial vectors produce an empty prior vector.

    Related tests
    -------------
    ``test_fixed_neutral_prior_is_exactly_half``.
    """

    if (
        not isinstance(reference, torch.Tensor)
        or not reference.is_floating_point()
    ):
        raise TypeError("reference must be a floating torch.Tensor")
    return torch.full_like(reference, 0.5)


def _scatter_valid(
    values: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Scatter differentiable valid values into a NaN full-timeline tensor."""

    return torch.full_like(reference, torch.nan).masked_scatter(
        mask,
        values.reshape(-1),
    )


def predict_m0_trials_with_parameters(
    duration_s: torch.Tensor,
    category_long: torch.Tensor,
    choice_long: torch.Tensor,
    parameters: DecodedParameters,
    moments: TaskMoments,
) -> MaskedM0Predictions:
    r"""Predict M0 on completed Bernoulli decisions while retaining the timeline.

    Purpose
    -------
    Preserve one row per task trial yet skip omissions and non-decisions before
    any scientific probability calculation. This prevents invalid durations or
    undefined choices from contaminating the forward pass or parameter gradient.

    Mathematical definition
    -----------------------
    Define

    .. math::

       \mathcal V=\{i:s_i\text{ finite},\ s_i>0,\ c_i,y_i\in\{0,1\}\}.

    Only :math:`i\in\mathcal V` are evaluated. For those trials,

    .. math::

       q_i=P[a_2X_i^2+a_1X_i+a_0>0\mid s_i],
       \epsilon_i=\ell_{\rm long}\mathbf 1(c_i=1)
       +\ell_{\rm short}\mathbf 1(c_i=0),\qquad
       p_i=(1-\epsilon_i)q_i+\epsilon_i/2.

    All other output rows remain NaN.

    Parameter interpretation
    ------------------------
    ``choice_long`` is used only to determine whether a completed binary action
    exists; its 0/1 value never enters the prediction equation. RewardNaive and
    PunishNaive rows already have reconstructed binary choices and are therefore
    evaluated like Reward and Punish while retaining their original outcome label.

    Input units and shapes
    ----------------------
    ``duration_s``, ``category_long``, and ``choice_long`` are same-shaped
    floating tensors. ``category_long`` is 0 for a short-category stimulus and
    1 for a long-category stimulus.
    Missing choices are NaN. ``parameters`` is one pooled decoded M0 vector.

    Returned quantities and shapes
    ------------------------------
    Full-timeline tensors match the trial input shape and contain NaN at every
    skipped row, plus the boolean mask and named constrained parameters.

    Constraints
    -----------
    Valid choices are exactly 0 or 1 and valid durations are finite positive
    seconds. At least one valid trial is not required for prediction.

    Numerical stability
    -------------------
    Boolean indexing occurs before square roots, roots, CDFs, or lapse mixing.
    Thus NaN placeholders do not enter those autograd graphs.

    Causal restrictions
    -------------------
    Choice values select likelihood eligibility only. Predictions use current
    duration, fixed task moments, exact prior 0.5, and pooled parameters—never
    outcome, reward, block, history, session ID, or trial position.

    Important edge cases
    --------------------
    An all-invalid input returns all-NaN prediction tensors. The later session
    objective must skip such a session or report zero valid choices.

    Related tests
    -------------
    ``test_masked_prediction_skips_nondecisions_before_computation`` and
    ``test_choice_values_do_not_change_predictions``.
    """

    if not isinstance(duration_s, torch.Tensor) or not duration_s.is_floating_point():
        raise TypeError("duration_s must be a floating torch.Tensor")
    if not isinstance(choice_long, torch.Tensor) or not choice_long.is_floating_point():
        raise TypeError("choice_long must be a floating torch.Tensor with NaN for missing")
    if not isinstance(category_long, torch.Tensor) or not category_long.is_floating_point():
        raise TypeError("category_long must be a floating torch.Tensor")
    if duration_s.shape != choice_long.shape or duration_s.shape != category_long.shape:
        raise ValueError("duration_s, category_long, and choice_long must have the same shape")
    valid = (
        torch.isfinite(duration_s)
        & (duration_s > 0.0)
        & torch.isfinite(category_long)
        & ((category_long == 0.0) | (category_long == 1.0))
        & torch.isfinite(choice_long)
        & ((choice_long == 0.0) | (choice_long == 1.0))
    )

    if not bool(torch.any(valid).detach().cpu()):
        missing = torch.full_like(duration_s, torch.nan)
        return MaskedM0Predictions(
            probability_long=missing,
            non_lapse_probability_long=missing.clone(),
            positive_threshold_s=missing.clone(),
            decision_lower_boundary_s=missing.clone(),
            decision_upper_boundary_s=missing.clone(),
            decision_regime=("not_evaluated_invalid_choice",) * duration_s.numel(),
            sensory_sd_s=missing.clone(),
            valid_choice_mask=valid,
            parameters=parameters,
        )

    valid_duration = duration_s[valid]
    valid_category = category_long[valid]
    prior = fixed_neutral_prior(parameters.w)
    bayesian = bayesian_choice_probability(valid_duration, parameters, moments, prior)
    probability = lapse_mixture(
        bayesian.probability_long, valid_category, parameters
    )
    threshold = torch.broadcast_to(bayesian.positive_threshold_s, valid_duration.shape)
    lower = torch.broadcast_to(bayesian.decision_region.lower_boundary_s, valid_duration.shape)
    regimes = tuple(
        bayesian.decision_region.regime if included else "not_evaluated_invalid_choice"
        for included in valid.reshape(-1).tolist()
    )
    return MaskedM0Predictions(
        probability_long=_scatter_valid(probability, valid, duration_s),
        non_lapse_probability_long=_scatter_valid(
            bayesian.probability_long, valid, duration_s
        ),
        positive_threshold_s=_scatter_valid(threshold, valid, duration_s),
        decision_lower_boundary_s=_scatter_valid(lower, valid, duration_s),
        decision_upper_boundary_s=_scatter_valid(threshold, valid, duration_s),
        decision_regime=regimes,
        sensory_sd_s=_scatter_valid(bayesian.sensory_sd_s, valid, duration_s),
        valid_choice_mask=valid,
        parameters=parameters,
    )


def predict_m0_trials(
    duration_s: torch.Tensor,
    category_long: torch.Tensor,
    choice_long: torch.Tensor,
    raw_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> MaskedM0Predictions:
    r"""Decode one raw pooled vector and predict an omission-safe trial timeline.

    Purpose
    -------
    Provide a convenient Part I entry point for a single trial tensor. Part II
    objectives instead decode once and call
    :func:`predict_m0_trials_with_parameters` separately for every session.

    Mathematical definition
    -----------------------
    The raw vector is decoded as
    :math:`(\widetilde w,\widetilde s_0,\widetilde\ell_short,
    \widetilde ell_long,\widetilde{\log\kappa})
    \mapsto(w,s_0,\ell_short,ell_long,\kappa)`, then valid rows use

    .. math::

       p_i=(1-\epsilon_i)\Phi\left(
       \frac{s_i-T}{\sqrt{ws_i^2+s_0^2}}\right)+\epsilon_i/2.

    Parameter interpretation
    ------------------------
    One raw vector represents the five parameters pooled across a mouse.

    Input units and shapes
    ----------------------
    Trial tensors share shape and use seconds/NaN conventions. Raw parameters
    have exactly shape ``(5,)``.

    Returned quantities and shapes
    ------------------------------
    Returns :class:`MaskedM0Predictions` with full-timeline diagnostic tensors.

    Constraints
    -----------
    Parameter constraints are applied only by :func:`decode_raw_parameters`.

    Numerical stability
    -------------------
    Missing decisions are indexed out before scientific calculations.

    Causal restrictions
    -------------------
    Choice values determine eligibility only and do not enter probabilities.

    Important edge cases
    --------------------
    An all-invalid timeline returns all-NaN predictions.

    Related tests
    -------------
    ``test_masked_prediction_skips_nondecisions_before_computation``.
    """

    if raw_parameters.shape != (5,):
        raise ValueError(
            "M0 currently requires one pooled raw parameter vector of shape (5,)"
        )
    parameters = decode_raw_parameters(
        raw_parameters,
        lapse_max=lapse_max,
        epsilon=epsilon,
    )
    return predict_m0_trials_with_parameters(
        duration_s,
        category_long,
        choice_long,
        parameters,
        moments,
    )
