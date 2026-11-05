r"""Shared M0-sensory + previous-trial-bias evaluator core.

This module holds the ONE readout/mixing/masking/likelihood implementation
shared by ``m1_previous_action_bias_v1`` (sensory_actionSingle_lapse: the
bias state is the mouse's own last valid choice) and
``m1_previous_wsls_bias_v1`` (sensory_wslsSingle_lapse: the bias state is the
last feedback-valid true category). The two models differ ONLY in
``history_driver``; every other line is shared so they can never drift.

It is ``M2_dynamic_previous_action_bias_v1`` /
``M2_dynamic_previous_wsls_bias_v1`` with the block/run-length HSMM removed:
the sensory prior is M0's fixed neutral ``0.5`` instead of the HSMM block
prior, and the same eta-mixing and lapse order are kept:

.. math::

   r_t &= \mathrm{BayesianReadout}(s_t;\,w,s_0,\kappa,\ \text{prior}=0.5)
          \quad\text{(M0's own readout; prior fixed)}\\
   m_t &= (1-\eta)\,r_t+\eta\,\pi_t\\
   p_t &= (1-\epsilon_t)\,m_t+\epsilon_t/2,\qquad
   \epsilon_t=\ell_{\mathrm{long}}\mathbf 1(C_t=1)+\ell_{\mathrm{short}}\mathbf 1(C_t=0)\\
   \pi_1 &= 0.5,\qquad
   \pi_{t+1}=\begin{cases}X_t,&G_t\\ \pi_t,&\text{otherwise}\end{cases}

with ``alpha`` fixed at exactly 1 (the delta rule collapses to assignment):

- ``history_driver="action"``: :math:`X_t=A_t` (``choice_long``), gate
  :math:`G_t` = choice validity (finite, binary), independent of
  ``feedback_valid``, category validity and duration validity -- the gate
  ``M1_action_kernel_v1``/``M2_dynamic_previous_action_bias_v1`` use.
- ``history_driver="category"``: :math:`X_t=C_t` (``category_long``), gate
  :math:`G_t` = ``feedback_valid``, independent of choice validity -- the
  gate ``M1_stimulus_kernel_v1``/``M2_dynamic_previous_wsls_bias_v1`` use.

Why the bias is mixed, not fed into the prior
---------------------------------------------
Feeding a fixed-alpha state :math:`\pi_t\in\{0,1\}` into the Bayesian prior
(i.e. ``M1_action_kernel_v1``/``M1_stimulus_kernel_v1`` with alpha=1) is
degenerate: the prior is clamped to ``[1e-10, 1-1e-10]`` before its logit,
which shifts the decision boundary by about 23 log-odds, so the readout is
exactly 0 or 1 for every duration and the ``w``/``s0`` gradients vanish.
Output mixing keeps the sensory readout intact.

Likelihood eligibility (one mask for both drivers)
--------------------------------------------------
Finite positive duration AND valid binary choice AND finite binary current
category (``lapse_mixture`` requires a binary category), matching
``M2_dynamic_previous_wsls_bias_v1``'s mask. In every registered real
dataset a valid choice always carries a binary category, so this only
matters for edge cases.

Vectorization
-------------
Because alpha is fixed, :math:`\pi_t` depends only on observed data (never on
a fitted parameter), and because the sensory prior is a constant, the
readout has scalar decision coefficients for the whole session. The state
trajectory is therefore built once in a plain loop and the readout, mixture
and likelihood are evaluated in one vectorized call -- mathematically
identical to a per-trial loop, and the same vectorized integration M0 uses.

Causal restrictions
-------------------
Trial :math:`t`'s own choice and category never enter :math:`\pi_t`; the
current category enters only the current lapse rate and (as for every model
in this repository) nothing else about the current trial is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import M1SessionTrials
from m0_sensory_v1.model.parameters import DecodedParameters


HistoryDriver = Literal["action", "category"]
FIXED_SENSORY_PRIOR_LONG = 0.5

__all__ = [
    "FIXED_SENSORY_PRIOR_LONG",
    "HistoryDriver",
    "SensoryPreviousBiasPredictions",
    "SensoryPreviousBiasSessionEvaluation",
    "evaluate_sensory_previous_bias_session",
    "previous_bias_state_trajectory",
]


@dataclass(frozen=True)
class SensoryPreviousBiasPredictions:
    """Full-timeline predictions, previous-trial bias state, and readout diagnostics."""

    probability_long: torch.Tensor
    non_lapse_probability_long: torch.Tensor
    bias_mixed_probability_long: torch.Tensor
    non_bias_probability_long: torch.Tensor
    prior_long: torch.Tensor
    history_state: torch.Tensor
    post_update_history_state: torch.Tensor
    positive_threshold_s: torch.Tensor
    decision_lower_boundary_s: torch.Tensor
    decision_upper_boundary_s: torch.Tensor
    decision_regime: tuple[str, ...]
    sensory_sd_s: torch.Tensor
    valid_choice_mask: torch.Tensor
    history_update_mask: torch.Tensor


@dataclass(frozen=True)
class SensoryPreviousBiasSessionEvaluation:
    """Differentiable session choice NLL and prediction trajectory."""

    mouse_id: str
    session_id: str
    split_role: str
    negative_log_likelihood: torch.Tensor
    valid_trial_count: int
    per_trial_negative_log_likelihood: torch.Tensor
    predictions: SensoryPreviousBiasPredictions
    duration_s: torch.Tensor
    choice_long: torch.Tensor


def _binary(values: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(values) & ((values == 0.0) | (values == 1.0))


def previous_bias_state_trajectory(
    session: M1SessionTrials,
    history_driver: HistoryDriver,
    *,
    initial_pi: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return ``(pi_pre, pi_post, update_mask)`` for one session.

    ``pi_pre[t]`` is the state used to predict trial ``t``; ``pi_post[t]`` the
    state carried into trial ``t+1``. With alpha fixed at 1 the update is the
    exact assignment ``pi_next = X_t`` (no floating-point arithmetic), so
    ``pi_pre[t]`` equals the driver value of the last gated trial before
    ``t`` exactly, or ``initial_pi`` if there is none. The state is reset at
    every call (one call per physical session).
    """

    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")
    if history_driver == "action":
        driver = session.choice_long
        update_mask = _binary(session.choice_long)
    elif history_driver == "category":
        driver = session.category_long
        update_mask = session.feedback_valid.to(torch.bool)
    else:
        raise ValueError(f"Unknown history_driver: {history_driver!r}")
    pre: list[float] = []
    post: list[float] = []
    current = float(initial_pi)
    for value, gated in zip(driver.detach().cpu().tolist(), update_mask.detach().cpu().tolist()):
        pre.append(current)
        if gated:
            if value not in (0.0, 1.0):
                raise ValueError("A gated previous-bias update requires a binary driver value")
            current = float(value)
        post.append(current)
    dtype, device = session.duration_s.dtype, session.duration_s.device
    return (
        torch.tensor(pre, dtype=dtype, device=device),
        torch.tensor(post, dtype=dtype, device=device),
        update_mask,
    )


def evaluate_sensory_previous_bias_session(
    session: M1SessionTrials,
    common: DecodedParameters,
    eta: torch.Tensor,
    moments: TaskMoments,
    history_driver: HistoryDriver,
    *,
    initial_pi: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    log_probability_epsilon: float = 1e-12,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> SensoryPreviousBiasSessionEvaluation:
    r"""Evaluate one session of the M0-sensory + previous-trial-bias model.

    Parameter interpretation
    ------------------------
    ``common`` holds M0's five decoded sensory/lapse parameters and ``eta``
    the 0-dim decoded bias-mixing weight ``sigmoid(eta_logit)``.

    Returned quantities and shapes
    ------------------------------
    Scalar choice NLL plus full-timeline (length ``T``) tensors. Trials that
    are not likelihood-eligible carry NaN probabilities/diagnostics and the
    regime label ``"not_evaluated_invalid_choice"``; the history state is
    defined on every trial.

    Important edge cases
    --------------------
    Zero eligible trials give an NLL of exactly 0.0 built from the
    parameters (gradients stay defined) and an all-NaN per-trial row.

    Related tests
    -------------
    ``test_eta_zero_reproduces_m0_exactly``,
    ``test_matches_per_trial_reference_loop``,
    ``test_history_driver_and_gate``.
    """

    if not 0.0 < log_probability_epsilon < 0.5:
        raise ValueError("log_probability_epsilon must lie in (0,0.5)")
    if eta.ndim != 0:
        raise ValueError("eta must be a 0-dimensional tensor")
    pi_pre, pi_post, update_mask = previous_bias_state_trajectory(
        session, history_driver, initial_pi=initial_pi
    )
    valid_mask = (
        torch.isfinite(session.duration_s)
        & (session.duration_s > 0.0)
        & _binary(session.choice_long)
        & _binary(session.category_long)
    )
    length = session.duration_s.numel()
    nan_row = torch.full_like(session.duration_s, torch.nan)
    prior_row = torch.full_like(session.duration_s, FIXED_SENSORY_PRIOR_LONG)
    valid_count = int(valid_mask.sum().detach().cpu())

    if valid_count == 0:
        zero = common.as_tensor().sum() * 0.0 + eta * 0.0
        predictions = SensoryPreviousBiasPredictions(
            probability_long=nan_row, non_lapse_probability_long=nan_row.clone(),
            bias_mixed_probability_long=nan_row.clone(), non_bias_probability_long=nan_row.clone(),
            prior_long=prior_row, history_state=pi_pre, post_update_history_state=pi_post,
            positive_threshold_s=nan_row.clone(), decision_lower_boundary_s=nan_row.clone(),
            decision_upper_boundary_s=nan_row.clone(),
            decision_regime=("not_evaluated_invalid_choice",) * length,
            sensory_sd_s=nan_row.clone(), valid_choice_mask=valid_mask,
            history_update_mask=update_mask,
        )
        return SensoryPreviousBiasSessionEvaluation(
            mouse_id=session.mouse_id, session_id=session.session_id,
            split_role=session.split_role, negative_log_likelihood=zero,
            valid_trial_count=0, per_trial_negative_log_likelihood=nan_row.clone(),
            predictions=predictions, duration_s=session.duration_s,
            choice_long=session.choice_long,
        )

    durations = session.duration_s[valid_mask]
    categories = session.category_long[valid_mask]
    readout = m1_bayesian_choice_probability(
        durations,
        common,
        moments,
        durations.new_tensor(FIXED_SENSORY_PRIOR_LONG),
        prior_probability_epsilon=prior_probability_epsilon,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    )
    r_t = readout.probability_long
    m_t = (1.0 - eta) * r_t + eta * pi_pre[valid_mask]
    probability = lapse_mixture(m_t, categories, common)
    non_bias = lapse_mixture(r_t, categories, common)
    safe = torch.clamp(probability, min=log_probability_epsilon, max=1.0 - log_probability_epsilon)
    observed = session.choice_long[valid_mask]
    valid_nll = -(observed * torch.log(safe) + (1.0 - observed) * torch.log1p(-safe))

    def scatter(values: torch.Tensor) -> torch.Tensor:
        return nan_row.clone().masked_scatter(valid_mask, values.expand(durations.shape))

    region = readout.decision_region
    regime = region.regime
    regimes_valid = (regime,) * valid_count if isinstance(regime, str) else tuple(regime)
    regimes_iter = iter(regimes_valid)
    decision_regime = tuple(
        next(regimes_iter) if bool(flag) else "not_evaluated_invalid_choice"
        for flag in valid_mask.detach().cpu().tolist()
    )
    predictions = SensoryPreviousBiasPredictions(
        probability_long=scatter(probability),
        non_lapse_probability_long=scatter(r_t),
        bias_mixed_probability_long=scatter(m_t),
        non_bias_probability_long=scatter(non_bias),
        prior_long=prior_row,
        history_state=pi_pre,
        post_update_history_state=pi_post,
        positive_threshold_s=scatter(region.upper_boundary_s),
        decision_lower_boundary_s=scatter(region.lower_boundary_s),
        decision_upper_boundary_s=scatter(region.upper_boundary_s),
        decision_regime=decision_regime,
        sensory_sd_s=scatter(readout.sensory_sd_s),
        valid_choice_mask=valid_mask,
        history_update_mask=update_mask,
    )
    return SensoryPreviousBiasSessionEvaluation(
        mouse_id=session.mouse_id,
        session_id=session.session_id,
        split_role=session.split_role,
        negative_log_likelihood=valid_nll.sum(),
        valid_trial_count=valid_count,
        per_trial_negative_log_likelihood=scatter(valid_nll),
        predictions=predictions,
        duration_s=session.duration_s,
        choice_long=session.choice_long,
    )
