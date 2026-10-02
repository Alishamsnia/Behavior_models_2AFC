"""Shared full Gaussian decision-region integration for M0 and context models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.task_moments import TaskMoments
from .bayesian import (
    CategoryVariances,
    LogOddsCoefficients,
    category_variances,
    log_odds_coefficients,
    sensory_standard_deviation,
)
from .parameters import DecodedParameters


@dataclass(frozen=True)
class GeneralDecisionRegion:
    """Scalar solution of ``a2*x^2+a1*x+a0 > 0``.

    Infinite boundaries represent an unbounded interval and NaNs represent a
    constant all-long or all-short region.  ``regime`` is diagnostic only; the
    probability tensor remains differentiable within the selected regime.
    """

    lower_boundary_s: torch.Tensor
    upper_boundary_s: torch.Tensor
    regime: str | tuple[str, ...]
    long_region: str | tuple[str, ...]
    discriminant: torch.Tensor | None


    @property
    def threshold_s(self) -> torch.Tensor:
        """Compatibility alias for the upper boundary, not a one-threshold rule."""
        return self.upper_boundary_s

    @property
    def solver_regime(self) -> str | tuple[str, ...]:
        """Compatibility alias for the full-region regime."""
        return self.regime

    @property
    def negative_root_policy(self) -> str:
        """All real measurement regions are integrated, including negative tails."""
        return "integrate_all_real_measurement_regions"


@dataclass(frozen=True)
class M1BayesianChoiceResult:
    """M1 non-lapse choice probability and full boundary diagnostics."""

    probability_long: torch.Tensor
    decision_region: GeneralDecisionRegion
    sensory_sd_s: torch.Tensor
    category_variances: CategoryVariances
    coefficients: LogOddsCoefficients


def _truth(value: torch.Tensor) -> bool:
    """Read a scalar tensor only to select its analytic integration regime."""

    return bool((value if value.ndim == 0 else torch.all(value)).detach().cpu())


def integrate_general_decision_region(
    duration_s: torch.Tensor,
    sensory_sd_s: torch.Tensor,
    coefficients: LogOddsCoefficients,
    *,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> tuple[torch.Tensor, GeneralDecisionRegion]:
    r"""Integrate ``P(a2*X^2+a1*X+a0 > 0)`` for Gaussian ``X``.

    Purpose
    -------
    Evaluate M1's complete Bayesian action region without requiring a positive
    threshold or discarding a mathematically valid lower measurement tail.

    Mathematical definition
    -----------------------
    For :math:`X\sim N(s,\sigma^2)` and

    .. math::

       \Lambda(X)=a_2X^2+a_1X+a_0,
       \qquad q=P[\Lambda(X)>0],

    integrate outside sorted roots when ``a2>0`` and between them when
    ``a2<0``; linear and constant-sign limits are handled explicitly.

    Parameter interpretation
    ------------------------
    Coefficients combine fixed task moments, pooled sensory parameters, and the
    causal pre-choice prior. Boundaries are derived, never fitted parameters.

    Input units and shapes
    ----------------------
    Duration and sensory SD are broadcastable tensors in seconds. Scalar
    coefficients use one vectorized integration; batched coefficients are
    grouped by decision regime and integrated with the same vectorized solver.

    Returned quantities and shapes
    ------------------------------
    Returns broadcast-shaped non-lapse probability ``q`` and a structured region with
    lower/upper boundaries, regime, sign region, and optional discriminant.

    Constraints
    -----------
    Sensory SD must be positive upstream; positive numerical tolerances are
    required. Batched regime labels are tuples in flattened row-major order.

    Numerical stability
    -------------------
    Near-zero quadratic terms use the linear limit and near-zero negative
    discriminants are clamped only before the square root.

    Causal restrictions
    -------------------
    This function receives only current duration, sensory SD, and already
    formed coefficients; category feedback and block truth are unavailable.

    Important edge cases
    --------------------
    No-real-root and double-root regimes return exact all-long/all-short
    probabilities. Constant ties use the declared strict short decision.

    Related tests
    -------------
    ``test_general_decision_regions_match_normal_probabilities`` and
    ``test_general_solver_handles_double_no_root_and_constant_cases``.
    """

    duration_s, sensory_sd_s = torch.broadcast_tensors(duration_s, sensory_sd_s)
    if quadratic_tolerance <= 0.0 or discriminant_tolerance <= 0.0:
        raise ValueError("Decision-region tolerances must be positive")
    a2, a1, a0 = coefficients.a2, coefficients.a1, coefficients.a0
    if a2.ndim != 0 or a1.ndim != 0 or a0.ndim != 0:
        return _integrate_batched_decision_regions(
            duration_s, sensory_sd_s, coefficients,
            quadratic_tolerance=quadratic_tolerance,
            discriminant_tolerance=discriminant_tolerance,
        )
    return _integrate_homogeneous_decision_region(
        duration_s, sensory_sd_s, coefficients,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    )


def _integrate_batched_decision_regions(
    duration_s: torch.Tensor,
    sensory_sd_s: torch.Tensor,
    coefficients: LogOddsCoefficients,
    *,
    quadratic_tolerance: float,
    discriminant_tolerance: float,
) -> tuple[torch.Tensor, GeneralDecisionRegion]:
    """Group broadcast coefficients into at most ten analytic sign regimes.

    Only each selected group enters its branch's divisions and square roots.
    This avoids NaN gradients from evaluating invalid unselected branches with
    torch.where, and avoids one Python/CDF/autograd call per trial.
    """
    duration, sd, a2, a1, a0 = torch.broadcast_tensors(
        duration_s, sensory_sd_s, coefficients.a2, coefficients.a1, coefficients.a0
    )
    shape = duration.shape
    # broadcast_tensors expands shapes but does not promote mixed dtypes.
    # Match arithmetic promotion before allocating index_copy destinations.
    dtype = duration.dtype
    for tensor in (sd, a2, a1, a0):
        dtype = torch.promote_types(dtype, tensor.dtype)
    duration, sd, a2, a1, a0 = (
        x.to(dtype=dtype).reshape(-1) for x in (duration, sd, a2, a1, a0)
    )
    # Classification is diagnostic control flow, exactly like scalar _truth.
    with torch.no_grad():
        discriminant = a1.square() - 4.0 * a2 * a0
        linear_key = torch.where(a1.abs() <= quadratic_tolerance,
                                 (a0 <= 0).long(), 2 + (a1 < 0).long())
        quadratic_key = torch.where(
            discriminant < -discriminant_tolerance, 4,
            torch.where(discriminant.abs() <= discriminant_tolerance, 6, 8),
        ) + (a2 < 0).long()
        keys = torch.where(a2.abs() <= quadratic_tolerance, linear_key, quadratic_key)
    probability = torch.zeros_like(duration)
    lower = torch.full_like(duration, torch.nan)
    upper = lower.clone()
    discriminants = lower.clone()
    regimes = [""] * duration.numel()
    long_regions = [""] * duration.numel()
    for key in torch.unique(keys).tolist():
        indices = torch.nonzero(keys == key, as_tuple=False).flatten()
        q, region = _integrate_homogeneous_decision_region(
            duration[indices], sd[indices],
            LogOddsCoefficients(a2[indices], a1[indices], a0[indices]),
            quadratic_tolerance=quadratic_tolerance,
            discriminant_tolerance=discriminant_tolerance,
        )
        probability = probability.index_copy(0, indices, q)
        lower = lower.index_copy(0, indices, region.lower_boundary_s.expand_as(q))
        upper = upper.index_copy(0, indices, region.upper_boundary_s.expand_as(q))
        if region.discriminant is not None:
            discriminants = discriminants.index_copy(0, indices, region.discriminant)
        for index in indices.tolist():
            regimes[index] = region.regime
            long_regions[index] = region.long_region
    return probability.reshape(shape), GeneralDecisionRegion(
        lower.reshape(shape), upper.reshape(shape), tuple(regimes),
        tuple(long_regions), discriminants.reshape(shape),
    )


def _integrate_homogeneous_decision_region(
    duration_s: torch.Tensor,
    sensory_sd_s: torch.Tensor,
    coefficients: LogOddsCoefficients,
    *,
    quadratic_tolerance: float,
    discriminant_tolerance: float,
) -> tuple[torch.Tensor, GeneralDecisionRegion]:
    """Original scalar branch equations, also applied to same-regime batches."""
    a2, a1, a0 = coefficients.a2, coefficients.a1, coefficients.a0
    normal = torch.distributions.Normal(duration_s, sensory_sd_s)
    nan = duration_s.new_tensor(torch.nan)
    neg_inf = duration_s.new_tensor(-torch.inf)
    pos_inf = duration_s.new_tensor(torch.inf)

    if _truth(torch.abs(a2) <= quadratic_tolerance):
        if _truth(torch.abs(a1) <= quadratic_tolerance):
            all_long = _truth(a0 > 0.0)
            probability = torch.full_like(duration_s, 1.0 if all_long else 0.0) + a0 * 0.0
            return probability, GeneralDecisionRegion(
                nan, nan, "constant", "all real x" if all_long else "empty", None
            )
        root = -a0 / a1
        cdf = normal.cdf(root)
        if _truth(a1 > 0.0):
            return 1.0 - cdf, GeneralDecisionRegion(
                root, pos_inf, "linear", "x > lower_boundary", None
            )
        return cdf, GeneralDecisionRegion(
            neg_inf, root, "linear", "x < upper_boundary", None
        )

    discriminant = a1.square() - 4.0 * a2 * a0
    if _truth(discriminant < -discriminant_tolerance):
        all_long = _truth(a2 > 0.0)
        probability = torch.full_like(duration_s, 1.0 if all_long else 0.0) + discriminant * 0.0
        return probability, GeneralDecisionRegion(
            nan,
            nan,
            "quadratic_no_real_roots",
            "all real x" if all_long else "empty",
            discriminant,
        )

    if _truth(torch.abs(discriminant) <= discriminant_tolerance):
        # A double (or near-double) root: both quadratic roots coincide at
        # -a1/(2*a2), computed directly with no square root involved. Doing
        # it this way -- rather than computing sqrt(discriminant) and then
        # discarding it via `* 0.0` -- matters for backward: sqrt's local
        # derivative 1/(2*sqrt(x)) diverges as x -> 0, and multiplying an
        # infinite local gradient by an exact-zero incoming gradient still
        # yields NaN (0 * inf), which would otherwise poison every fitted
        # parameter's gradient for this trial even though the forward
        # probability is perfectly well defined. Routing the constant
        # output through `discriminant` instead (whose own gradient is
        # finite everywhere) keeps the zero-measure-boundary property
        # (probability does not depend on which side of the point we are on)
        # without the singular intermediate.
        all_long = _truth(a2 > 0.0)
        root = -a1 / (2.0 * a2)
        probability = torch.full_like(duration_s, 1.0 if all_long else 0.0) + discriminant * 0.0
        return probability, GeneralDecisionRegion(
            root,
            root,
            "quadratic_double_root",
            "all except boundary" if all_long else "empty except boundary",
            discriminant,
        )

    safe_discriminant = torch.clamp_min(discriminant, 0.0)
    square_root = torch.sqrt(safe_discriminant)
    root_a = (-a1 - square_root) / (2.0 * a2)
    root_b = (-a1 + square_root) / (2.0 * a2)
    lower = torch.minimum(root_a, root_b)
    upper = torch.maximum(root_a, root_b)
    lower_cdf = normal.cdf(lower)
    upper_cdf = normal.cdf(upper)
    if _truth(a2 > 0.0):
        probability = lower_cdf + (1.0 - upper_cdf)
        long_region = "x < lower_boundary or x > upper_boundary"
    else:
        probability = upper_cdf - lower_cdf
        long_region = "lower_boundary < x < upper_boundary"
    return probability, GeneralDecisionRegion(
        lower, upper, "quadratic_two_roots", long_region, discriminant
    )


def m1_bayesian_choice_probability(
    duration_s: torch.Tensor,
    parameters: DecodedParameters,
    moments: TaskMoments,
    prior_long: torch.Tensor,
    *,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> M1BayesianChoiceResult:
    r"""Return M1's exact Gaussian decision-region probability for one trial.

    Purpose
    -------
    Compose the shared sensory equations with M1's robust region integration.

    Mathematical definition
    -----------------------
    The function calculates

    .. math::

       \sigma(s)=\sqrt{ws^2+s_0^2},\qquad
       q=P[\Lambda(X;\pi)>0\mid s].

    Parameter interpretation
    ------------------------
    ``prior_long`` is the causal HMM prior; ``w`` and ``s0`` are pooled sensory
    parameters. Lapse mixing is deliberately performed by the caller.

    Input units and shapes
    ----------------------
    Duration is a scalar second value; parameters, prior, and outputs are
    scalar float64 tensors in registered fitting.

    Returned quantities and shapes
    ------------------------------
    Returns scalar ``q``, region diagnostics, sensory SD, variances, and
    log-odds coefficients.

    Constraints
    -----------
    Duration must be finite and positive and the numerical prior epsilon must
    lie in ``(0,0.5)``.

    Numerical stability
    -------------------
    The prior is symmetrically clamped to ``[epsilon,1-epsilon]`` only for its
    logit. Registered true-rate M1 priors naturally remain in ``[0.2,0.8]``.

    Causal restrictions
    -------------------
    No current category, reward, true block, run length, or choice is accepted.

    Important edge cases
    --------------------
    A missing positive root is not an error because the delegated full-region
    solver integrates all valid sign regimes.

    Related tests
    -------------
    ``test_all_five_objective_gradients_match_finite_differences`` and the
    general decision-region tests.
    """

    if not 0.0 < prior_probability_epsilon < 0.5:
        raise ValueError("prior_probability_epsilon must lie in (0,0.5)")
    prior = torch.clamp(
        prior_long,
        min=prior_probability_epsilon,
        max=1.0 - prior_probability_epsilon,
    )
    variances = category_variances(parameters, moments)
    coefficients = log_odds_coefficients(
        variances, moments, prior, parameters.log_kappa
    )
    sensory_sd = sensory_standard_deviation(duration_s, parameters)
    probability, region = integrate_general_decision_region(
        duration_s,
        sensory_sd,
        coefficients,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    )
    return M1BayesianChoiceResult(
        probability_long=probability,
        decision_region=region,
        sensory_sd_s=sensory_sd,
        category_variances=variances,
        coefficients=coefficients,
    )
