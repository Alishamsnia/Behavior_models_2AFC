"""Differentiable positive-threshold Bayesian sensory readout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.task_moments import TaskMoments
from .parameters import DecodedParameters


@dataclass(frozen=True)
class CategoryVariances:
    """Gaussian-approximation variances for short and long categories."""

    short: torch.Tensor
    long: torch.Tensor


@dataclass(frozen=True)
class LogOddsCoefficients:
    """Coefficients of ``a2*x**2 + a1*x + a0``."""

    a2: torch.Tensor
    a1: torch.Tensor
    a0: torch.Tensor


@dataclass(frozen=True)
class PositiveDecisionRegion:
    """Structured positive-threshold region with no negative-root output."""

    threshold_s: torch.Tensor
    solver_regime: str
    long_region: str
    negative_root_policy: str
    discriminant: torch.Tensor | None


@dataclass(frozen=True)
class BayesianChoiceResult:
    """Non-lapse probability and diagnostics for evaluated valid trials."""

    probability_long: torch.Tensor
    decision_region: PositiveDecisionRegion
    sensory_sd_s: torch.Tensor
    category_variances: CategoryVariances
    coefficients: LogOddsCoefficients

    @property
    def positive_threshold_s(self) -> torch.Tensor:
        """Return the structured region's positive threshold for compatibility."""

        return self.decision_region.threshold_s


def _moment_tensor(value: float, like: torch.Tensor) -> torch.Tensor:
    """Create a fixed moment on the dtype/device of an autograd tensor."""

    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def category_variances(
    parameters: DecodedParameters,
    moments: TaskMoments,
) -> CategoryVariances:
    r"""Calculate the category-conditioned sensory approximation variances.

    Purpose
    -------
    Combine fixed empirical task moments with the fitted sensory-noise terms.

    Mathematical definition
    -----------------------
    For category :math:`k\in\{0,1\}`,

    .. math::

       v_k = \tau_k^2 + w\mu_k^2 + s_0^2.

    Parameter interpretation
    ------------------------
    :math:`\mu_k` and :math:`\tau_k^2` are fixed real-data task quantities;
    :math:`w` and :math:`s_0` are pooled mouse-level fitted parameters.

    Known approximation, not a bug
    -------------------------------
    If the measurement model :math:`\operatorname{Var}(X\mid S)=wS^2+s_0^2`
    is read literally as a hierarchical generative model -- :math:`S` itself
    drawn from category :math:`k` with mean :math:`\mu_k` and variance
    :math:`\tau_k^2`, then :math:`X` measured given that :math:`S` -- the
    law of total variance gives the exact category-conditioned variance

    .. math::

       \operatorname{Var}(X\mid C=k) = E[\operatorname{Var}(X\mid S)]
           + \operatorname{Var}(E[X\mid S])
         = w(\tau_k^2+\mu_k^2) + s_0^2 + \tau_k^2
         = (1+w)\tau_k^2 + w\mu_k^2 + s_0^2,

    which has an extra :math:`w\tau_k^2` term the registered :math:`v_k`
    above omits. This is the model's registered, intentional approximation
    (not derived here as the exact hierarchical variance), presumably
    chosen because :math:`\tau_k^2` is an empirical, already-pooled task
    quantity rather than an explicit per-trial latent :math:`S` in this
    model's own causal structure. Whether the omitted :math:`w\tau_k^2`
    term is material depends on the fitted :math:`w` and each category's
    :math:`\tau_k^2/\mu_k^2` ratio -- check against this model's own
    fitted values before assuming it is negligible.

    Input units and shapes
    ----------------------
    Means and ``s0`` are seconds; variances are seconds squared. M0 fitting
    uses scalar parameters, though leading parameter batch shapes are retained.

    Returned quantities and shapes
    ------------------------------
    Returns ``short`` and ``long`` variance tensors with parameter batch shape.

    Constraints
    -----------
    Decoded ``w`` and ``s0`` must be positive; empirical variances must be
    nonnegative.

    Numerical stability
    -------------------
    All operations retain the input Torch dtype; registered fitting uses
    ``float64``.

    Causal restrictions
    -------------------
    No choice, outcome, history, block, or session field enters this equation.

    Important edge cases
    --------------------
    Zero empirical task variance is permitted because positive ``s0`` keeps
    both category variances positive.

    Related tests
    -------------
    ``test_reference_category_variances_coefficients_and_roots``.
    """

    mu0 = _moment_tensor(moments.mu_short_s, parameters.w)
    mu1 = _moment_tensor(moments.mu_long_s, parameters.w)
    tau0_sq = _moment_tensor(moments.var_short_s2, parameters.w)
    tau1_sq = _moment_tensor(moments.var_long_s2, parameters.w)
    return CategoryVariances(
        short=tau0_sq + parameters.w * mu0.square() + parameters.s0.square(),
        long=tau1_sq + parameters.w * mu1.square() + parameters.s0.square(),
    )


def log_odds_coefficients(
    variances: CategoryVariances,
    moments: TaskMoments,
    prior_long: torch.Tensor,
    log_kappa: torch.Tensor,
) -> LogOddsCoefficients:
    r"""Form the quadratic Gaussian-category posterior log-odds coefficients.

    Purpose
    -------
    Represent :math:`\log[f_1(x)/f_0(x)]+\operatorname{logit}(\pi)` as a
    quadratic in the unobserved internal measurement ``x``.

    Mathematical definition
    -----------------------
    .. math::

       a_2 &= \frac{1}{2v_0}-\frac{1}{2v_1},\\
       a_1 &= \frac{\mu_1}{v_1}-\frac{\mu_0}{v_0},\\
       a_0 &= \frac{\mu_0^2}{2v_0}-\frac{\mu_1^2}{2v_1}
              +\frac{1}{2}\log\frac{v_0}{v_1}
              +\operatorname{logit}(\pi)-\log\kappa.

    Parameter interpretation
    ------------------------
    ``prior_long`` is a category prior, not a response bias. ``log_kappa`` is
    the pooled internal decision-cost bias. M0 supplies prior ``0.5``; later
    models may provide trial-varying priors.

    Input units and shapes
    ----------------------
    Variances are seconds squared. ``prior_long`` may be scalar or have one
    element per valid trial and must share dtype/device with the variances.

    Returned quantities and shapes
    ------------------------------
    ``a2`` and ``a1`` follow parameter shape; ``a0`` broadcasts with the prior.

    Constraints
    -----------
    Category variances must be positive and priors must lie strictly in ``(0,1)``.

    Numerical stability
    -------------------
    The prior logit is calculated as ``log(pi)-log1p(-pi)`` in ``float64``.

    Causal restrictions
    -------------------
    The common function accepts only a prior. M0's prior function cannot
    inspect block, category, history, reward, trial position, or session ID.

    Important edge cases
    --------------------
    Unequal variances make the boundary quadratic. A near-zero quadratic term
    is handled by the threshold solver's linear limit.

    Related tests
    -------------
    ``test_reference_category_variances_coefficients_and_roots`` and
    ``test_fixed_neutral_prior_is_exactly_half``.
    """

    if torch.any(variances.short <= 0.0) or torch.any(variances.long <= 0.0):
        raise ValueError("Category variances must be positive")
    if torch.any(prior_long <= 0.0) or torch.any(prior_long >= 1.0):
        raise ValueError("prior_long must lie strictly in (0, 1)")
    if torch.any(~torch.isfinite(log_kappa)):
        raise ValueError("log_kappa must be finite")

    mu0 = _moment_tensor(moments.mu_short_s, variances.short)
    mu1 = _moment_tensor(moments.mu_long_s, variances.short)
    logit_prior = torch.log(prior_long) - torch.log1p(-prior_long)
    a2 = 0.5 / variances.short - 0.5 / variances.long
    a1 = mu1 / variances.long - mu0 / variances.short
    a0 = (
        mu0.square() / (2.0 * variances.short)
        - mu1.square() / (2.0 * variances.long)
        + 0.5 * torch.log(variances.short / variances.long)
        + logit_prior
        - log_kappa
    )
    return LogOddsCoefficients(a2=a2, a1=a1, a0=a0)


def solve_positive_decision_region(
    coefficients: LogOddsCoefficients,
    *,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> PositiveDecisionRegion:
    r"""Return the structured positive-threshold region used by this project.

    Purpose
    -------
    Reduce the Gaussian log-odds boundary to a single physically meaningful,
    positive duration threshold. The negative mathematical root is deliberately
    discarded according to the registered M0 implementation decision.

    Mathematical definition
    -----------------------
    For

    .. math::

       a_2x^2+a_1x+a_0=0,\qquad
       \Delta=a_1^2-4a_2a_0,

    calculate

    .. math::

       x_{\pm}=\frac{-a_1\pm\sqrt{\Delta}}{2a_2}

    and return the unique root :math:`T>0`. If :math:`|a_2|` is numerically
    small, use the linear limit :math:`T=-a_0/a_1`.

    Parameter interpretation
    ------------------------
    ``T`` is the internal-measurement boundary: the non-lapse action is long
    exactly when :math:`X>T`.

    Input units and shapes
    ----------------------
    Coefficients may contain a scalar or trial-shaped ``a0``. Returned ``T``
    broadcasts to that shape and is measured in seconds.

    Returned quantities and shapes
    ------------------------------
    A :class:`PositiveDecisionRegion` containing the differentiable positive
    threshold, numerical solver regime, upper-threshold rule, explicit negative
    root policy, and discriminant. No negative root value is returned.

    Constraints
    -----------
    Every evaluated prior/parameter combination must yield exactly one positive
    real root, or a positive linear threshold.

    Numerical stability
    -------------------
    Discriminants within ``discriminant_tolerance`` below zero are clamped to
    zero before ``sqrt``. The linear limit avoids division by near-zero ``a2``.

    Causal restrictions
    -------------------
    This solver sees mathematical coefficients only.

    Important edge cases
    --------------------
    Unlike the original broad plan, this implementation does not integrate the
    lower tail below a negative root and does not support two distinct positive
    roots, no-real-root, or constant-region policies. A positive double root is
    a single unique threshold and is accepted. Unsupported cases raise visibly
    rather than silently changing the scientific decision rule.

    Related tests
    -------------
    ``test_reference_category_variances_coefficients_and_roots`` and
    ``test_positive_threshold_rejects_nonphysical_root_structures``.
    """

    if quadratic_tolerance <= 0.0 or discriminant_tolerance <= 0.0:
        raise ValueError("Numerical tolerances must be positive")

    a2, a1, a0 = coefficients.a2, coefficients.a1, coefficients.a0
    if bool(torch.all(torch.abs(a2) < quadratic_tolerance).detach().cpu()):
        if bool(torch.any(torch.abs(a1) < quadratic_tolerance).detach().cpu()):
            raise ValueError("Constant log odds do not define a positive threshold")
        if bool(torch.any(a1 <= 0.0).detach().cpu()):
            raise ValueError("Linear boundary is not an upper long-choice threshold")
        threshold = -a0 / a1
        if bool(torch.any(threshold <= 0.0).detach().cpu()):
            raise ValueError("Linear decision boundary is not positive")
        return PositiveDecisionRegion(
            threshold_s=threshold,
            solver_regime="linear",
            long_region="x > threshold",
            negative_root_policy="discard",
            discriminant=None,
        )

    discriminant = a1.square() - 4.0 * a2 * a0
    if bool(torch.any(discriminant < -discriminant_tolerance).detach().cpu()):
        raise ValueError("Posterior log odds have no real decision root")
    square_root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    denominator = 2.0 * a2
    roots = torch.stack(
        ((-a1 - square_root) / denominator, (-a1 + square_root) / denominator),
        dim=0,
    )
    positive = roots > 0.0
    positive_count = positive.sum(dim=0)
    is_double_root = torch.abs(discriminant) <= discriminant_tolerance
    double_root = -a1 / denominator
    simple_root = ~is_double_root
    if bool(torch.any(simple_root & (a2 <= 0.0)).detach().cpu()):
        raise ValueError("Quadratic boundary is not an upper long-choice threshold")
    valid_structure = torch.where(
        is_double_root,
        double_root > 0.0,
        positive_count == 1,
    )
    if not bool(torch.all(valid_structure).detach().cpu()):
        raise ValueError("Expected exactly one positive real decision root")
    infinity = torch.full_like(roots, torch.inf)
    selected_root = torch.where(positive, roots, infinity).amin(dim=0)
    threshold = torch.where(is_double_root, double_root, selected_root)
    if bool(torch.all(is_double_root).detach().cpu()):
        regime = "quadratic_double_root"
    elif bool(torch.any(is_double_root).detach().cpu()):
        regime = "quadratic_mixed_simple_and_double"
    else:
        regime = "quadratic_simple_root"
    return PositiveDecisionRegion(
        threshold_s=threshold,
        solver_regime=regime,
        long_region="x > threshold",
        negative_root_policy="discard",
        discriminant=discriminant,
    )


def positive_decision_threshold(
    coefficients: LogOddsCoefficients,
    *,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> torch.Tensor:
    r"""Return only the registered unique positive decision threshold.

    Purpose
    -------
    Provide a tensor-only convenience wrapper around the structured positive
    decision-region solver used by the project's modified M0 rule.

    Mathematical definition
    -----------------------
    For :math:`\Lambda(x)=a_2x^2+a_1x+a_0`, return its unique positive root

    .. math::

       T\in(0,\infty),\qquad \Lambda(T)=0,

    with the registered action rule ``long iff x > T``.

    Parameter interpretation
    ------------------------
    Coefficients encode fixed task moments, pooled sensory parameters, and the
    prior. The threshold is derived, not independently fitted.

    Input units and shapes
    ----------------------
    Coefficients are scalar ``float64`` tensors. Tolerances are scalar numerical
    controls; the implied threshold unit is seconds.

    Returned quantities and shapes
    ------------------------------
    Returns one scalar threshold tensor in seconds and preserves autograd.

    Constraints
    -----------
    There must be exactly one unique positive root. The negative mathematical
    root is discarded and never returned, per the positive-root-only policy.

    Numerical stability
    -------------------
    The delegated solver handles linear, simple-quadratic, and double-root
    regimes using registered coefficient and discriminant tolerances.

    Causal restrictions
    -------------------
    This wrapper sees coefficients only; it cannot access choice, outcome,
    reward, block, session identity, or trial history.

    Important edge cases
    --------------------
    No-real-root, two-positive-root, nonpositive-root, and constant-region cases
    raise instead of silently changing the registered decision rule.

    Related tests
    -------------
    ``test_positive_root_matches_reference`` and
    ``test_structured_positive_region_has_no_negative_root_output``.
    """

    return solve_positive_decision_region(
        coefficients,
        quadratic_tolerance=quadratic_tolerance,
        discriminant_tolerance=discriminant_tolerance,
    ).threshold_s


def sensory_standard_deviation(
    duration_s: torch.Tensor,
    parameters: DecodedParameters,
) -> torch.Tensor:
    r"""Calculate trial-level sensory uncertainty in seconds.

    Purpose
    -------
    Convert physical stimulus duration into the Gaussian measurement SD.

    Mathematical definition
    -----------------------
    .. math::

       \sigma(s)=\sqrt{ws^2+s_0^2}.

    Parameter interpretation
    ------------------------
    ``w`` controls duration-dependent variance and ``s0`` is baseline SD.

    Input units and shapes
    ----------------------
    ``duration_s`` is a floating Torch tensor in seconds, containing only valid
    trials selected before this call.

    Returned quantities and shapes
    ------------------------------
    Tensor of positive SD values in seconds with broadcast input shape.

    Constraints
    -----------
    Durations must be finite and strictly positive.

    Numerical stability
    -------------------
    Positive decoded ``s0`` prevents a zero square-root argument.

    Causal restrictions
    -------------------
    Only current physical duration and pooled sensory parameters are used.

    Important edge cases
    --------------------
    Missing/omission rows must be removed by the masked M0 wrapper beforehand.

    Related tests
    -------------
    ``test_masked_prediction_skips_nondecisions_before_computation``.
    """

    if torch.any(~torch.isfinite(duration_s)) or torch.any(duration_s <= 0.0):
        raise ValueError("Evaluated durations must be finite positive seconds")
    return torch.sqrt(parameters.w * duration_s.square() + parameters.s0.square())


def bayesian_choice_probability(
    duration_s: torch.Tensor,
    parameters: DecodedParameters,
    moments: TaskMoments,
    prior_long: torch.Tensor,
) -> BayesianChoiceResult:
    r"""Integrate sensory noise above the positive decision threshold.

    Purpose
    -------
    Compute the non-lapse probability of a long response analytically without
    Monte Carlo sampling.

    Mathematical definition
    -----------------------
    With :math:`X\mid s\sim\mathcal N(s,\sigma(s)^2)` and the registered
    positive threshold :math:`T`, long is chosen when :math:`X>T`; therefore

    .. math::

       q(s)=P(X>T\mid s)
           =1-\Phi\left(\frac{T-s}{\sigma(s)}\right)
           =\Phi\left(\frac{s-T}{\sigma(s)}\right).

    The negative quadratic root and its extreme lower tail are not used.

    Parameter interpretation
    ------------------------
    ``prior_long`` affects the Bayesian boundary; M0 always passes exactly 0.5.

    Input units and shapes
    ----------------------
    Durations and threshold are seconds. Inputs may be one-dimensional valid
    trial vectors; the prior may be scalar or trial-shaped.

    Returned quantities and shapes
    ------------------------------
    Returns trial-shaped ``q``, positive threshold, sensory SD, category
    variances, and log-odds coefficients.

    Constraints
    -----------
    Inputs must satisfy the unique-positive-root contract.

    Numerical stability
    -------------------
    Uses Torch's differentiable standard-normal CDF in the input dtype. No
    probability clamp is applied to diagnostic predictions.

    Causal restrictions
    -------------------
    Choice, reward, outcome, true block, run length, and history are absent.

    Important edge cases
    --------------------
    Invalid trials must be skipped before calling this function. Root-structure
    violations raise rather than invoking an unregistered fallback region.

    Related tests
    -------------
    ``test_choice_probability_matches_normal_cdf`` and the autograd test.
    """

    variances = category_variances(parameters, moments)
    coefficients = log_odds_coefficients(
        variances, moments, prior_long, parameters.log_kappa
    )
    decision_region = solve_positive_decision_region(coefficients)
    threshold = decision_region.threshold_s
    sensory_sd = sensory_standard_deviation(duration_s, parameters)
    probability_long = torch.special.ndtr((duration_s - threshold) / sensory_sd)
    return BayesianChoiceResult(
        probability_long=probability_long,
        decision_region=decision_region,
        sensory_sd_s=sensory_sd,
        category_variances=variances,
        coefficients=coefficients,
    )


def lapse_mixture(
    non_lapse_probability_long: torch.Tensor,
    category_long: torch.Tensor,
    parameters: DecodedParameters,
) -> torch.Tensor:
    r"""Mix the Bayesian response with a category-dependent random-choice lapse.

    Purpose
    -------
    Convert non-lapse long-choice probability into the observable Bernoulli
    response probability.

    Mathematical definition
    -----------------------
    .. math::

       \epsilon_t &= \ell_{\mathrm{long}}\,\mathbf 1(c_t=1)
       +\ell_{\mathrm{short}}\,\mathbf 1(c_t=0),\\
       p_t &= (1-\epsilon_t)q_t+\epsilon_t/2.

    Parameter interpretation
    ------------------------
    ``category_long`` is the signed-category equivalent encoded as 0 for short
    and 1 for long. ``ell_short`` and ``ell_long`` are category-specific lapse
    rates. A lapse draws uniformly from the two response alternatives, so its
    long-choice probability is exactly one half.

    Input units and shapes
    ----------------------
    ``q`` is a dimensionless valid-trial probability tensor and
    ``category_long`` has the same shape with binary values.

    Returned quantities and shapes
    ------------------------------
    Returns the unclamped observable probability ``p`` with the same shape.

    Constraints
    -----------
    Decoding guarantees both lapse rates lie strictly between zero and one.

    Numerical stability
    -------------------
    No clamp is applied here. A later likelihood may clamp only its logarithm
    input while preserving this diagnostic probability.

    Causal restrictions
    -------------------
    Lapse mixing uses only ``q``, the current physical stimulus category, and
    pooled parameters. It never uses the hidden block state or choice outcome.

    Important edge cases
    --------------------
    At either lapse rate equal to one the corresponding category's prediction
    is exactly 0.5; at a zero lapse rate it is exactly ``q``.

    Related tests
    -------------
    ``test_category_specific_random_choice_lapse`` and the autograd test.
    """

    if torch.any(non_lapse_probability_long < 0.0) or torch.any(non_lapse_probability_long > 1.0):
        raise ValueError("Non-lapse probabilities must lie in [0, 1]")
    if category_long.shape != non_lapse_probability_long.shape:
        raise ValueError("category_long and non-lapse probabilities must share shape")
    if not bool(torch.all((category_long == 0.0) | (category_long == 1.0)).detach().cpu()):
        raise ValueError("category_long must contain only binary short/long codes")
    lapse_rate = torch.where(
        category_long == 1.0,
        parameters.ell_long,
        parameters.ell_short,
    )
    return (1.0 - lapse_rate) * non_lapse_probability_long + 0.5 * lapse_rate
