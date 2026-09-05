"""Empirical task-distribution moments used by the M0 sensory model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskMoments:
    r"""Fixed empirical moments of the short and long stimulus distributions.

    The variances are population variances (``ddof=0``), because M0 requires
    the task-distribution quantity :math:`\operatorname{Var}(s\mid C=k)`, not
    an unbiased estimate of an unknown sampling variance.
    """

    mu_short_s: float
    mu_long_s: float
    var_short_s2: float
    var_long_s2: float
    n_short: int
    n_long: int
    min_short_s: float
    max_short_s: float
    min_long_s: float
    max_long_s: float
    variance_ddof: int = 0
    source: str = "canonical presented trials"

    def __post_init__(self) -> None:
        """Enforce the registered positive, truncated, non-overlapping support."""

        numeric = (
            self.mu_short_s,
            self.mu_long_s,
            self.var_short_s2,
            self.var_long_s2,
            self.min_short_s,
            self.max_short_s,
            self.min_long_s,
            self.max_long_s,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Task moments and support bounds must be finite")
        if self.n_short <= 0 or self.n_long <= 0:
            raise ValueError("Both task categories require positive counts")
        if self.var_short_s2 < 0.0 or self.var_long_s2 < 0.0:
            raise ValueError("Task variances must be nonnegative")
        if self.min_short_s <= 0.0 or self.min_long_s <= 0.0:
            raise ValueError("Truncated task support must be strictly positive")
        if not self.min_short_s <= self.mu_short_s <= self.max_short_s:
            raise ValueError("Short-category mean lies outside its observed support")
        if not self.min_long_s <= self.mu_long_s <= self.max_long_s:
            raise ValueError("Long-category mean lies outside its observed support")
        if not self.max_short_s < self.min_long_s:
            raise ValueError("Short and long task supports must not overlap")
        if self.variance_ddof != 0:
            raise ValueError("Registered task moments require population variance ddof=0")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with units in field names."""

        return asdict(self)


def estimate_task_moments(
    canonical_trials: pd.DataFrame,
    *,
    require_positive_support: bool = True,
    require_nonoverlap: bool = True,
    source: str = "canonical presented trials",
) -> TaskMoments:
    r"""Estimate fixed category moments from real presented stimuli.

    Purpose
    -------
    Estimate the task quantities used in the category approximation directly
    from the real, truncated stimulus distributions. Outcome and choice are not
    used. Consequently, a presented stimulus on an omission trial contributes
    to the task-distribution audit but never to the Bernoulli likelihood.

    Mathematical definition
    -----------------------
    For category :math:`k\in\{0,1\}` and its :math:`N_k` observed presented
    durations, this function calculates

    .. math::

       \mu_k = \frac{1}{N_k}\sum_{i:C_i=k}s_i,
       \qquad
       \tau_k^2 = \frac{1}{N_k}\sum_{i:C_i=k}(s_i-\mu_k)^2.

    Parameter interpretation
    ------------------------
    ``category_long=0`` is the short task distribution and
    ``category_long=1`` is the long task distribution.

    Input units and shapes
    ----------------------
    ``canonical_trials`` contains one row per presented task trial;
    ``duration_s`` is measured in seconds and ``category_long`` is binary.

    Returned quantities and shapes
    ------------------------------
    Returns one :class:`TaskMoments` record containing scalar means, population
    variances, counts, and observed support bounds.

    Constraints
    -----------
    Both categories must contain finite durations. By default all durations
    must be strictly positive and the observed supports must not overlap:
    :math:`\max(s\mid C=0)<\min(s\mid C=1)`.

    Numerical stability
    -------------------
    Calculations use NumPy ``float64`` and population variance ``ddof=0``.

    Causal restrictions
    -------------------
    Reward, outcome, block, and choice columns are never read. These fixed
    moments describe the stimulus generator, not mouse behavior.

    Important edge cases
    --------------------
    Missing category/duration pairs are excluded from moment estimation.
    Overlapping supports or nonpositive durations raise instead of silently
    contradicting the registered truncated-distribution assumption.

    Related tests
    -------------
    ``test_estimate_task_moments_uses_population_variance_and_checks_support``
    and the real-data integration audit.
    """

    required = {"category_long", "duration_s"}
    missing = sorted(required.difference(canonical_trials.columns))
    if missing:
        raise KeyError(f"Canonical trial table is missing columns: {missing}")

    category = pd.to_numeric(canonical_trials["category_long"], errors="coerce")
    duration = pd.to_numeric(canonical_trials["duration_s"], errors="coerce")
    category_array = category.to_numpy(dtype=float, na_value=np.nan)
    duration_array = duration.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(duration_array) & np.isin(category_array, [0.0, 1.0])
    category_values = category_array[finite].astype(int)
    duration_values = duration_array[finite]

    short = duration_values[category_values == 0]
    long = duration_values[category_values == 1]
    if short.size == 0 or long.size == 0:
        raise ValueError("Both short and long categories require finite durations")
    if require_positive_support and (np.any(short <= 0.0) or np.any(long <= 0.0)):
        raise ValueError("Truncated task durations must be strictly positive")
    if require_nonoverlap and not float(np.max(short)) < float(np.min(long)):
        raise ValueError(
            "Observed short/long supports overlap: "
            f"max(short)={np.max(short):.9g}, min(long)={np.min(long):.9g}"
        )

    return TaskMoments(
        mu_short_s=float(np.mean(short, dtype=np.float64)),
        mu_long_s=float(np.mean(long, dtype=np.float64)),
        var_short_s2=float(np.var(short, ddof=0, dtype=np.float64)),
        var_long_s2=float(np.var(long, ddof=0, dtype=np.float64)),
        n_short=int(short.size),
        n_long=int(long.size),
        min_short_s=float(np.min(short)),
        max_short_s=float(np.max(short)),
        min_long_s=float(np.min(long)),
        max_long_s=float(np.max(long)),
        source=source,
    )
