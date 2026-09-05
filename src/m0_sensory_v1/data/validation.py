"""Validation checks for prepared M0 behavioral tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preparation import CANONICAL_TRIAL_COLUMNS, DataPreparationResult


def _require_binary_or_missing(frame: pd.DataFrame, column: str) -> None:
    """Raise when a nullable binary column contains a value outside ``0/1``."""

    values = set(frame[column].dropna().astype(int).unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Column {column!r} is not binary: {sorted(values)}")


def validate_prepared_data(result: DataPreparationResult) -> None:
    """Validate canonical and model tables before splitting or fitting.

    Purpose
    -------
    Fail early when identifiers, temporal ordering, binary encodings, duration
    units, or deterministic-feedback reconstruction are inconsistent.

    Parameters
    ----------
    result:
        Output of :func:`m0_sensory_v1.data.prepare_dataset`.

    Returns
    -------
    None
        Successful return means every registered invariant passed.

    Constraints
    -----------
    Durations entering M0 must be finite and positive seconds. Session/trial
    pairs must be unique. Canonical trial indices must be consecutive and
    one-based within each session.

    Numerical stability
    -------------------
    Nonfinite model durations are rejected before they can enter a Torch loss.

    Causal restrictions
    -------------------
    This validates feedback reconstruction but does not make category, reward,
    or true block legal pre-choice predictors.

    Important edge cases
    --------------------
    Omission rows may have missing choices and rewards; model rows may not.

    Related tests
    -------------
    The synthetic preparation tests call this validator on a mixed valid and
    omission session.
    """

    canonical = result.canonical_trials
    model = result.model_trials
    missing = sorted(set(CANONICAL_TRIAL_COLUMNS).difference(canonical.columns))
    if missing:
        raise KeyError(f"Canonical table is missing columns: {missing}")
    if canonical.empty:
        raise ValueError("Canonical trial table is empty")
    if result.sessions.empty:
        raise ValueError("Session audit table is empty")

    if canonical.duplicated(["mouse_id", "session_id", "trial_index"]).any():
        raise ValueError("Canonical trial identifiers are not unique")

    for session_key, group in canonical.groupby(
        ["mouse_id", "session_id"], sort=False, observed=False
    ):
        observed = group.sort_values("trial_index")["trial_index"].astype(int).to_numpy()
        expected = np.arange(1, len(group) + 1)
        if not np.array_equal(observed, expected):
            raise ValueError(f"Nonconsecutive trial indices in session {session_key}")

    for column in (
        "category_long",
        "choice_long",
        "rewarded",
        "omission",
        "choice_valid",
        "feedback_valid",
        "warm_up",
        "jitter_flag",
    ):
        _require_binary_or_missing(canonical, column)

    if not model.empty:
        durations = model["duration_s"].to_numpy(dtype=float)
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("M0 model durations must be finite positive seconds")
        for column in ("category_long", "choice_long", "rewarded"):
            if model[column].isna().any():
                raise ValueError(f"Model column {column!r} contains missing values")

        reconstructed_category = np.where(
            model["rewarded"].astype(int).to_numpy() == 1,
            model["choice_long"].astype(int).to_numpy(),
            1 - model["choice_long"].astype(int).to_numpy(),
        )
        recorded_category = model["category_long"].astype(int).to_numpy()
        if not np.array_equal(reconstructed_category, recorded_category):
            raise ValueError("Choice/reward reconstruction disagrees with category_long")

