"""Hazard, duration, tail-occupancy, and numerical diagnostics for
M2_dynamic_previous_wsls_bias_v1.

The hazard/duration machinery is genuinely shared with
M2_dynamic_block_length_v1 (``ah``, ``bh``, and ``delta0`` mean exactly the
same thing here, unchanged) -- this module slices the first eight raw
coordinates and delegates to the registered parent registry rather than
reimplementing the hazard/duration mathematics. Only the truncation-
sensitivity check is reimplemented natively, since it must hold the
bias-mixing coordinate (``eta_logit``) fixed too, which requires this
model's own nine-coordinate objective.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch

from m2_dynamic_block_length_v1.diagnostics import (
    M2DynamicBlockLengthHazardDurationSummary,
    build_m2_dynamic_block_length_implied_duration_distribution,
    summarize_m2_dynamic_block_length_hazard_duration,
    summarize_m2_dynamic_block_length_prediction_diagnostics,
)

from .session import evaluate_m2_dynamic_previous_wsls_bias_mouse_objective


M2DynamicPreviousWslsBiasHazardDurationSummary = M2DynamicBlockLengthHazardDurationSummary
summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics = summarize_m2_dynamic_block_length_prediction_diagnostics


def build_m2_dynamic_previous_wsls_bias_implied_duration_distribution(
    raw_parameters: Iterable[float],
    registry,
    *,
    block_state: str = "biased",
    maximum_duration: int = 5000,
    remaining_probability_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return the duration PMF for the tied biased or neutral hazard, via
    M2_dynamic_block_length_v1's own unchanged implementation."""

    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (9,):
        raise ValueError("M2_dynamic_previous_wsls_bias raw_parameters must contain exactly nine values")
    return build_m2_dynamic_block_length_implied_duration_distribution(
        raw[:8], registry.source_m2_dynamic_block_length_registry,
        block_state=block_state, maximum_duration=maximum_duration,
        remaining_probability_tolerance=remaining_probability_tolerance,
    )


def summarize_m2_dynamic_previous_wsls_bias_hazard_duration(
    raw_parameters: Iterable[float], registry
) -> M2DynamicBlockLengthHazardDurationSummary:
    """Summarize landmark hazards and quantiles of the implied duration law,
    via M2_dynamic_block_length_v1's own unchanged implementation."""

    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (9,):
        raise ValueError("M2_dynamic_previous_wsls_bias raw_parameters must contain exactly nine values")
    return summarize_m2_dynamic_block_length_hazard_duration(
        raw[:8], registry.source_m2_dynamic_block_length_registry
    )


def evaluate_m2_dynamic_previous_wsls_bias_truncation_sensitivity(
    sessions,
    raw_parameters: Iterable[float],
    registry,
    *,
    maximum_run_lengths: Iterable[int] = (65, 75, 100),
) -> pd.DataFrame:
    """Re-evaluate a fitted checkpoint under larger capped-tail state spaces.

    All nine parameters (including the bias-mixing weight; the
    previous-category learning rate is a fixed constant, not fitted) are
    held fixed. This is a diagnostic of the numerical representation, not a
    refit and not a change to the registered Rmax=65 model.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("Truncation sensitivity requires nonempty sessions")
    role = records[0].split_role
    if any(item.split_role != role for item in records) or role == "test":
        raise ValueError("Truncation sensitivity requires one non-test split")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (9,):
        raise ValueError("M2_dynamic_previous_wsls_bias raw_parameters must contain exactly nine values")
    maxima = tuple(int(value) for value in maximum_run_lengths)
    if not maxima or any(value < registry.run_length_center for value in maxima):
        raise ValueError("Every sensitivity cap must be at least the run-length center")
    rows = []
    with torch.no_grad():
        for maximum in maxima:
            objective = evaluate_m2_dynamic_previous_wsls_bias_mouse_objective(
                records, raw, registry.task_moments,
                objective_weighting=registry.objective_weighting,
                required_split_role=role, lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                run_length_center=registry.run_length_center, maximum_run_length=maximum,
                first_long_probability=registry.first_long_probability,
                initial_pi=registry.initial_pi,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
            rows.append({
                "split_role": role, "maximum_run_length": maximum, "mean_nll": float(objective.loss),
                "mean_tail_probability_pre": objective.mean_tail_probability_pre,
                "maximum_tail_probability_pre": objective.maximum_tail_probability_pre,
                "valid_trial_count": objective.total_valid_trial_count,
            })
    result = pd.DataFrame(rows)
    registered = result.loc[result.maximum_run_length.eq(registry.maximum_represented_run_length), "mean_nll"]
    baseline = float(registered.iloc[0]) if not registered.empty else float(result.mean_nll.iloc[0])
    result["delta_mean_nll_from_registered_rmax65"] = result.mean_nll - baseline
    return result
