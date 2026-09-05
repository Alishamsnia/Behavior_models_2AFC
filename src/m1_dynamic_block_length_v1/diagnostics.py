"""Hazard, duration, tail-occupancy, and numerical diagnostics for
M1_dynamic_block_length_v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .analysis import build_m1_dynamic_block_length_hazard_curve
from .session import evaluate_m1_dynamic_block_length_mouse_objective


@dataclass(frozen=True)
class M1DynamicBlockLengthHazardDurationSummary:
    """Human-readable summaries of the tied biased and neutral duration laws."""

    hazard_r1: float
    hazard_r10: float
    hazard_r20: float
    hazard_r23: float
    hazard_r65: float
    implied_mean_duration: float
    implied_duration_q10: int
    implied_duration_q25: int
    implied_duration_median: int
    implied_duration_q75: int
    implied_duration_q90: int
    implied_duration_q95: int
    probability_duration_at_least_65: float
    neutral_hazard_r1: float
    neutral_hazard_r10: float
    neutral_hazard_r20: float
    neutral_hazard_r23: float
    neutral_hazard_r65: float
    neutral_implied_mean_duration: float
    neutral_implied_duration_q10: int
    neutral_implied_duration_q25: int
    neutral_implied_duration_median: int
    neutral_implied_duration_q75: int
    neutral_implied_duration_q90: int
    neutral_implied_duration_q95: int
    neutral_probability_duration_at_least_65: float
    truncation_approximation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_m1_dynamic_block_length_implied_duration_distribution(
    raw_parameters: Iterable[float],
    registry,
    *,
    block_state: str = "biased",
    maximum_duration: int = 5000,
    remaining_probability_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return the duration PMF for the tied biased or neutral hazard.

    Durations 1--64 use their individual hazards. From duration 65 onward the
    registered approximation freezes the hazard at h(65).
    """

    if block_state not in {"biased", "neutral"}:
        raise ValueError("block_state must be 'biased' or 'neutral'")
    if maximum_duration < registry.maximum_represented_run_length:
        raise ValueError("maximum_duration must include the represented tail")
    if not 0.0 < remaining_probability_tolerance < 1.0:
        raise ValueError("remaining_probability_tolerance must lie in (0,1)")
    curve = build_m1_dynamic_block_length_hazard_curve(raw_parameters, registry)
    hazards = curve[f"{block_state}_hazard"].to_numpy(float)
    cap = registry.maximum_represented_run_length
    survival = 1.0
    rows: list[dict[str, float | int | bool | str]] = []
    duration = 0
    effective_maximum_duration = maximum_duration
    while duration < effective_maximum_duration:
        duration += 1
        index = min(duration, cap) - 1
        hazard = float(hazards[index])
        mass = survival * hazard
        rows.append({
            "duration_trials": duration, "hazard": hazard, "block_state": block_state,
            "probability": mass, "survival_probability_at_start": survival,
            "uses_frozen_tail_hazard": duration >= cap,
        })
        survival *= 1.0 - hazard
        if survival <= remaining_probability_tolerance:
            break
        if duration >= cap and duration == effective_maximum_duration:
            # From here on the hazard is frozen at a single constant value
            # (index has saturated at cap-1), so the remaining tail is
            # exactly geometric. Solve analytically for how many more
            # trials are needed to reach the tolerance instead of failing a
            # legitimate low-hazard fit against an arbitrary fixed row cap
            # -- a valid fitted hazard can require far more than the
            # default 5000 trials to reach an arbitrarily small tolerance.
            if hazard <= 0.0:
                raise ValueError(
                    "Frozen tail hazard must be positive to reach any tail tolerance"
                )
            additional_steps = int(
                np.ceil(
                    np.log(remaining_probability_tolerance / survival)
                    / np.log(1.0 - hazard)
                )
            )
            effective_maximum_duration = duration + max(1, additional_steps)
    if survival > remaining_probability_tolerance:
        raise RuntimeError("Duration distribution did not reach the requested tail tolerance")
    result = pd.DataFrame(rows)
    result["cumulative_probability"] = result.probability.cumsum()
    result.attrs["unallocated_tail_probability"] = survival
    return result


def summarize_m1_dynamic_block_length_hazard_duration(
    raw_parameters: Iterable[float], registry
) -> M1DynamicBlockLengthHazardDurationSummary:
    """Summarize landmark hazards and quantiles of the implied duration law."""

    curve = build_m1_dynamic_block_length_hazard_curve(raw_parameters, registry).set_index("represented_run_length")

    def duration_statistics(block_state: str) -> tuple[float, int, int, int, int, int, int, float]:
        distribution = build_m1_dynamic_block_length_implied_duration_distribution(
            raw_parameters, registry, block_state=block_state
        )
        cumulative = distribution.cumulative_probability.to_numpy(float)
        durations = distribution.duration_trials.to_numpy(int)

        def quantile(probability: float) -> int:
            index = min(np.searchsorted(cumulative, probability), len(durations) - 1)
            return int(durations[index])

        mean = float(np.sum(distribution.duration_trials.to_numpy(float) * distribution.probability.to_numpy(float)))
        landmark = distribution.loc[
            distribution.duration_trials.eq(registry.maximum_represented_run_length), "survival_probability_at_start",
        ]
        probability_at_least_65 = (
            float(landmark.iloc[0]) if not landmark.empty
            else float(distribution.attrs.get("unallocated_tail_probability", 0.0))
        )
        return (mean, quantile(.10), quantile(.25), quantile(.50), quantile(.75), quantile(.90), quantile(.95), probability_at_least_65)

    biased = duration_statistics("biased")
    neutral = duration_statistics("neutral")
    return M1DynamicBlockLengthHazardDurationSummary(
        hazard_r1=float(curve.loc[1, "biased_hazard"]), hazard_r10=float(curve.loc[10, "biased_hazard"]),
        hazard_r20=float(curve.loc[20, "biased_hazard"]), hazard_r23=float(curve.loc[23, "biased_hazard"]),
        hazard_r65=float(curve.loc[65, "biased_hazard"]),
        implied_mean_duration=biased[0], implied_duration_q10=biased[1], implied_duration_q25=biased[2],
        implied_duration_median=biased[3], implied_duration_q75=biased[4], implied_duration_q90=biased[5],
        implied_duration_q95=biased[6], probability_duration_at_least_65=biased[7],
        neutral_hazard_r1=float(curve.loc[1, "neutral_hazard"]), neutral_hazard_r10=float(curve.loc[10, "neutral_hazard"]),
        neutral_hazard_r20=float(curve.loc[20, "neutral_hazard"]), neutral_hazard_r23=float(curve.loc[23, "neutral_hazard"]),
        neutral_hazard_r65=float(curve.loc[65, "neutral_hazard"]),
        neutral_implied_mean_duration=neutral[0], neutral_implied_duration_q10=neutral[1],
        neutral_implied_duration_q25=neutral[2], neutral_implied_duration_median=neutral[3],
        neutral_implied_duration_q75=neutral[4], neutral_implied_duration_q90=neutral[5],
        neutral_implied_duration_q95=neutral[6], neutral_probability_duration_at_least_65=neutral[7],
        truncation_approximation="hazard_frozen_at_h65_for_all_true_run_lengths_at_or_above_65",
    )


def summarize_m1_dynamic_block_length_prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tail use, support checks, and numerical errors per split/session."""

    required = {
        "split_role", "session_id", "model_tail_probability_pre",
        "model_pre_state_normalization_error", "model_post_state_normalization_error",
        "model_transition_normalization_error", "model_unreachable_support_mass_pre",
        "model_unreachable_support_mass_post",
    }
    if missing := required.difference(predictions.columns):
        raise KeyError(f"M1_dynamic_block_length prediction diagnostics missing columns: {sorted(missing)}")
    data = predictions.copy()
    data["absolute_transition_normalization_error"] = pd.to_numeric(
        data.model_transition_normalization_error, errors="coerce"
    ).abs()
    return data.groupby(["split_role", "session_id"], observed=False).agg(
        timeline_trials=("model_tail_probability_pre", "size"),
        mean_tail_probability_pre=("model_tail_probability_pre", "mean"),
        maximum_tail_probability_pre=("model_tail_probability_pre", "max"),
        maximum_pre_state_normalization_error=("model_pre_state_normalization_error", "max"),
        maximum_post_state_normalization_error=("model_post_state_normalization_error", "max"),
        maximum_transition_normalization_error=("absolute_transition_normalization_error", "max"),
        maximum_unreachable_support_mass_pre=("model_unreachable_support_mass_pre", "max"),
        maximum_unreachable_support_mass_post=("model_unreachable_support_mass_post", "max"),
    ).reset_index()


def evaluate_m1_dynamic_block_length_truncation_sensitivity(
    sessions,
    raw_parameters: Iterable[float],
    registry,
    *,
    maximum_run_lengths: Iterable[int] = (65, 75, 100),
) -> pd.DataFrame:
    """Re-evaluate a fitted checkpoint under larger capped-tail state spaces.

    Parameters are held fixed. This is a diagnostic of the numerical
    representation, not a refit and not a change to the registered Rmax=65
    model.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("Truncation sensitivity requires nonempty sessions")
    role = records[0].split_role
    if any(item.split_role != role for item in records) or role == "test":
        raise ValueError("Truncation sensitivity requires one non-test split")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    maxima = tuple(int(value) for value in maximum_run_lengths)
    if not maxima or any(value < registry.run_length_center for value in maxima):
        raise ValueError("Every sensitivity cap must be at least the run-length center")
    rows = []
    with torch.no_grad():
        for maximum in maxima:
            objective = evaluate_m1_dynamic_block_length_mouse_objective(
                records, raw, registry.task_moments,
                objective_weighting=registry.objective_weighting,
                required_split_role=role, lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                run_length_center=registry.run_length_center, maximum_run_length=maximum,
                first_long_probability=registry.first_long_probability,
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
