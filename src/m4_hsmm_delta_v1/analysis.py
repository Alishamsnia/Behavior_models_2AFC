"""Auditable M4 trial exports and paired M4-versus-M3 comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import M4_PARAMETER_ORDER, decode_m4_raw_parameters
from .session import evaluate_m4_session, m4_session_trials_from_frame


def build_m4_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry,
    *,
    allow_test: bool = False,
    include_full_run_length_distributions: bool = False,
    full_distribution_session_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rebuild reset sessions and export causal beliefs without using true labels.

    The default table contains compact duration summaries.  The 100 marginal
    columns are emitted only when explicitly requested, optionally for selected
    sessions, so routine result files remain tractable.
    """

    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "feedback_valid", "true_block", "split_role",
    }
    if missing := sorted(required.difference(canonical_trials.columns)):
        raise KeyError(f"Canonical M4 table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("M4 test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown M4 split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (9,):
        raise ValueError("M4 raw_parameters must contain exactly nine values")
    parameters = decode_m4_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    selected_sessions = (
        None if full_distribution_session_ids is None
        else {str(value) for value in full_distribution_session_ids}
    )
    parts: list[pd.DataFrame] = []
    for (_, _), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        session = m4_session_trials_from_frame(frame, split_role=str(frame.split_role.iloc[0]))
        with torch.no_grad():
            evaluation = evaluate_m4_session(
                session, parameters, registry.task_moments,
                run_length_center=registry.run_length_center,
                maximum_run_length=registry.maximum_represented_run_length,
                first_long_probability=registry.first_long_probability,
                delta_min=registry.delta_min,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        direct = {
            "model_probability_long": prediction.probability_long,
            "model_non_lapse_probability_long": prediction.non_lapse_probability_long,
            "model_prior_long": prediction.prior_long,
            "model_decision_lower_boundary_s": prediction.decision_lower_boundary_s,
            "model_decision_upper_boundary_s": prediction.decision_upper_boundary_s,
            "model_sensory_sd_s": prediction.sensory_sd_s,
            "model_emission_normalizer": prediction.emission_normalizer,
            "model_delta_pre": prediction.delta_pre,
            "model_delta_update_signal": prediction.delta_update_signal,
            "model_delta_unprojected_proposal": prediction.delta_unprojected_proposal,
            "model_delta_post": prediction.delta_post,
            "model_projection_activated": prediction.projection_activated,
            "model_expected_capped_run_length_pre": prediction.expected_run_length_pre,
            "model_expected_capped_run_length_post": prediction.expected_run_length_post,
            "model_map_capped_run_length_pre": prediction.map_run_length_pre,
            "model_map_capped_run_length_post": prediction.map_run_length_post,
            "model_run_length_entropy_pre": prediction.run_length_entropy_pre,
            "model_run_length_entropy_post": prediction.run_length_entropy_post,
            "model_expected_hazard_pre": prediction.expected_hazard_pre,
            "model_expected_hazard_post": prediction.expected_hazard_post,
            "model_hazard_at_map_run_length_pre": prediction.hazard_at_map_run_length_pre,
            "model_total_switch_mass_post": prediction.total_switch_mass_post,
            "model_total_survival_mass_post": prediction.total_survival_mass_post,
            "model_tail_probability_pre": prediction.tail_probability_pre,
            "model_tail_probability_post": prediction.tail_probability_post,
            "model_maximum_reachable_run_length": prediction.maximum_reachable_run_length,
            "model_pre_state_normalization_error": prediction.pre_state_normalization_error,
            "model_post_state_normalization_error": prediction.post_state_normalization_error,
            "model_transition_normalization_error": prediction.transition_normalization_error,
            "model_unreachable_support_mass_pre": prediction.unreachable_support_mass_pre,
            "model_unreachable_support_mass_post": prediction.unreachable_support_mass_post,
        }
        model_columns = {name: tensor.cpu().numpy() for name, tensor in direct.items()}
        model_columns["model_decision_regime"] = prediction.decision_regime
        model_columns["model_projection_cumulative_count"] = (
            prediction.projection_activated.to(torch.int64).cumsum(0).cpu().numpy()
        )
        for index, state in enumerate(registry.state_order):
            model_columns[f"model_pre_choice_probability_{state}"] = prediction.pre_choice_block_probability[:, index].cpu().numpy()
            model_columns[f"model_post_feedback_probability_{state}"] = prediction.post_feedback_block_probability[:, index].cpu().numpy()
            model_columns[f"model_responsibility_{state}"] = prediction.posterior_block_responsibility[:, index].cpu().numpy()
            model_columns[f"model_subjective_prototype_{state}"] = prediction.subjective_prototypes_pre[:, index].cpu().numpy()
            model_columns[f"model_switch_mass_post_{state}"] = prediction.block_specific_switch_mass_post[:, index].cpu().numpy()
        include_marginals = include_full_run_length_distributions and (
            selected_sessions is None or str(frame.session_id.iloc[0]) in selected_sessions
        )
        if include_marginals:
            for index in range(registry.maximum_represented_run_length):
                run = index + 1
                model_columns[f"model_pre_choice_run_probability_r{run:02d}"] = prediction.pre_choice_run_length_probability[:, index].cpu().numpy()
                model_columns[f"model_post_feedback_run_probability_r{run:02d}"] = prediction.post_feedback_run_length_probability[:, index].cpu().numpy()
        model_columns["per_trial_negative_log_likelihood"] = evaluation.per_trial_negative_log_likelihood.cpu().numpy()
        model_columns["model_evaluated"] = prediction.valid_choice_mask.cpu().numpy()
        model_columns["model_feedback_updated"] = prediction.feedback_update_mask.cpu().numpy()
        part = pd.concat(
            (frame.copy(), pd.DataFrame(model_columns, index=frame.index)), axis=1
        )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    probability = pd.to_numeric(result.model_probability_long, errors="coerce")
    choice = pd.to_numeric(result.choice_long, errors="coerce")
    category = pd.to_numeric(result.category_long, errors="coerce")
    derived = pd.DataFrame(index=result.index)
    derived["trial_type"] = np.select(
        [category.eq(0), category.eq(1)], ["short", "long"], default="undefined"
    )
    derived["block_type"] = result["true_block"].map(
        {"N": "neutral", "S": "short_block", "L": "long_block"}
    ).fillna("undefined")
    derived["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1-probability, np.nan)
    )
    derived["mouse_correct"] = np.where(
        choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan
    )
    derived["model_choice_long"] = np.where(
        probability.notna(), (probability >= 0.5).astype(float), np.nan
    )
    derived["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (derived["model_choice_long"] == category).astype(float),
        np.nan,
    )
    result["model_probability_true_block"] = np.nan
    for state in registry.state_order:
        selected = result.true_block.eq(state)
        result.loc[selected, "model_probability_true_block"] = result.loc[
            selected, f"model_pre_choice_probability_{state}"
        ]
    for name, value in zip(M4_PARAMETER_ORDER, parameters.as_tensor().tolist(), strict=True):
        derived[f"fitted_{name}"] = float(value)
    result = pd.concat((result, derived), axis=1)
    sort_columns = [name for name in ("split_role", "session_date", "session_id", "trial_index") if name in result]
    return result.sort_values(sort_columns).reset_index(drop=True)


def build_m4_hazard_curve(raw_parameters: Iterable[float], registry) -> pd.DataFrame:
    """Return tied neutral and biased hazards across represented run lengths."""

    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    parameters = decode_m4_raw_parameters(raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon)
    run = torch.arange(1, registry.maximum_represented_run_length + 1, dtype=torch.float64)
    with torch.no_grad():
        biased_hazard = parameters.hazard(run, center=registry.run_length_center)
        neutral_hazard = biased_hazard / 2.0
    return pd.DataFrame({
        "represented_run_length": run.to(torch.int64).numpy(),
        # ``hazard`` remains a compatibility alias for the fitted biased hazard.
        "hazard": biased_hazard.numpy(),
        "biased_hazard": biased_hazard.numpy(),
        "neutral_hazard": neutral_hazard.numpy(),
        "is_center": run.eq(registry.run_length_center).numpy(),
        "is_capped_tail": run.eq(registry.maximum_represented_run_length).numpy(),
    })


def compare_m4_to_m3_predictions(
    m4_predictions: pd.DataFrame,
    m3_predictions: pd.DataFrame,
    *,
    split_role: str = "validation",
) -> pd.DataFrame:
    """Return aligned session-level held-out M4 gains over M3 in bits/trial."""

    keys = [name for name in ("mouse_id", "split_role", "session_id", "trial_index") if name in m4_predictions and name in m3_predictions]
    columns = keys + ["per_trial_negative_log_likelihood", "model_evaluated"]
    for label, frame in (("M4", m4_predictions), ("M3", m3_predictions)):
        if missing := set(columns).difference(frame.columns):
            raise KeyError(f"{label} predictions missing columns: {sorted(missing)}")
    merged = m4_predictions[columns].merge(
        m3_predictions[columns], on=keys, how="outer", validate="one_to_one",
        suffixes=("_m4", "_m3"), indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise ValueError("M4 and M3 prediction timelines are not identical")
    merged = merged.loc[
        merged.split_role.eq(split_role)
        & merged.model_evaluated_m4.astype(bool)
        & merged.model_evaluated_m3.astype(bool)
    ].copy()
    if merged.empty:
        raise ValueError(f"No aligned evaluated {split_role} trials")
    merged["delta_bits_m4_vs_m3"] = (
        merged.per_trial_negative_log_likelihood_m3
        - merged.per_trial_negative_log_likelihood_m4
    ) / np.log(2)
    return merged.groupby([name for name in ("mouse_id", "split_role", "session_id") if name in merged], observed=False).agg(
        n_valid_trials=("delta_bits_m4_vs_m3", "size"),
        total_delta_bits_m4_vs_m3=("delta_bits_m4_vs_m3", "sum"),
        mean_delta_bits_per_trial_m4_vs_m3=("delta_bits_m4_vs_m3", "mean"),
    ).reset_index()


@dataclass(frozen=True)
class M4M3ValidationComparison:
    """Complete aligned held-out comparison and conservative selection rule."""

    overall: pd.DataFrame
    per_session: pd.DataFrame
    calibration: pd.DataFrame
    context: pd.DataFrame
    transition_aligned: pd.DataFrame
    rare_common: pd.DataFrame


def build_m4_m3_validation_comparison(
    m4_predictions: pd.DataFrame,
    m3_predictions: pd.DataFrame,
    *,
    split_role: str = "validation",
) -> M4M3ValidationComparison:
    """Compare models on exactly matched evaluated trials.

    Selection is validation-only. M4 is selected only when its mean trialwise
    log-score gain is greater than one session-level standard error; otherwise
    the nested, simpler M3 is retained. Test data may be summarized only after
    an explicit unlock in the caller and never participate in model selection.
    """

    keys = ["mouse_id", "split_role", "session_id", "trial_index"]
    shared = [
        *keys, "choice_long", "category_long", "true_block", "model_evaluated",
        "model_probability_long", "per_trial_negative_log_likelihood",
    ]
    for label, frame in (("M4", m4_predictions), ("M3", m3_predictions)):
        if missing := set(shared).difference(frame.columns):
            raise KeyError(f"{label} predictions missing columns: {sorted(missing)}")
    left = m4_predictions[shared].copy()
    right = m3_predictions[keys + [
        "choice_long", "category_long", "true_block", "model_evaluated",
        "model_probability_long", "per_trial_negative_log_likelihood",
    ]].copy()
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one",
                        suffixes=("_m4", "_m3"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("M4 and M3 prediction timelines are not identical")
    for name in ("choice_long", "category_long", "true_block"):
        left_value = merged[f"{name}_m4"]
        right_value = merged[f"{name}_m3"]
        equal = left_value.eq(right_value) | (
            left_value.isna() & right_value.isna()
        )
        if not bool(equal.fillna(False).all()):
            raise ValueError(f"M4/M3 aligned observations disagree for {name}")
    mask = (
        merged.split_role.eq(split_role)
        & merged.model_evaluated_m4.astype(bool)
        & merged.model_evaluated_m3.astype(bool)
    )
    aligned = merged.loc[mask].copy()
    if aligned.empty:
        raise ValueError(f"No aligned evaluated {split_role} trials")
    aligned["delta_elpd_m4_vs_m3"] = (
        aligned.per_trial_negative_log_likelihood_m3
        - aligned.per_trial_negative_log_likelihood_m4
    )
    aligned["delta_bits_m4_vs_m3"] = aligned.delta_elpd_m4_vs_m3 / np.log(2.0)
    aligned["squared_error_m4"] = (
        aligned.model_probability_long_m4 - aligned.choice_long_m4
    ) ** 2
    aligned["squared_error_m3"] = (
        aligned.model_probability_long_m3 - aligned.choice_long_m4
    ) ** 2
    per_session = aligned.groupby(["mouse_id", "split_role", "session_id"], observed=False).agg(
        n_valid_trials=("delta_elpd_m4_vs_m3", "size"),
        delta_elpd_m4_vs_m3=("delta_elpd_m4_vs_m3", "sum"),
        mean_delta_elpd_per_trial_m4_vs_m3=("delta_elpd_m4_vs_m3", "mean"),
        mean_delta_bits_per_trial_m4_vs_m3=("delta_bits_m4_vs_m3", "mean"),
        brier_m4=("squared_error_m4", "mean"),
        brier_m3=("squared_error_m3", "mean"),
    ).reset_index()
    session_values = per_session.mean_delta_elpd_per_trial_m4_vs_m3.to_numpy(float)
    session_se = (
        float(np.std(session_values, ddof=1) / np.sqrt(len(session_values)))
        if len(session_values) > 1 else float("nan")
    )
    mean_gain = float(aligned.delta_elpd_m4_vs_m3.mean())
    threshold = session_se if np.isfinite(session_se) else float("inf")
    selected = "M4_hsmm_delta_v1" if mean_gain > threshold else "M3_hmm_delta_v1"
    overall = pd.DataFrame([{
        "split_role": split_role,
        "n_valid_trials": len(aligned),
        "n_sessions": len(per_session),
        "delta_elpd_m4_vs_m3": float(aligned.delta_elpd_m4_vs_m3.sum()),
        "mean_delta_elpd_per_trial_m4_vs_m3": mean_gain,
        "mean_delta_bits_per_trial_m4_vs_m3": float(aligned.delta_bits_m4_vs_m3.mean()),
        "session_level_standard_error": session_se,
        "selection_rule": "select_M4_only_if_mean_gain_exceeds_one_session_level_SE",
        "selected_model": selected,
        "brier_m4": float(aligned.squared_error_m4.mean()),
        "brier_m3": float(aligned.squared_error_m3.mean()),
    }])
    long = pd.concat([
        aligned.assign(model="M4", probability=aligned.model_probability_long_m4),
        aligned.assign(model="M3", probability=aligned.model_probability_long_m3),
    ], ignore_index=True)
    long["calibration_bin"] = pd.cut(long.probability, np.linspace(0, 1, 11), include_lowest=True)
    calibration = long.groupby(["model", "calibration_bin"], observed=False).agg(
        mean_predicted=("probability", "mean"), observed_long=("choice_long_m4", "mean"),
        n=("choice_long_m4", "size"),
    ).dropna().reset_index()
    aligned = aligned.sort_values(["session_id", "trial_index"]).reset_index(drop=True)
    aligned["block_run_position"] = aligned.groupby("session_id", observed=False)["true_block_m4"].transform(
        lambda value: value.groupby(value.ne(value.shift()).cumsum()).cumcount() + 1
    )
    aligned["context"] = np.where(aligned.block_run_position <= 10, "early_1_10", "late_11_plus")
    context = aligned.groupby(["true_block_m4", "context"], observed=False).agg(
        n=("delta_elpd_m4_vs_m3", "size"),
        mean_delta_bits_per_trial_m4_vs_m3=("delta_bits_m4_vs_m3", "mean"),
    ).reset_index().rename(columns={"true_block_m4": "true_block"})
    rare_mask = (
        (aligned.true_block_m4.eq("S") & aligned.category_long_m4.eq(1))
        | (aligned.true_block_m4.eq("L") & aligned.category_long_m4.eq(0))
    )
    aligned["event_frequency"] = np.where(rare_mask, "rare", "common")
    rare_common = aligned.groupby("event_frequency", observed=False).agg(
        n=("delta_elpd_m4_vs_m3", "size"),
        mean_delta_bits_per_trial_m4_vs_m3=("delta_bits_m4_vs_m3", "mean"),
    ).reset_index()
    records: list[dict[str, float | int | str]] = []
    for session_id, frame in aligned.groupby("session_id", sort=False, observed=False):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        switches = np.flatnonzero(frame.true_block_m4.ne(frame.true_block_m4.shift()).to_numpy())[1:]
        for event, center in enumerate(switches):
            for index in range(max(0, center - 10), min(len(frame), center + 11)):
                records.append({
                    "session_id": session_id, "event": event,
                    "relative_trial": index - center,
                    "delta_bits_m4_vs_m3": float(frame.loc[index, "delta_bits_m4_vs_m3"]),
                })
    transition = pd.DataFrame(records)
    if not transition.empty:
        transition = transition.groupby("relative_trial", observed=False).agg(
            n=("delta_bits_m4_vs_m3", "size"),
            mean_delta_bits_per_trial_m4_vs_m3=("delta_bits_m4_vs_m3", "mean"),
        ).reset_index()
    return M4M3ValidationComparison(overall, per_session, calibration, context, transition, rare_common)


def save_m4_trial_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write trialwise M4 predictions to CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return destination
