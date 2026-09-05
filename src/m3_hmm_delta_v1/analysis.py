"""Auditable trial-level exports and M3-versus-M2 paired comparison."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import M3_PARAMETER_ORDER, decode_m3_raw_parameters
from .session import evaluate_m3_session, m3_session_trials_from_frame


@dataclass(frozen=True)
class M3M2ValidationComparison:
    """Aligned validation evidence and the prespecified scientific strata."""

    overall: pd.DataFrame
    per_session: pd.DataFrame
    context: pd.DataFrame
    transition_aligned: pd.DataFrame
    surprising_category_aligned: pd.DataFrame
    psychometric_by_block: pd.DataFrame
    calibration: pd.DataFrame


def build_m3_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Rebuild every session independently and export all learning states."""

    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "feedback_valid", "true_block", "split_role",
    }
    missing = sorted(required.difference(canonical_trials.columns))
    if missing:
        raise KeyError(f"Canonical M3 table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("M3 test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown M3 split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (8,):
        raise ValueError("M3 raw_parameters must contain exactly eight values")
    parameters = decode_m3_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    fitted = [float(value) for value in parameters.as_tensor()]
    parts: list[pd.DataFrame] = []
    for (_, _), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        role = str(frame["split_role"].iloc[0])
        session = m3_session_trials_from_frame(frame, split_role=role)
        with torch.no_grad():
            evaluation = evaluate_m3_session(
                session,
                parameters,
                registry.task_moments,
                first_long_probability=registry.first_long_probability,
                delta_min=registry.delta_min,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        part["model_probability_long"] = prediction.probability_long.cpu().numpy()
        part["model_non_lapse_probability_long"] = prediction.non_lapse_probability_long.cpu().numpy()
        part["model_prior_long"] = prediction.prior_long.cpu().numpy()
        part["model_decision_lower_boundary_s"] = prediction.decision_lower_boundary_s.cpu().numpy()
        part["model_decision_upper_boundary_s"] = prediction.decision_upper_boundary_s.cpu().numpy()
        part["model_decision_regime"] = prediction.decision_regime
        part["model_sensory_sd_s"] = prediction.sensory_sd_s.cpu().numpy()
        part["model_emission_normalizer"] = prediction.emission_normalizer.cpu().numpy()
        part["model_delta_pre"] = prediction.delta_pre.cpu().numpy()
        part["model_delta_update_signal"] = prediction.delta_update_signal.cpu().numpy()
        part["model_delta_unprojected_proposal"] = prediction.delta_unprojected_proposal.cpu().numpy()
        part["model_delta_post"] = prediction.delta_post.cpu().numpy()
        part["model_projection_activated"] = prediction.projection_activated.cpu().numpy()
        part["model_projection_cumulative_count"] = (
            prediction.projection_activated.to(torch.int64).cumsum(0).cpu().numpy()
        )
        for index, state in enumerate(registry.state_order):
            part[f"model_pre_choice_probability_{state}"] = prediction.pre_choice_block_probability[:, index].cpu().numpy()
            part[f"model_post_feedback_probability_{state}"] = prediction.post_feedback_block_probability[:, index].cpu().numpy()
            part[f"model_responsibility_{state}"] = prediction.posterior_block_responsibility[:, index].cpu().numpy()
            part[f"model_subjective_prototype_{state}"] = prediction.subjective_prototypes_pre[:, index].cpu().numpy()
        part["per_trial_negative_log_likelihood"] = evaluation.per_trial_negative_log_likelihood.cpu().numpy()
        part["model_evaluated"] = prediction.valid_choice_mask.cpu().numpy()
        part["model_feedback_updated"] = prediction.feedback_update_mask.cpu().numpy()
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    probability = result["model_probability_long"].astype(float)
    choice = pd.to_numeric(result["choice_long"], errors="coerce")
    category = pd.to_numeric(result["category_long"], errors="coerce")
    result["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan)
    )
    result["mouse_correct"] = np.where(
        choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan
    )
    result["model_probability_true_block"] = np.nan
    for state in registry.state_order:
        selected = result["true_block"].eq(state)
        result.loc[selected, "model_probability_true_block"] = result.loc[
            selected, f"model_pre_choice_probability_{state}"
        ]
    for name, value in zip(M3_PARAMETER_ORDER, fitted, strict=True):
        result[f"fitted_{name}"] = value
    hazard = float(parameters.hazard)
    result["fitted_hazard"] = hazard
    result["fitted_biased_hazard"] = hazard
    result["derived_neutral_hazard"] = hazard / 2.0
    result["fitted_biased_mean_block_duration_trials"] = 1.0 / hazard
    result["derived_neutral_mean_block_duration_trials"] = 2.0 / hazard
    sort_columns = [
        name for name in ("split_role", "session_date", "session_id", "trial_index")
        if name in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def compare_m3_to_m2_predictions(
    m3_predictions: pd.DataFrame,
    m2_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return paired session-level choice-likelihood gains in bits per trial."""

    keys = [name for name in ("mouse_id", "split_role", "session_id", "trial_index")
            if name in m3_predictions.columns and name in m2_predictions.columns]
    needed = {*keys, "per_trial_negative_log_likelihood", "model_evaluated"}
    for name, frame in (("M3", m3_predictions), ("M2", m2_predictions)):
        if missing := needed.difference(frame.columns):
            raise KeyError(f"{name} predictions missing columns: {sorted(missing)}")
    ordered = keys + ["per_trial_negative_log_likelihood", "model_evaluated"]
    merged = m3_predictions[ordered].merge(
        m2_predictions[ordered], on=keys, suffixes=("_m3", "_m2"),
        validate="one_to_one", how="outer", indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("M3 and M2 prediction timelines are not identical")
    valid = merged["model_evaluated_m3"].astype(bool) & merged["model_evaluated_m2"].astype(bool)
    merged = merged.loc[valid].copy()
    merged["delta_bits_m3_vs_m2"] = (
        merged["per_trial_negative_log_likelihood_m2"]
        - merged["per_trial_negative_log_likelihood_m3"]
    ) / np.log(2.0)
    return merged.groupby(["split_role", "session_id"], observed=False).agg(
        n_valid_trials=("delta_bits_m3_vs_m2", "size"),
        total_delta_bits_m3_vs_m2=("delta_bits_m3_vs_m2", "sum"),
        mean_delta_bits_per_trial_m3_vs_m2=("delta_bits_m3_vs_m2", "mean"),
    ).reset_index()


def _aligned_m3_m2_trials(
    m3_predictions: pd.DataFrame,
    m2_predictions: pd.DataFrame,
    *,
    split_role: str = "validation",
) -> pd.DataFrame:
    """Align predictions and prove that both models saw identical observations."""

    keys = [name for name in ("mouse_id", "split_role", "session_id", "trial_index")
            if name in m3_predictions.columns and name in m2_predictions.columns]
    observation_columns = [
        name for name in (
            "duration_s", "choice_long", "category_long", "feedback_valid", "true_block"
        ) if name in m3_predictions.columns and name in m2_predictions.columns
    ]
    model_columns = [
        "model_probability_long", "per_trial_negative_log_likelihood", "model_evaluated"
    ]
    for label, frame in (("M3", m3_predictions), ("M2", m2_predictions)):
        missing = set(keys + model_columns).difference(frame.columns)
        if missing:
            raise KeyError(f"{label} predictions missing columns: {sorted(missing)}")
    left = m3_predictions.loc[
        m3_predictions["split_role"].astype(str).eq(split_role),
        keys + observation_columns + model_columns,
    ]
    right = m2_predictions.loc[
        m2_predictions["split_role"].astype(str).eq(split_role),
        keys + observation_columns + model_columns,
    ]
    merged = left.merge(
        right, on=keys, how="outer", suffixes=("_m3", "_m2"),
        validate="one_to_one", indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        raise ValueError(f"M3/M2 {split_role} timelines are empty or misaligned")
    for column in observation_columns:
        lvalue, rvalue = merged[f"{column}_m3"], merged[f"{column}_m2"]
        equal = lvalue.eq(rvalue) | (lvalue.isna() & rvalue.isna())
        if not bool(equal.all()):
            raise ValueError(f"M3/M2 observations disagree in {column}")
        merged[column] = lvalue
    if not bool(
        merged["model_evaluated_m3"].astype(bool)
        .eq(merged["model_evaluated_m2"].astype(bool)).all()
    ):
        raise ValueError("M3/M2 evaluation masks disagree")
    merged = merged.loc[merged["model_evaluated_m3"].astype(bool)].copy()
    merged["delta_elpd_m3_vs_m2"] = (
        merged["per_trial_negative_log_likelihood_m2"]
        - merged["per_trial_negative_log_likelihood_m3"]
    )
    merged["delta_bits_m3_vs_m2"] = merged.delta_elpd_m3_vs_m2 / np.log(2.0)
    return merged


def build_validation_model_metrics(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    split_role: str = "validation",
) -> pd.DataFrame:
    """Compute directly comparable NLL, ELPD, Brier, and calibration error."""

    rows = []
    for model, frame in predictions_by_model.items():
        required = {
            "split_role", "model_evaluated", "model_probability_long",
            "per_trial_negative_log_likelihood", "choice_long",
        }
        if missing := required.difference(frame.columns):
            raise KeyError(f"{model} predictions missing columns: {sorted(missing)}")
        selected = frame.loc[
            frame.split_role.astype(str).eq(split_role) & frame.model_evaluated.astype(bool)
        ].copy()
        if selected.empty:
            raise ValueError(f"{model} has no evaluated {split_role} trials")
        probability = pd.to_numeric(selected.model_probability_long).to_numpy(float)
        choice = pd.to_numeric(selected.choice_long).to_numpy(float)
        nll = pd.to_numeric(selected.per_trial_negative_log_likelihood).to_numpy(float)
        bins = pd.cut(probability, np.linspace(0.0, 1.0, 11), include_lowest=True)
        calibration = pd.DataFrame({"bin": bins, "p": probability, "y": choice}).groupby(
            "bin", observed=False
        ).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).dropna()
        ece = float(np.sum(calibration.n * np.abs(calibration.p - calibration.y)) / len(selected))
        rows.append({
            "model": model, "split_role": split_role, "n_valid_trials": len(selected),
            "total_nll": float(nll.sum()), "mean_nll": float(nll.mean()),
            "elpd": float(-nll.sum()), "brier_score": float(np.mean((probability-choice) ** 2)),
            "expected_calibration_error_10_bins": ece,
        })
    return pd.DataFrame(rows)


def build_m3_m2_validation_comparison(
    m3_predictions: pd.DataFrame,
    m2_predictions: pd.DataFrame,
    *,
    split_role: str = "validation",
    early_trial_cutoff: int = 10,
    switch_window: int = 10,
) -> M3M2ValidationComparison:
    """Build all prespecified paired M3-versus-M2 validation comparisons."""

    if early_trial_cutoff <= 0 or switch_window <= 0:
        raise ValueError("Context cutoffs must be positive")
    paired = _aligned_m3_m2_trials(
        m3_predictions, m2_predictions, split_role=split_role
    )
    session_keys = [name for name in ("mouse_id", "split_role", "session_id") if name in paired]
    per_session = paired.groupby(session_keys, observed=False).agg(
        n_valid_trials=("delta_elpd_m3_vs_m2", "size"),
        delta_elpd_m3_vs_m2=("delta_elpd_m3_vs_m2", "sum"),
        total_delta_bits_m3_vs_m2=("delta_bits_m3_vs_m2", "sum"),
    ).reset_index()
    per_session["mean_delta_bits_per_trial_m3_vs_m2"] = (
        per_session.total_delta_bits_m3_vs_m2 / per_session.n_valid_trials
    )
    session_gain = per_session.delta_elpd_m3_vs_m2.to_numpy(float)
    mean_session_se = float(np.std(session_gain, ddof=1) / np.sqrt(len(session_gain))) if len(session_gain) > 1 else np.nan
    total_elpd_se = float(np.std(session_gain, ddof=1) * np.sqrt(len(session_gain))) if len(session_gain) > 1 else np.nan
    total_gain = float(paired.delta_elpd_m3_vs_m2.sum())
    overall = pd.DataFrame([{
        "split_role": split_role, "n_valid_trials": len(paired),
        "n_sessions": len(per_session), "delta_elpd_m3_vs_m2": total_gain,
        "delta_bits_per_trial_m3_vs_m2": float(paired.delta_bits_m3_vs_m2.mean()),
        "mean_session_delta_elpd_standard_error": mean_session_se,
        "total_delta_elpd_clustered_standard_error": total_elpd_se,
        "exceeds_one_clustered_standard_error": (
            bool(total_gain > total_elpd_se) if np.isfinite(total_elpd_se) else False
        ),
    }])

    paired = paired.sort_values(session_keys + ["trial_index"]).copy()
    paired["block_run_position"] = paired.groupby(session_keys, observed=False)["true_block"].transform(
        lambda values: values.groupby(values.ne(values.shift()).cumsum()).cumcount() + 1
    )
    paired["timing"] = np.where(
        paired.block_run_position <= early_trial_cutoff,
        f"early_1_{early_trial_cutoff}", f"late_{early_trial_cutoff + 1}_plus",
    )
    paired["congruence"] = np.select(
        [
            ((paired.true_block == "L") & (paired.category_long == 1))
            | ((paired.true_block == "S") & (paired.category_long == 0)),
            paired.true_block.isin(["S", "L"]),
        ], ["common", "rare"], default="neutral_block",
    )
    context = paired.groupby(["timing", "congruence"], observed=False).agg(
        n_valid_trials=("delta_elpd_m3_vs_m2", "size"),
        delta_elpd_m3_vs_m2=("delta_elpd_m3_vs_m2", "sum"),
        mean_delta_bits_per_trial_m3_vs_m2=("delta_bits_m3_vs_m2", "mean"),
    ).reset_index()

    aligned_rows = []
    for session_key, frame in paired.groupby(session_keys, sort=False, observed=False):
        frame = frame.reset_index(drop=True)
        switches = np.flatnonzero(frame.true_block.ne(frame.true_block.shift()).to_numpy())[1:]
        for event, center in enumerate(switches):
            for index in range(max(0, center-switch_window), min(len(frame), center+switch_window+1)):
                aligned_rows.append({
                    "session_key": str(session_key), "event": event,
                    "relative_trial": index-center,
                    "delta_elpd_m3_vs_m2": frame.loc[index, "delta_elpd_m3_vs_m2"],
                    "delta_bits_m3_vs_m2": frame.loc[index, "delta_bits_m3_vs_m2"],
                })
    aligned = pd.DataFrame(aligned_rows)
    transition = (
        aligned.groupby("relative_trial", observed=False).agg(
            n=("event", "size"), delta_elpd_m3_vs_m2=("delta_elpd_m3_vs_m2", "sum"),
            mean_delta_bits_per_trial_m3_vs_m2=("delta_bits_m3_vs_m2", "mean"),
        ).reset_index() if not aligned.empty else pd.DataFrame(
            columns=["relative_trial", "n", "delta_elpd_m3_vs_m2", "mean_delta_bits_per_trial_m3_vs_m2"]
        )
    )

    surprising_rows = []
    for session_key, frame in paired.groupby(session_keys, sort=False, observed=False):
        frame = frame.reset_index(drop=True)
        events = np.flatnonzero(frame.congruence.eq("rare").to_numpy())
        for event, center in enumerate(events):
            for index in range(max(0, center-3), min(len(frame), center+4)):
                surprising_rows.append({
                    "session_key": str(session_key), "event": event,
                    "relative_trial": index-center,
                    "delta_elpd_m3_vs_m2": frame.loc[index, "delta_elpd_m3_vs_m2"],
                    "delta_bits_m3_vs_m2": frame.loc[index, "delta_bits_m3_vs_m2"],
                    "probability_change_m3_minus_m2": (
                        frame.loc[index, "model_probability_long_m3"]
                        - frame.loc[index, "model_probability_long_m2"]
                    ),
                })
    surprising_raw = pd.DataFrame(surprising_rows)
    surprising = (
        surprising_raw.groupby("relative_trial", observed=False).agg(
            n=("event", "size"), delta_elpd_m3_vs_m2=("delta_elpd_m3_vs_m2", "sum"),
            mean_delta_bits_per_trial_m3_vs_m2=("delta_bits_m3_vs_m2", "mean"),
            mean_probability_change_m3_minus_m2=("probability_change_m3_minus_m2", "mean"),
        ).reset_index() if not surprising_raw.empty else pd.DataFrame(columns=[
            "relative_trial", "n", "delta_elpd_m3_vs_m2",
            "mean_delta_bits_per_trial_m3_vs_m2", "mean_probability_change_m3_minus_m2",
        ])
    )

    psychometric = []
    calibration = []
    for label, probability_column in (
        ("M2", "model_probability_long_m2"), ("M3", "model_probability_long_m3")
    ):
        working = paired.copy()
        working["duration_rounded"] = pd.to_numeric(working.duration_s).round(3)
        block = working.groupby(["true_block", "duration_rounded"], observed=False).agg(
            observed_long=("choice_long", "mean"), predicted_long=(probability_column, "mean"),
            n=("choice_long", "size"),
        ).reset_index()
        block["model"] = label
        psychometric.append(block)
        working["calibration_bin"] = pd.cut(
            working[probability_column], np.linspace(0, 1, 11), include_lowest=True
        )
        calibrated = working.groupby("calibration_bin", observed=False).agg(
            mean_predicted=(probability_column, "mean"), observed_long=("choice_long", "mean"),
            n=("choice_long", "size"),
        ).dropna().reset_index()
        calibrated["model"] = label
        calibration.append(calibrated)
    return M3M2ValidationComparison(
        overall=overall, per_session=per_session, context=context,
        transition_aligned=transition, surprising_category_aligned=surprising,
        psychometric_by_block=pd.concat(psychometric, ignore_index=True),
        calibration=pd.concat(calibration, ignore_index=True),
    )


def save_m3_trial_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return destination
