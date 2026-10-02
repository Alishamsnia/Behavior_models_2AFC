"""Auditable trial-level exports for M2_dynamic_previous_wsls_bias_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER, decode_m2_dynamic_previous_wsls_bias_raw_parameters
from .session import evaluate_m2_dynamic_previous_wsls_bias_session, m2_dynamic_previous_wsls_bias_session_trials_from_frame


M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES = M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER


def build_m2_dynamic_previous_wsls_bias_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry,
    *,
    allow_test: bool = False,
    include_full_run_length_distributions: bool = False,
    full_distribution_session_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rebuild M2_dynamic_previous_wsls_bias independently per session and
    export its causal trajectory. The 130 marginal run-length columns are
    emitted only when explicitly requested, mirroring
    M2_dynamic_block_length_v1's own analysis.py convention."""

    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "feedback_valid", "true_block", "split_role",
    }
    if missing := sorted(required.difference(canonical_trials.columns)):
        raise KeyError(f"Canonical M2_dynamic_previous_wsls_bias table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("M2_dynamic_previous_wsls_bias test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown M2_dynamic_previous_wsls_bias split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (9,):
        raise ValueError("M2_dynamic_previous_wsls_bias raw_parameters must contain exactly nine values")
    parameters = decode_m2_dynamic_previous_wsls_bias_raw_parameters(
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
        session = m2_dynamic_previous_wsls_bias_session_trials_from_frame(
            frame, split_role=str(frame.split_role.iloc[0])
        )
        with torch.no_grad():
            evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(
                session, parameters, registry.task_moments,
                run_length_center=registry.run_length_center,
                maximum_run_length=registry.maximum_represented_run_length,
                first_long_probability=registry.first_long_probability,
                initial_pi=registry.initial_pi,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        direct = {
            "model_probability_long": prediction.probability_long,
            "model_non_lapse_probability_long": prediction.non_lapse_probability_long,
            "model_bias_mixed_probability_long": prediction.bias_mixed_probability_long,
            "model_non_bias_probability_long": prediction.non_bias_probability_long,
            "model_prior_long": prediction.prior_long,
            "model_previous_wsls_state": prediction.previous_wsls_state,
            "model_decision_lower_boundary_s": prediction.decision_lower_boundary_s,
            "model_decision_upper_boundary_s": prediction.decision_upper_boundary_s,
            "model_sensory_sd_s": prediction.sensory_sd_s,
            "model_emission_normalizer": prediction.emission_normalizer,
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
        prototypes = prediction.subjective_prototypes.detach().cpu().numpy()
        for index, state in enumerate(registry.state_order):
            model_columns[f"model_pre_choice_probability_{state}"] = prediction.pre_choice_block_probability[:, index].cpu().numpy()
            model_columns[f"model_post_feedback_probability_{state}"] = prediction.post_feedback_block_probability[:, index].cpu().numpy()
            model_columns[f"model_subjective_prototype_{state}"] = float(prototypes[index])
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
        part = pd.concat((frame.copy(), pd.DataFrame(model_columns, index=frame.index)), axis=1)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    probability = pd.to_numeric(result.model_probability_long, errors="coerce")
    choice = pd.to_numeric(result.choice_long, errors="coerce")
    category = pd.to_numeric(result.category_long, errors="coerce")
    derived = pd.DataFrame(index=result.index)
    derived["trial_type"] = np.select([category.eq(0), category.eq(1)], ["short", "long"], default="undefined")
    derived["block_type"] = result["true_block"].map(
        {"N": "neutral", "S": "short_block", "L": "long_block"}
    ).fillna("undefined")
    derived["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1 - probability, np.nan)
    )
    derived["mouse_correct"] = np.where(
        choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan
    )
    derived["model_choice_long"] = np.where(probability.notna(), (probability >= 0.5).astype(float), np.nan)
    derived["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (derived["model_choice_long"] == category).astype(float), np.nan,
    )
    result["model_probability_true_block"] = np.nan
    for state in registry.state_order:
        selected = result.true_block.eq(state)
        result.loc[selected, "model_probability_true_block"] = result.loc[selected, f"model_pre_choice_probability_{state}"]
    for name, value in zip(M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES, parameters.as_tensor().tolist(), strict=True):
        derived[f"fitted_{name}"] = float(value)
    hazard_at_center = float(parameters.hazard(
        torch.tensor(float(registry.run_length_center), dtype=torch.float64), center=registry.run_length_center
    ))
    derived["fitted_hazard_at_center"] = hazard_at_center
    derived["fitted_delta0_distance_from_log4"] = abs(float(parameters.delta0) - np.log(4.0))
    derived["fitted_alpha"] = float(parameters.alpha)
    derived["fitted_eta"] = float(parameters.eta)
    result = pd.concat((result, derived), axis=1)
    sort_columns = [name for name in ("split_role", "session_date", "session_id", "trial_index") if name in result]
    return result.sort_values(sort_columns).reset_index(drop=True)


def build_m2_dynamic_previous_wsls_bias_hazard_curve(raw_parameters: Iterable[float], registry) -> pd.DataFrame:
    """Return tied neutral and biased hazards across represented run lengths."""

    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    parameters = decode_m2_dynamic_previous_wsls_bias_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    run = torch.arange(1, registry.maximum_represented_run_length + 1, dtype=torch.float64)
    with torch.no_grad():
        biased_hazard = parameters.hazard(run, center=registry.run_length_center)
        neutral_hazard = biased_hazard / 2.0
    return pd.DataFrame({
        "represented_run_length": run.to(torch.int64).numpy(),
        "hazard": biased_hazard.numpy(),
        "biased_hazard": biased_hazard.numpy(),
        "neutral_hazard": neutral_hazard.numpy(),
        "is_center": run.eq(registry.run_length_center).numpy(),
        "is_capped_tail": run.eq(registry.maximum_represented_run_length).numpy(),
    })


def evaluate_m2_dynamic_previous_wsls_bias_bias_evidence_dependence(
    predictions: pd.DataFrame, *, duration_bins: int = 10,
) -> pd.DataFrame:
    """Bin evaluated trials by sensory-evidence strength and report the
    bias's own marginal contribution per bin.

    Evidence strength is approximated by the trial's absolute deviation of
    ``duration_s`` from the fitted decision boundary
    (``model_decision_upper_boundary_s``, the same quantity the causal
    per-trial loop already exports); the bias magnitude reported is
    ``|model_bias_mixed_probability_long - model_non_lapse_probability_long|``,
    i.e. ``eta*|pi_t-r_t|``. Because the bias is mixed with the non-lapse
    probability *after* the prior has already been Bayes-combined with the
    evidence but *before* lapse, Bayes' rule itself never down-weights the
    ``eta`` coefficient by evidence strength the way it would a
    prior-mediated effect. This does not by itself imply
    ``|pi_t-r_t|`` is flat in evidence strength -- ``r_t`` saturates toward
    0 or 1 as evidence strengthens while ``pi_t`` is fixed per trial
    (previous-category, not duration-dependent), so their disagreement can
    grow or shrink with evidence depending on which side ``pi_t`` sits on.
    There is no rigorously derived null shape here; this function is an
    exploratory diagnostic, not a test that must pass or match a specific
    expected signature.
    """

    valid = predictions.loc[predictions["model_evaluated"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["evidence_bin", "mean_evidence_strength", "mean_bias_magnitude", "trial_count"])
    valid["evidence_strength"] = (valid["duration_s"] - valid["model_decision_upper_boundary_s"]).abs()
    valid["bias_magnitude"] = (
        valid["model_bias_mixed_probability_long"] - valid["model_non_lapse_probability_long"]
    ).abs()
    valid["evidence_bin"] = pd.qcut(valid["evidence_strength"], q=duration_bins, duplicates="drop")
    summary = valid.groupby("evidence_bin", observed=True).agg(
        mean_evidence_strength=("evidence_strength", "mean"),
        mean_bias_magnitude=("bias_magnitude", "mean"),
        trial_count=("bias_magnitude", "size"),
    ).reset_index()
    summary["evidence_bin"] = summary["evidence_bin"].astype(str)
    return summary


def save_m2_dynamic_previous_wsls_bias_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Save M2_dynamic_previous_wsls_bias predictions, per-session
    performance, and model inventory."""

    output = Path(results_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "trial_predictions.csv", index=False)
    valid = predictions.loc[predictions["model_evaluated"].astype(bool)]
    performance = valid.groupby(["split_role", "session_id"], observed=False).agg(
        n_valid_decisions=("model_evaluated", "size"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
        model_mean_nll=("per_trial_negative_log_likelihood", "mean"),
        mean_prior_long=("model_prior_long", "mean"),
        mean_probability_true_block=("model_probability_true_block", "mean"),
        mean_previous_wsls_state=("model_previous_wsls_state", "mean"),
        mean_tail_probability_pre=("model_tail_probability_pre", "mean"),
        max_tail_probability_pre=("model_tail_probability_pre", "max"),
    ).reset_index()
    performance.to_csv(output / "performance_summary.csv", index=False)
    winner_id = fit_summary.get("winner_start_id")
    winner = next((run for run in fit_summary.get("runs", []) if run["start_id"] == winner_id), None)
    fitted = None
    if winner is not None:
        fitted = dict(zip(M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES, winner["best_constrained_parameters"]))
        fitted["hazard_at_center"] = float(1.0 / (1.0 + np.exp(-fitted["ah"])))
        fitted["theta_N"] = 0.5
        fitted["theta_S"] = float(1.0 / (1.0 + np.exp(fitted["delta0"])))
        fitted["theta_L"] = 1.0 - fitted["theta_S"]
        fitted["alpha"] = 1.0
        fitted["eta"] = float(1.0 / (1.0 + np.exp(-fitted["eta_logit"])))
    summary = {
        "output_schema_version": "m2_dynamic_previous_wsls_bias_output_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": winner_id,
        "fitted_parameters": fitted,
        "state_order": ["N", "S", "L"],
        "prototype_policy": "static fitted prototypes (delta0), no learning; duration-dependent HSMM hazard, unchanged from M2_dynamic_block_length_v1",
        "state_reset": "xi_pre=one-hot(N,r=1) and pi_1=0.5 independently at every session",
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t) -> lapse_mixture(m_t)",
        "feedback_order": "predict_choice_then_category_update_then_hsmm_transition; previous-wsls-bias update (pi_next=category) is gated by feedback_valid, independent of choice validity",
        "outer_objective": "choice_likelihood_only",
        "independent_of_m4_hsmm_delta_v1": True,
        "independent_of_m1_action_kernel_v1": True,
        "independent_of_wsls_core_v1": True,
        "test_unlocked": test_unlocked,
        "saved_trial_count": int(len(predictions)),
        "saved_valid_decision_count": int(predictions["model_evaluated"].sum()),
    }
    destination = output / "model_output_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return summary
