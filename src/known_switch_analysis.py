"""Shared trial export helpers for the two oracle-switch models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch


def fit_summary_has_trajectory_data(fit_summary: dict) -> bool:
    """Return whether any recorded history row actually carries a per-step
    constrained-parameter snapshot.

    A fit summary produced by a pre-upgrade ``fit.py`` (or one whose saved
    JSON was reused across a ``--resume`` from before this field existed)
    has none of these: its trajectory figure would render empty even though
    the figure-schema version on the manifest looks current. Callers should
    fold this into their own ``figure_outputs_are_current`` check (alongside
    the schema-version string) so that mixing an old fit result with new
    figure code is never mistaken for a complete, current figure set.
    """
    return any(
        "constrained_parameters_after_step" in row
        for run in fit_summary.get("runs", [])
        for row in run.get("history", [])
    )


def generate_known_switch_fit_figures(
    fit_summary: dict, output: Path, *, parameter_names: tuple[str, ...], model_label: str
) -> dict[str, list[str]]:
    """Plot the fit fields actually recorded by the known-switch optimizers."""
    import matplotlib.pyplot as plt
    from m0_sensory_v1.analysis import _save_figure

    output.mkdir(parents=True, exist_ok=True)
    winner = fit_summary.get("winner_start_id")
    nll_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in fit_summary.get("runs", []):
        start = int(run["start_id"])
        history = pd.DataFrame(run.get("history", []))
        if history.empty:
            continue
        style = {"alpha": 1.0 if start == winner else 0.32, "linewidth": 2.4 if start == winner else 1.0}
        axes[0].plot(history["step"], history["fresh_training_mean_nll"], label=f"Start {start}", **style)
        axes[1].plot(history["step"], history["validation_mean_nll"], **style)
        for row in history.to_dict("records"):
            nll_rows.append({"start_id": start, **row})
    pd.DataFrame(nll_rows).to_csv(output / "nll_history.csv", index=False)
    axes[0].set_title("Fresh training mean NLL")
    axes[1].set_title("Validation mean NLL")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25, linestyle=":")
    if axes[0].lines:
        axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"{model_label} optimization history")
    nll_files = _save_figure(fig, output / "nll_history")

    parameter_rows = []
    for run in fit_summary.get("runs", []):
        values = run.get("best_constrained_parameters", [])
        if len(values) == len(parameter_names):
            parameter_rows.append({"start_id": int(run["start_id"]), **dict(zip(parameter_names, values, strict=True))})
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(output / "best_parameters_by_start.csv", index=False)
    columns = 2 if len(parameter_names) <= 4 else 3
    rows = int(np.ceil(len(parameter_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for index, name in enumerate(parameter_names):
        if not parameters.empty:
            colors = ["#c62828" if value == winner else "#777777" for value in parameters["start_id"]]
            flat[index].scatter(parameters["start_id"], parameters[name], c=colors)
        flat[index].set_title(name)
        flat[index].set_xlabel("Start ID")
        flat[index].grid(alpha=0.25, linestyle=":")
    for axis in flat[len(parameter_names):]:
        axis.set_visible(False)
    fig.suptitle(f"{model_label} restored-best parameters (winner in red)")
    parameter_files = _save_figure(fig, output / "best_parameters_by_start")

    trajectory_files = _plot_known_switch_parameter_trajectories(
        fit_summary, output, parameter_names=parameter_names, model_label=model_label,
    )
    return {"nll": nll_files, "parameters": parameter_files, "trajectories": trajectory_files}


def _plot_known_switch_parameter_trajectories(
    fit_summary: dict, output: Path, *, parameter_names: tuple[str, ...], model_label: str
) -> list[str]:
    """Plot constrained parameter values after every optimizer step, every start.

    Requires each history row to carry ``constrained_parameters_after_step``
    (added alongside the scalar NLL/gradient fields once known-switch's own
    ``fit.py`` started recording it); older saved fit summaries without that
    field simply produce an empty figure rather than raising.

    The x-axis is the row's position in ``history`` (a running count across
    the whole run), not the recorded ``step`` field: hybrid mode numbers its
    Adam-warmup and L-BFGS phases independently from 1, so plotting against
    ``step`` directly would draw a backward, duplicated-x segment where the
    second phase begins.
    """
    import matplotlib.pyplot as plt
    from m0_sensory_v1.analysis import _save_figure

    winner = fit_summary.get("winner_start_id")
    columns = 2 if len(parameter_names) <= 4 else 3
    rows = int(np.ceil(len(parameter_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    records: list[dict[str, object]] = []
    for run in fit_summary.get("runs", []):
        start = int(run["start_id"])
        initial = run.get("initial_constrained_parameters")
        points = [(0, initial)] if initial is not None and len(initial) == len(parameter_names) else []
        history_rows = [
            row for row in run.get("history", []) if "constrained_parameters_after_step" in row
        ]
        points += [
            (index + 1, row["constrained_parameters_after_step"])
            for index, row in enumerate(history_rows)
        ]
        if not points:
            continue
        for history_index, values in points:
            records.append({
                "start_id": start, "history_index": history_index,
                **dict(zip(parameter_names, values, strict=True)),
            })
        frame = pd.DataFrame([record for record in records if record["start_id"] == start])
        style = {"alpha": 0.95 if start == winner else 0.32, "linewidth": 2.3 if start == winner else 1.0}
        for index, name in enumerate(parameter_names):
            flat[index].plot(frame["history_index"], frame[name], label=f"Start {start}" if index == 0 else None, **style)
    pd.DataFrame(records).to_csv(output / "parameter_history.csv", index=False)
    for axis, name in zip(flat, parameter_names):
        axis.set_title(name)
        axis.set_xlabel("Optimizer step (sequential across phases)")
        axis.set_ylabel("Parameter value")
        axis.grid(alpha=0.25, linestyle=":")
    for axis in flat[len(parameter_names):]:
        axis.set_visible(False)
    if flat[0].lines:
        flat[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"{model_label} constrained parameter trajectories (winner emphasized)")
    return _save_figure(fig, output / "parameter_trajectories")


def build_known_switch_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry,
    *,
    parameter_names: tuple[str, ...],
    decode: Callable,
    make_session: Callable,
    evaluate_session: Callable,
    allow_test: bool = False,
) -> pd.DataFrame:
    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "feedback_valid", "true_block", "split_role",
    }
    if missing := sorted(required.difference(canonical_trials.columns)):
        raise KeyError(f"Known-switch analysis table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("Known-switch test predictions require allow_test=True")
    if unknown := roles.difference({"train", "validation", "test"}):
        raise ValueError(f"Unknown split roles: {sorted(unknown)}")
    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (len(parameter_names),):
        raise ValueError(f"Expected {len(parameter_names)} known-switch raw parameters")
    parameters = decode(raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon)
    constrained = [float(value) for value in parameters.as_tensor().detach().cpu()]
    parts: list[pd.DataFrame] = []
    for (_, _), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        role = str(frame["split_role"].iloc[0])
        session = make_session(frame, split_role=role)
        with torch.no_grad():
            evaluation = evaluate_session(
                session, parameters, registry.task_moments,
                prior_probability_epsilon=registry.prior_probability_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
                quadratic_tolerance=registry.quadratic_tolerance,
                discriminant_tolerance=registry.discriminant_tolerance,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        direct = {
            "model_probability_long": prediction.probability_long,
            "model_non_lapse_probability_long": prediction.non_lapse_probability_long,
            "model_prior_long": prediction.prior_long,
            "model_decision_lower_boundary_s": prediction.decision_lower_boundary_s,
            "model_decision_upper_boundary_s": prediction.decision_upper_boundary_s,
            "model_sensory_sd_s": prediction.sensory_sd_s,
            "model_biased_log_odds_pre": prediction.biased_log_odds_pre,
            "model_biased_log_odds_post": prediction.biased_log_odds_post,
            "model_in_neutral": prediction.in_neutral,
            "model_is_switch": prediction.is_switch,
        }
        for name, values in direct.items():
            part[name] = values.detach().cpu().numpy()
        part["model_decision_regime"] = prediction.decision_regime
        for index, state in enumerate(("N", "S", "L")):
            part[f"model_pre_choice_probability_{state}"] = (
                prediction.pre_choice_block_probability[:, index].detach().cpu().numpy()
            )
            part[f"model_post_feedback_probability_{state}"] = (
                prediction.post_feedback_block_probability[:, index].detach().cpu().numpy()
            )
        part["per_trial_negative_log_likelihood"] = (
            evaluation.per_trial_negative_log_likelihood.detach().cpu().numpy()
        )
        part["model_evaluated"] = prediction.valid_choice_mask.detach().cpu().numpy()
        part["model_feedback_updated"] = prediction.feedback_update_mask.detach().cpu().numpy()
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    category = pd.to_numeric(result["category_long"], errors="coerce")
    choice = pd.to_numeric(result["choice_long"], errors="coerce")
    probability = result["model_probability_long"].astype(float)
    result["trial_type"] = np.select(
        [category.eq(0), category.eq(1)], ["short", "long"], default="undefined"
    )
    result["block_type"] = result["true_block"].map(
        {"N": "neutral", "S": "short_block", "L": "long_block"}
    ).fillna("undefined")
    result["mouse_correct"] = np.where(
        choice.isin([0, 1]) & category.isin([0, 1]), (choice == category).astype(float), np.nan
    )
    result["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan)
    )
    result["model_choice_long"] = np.where(
        probability.notna(), (probability >= 0.5).astype(float), np.nan
    )
    result["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (result["model_choice_long"] == category).astype(float), np.nan,
    )
    result["model_probability_true_block"] = np.nan
    for state in ("N", "S", "L"):
        selected = result["true_block"].eq(state)
        result.loc[selected, "model_probability_true_block"] = result.loc[
            selected, f"model_pre_choice_probability_{state}"
        ]
    for name, value in zip(parameter_names, constrained, strict=True):
        result[f"fitted_{name}"] = value
    sort_columns = [
        name for name in ("split_role", "session_date", "session_id", "trial_index")
        if name in result.columns
    ]
    return result.sort_values(sort_columns).reset_index(drop=True)


def save_known_switch_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    parameter_names: tuple[str, ...],
    test_unlocked: bool,
) -> dict:
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
    ).reset_index()
    performance.to_csv(output / "performance_summary.csv", index=False)
    winner_id = fit_summary.get("winner_start_id")
    winner = next(
        (run for run in fit_summary.get("runs", []) if run.get("start_id") == winner_id), None
    )
    fitted = None if winner is None else dict(
        zip(parameter_names, winner["best_constrained_parameters"], strict=True)
    )
    summary = {
        "output_schema_version": "known_switch_output_v1",
        "model_id": fit_summary.get("model_id"), "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": winner_id, "fitted_parameters": fitted,
        "switch_timing": "oracle switch indicator; block identity unavailable to model",
        "session_state_reset": "neutral independently at every physical session start",
        "test_unlocked": test_unlocked, "saved_trial_count": int(len(predictions)),
        "saved_valid_decision_count": int(predictions["model_evaluated"].sum()),
    }
    destination = output / "model_output_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return summary
