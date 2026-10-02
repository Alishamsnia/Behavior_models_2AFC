"""Trial-level exports and preliminary diagnostics for fitted M0 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model.parameters import decode_raw_parameters
from .model.session import evaluate_session, session_trials_from_frame
from .registry import ModelRegistry


PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa")
ROLE_ORDER = ("train", "validation", "test")
ROLE_LABELS = {"train": "Training", "validation": "Validation", "test": "Test"}


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_trial_predictions(
    canonical_trials: pd.DataFrame,
    raw_parameters: Iterable[float],
    registry: ModelRegistry,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Create one auditable mouse/model record per original task trial.

    Invalid decisions remain in the returned timeline with NaN prediction and
    likelihood fields. The M0 equation used on valid decisions is
    ``p(long)=(1-epsilon_t)*Phi((s-T)/sqrt(w*s**2+s0**2))+epsilon_t/2``,
    where the physical stimulus category selects ``ell_short`` or ``ell_long``.
    """

    required = {
        "mouse_id", "session_id", "trial_index", "duration_s", "choice_long",
        "category_long", "true_block", "outcome", "split_role",
    }
    missing = sorted(required.difference(canonical_trials.columns))
    if missing:
        raise KeyError(f"Canonical analysis table is missing columns: {missing}")
    roles = set(canonical_trials["split_role"].astype(str))
    if "test" in roles and not allow_test:
        raise PermissionError("Test predictions require allow_test=True")
    if unknown := roles.difference(ROLE_ORDER):
        raise ValueError(f"Unknown split roles: {sorted(unknown)}")

    raw = torch.as_tensor(tuple(raw_parameters), dtype=torch.float64)
    if raw.shape != (5,):
        raise ValueError("raw_parameters must contain exactly five values")
    parameters = decode_raw_parameters(
        raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    )
    constrained = [float(value) for value in parameters.as_tensor().detach().cpu()]
    parts: list[pd.DataFrame] = []
    for (_, session_id), frame in canonical_trials.groupby(
        ["split_role", "session_id"], sort=False, observed=False
    ):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        role = str(frame["split_role"].iloc[0])
        session = session_trials_from_frame(frame, split_role=role)  # type: ignore[arg-type]
        with torch.no_grad():
            evaluation = evaluate_session(
                session,
                parameters,
                registry.task_moments,
                log_probability_epsilon=registry.log_probability_epsilon,
            )
        prediction = evaluation.predictions
        part = frame.copy()
        part["model_probability_long"] = prediction.probability_long.cpu().numpy()
        part["model_non_lapse_probability_long"] = (
            prediction.non_lapse_probability_long.cpu().numpy()
        )
        part["model_positive_threshold_s"] = prediction.positive_threshold_s.cpu().numpy()
        part["model_decision_lower_boundary_s"] = prediction.decision_lower_boundary_s.cpu().numpy()
        part["model_decision_upper_boundary_s"] = prediction.decision_upper_boundary_s.cpu().numpy()
        part["model_decision_regime"] = prediction.decision_regime
        part["model_sensory_sd_s"] = prediction.sensory_sd_s.cpu().numpy()
        part["per_trial_negative_log_likelihood"] = (
            evaluation.per_trial_negative_log_likelihood.cpu().numpy()
        )
        part["model_evaluated"] = prediction.valid_choice_mask.cpu().numpy()
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
        choice.isin([0, 1]) & category.isin([0, 1]),
        (choice == category).astype(float),
        np.nan,
    )
    result["model_probability_correct"] = np.where(
        category.eq(1), probability, np.where(category.eq(0), 1.0 - probability, np.nan)
    )
    result["model_choice_long"] = np.where(
        probability.notna(), (probability >= 0.5).astype(float), np.nan
    )
    result["model_choice_correct"] = np.where(
        probability.notna() & category.isin([0, 1]),
        (result["model_choice_long"] == category).astype(float),
        np.nan,
    )
    for name, value in zip(PARAMETER_NAMES, constrained):
        result[f"fitted_{name}"] = value
    return result.sort_values(
        ["split_role", "session_date", "session_id", "trial_index"]
    ).reset_index(drop=True)


def four_bin_psychometric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return session-balanced psychometrics in four bins per trial type.

    This reproduces the previous notebook convention: short and long trials are
    quantile-binned separately after deterministic N(0, 1e-6) jitter (seed 42).
    """

    work = data.loc[
        data["model_evaluated"].astype(bool)
        & data["trial_type"].isin(["short", "long"])
    ].copy()
    work = work.sort_values(["session_date", "session_id", "trial_index"])
    parts: list[pd.DataFrame] = []
    for offset, trial_type in ((0, "short"), (4, "long")):
        subset = work.loc[work["trial_type"].eq(trial_type)].copy()
        if subset.empty:
            continue
        jitter = np.random.RandomState(42).normal(0.0, 1e-6, len(subset))
        labels = pd.qcut(
            subset["duration_s"].to_numpy(float) + jitter,
            q=4,
            labels=False,
            duplicates="drop",
        )
        subset["within_type_bin"] = np.asarray(labels, dtype=int)
        if subset["within_type_bin"].nunique() != 4:
            raise ValueError(f"Expected four {trial_type} bins")
        subset["bin_order"] = subset["within_type_bin"] + offset
        parts.append(subset)
    if not parts:
        return pd.DataFrame()
    binned = pd.concat(parts, ignore_index=True)
    metadata = binned.groupby("bin_order", observed=False).agg(
        bin_center_s=("duration_s", "mean"),
        bin_min_s=("duration_s", "min"),
        bin_max_s=("duration_s", "max"),
        n_trials=("duration_s", "size"),
    ).reset_index()
    per_session = binned.groupby(
        ["bin_order", "session_id"], observed=False
    ).agg(
        mouse_probability_long=("choice_long", "mean"),
        model_probability_long=("model_probability_long", "mean"),
    ).reset_index()
    summary = per_session.groupby("bin_order", observed=False).agg(
        mouse_probability_long=("mouse_probability_long", "mean"),
        mouse_sem=("mouse_probability_long", "sem"),
        model_probability_long=("model_probability_long", "mean"),
        model_sem=("model_probability_long", "sem"),
        n_sessions=("session_id", "nunique"),
    ).reset_index()
    summary = summary.merge(metadata, on="bin_order", validate="one_to_one")
    summary["trial_type"] = np.where(summary["bin_order"] < 4, "short", "long")
    return summary.sort_values("bin_order").reset_index(drop=True)


def event_aligned_performance(
    data: pd.DataFrame,
    event_type: str,
    *,
    window: int,
) -> pd.DataFrame:
    """Align observed and expected correctness to rare trials or block changes."""

    allowed = {
        "rare_long_in_short", "rare_short_in_long", "short_to_long", "long_to_short"
    }
    if event_type not in allowed:
        raise ValueError(f"Unknown event type: {event_type}")
    if window < 0:
        raise ValueError("window must be nonnegative")
    records: list[dict[str, object]] = []
    for session_id, frame in data.groupby("session_id", sort=False, observed=False):
        valid = frame.loc[
            frame["model_evaluated"].astype(bool)
            & frame["block_type"].isin(["short_block", "long_block"])
            & frame["trial_type"].isin(["short", "long"])
        ].sort_values("trial_index").reset_index(drop=True)
        if valid.empty:
            continue
        if event_type == "rare_long_in_short":
            event_indices = np.flatnonzero(
                valid["block_type"].eq("short_block") & valid["trial_type"].eq("long")
            )
        elif event_type == "rare_short_in_long":
            event_indices = np.flatnonzero(
                valid["block_type"].eq("long_block") & valid["trial_type"].eq("short")
            )
        else:
            previous = valid["block_type"].shift(1)
            if event_type == "short_to_long":
                event_indices = np.flatnonzero(
                    previous.eq("short_block") & valid["block_type"].eq("long_block")
                )
            else:
                event_indices = np.flatnonzero(
                    previous.eq("long_block") & valid["block_type"].eq("short_block")
                )
        for event_number, center in enumerate(event_indices):
            event_id = f"{session_id}::{event_type}::{event_number}"
            start, stop = max(0, center - window), min(len(valid), center + window + 1)
            for index in range(start, stop):
                row = valid.iloc[index]
                records.append(
                    {
                        "event_type": event_type,
                        "event_id": event_id,
                        "session_id": session_id,
                        "relative_trial": int(index - center),
                        "mouse_correct": float(row["mouse_correct"]),
                        "model_probability_correct": float(row["model_probability_correct"]),
                    }
                )
    if not records:
        return pd.DataFrame()
    aligned = pd.DataFrame(records)
    summary = aligned.groupby(["event_type", "relative_trial"], observed=False).agg(
        mouse_correct=("mouse_correct", "mean"),
        mouse_sem=("mouse_correct", "sem"),
        model_probability_correct=("model_probability_correct", "mean"),
        model_sem=("model_probability_correct", "sem"),
        n_events=("event_id", "nunique"),
        n_observations=("event_id", "size"),
    ).reset_index()
    return summary


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        destination = stem.with_suffix(suffix)
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        outputs.append(destination.name)
    plt.close(fig)
    return outputs


def _role_subset(predictions: pd.DataFrame, role: str) -> pd.DataFrame:
    if role == "all":
        return predictions
    return predictions.loc[predictions["split_role"].eq(role)]


def _locked_panel(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "TEST LOCKED\nuse --unlock-test", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_psychometrics(
    predictions: pd.DataFrame,
    output: Path,
    *,
    model_label: str = "M0",
) -> list[str]:
    """Plot mouse versus M0 psychometrics by split and block context."""

    all_label = "All data" if predictions["split_role"].eq("test").any() else "All available (test locked)"
    rows = (
        ("all", all_label),
        ("train", "Training"),
        ("validation", "Validation"),
        ("test", "Test"),
    )
    columns = (
        ("all", "Short + long blocks"),
        ("short_block", "Short block"),
        ("long_block", "Long block"),
        ("neutral", "Neutral block"),
    )
    fig, axes = plt.subplots(4, 4, figsize=(17, 13), sharey=True)
    summaries = []
    for row_index, (role, row_label) in enumerate(rows):
        role_data = _role_subset(predictions, role)
        for column_index, (block, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            if role == "test" and role_data.empty:
                _locked_panel(ax)
                continue
            if block == "all":
                panel = role_data.loc[
                    role_data["block_type"].isin(["short_block", "long_block"])
                ]
            else:
                panel = role_data
                panel = panel.loc[panel["block_type"].eq(block)]
            summary = four_bin_psychometric_summary(panel)
            summary["data_subset"] = role
            summary["block_subset"] = block
            summaries.append(summary)
            if summary.empty:
                ax.text(0.5, 0.5, "No trials", ha="center", va="center")
            else:
                ax.errorbar(summary["bin_center_s"], summary["mouse_probability_long"], yerr=summary["mouse_sem"], color="black", marker="o", label="Mouse")
                ax.errorbar(summary["bin_center_s"], summary["model_probability_long"], yerr=summary["model_sem"], color="#c62828", marker="o", label=model_label)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, linestyle=":")
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\nP(long choice)")
            if row_index == len(rows) - 1:
                ax.set_xlabel("Stimulus duration (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / "psychometric_summary.csv", index=False)
    fig.suptitle(
        f"{model_label} psychometrics: four quantile bins within each trial type"
    )
    return _save_figure(fig, output / "psychometrics_mouse_vs_model")


def _plot_event_grid(
    predictions: pd.DataFrame,
    output: Path,
    *,
    events: tuple[tuple[str, str], tuple[str, str]],
    window: int,
    stem: str,
    title: str,
    model_label: str = "M0",
) -> list[str]:
    all_label = "All data" if predictions["split_role"].eq("test").any() else "All available (test locked)"
    rows = (("all", all_label), ("train", "Training"), ("validation", "Validation"), ("test", "Test"))
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
    summaries = []
    for row_index, (role, row_label) in enumerate(rows):
        subset = _role_subset(predictions, role)
        for column_index, (event, event_label) in enumerate(events):
            ax = axes[row_index, column_index]
            if role == "test" and subset.empty:
                _locked_panel(ax)
                continue
            summary = event_aligned_performance(subset, event, window=window)
            if summary.empty:
                ax.text(0.5, 0.5, "No events", ha="center", va="center")
            else:
                summary["data_subset"] = role
                summaries.append(summary)
                ax.plot(summary["relative_trial"], summary["mouse_correct"], color="black", marker="o", ms=3, label="Mouse")
                ax.fill_between(summary["relative_trial"], summary["mouse_correct"] - summary["mouse_sem"].fillna(0), summary["mouse_correct"] + summary["mouse_sem"].fillna(0), color="black", alpha=0.12)
                ax.plot(summary["relative_trial"], summary["model_probability_correct"], color="#c62828", marker="o", ms=3, label=f"{model_label} expected")
                ax.fill_between(summary["relative_trial"], summary["model_probability_correct"] - summary["model_sem"].fillna(0), summary["model_probability_correct"] + summary["model_sem"].fillna(0), color="#c62828", alpha=0.12)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, linestyle=":")
            if row_index == 0:
                ax.set_title(event_label)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\nP(correct)")
            if row_index == 3:
                ax.set_xlabel("Trials from event")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output / f"{stem}_summary.csv", index=False)
    fig.suptitle(title)
    return _save_figure(fig, output / stem)


def plot_parameter_history(
    fit_summary: dict,
    output: Path,
    *,
    parameter_names: tuple[str, ...] = PARAMETER_NAMES,
    model_label: str | None = None,
) -> list[str]:
    """Plot constrained parameter values after each optimizer step."""

    winner = fit_summary.get("winner_start_id")
    if not parameter_names:
        raise ValueError("At least one parameter name is required")
    columns = 2 if len(parameter_names) <= 4 else 3
    rows = int(np.ceil(len(parameter_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows))
    flat_axes = np.atleast_1d(axes).ravel()
    records = []
    for run in fit_summary["runs"]:
        start = int(run["start_id"])
        history = [(0, run["initial_constrained_parameters"])] + [
            (int(row["epoch"]), row["constrained_parameters_after_step"])
            for row in run["history"]
            if "constrained_parameters_after_step" in row
        ]
        for epoch, values in history:
            records.append({"start_id": start, "epoch": epoch, **dict(zip(parameter_names, values))})
        frame = pd.DataFrame([record for record in records if record["start_id"] == start])
        for index, name in enumerate(parameter_names):
            flat_axes[index].plot(frame["epoch"], frame[name], alpha=0.95 if start == winner else 0.32, linewidth=2.3 if start == winner else 1.0, label=f"Start {start}" if index == 0 else None)
    history_frame = pd.DataFrame(records)
    history_frame.to_csv(output / "parameter_history.csv", index=False)
    for ax, name in zip(flat_axes, parameter_names):
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25, linestyle=":")
    for ax in flat_axes[:len(parameter_names)]:
        ax.set_ylabel("Parameter value")
    for ax in flat_axes[len(parameter_names):]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, ncol=2, fontsize=8)
    title = (
        "Constrained parameter trajectories (winner emphasized)"
        if model_label is None
        else f"{model_label} constrained parameter trajectories (winner emphasized)"
    )
    fig.suptitle(title)
    return _save_figure(fig, output / "parameter_trajectories")


def plot_nll_history(fit_summary: dict, output: Path) -> list[str]:
    """Plot fresh training and validation mean NLL after every fit step."""

    winner = fit_summary.get("winner_start_id")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    records = []
    for run in fit_summary["runs"]:
        start = int(run["start_id"])
        points = [{"epoch": 0, "train_mean_nll": run["initial_training_mean_nll"], "validation_mean_nll": run["initial_validation_mean_nll"]}]
        points.extend({"epoch": row["epoch"], "train_mean_nll": row["fresh_training_mean_nll"], "validation_mean_nll": row["validation_mean_nll"]} for row in run["history"])
        frame = pd.DataFrame(points)
        frame["start_id"] = start
        records.append(frame)
        style = {"alpha": 1.0 if start == winner else 0.32, "linewidth": 2.4 if start == winner else 1.0}
        axes[0].plot(frame["epoch"], frame["train_mean_nll"], label=f"Start {start}", **style)
        axes[1].plot(frame["epoch"], frame["validation_mean_nll"], **style)
    pd.concat(records, ignore_index=True).to_csv(output / "nll_history.csv", index=False)
    for ax, label in zip(axes, ("Fresh training mean NLL", "Validation mean NLL")):
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean NLL / valid decision")
        ax.grid(alpha=0.25, linestyle=":")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Optimization metrics (validation is reporting-only)")
    return _save_figure(fig, output / "training_validation_nll")


def generate_preliminary_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    output_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Save all requested preliminary figures, summaries, and a manifest."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_psychometrics(predictions, output)
    files["rare_trials"] = _plot_event_grid(
        predictions,
        output,
        events=(("rare_long_in_short", "Rare long in short block"), ("rare_short_in_long", "Rare short in long block")),
        window=3,
        stem="rare_trial_performance",
        title="Performance around rare trials (±3 retained decisions)",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions,
        output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10,
        stem="block_transition_performance",
        title="Performance around block transitions (±10 retained decisions)",
    )
    files["parameters"] = plot_parameter_history(fit_summary, output)
    files["nll"] = plot_nll_history(fit_summary, output)
    manifest = {
        "analysis_schema_version": "m0_preliminary_analysis_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": test_unlocked,
        "all_data_definition": "train+validation+test" if test_unlocked else "train+validation; test remains locked",
        "psychometric_binning": "four quantile bins separately for short and long trials; deterministic N(0,1e-6) jitter, seed 42; session-balanced",
        "psychometric_panels": {
            "rows": ["all", "train", "validation", "test"],
            "columns": ["short_and_long_blocks", "short_block", "long_block", "neutral_block"],
        },
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "performance_definition": {"mouse": "binary correct choice", "model": "analytical expected P(correct), not a sampled choice"},
        "invalid_decision_policy": "retained in trial_predictions.csv with NaN model fields; excluded from plots",
        "files": files,
    }
    _atomic_json(manifest, output / "figure_manifest.json")
    return manifest


def save_model_analysis_outputs(
    predictions: pd.DataFrame,
    fit_summary: dict,
    results_directory: str | Path,
    *,
    test_unlocked: bool,
) -> dict:
    """Save the clean trial table and compact model/output inventory."""

    results = Path(results_directory).expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results / "trial_predictions.csv", index=False)
    valid = predictions.loc[predictions["model_evaluated"].astype(bool)].copy()
    session_performance = valid.groupby(
        ["split_role", "session_id"], observed=False
    ).agg(
        n_valid_decisions=("model_evaluated", "size"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
        model_threshold_accuracy=("model_choice_correct", "mean"),
        model_mean_nll=("per_trial_negative_log_likelihood", "mean"),
    ).reset_index()
    split_performance = valid.groupby("split_role", observed=False).agg(
        n_valid_decisions=("model_evaluated", "size"),
        mouse_accuracy=("mouse_correct", "mean"),
        model_expected_accuracy=("model_probability_correct", "mean"),
        model_threshold_accuracy=("model_choice_correct", "mean"),
        model_mean_nll=("per_trial_negative_log_likelihood", "mean"),
    ).reset_index()
    split_performance["session_id"] = "ALL_SESSIONS"
    performance = pd.concat([split_performance, session_performance], ignore_index=True)
    performance.to_csv(results / "performance_summary.csv", index=False)
    winner_id = fit_summary.get("winner_start_id")
    winner = next((run for run in fit_summary.get("runs", []) if run["start_id"] == winner_id), None)
    summary = {
        "output_schema_version": "m0_complete_output_v1",
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": winner_id,
        "static_parameters": None if winner is None else dict(zip(PARAMETER_NAMES, winner["best_constrained_parameters"])),
        "adaptive_parameters": {},
        "adaptive_parameter_note": "M0 is static and has no adaptive or trial-varying fitted parameters.",
        "model_decision_rule": "model_choice_long = 1 when P(long) >= 0.5",
        "trial_table": "trial_predictions.csv",
        "performance_table": "performance_summary.csv",
        "multistart_fit": "starts/multistart_summary.json",
        "test_unlocked": test_unlocked,
        "saved_trial_count": int(len(predictions)),
        "saved_valid_decision_count": int(predictions["model_evaluated"].sum()),
        "saved_split_roles": sorted(predictions["split_role"].unique().tolist()),
        "naive_outcomes": {
            "RewardNaive": int(predictions["outcome"].eq("RewardNaive").sum()),
            "PunishNaive": int(predictions["outcome"].eq("PunishNaive").sum()),
            "model_treatment": "used as Reward and Punish respectively; original labels retained",
        },
    }
    _atomic_json(summary, results / "model_output_summary.json")
    return summary
