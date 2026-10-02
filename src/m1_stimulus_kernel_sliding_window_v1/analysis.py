"""Experiment-only figures and pooled exports for M1_stimulus_kernel sliding-window fits.

Native implementation, not a reuse of ``known_switch_analysis``/
``m1_known_switch_sliding_window_v1.analysis``: this model has no fixed
category-rate prototypes and no oracle switch signal, so its own fit-history
figures are written directly against the fields this model's ``fit.py``
actually records, rather than importing the oracle-switch models' helper.

The rare-trial/block-transition event-grid panels (``_plot_event_grid``,
shared with every other model in this family) are reused here: they
summarize mouse-vs-model performance around task-defined block transitions
and rare trials, which is a property of the task timeline and the model's
predicted choice probability, not of whether the model itself represents a
discrete block or switch concept -- exactly the same reasoning that already
lets the model-agnostic psychometric grid apply here. ``pi_t`` never depends
on block identity, so these panels do not diagnose anything about the
scalar-prior mechanism itself, but they do show whether the fitted choice
probabilities track the mouse as well as the block-aware models do at
these events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m0_sensory_v1.analysis import _plot_event_grid, _save_figure
from m0_sliding_window_v1.analysis import (
    figure_ready,
    plot_adaptive_psychometrics,
    pool_prediction_tables,
)

from .experiment import M1_STIMULUS_KERNEL_PARAMETER_NAMES


# Bumped to _v2 when the real parameter-trajectory figure and the
# rare-trial/block-transition event-grid panels were added: a manifest saved
# under _v1 is missing both, so figure_outputs_are_current() must treat it as
# stale and regenerate, not resume_skip, on --resume.
FIGURE_ANALYSIS_SCHEMA_VERSION = "m1_stimulus_kernel_sliding_experiment_analysis_v2"
GLOBAL_ANALYSIS_SCHEMA_VERSION = "m1_stimulus_kernel_sliding_global_analysis_v2"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fit_history_figures(fit_summary: dict, output: Path) -> dict[str, list[str]]:
    """Plot NLL/parameter trajectories and best-constrained-parameters-by-start scatter.

    Native equivalent of ``known_switch_analysis.generate_known_switch_fit_figures``
    (that shared helper is fully generic in its own right -- it makes no
    known_switch-specific assumption -- but is kept out of this model's
    dependency graph so a future change to the two oracle-switch models
    cannot silently affect this one).
    """

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
    fig.suptitle("M1 stimulus kernel optimization history")
    nll_files = _save_figure(fig, output / "nll_history")

    parameter_rows = []
    for run in fit_summary.get("runs", []):
        values = run.get("best_constrained_parameters", [])
        if len(values) == len(M1_STIMULUS_KERNEL_PARAMETER_NAMES):
            parameter_rows.append({
                "start_id": int(run["start_id"]),
                **dict(zip(M1_STIMULUS_KERNEL_PARAMETER_NAMES, values, strict=True)),
            })
    parameters = pd.DataFrame(parameter_rows)
    if not parameters.empty:
        parameters["alpha"] = 1.0 / (1.0 + np.exp(-parameters["a"]))
    parameters.to_csv(output / "best_parameters_by_start.csv", index=False)
    names = (*M1_STIMULUS_KERNEL_PARAMETER_NAMES, "alpha")
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for index, name in enumerate(names):
        if not parameters.empty:
            colors = ["#c62828" if value == winner else "#777777" for value in parameters["start_id"]]
            flat[index].scatter(parameters["start_id"], parameters[name], c=colors)
        flat[index].set_title(name)
        flat[index].set_xlabel("Start ID")
        flat[index].grid(alpha=0.25, linestyle=":")
    for axis in flat[len(names):]:
        axis.set_visible(False)
    fig.suptitle("M1 stimulus kernel restored-best parameters (winner in red)")
    parameter_files = _save_figure(fig, output / "best_parameters_by_start")

    trajectory_files = _parameter_trajectory_figure(fit_summary, output)
    return {"nll": nll_files, "parameters": parameter_files, "trajectories": trajectory_files}


def _parameter_trajectory_figure(fit_summary: dict, output: Path) -> list[str]:
    """Plot constrained parameter (plus derived alpha) values after every step, every start.

    Requires ``constrained_parameters_after_step`` on each history row (added
    once ``m1_stimulus_kernel_v1/fit.py`` started recording it every step);
    a fit summary predating that field simply produces an empty figure.

    The x-axis is the row's position in ``history`` (a running count across
    the whole run), not the recorded ``step`` field: hybrid mode numbers its
    Adam-warmup and L-BFGS phases independently from 1, so plotting against
    ``step`` directly would draw a backward, duplicated-x segment where the
    second phase begins.
    """
    winner = fit_summary.get("winner_start_id")
    names = (*M1_STIMULUS_KERNEL_PARAMETER_NAMES, "alpha")
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    records: list[dict[str, object]] = []
    for run in fit_summary.get("runs", []):
        start = int(run["start_id"])
        initial = run.get("initial_constrained_parameters")
        points = (
            [(0, initial)] if initial is not None and len(initial) == len(M1_STIMULUS_KERNEL_PARAMETER_NAMES) else []
        )
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
            row_values = dict(zip(M1_STIMULUS_KERNEL_PARAMETER_NAMES, values, strict=True))
            row_values["alpha"] = 1.0 / (1.0 + np.exp(-row_values["a"]))
            records.append({"start_id": start, "history_index": history_index, **row_values})
        frame = pd.DataFrame([record for record in records if record["start_id"] == start])
        style = {"alpha": 0.95 if start == winner else 0.32, "linewidth": 2.3 if start == winner else 1.0}
        for index, name in enumerate(names):
            flat[index].plot(frame["history_index"], frame[name], label=f"Start {start}" if index == 0 else None, **style)
    pd.DataFrame(records).to_csv(output / "parameter_history.csv", index=False)
    for axis, name in zip(flat, names):
        axis.set_title(name)
        axis.set_xlabel("Optimizer step (sequential across phases)")
        axis.set_ylabel("Parameter value")
        axis.grid(alpha=0.25, linestyle=":")
    for axis in flat[len(names):]:
        axis.set_visible(False)
    if flat[0].lines:
        flat[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("M1 stimulus kernel constrained parameter trajectories (winner emphasized)")
    return _save_figure(fig, output / "parameter_trajectories")


def _prediction_figures(predictions: pd.DataFrame, output: Path) -> dict[str, list[str]]:
    """Model-agnostic psychometric grid plus rare-trial/block-transition panels.

    These event-grid panels summarize mouse-vs-model performance around
    task-defined events (block transitions, rare trials); that timeline
    structure comes from the canonical trial table, not from the model, so
    they apply here exactly as they do for every other model in this family
    even though ``pi_t`` itself has no notion of block identity.
    """
    files: dict[str, list[str]] = {}
    files["psychometrics"] = plot_adaptive_psychometrics(predictions, output, model_label="M1 stimulus kernel")
    files["rare_trials"] = _plot_event_grid(
        predictions,
        output,
        events=(
            ("rare_long_in_short", "Rare long in short block"),
            ("rare_short_in_long", "Rare short in long block"),
        ),
        window=3,
        stem="rare_trial_performance",
        title="M1 stimulus kernel performance around rare trials (±3 retained decisions)",
        model_label="M1 stimulus kernel",
    )
    files["block_transitions"] = _plot_event_grid(
        predictions,
        output,
        events=(("short_to_long", "Short → long"), ("long_to_short", "Long → short")),
        window=10,
        stem="block_transition_performance",
        title="M1 stimulus kernel performance around block transitions (±10 retained decisions)",
        model_label="M1 stimulus kernel",
    )
    return files


def generate_m1_stimulus_kernel_experiment_figures(
    predictions: pd.DataFrame,
    fit_summary: dict,
    m0_style_output_directory: str | Path,
    diagnostic_output_directory: str | Path,
) -> dict:
    """Generate the psychometric/event-grid panels plus native fit-history diagnostics.

    ``diagnostic_output_directory`` is accepted for call-site parity with
    every sibling model's ``generate_<model>_experiment_figures`` signature
    (``run_m1_stimulus_kernel_sliding_window.py`` passes both directories
    uniformly with the rest of the family) but is currently unused: this
    model has no native block-belief/switch/hazard diagnostics that would
    warrant a second figure directory the way M1's block-belief trajectory
    or M1_known_switch's biased log-odds trajectory do.
    """

    if fit_summary.get("model_id") != "M1_stimulus_kernel_v1":
        raise ValueError("M1_stimulus_kernel experiment figures require an M1_stimulus_kernel fit summary")
    output = Path(m0_style_output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = figure_ready(predictions)
    files = _prediction_figures(ready, output)
    files.update(_fit_history_figures(fit_summary, output))
    manifest = {
        "analysis_schema_version": FIGURE_ANALYSIS_SCHEMA_VERSION,
        "model_id": fit_summary.get("model_id"),
        "mouse_id": fit_summary.get("mouse_id"),
        "winner_start_id": fit_summary.get("winner_start_id"),
        "test_unlocked": True,
        "psychometric_binning": (
            "up to four quantile bins separately for short and long trials; "
            "sparse panels use fewer bins; deterministic N(0,1e-6) jitter, "
            "seed 42; session-balanced; no logistic fit"
        ),
        "prior_update_rule": "pi_next = pi + sigmoid(a) * (category - pi); pi_1 = 0.5",
        "rare_trial_window": [-3, 3],
        "block_transition_window": [-10, 10],
        "block_and_switch_diagnostics": "not_applicable_this_model_has_no_block_or_switch_concept",
        "parameter_trajectory_populated": _fit_summary_has_trajectory_data(fit_summary),
        "files": files,
    }
    _atomic_json(manifest, output / "m1_stimulus_kernel_sliding_figure_manifest.json")
    return manifest


def _fit_summary_has_trajectory_data(fit_summary: dict) -> bool:
    """Return whether any recorded history row carries a per-step constrained-
    parameter snapshot (see the docstring on the identically-named helper in
    ``known_switch_analysis.py`` for why this must gate resume freshness
    separately from the schema-version string).
    """
    return any(
        "constrained_parameters_after_step" in row
        for run in fit_summary.get("runs", [])
        for row in run.get("history", [])
    )


def figure_outputs_are_current(output_directory: str | Path) -> bool:
    """Return whether a run already has the current M1_stimulus_kernel experiment figures.

    Requires both the schema-version string and ``parameter_trajectory_populated``:
    a fit result saved by a pre-upgrade ``fit.py`` (e.g. reused across
    ``--resume`` from before per-step parameter recording existed) can still
    produce a manifest stamped with the current schema version, but its
    trajectory figure would be empty -- that must not be reported as current.
    """

    path = Path(output_directory) / "m1_stimulus_kernel_sliding_figure_manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("analysis_schema_version") == FIGURE_ANALYSIS_SCHEMA_VERSION
        and manifest.get("parameter_trajectory_populated") is True
    )


def generate_global_prediction_figures(
    sources: Iterable[tuple[str, str | Path]],
    output_directory: str | Path,
    *,
    source_kind: str,
) -> dict:
    """Pool all M1_stimulus_kernel prediction instances and plot every split separately."""

    source_list = list(sources)
    pooled = pool_prediction_tables(source_list)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "pooled_trial_predictions.csv", index=False)
    analysis = figure_ready(pooled)
    analysis["session_id"] = analysis["analysis_session_id"]
    files = _prediction_figures(analysis, output)
    role_counts = {
        role: int(pooled["split_role"].eq(role).sum())
        for role in ("train", "validation", "test")
    }
    role_source_units = {
        role: int(
            pooled.loc[
                pooled["split_role"].eq(role), "analysis_session_id"
            ].nunique()
        )
        for role in ("train", "validation", "test")
    }
    manifest = {
        "analysis_schema_version": GLOBAL_ANALYSIS_SCHEMA_VERSION,
        "model_id": "M1_stimulus_kernel_v1",
        "source_kind": source_kind,
        "source_count": len(source_list),
        "source_run_ids": [source_id for source_id, _ in source_list],
        "split_trial_counts": role_counts,
        "split_prediction_unit_counts": role_source_units,
        "pooling_policy": (
            "all predictions retained within each split; repeated physical sessions "
            "from different rolling windows are distinct window-session analysis units"
        ),
        "psychometric_binning": (
            "up to four within-type quantile bins; sparse panels use fewer bins; "
            "session-balanced across prediction units; no logistic fit"
        ),
        "files": {
            "pooled_predictions": ["pooled_trial_predictions.csv"],
            **files,
        },
    }
    _atomic_json(manifest, output / "global_figure_manifest.json")
    return manifest


__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION",
    "figure_outputs_are_current",
    "figure_ready",
    "generate_global_prediction_figures",
    "generate_m1_stimulus_kernel_experiment_figures",
]
