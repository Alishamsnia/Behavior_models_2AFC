"""Focused learning-trajectory and identifiability figures for M3."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import (
    build_m3_m2_validation_comparison,
    build_validation_model_metrics,
)


def _save(figure, output: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        destination = output / f"{stem}.{extension}"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        names.append(destination.name)
    plt.close(figure)
    return names


def generate_m3_learning_figures(
    predictions: pd.DataFrame,
    output_directory: str | Path,
    *,
    maximum_sessions: int = 6,
) -> tuple[Path, ...]:
    """Plot delta/prototype/update trajectories with projection activations."""

    required = {
        "session_id", "trial_index", "model_delta_pre", "model_delta_post",
        "model_delta_update_signal", "model_projection_activated",
        "model_subjective_prototype_S", "model_subjective_prototype_L",
    }
    if missing := required.difference(predictions.columns):
        raise KeyError(f"M3 figure input missing columns: {sorted(missing)}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    session_ids = predictions["session_id"].drop_duplicates().head(maximum_sessions)
    for session_id in session_ids:
        frame = predictions.loc[predictions["session_id"].eq(session_id)].sort_values("trial_index")
        trial = frame["trial_index"].to_numpy()
        figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        axes[0].plot(trial, frame["model_delta_pre"], label="delta pre", color="black")
        axes[0].plot(trial, frame["model_delta_post"], label="delta post", color="tab:blue", alpha=.7)
        projected = frame["model_projection_activated"].astype(bool)
        axes[0].scatter(trial[projected], frame.loc[projected, "model_delta_post"], color="red", marker="x", label="projection")
        axes[0].set_ylabel("delta")
        axes[0].legend(frameon=False, ncol=3)
        axes[1].plot(trial, frame["model_subjective_prototype_S"], label="theta S")
        axes[1].plot(trial, frame["model_subjective_prototype_L"], label="theta L")
        axes[1].set_ylabel("prototype")
        axes[1].legend(frameon=False)
        axes[2].axhline(0.0, color="0.7", linewidth=1)
        axes[2].plot(trial, frame["model_delta_update_signal"], color="tab:purple")
        axes[2].set_ylabel("update signal")
        axes[2].set_xlabel("trial")
        figure.suptitle(f"M3 learning trajectory: {session_id}")
        destination = output / f"learning_trajectory_{session_id}.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        saved.append(destination)
    return tuple(saved)


def generate_m3_identifiability_figure(
    surface: pd.DataFrame,
    output_directory: str | Path,
) -> Path:
    """Plot the fixed-other-five delta0/alpha_theta objective slice."""

    dynamic = [name for name in ("ah", "delta0", "alpha_theta") if name in surface.columns]
    if len(dynamic) != 2:
        raise KeyError("M3 surface figure requires exactly two dynamic axes")
    required = {*dynamic, "delta_mean_nll_from_grid_minimum"}
    if missing := required.difference(surface.columns):
        raise KeyError(f"M3 surface missing columns: {sorted(missing)}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    horizontal, vertical = dynamic[0], dynamic[1]
    table = surface.pivot(index=vertical, columns=horizontal, values="delta_mean_nll_from_grid_minimum")
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(
        table.to_numpy(), origin="lower", aspect="auto",
        extent=[table.columns.min(), table.columns.max(), table.index.min(), table.index.max()],
    )
    axis.set_xlabel(horizontal)
    axis.set_ylabel(vertical)
    axis.set_title("M3 local identifiability surface")
    figure.colorbar(image, ax=axis, label="mean NLL above grid minimum")
    destination = output / f"{horizontal}_{vertical}_identifiability.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def generate_m3_diagnostic_figures(
    m3_predictions: pd.DataFrame,
    output_directory: str | Path,
    *,
    m2_predictions: pd.DataFrame | None = None,
    m1_predictions: pd.DataFrame | None = None,
    m0_predictions: pd.DataFrame | None = None,
) -> dict:
    """Generate the full M3 scientific diagnostic set and optional comparisons."""

    required = {
        "session_id", "trial_index", "split_role", "choice_long", "category_long",
        "true_block", "model_evaluated", "model_probability_long", "model_prior_long",
        "model_delta_pre", "model_delta_update_signal", "model_projection_activated",
        "model_subjective_prototype_S", "model_subjective_prototype_L",
        "model_responsibility_S", "model_responsibility_L",
        "per_trial_negative_log_likelihood", "fitted_ah", "fitted_delta0",
        "fitted_alpha_theta", "fitted_hazard",
    }
    if missing := required.difference(m3_predictions.columns):
        raise KeyError(f"M3 diagnostics missing columns: {sorted(missing)}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = m3_predictions.copy()
    files: dict[str, list[str]] = {}

    example_id = str(data.session_id.iloc[0])
    example = data.loc[data.session_id.astype(str).eq(example_id)].sort_values("trial_index")
    trial = example.trial_index.to_numpy()
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(trial, example.model_prior_long, color="black", label="prior long")
    for state, color in zip(("N", "S", "L"), ("#777777", "#1565c0", "#c62828")):
        axes[1].plot(trial, example[f"model_pre_choice_probability_{state}"], color=color, label=state)
    axes[2].plot(trial, example.model_responsibility_S, label="gamma S", color="#1565c0")
    axes[2].plot(trial, example.model_responsibility_L, label="gamma L", color="#c62828")
    axes[0].set(ylabel="P(long)", title=f"M3 causal prior and HMM state: {example_id}")
    axes[1].set(ylabel="pre-choice belief"); axes[2].set(ylabel="responsibility", xlabel="trial")
    for axis in axes: axis.legend(frameon=False, ncol=3); axis.grid(alpha=.2)
    files["belief_prior_responsibility"] = _save(figure, output, "m3_belief_prior_responsibility")

    valid = data.loc[data.model_evaluated.astype(bool)].copy()
    valid["calibration_bin"] = pd.cut(valid.model_probability_long, np.linspace(0, 1, 11), include_lowest=True)
    calibration = valid.groupby("calibration_bin", observed=False).agg(
        mean_predicted=("model_probability_long", "mean"), observed_long=("choice_long", "mean"),
        n=("choice_long", "size"),
    ).dropna().reset_index()
    calibration.to_csv(output / "m3_calibration.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 6)); axis.plot([0, 1], [0, 1], "k--")
    axis.plot(calibration.mean_predicted, calibration.observed_long, "o-", color="#6a1b9a")
    axis.set(xlabel="Predicted P(long)", ylabel="Observed long rate", title="M3 calibration", xlim=(0, 1), ylim=(0, 1))
    files["calibration"] = _save(figure, output, "m3_calibration")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(["delta0", "median delta"], [float(data.fitted_delta0.iloc[0]), float(data.model_delta_pre.median())])
    axes[1].bar(["alpha_theta"], [float(data.fitted_alpha_theta.iloc[0])], color="#6a1b9a")
    biased_hazard = float(data.fitted_hazard.iloc[0])
    axes[2].bar(
        ["h_B", "h_N", "1/23"],
        [biased_hazard, biased_hazard / 2.0, 1/23],
        color=["#00897b", "#2e7d32", "#777777"],
    )
    axes[0].set(title="Contrast"); axes[1].set(title="Learning rate"); axes[2].set(title="Hazard reference")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    files["fitted_dynamic_parameters"] = _save(figure, output, "m3_fitted_dynamic_parameters")

    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(trial, example.model_delta_pre, color="black", label="delta")
    axes[0].plot(trial, example.model_subjective_prototype_S, label="theta S")
    axes[0].plot(trial, example.model_subjective_prototype_L, label="theta L")
    axes[1].plot(trial, example.model_delta_update_signal, color="#6a1b9a", label="update signal")
    projected = example.model_projection_activated.astype(bool)
    axes[0].scatter(trial[projected], example.loc[projected, "model_delta_pre"], marker="x", color="red", label="projection")
    axes[0].legend(ncol=4); axes[1].legend(); axes[1].set(xlabel="trial")
    files["delta_prototypes_signal_projection"] = _save(figure, output, "m3_delta_prototypes_signal_projection")

    projection_by_session = data.groupby("session_id", observed=False).agg(
        projection_count=("model_projection_activated", "sum"), trials=("trial_index", "size")
    ).reset_index()
    projection_by_session["projection_fraction"] = projection_by_session.projection_count / projection_by_session.trials
    projection_by_session.to_csv(output / "m3_projection_by_session.csv", index=False)
    figure, axis = plt.subplots(figsize=(max(7, len(projection_by_session)*.3), 4))
    axis.bar(np.arange(len(projection_by_session)), projection_by_session.projection_count, color="#c62828")
    axis.set(xlabel="session", ylabel="projection count", title="M3 hard-floor activations")
    files["projection_audit"] = _save(figure, output, "m3_projection_audit")

    # Switch- and rare-event-aligned learning dynamics.
    aligned, rare = [], []
    for sid, frame in data.groupby("session_id", sort=False):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        switches = np.flatnonzero(frame.true_block.ne(frame.true_block.shift()).to_numpy())[1:]
        rare_events = np.flatnonzero((((frame.true_block == "S") & (frame.category_long == 1)) |
                                      ((frame.true_block == "L") & (frame.category_long == 0))).to_numpy())
        for target, events, window in ((aligned, switches, 10), (rare, rare_events, 3)):
            for event, center in enumerate(events):
                for index in range(max(0, center-window), min(len(frame), center+window+1)):
                    target.append({
                        "session_id": sid, "event": event, "relative_trial": index-center,
                        "delta": frame.loc[index, "model_delta_pre"],
                        "signal": frame.loc[index, "model_delta_update_signal"],
                        "prior": frame.loc[index, "model_prior_long"],
                    })
    for label, records in (("switch_aligned", aligned), ("rare_trial_aligned", rare)):
        table = pd.DataFrame(records)
        if table.empty:
            continue
        summary = table.groupby("relative_trial", observed=False).agg(
            delta=("delta", "mean"), signal=("signal", "mean"), prior=("prior", "mean"), n=("event", "size")
        ).reset_index()
        summary.to_csv(output / f"m3_{label}.csv", index=False)
        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, column in zip(axes, ("delta", "signal", "prior")):
            axis.plot(summary.relative_trial, summary[column]); axis.axvline(0, color="black", ls="--")
            axis.set(xlabel="relative trial", ylabel=column)
        files[label] = _save(figure, output, f"m3_{label}")

    context = data.sort_values(["session_id", "trial_index"]).copy()
    context["block_run_position"] = context.groupby("session_id", observed=False)["true_block"].transform(
        lambda values: values.groupby(values.ne(values.shift()).cumsum()).cumcount() + 1
    )
    context["timing"] = np.where(context.block_run_position <= 10, "early_1_10", "late_11_plus")
    contrast = context.groupby(["true_block", "timing"], observed=False).agg(
        mean_delta=("model_delta_pre", "mean"),
        mean_theta_short=("model_subjective_prototype_S", "mean"),
        mean_theta_long=("model_subjective_prototype_L", "mean"),
        n=("trial_index", "size"),
    ).reset_index()
    contrast.to_csv(output / "m3_early_late_prototype_contrast.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 4))
    for timing, frame in contrast.groupby("timing", observed=False):
        axis.plot(frame.true_block, frame.mean_delta, marker="o", label=timing)
    axis.set(xlabel="true block (retrospective)", ylabel="mean delta", title="Early versus late prototype contrast")
    axis.legend()
    files["early_late_prototype_contrast"] = _save(figure, output, "m3_early_late_prototype_contrast")

    references = {"M3": data}
    if m2_predictions is not None: references["M2"] = m2_predictions
    if m1_predictions is not None: references["M1"] = m1_predictions
    if m0_predictions is not None: references["M0"] = m0_predictions
    if len(references) > 1:
        metrics = build_validation_model_metrics(dict(reversed(list(references.items()))))
        metrics.to_csv(output / "m0_m1_m2_m3_validation_metrics.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(metrics.model, metrics.mean_nll); axes[1].bar(metrics.model, metrics.brier_score)
        axes[0].set(title="Validation NLL", ylabel="mean NLL")
        axes[1].set(title="Validation Brier", ylabel="Brier score")
        files["all_model_validation"] = _save(figure, output, "m0_m1_m2_m3_validation_nll_brier")

    if m2_predictions is not None:
        comparison = build_m3_m2_validation_comparison(data, m2_predictions)
        for name in (
            "overall", "per_session", "context", "transition_aligned",
            "surprising_category_aligned", "psychometric_by_block", "calibration",
        ):
            getattr(comparison, name).to_csv(output / f"m3_vs_m2_{name}.csv", index=False)
        session = comparison.per_session.sort_values("delta_elpd_m3_vs_m2")
        figure, axis = plt.subplots(figsize=(max(7, len(session)*.3), 4))
        axis.bar(np.arange(len(session)), session.delta_elpd_m3_vs_m2,
                 color=np.where(session.delta_elpd_m3_vs_m2 >= 0, "#2e7d32", "#c62828"))
        axis.axhline(0, color="black"); axis.set(ylabel="ΔELPD M3−M2", xlabel="session")
        files["m3_m2_per_session_elpd"] = _save(figure, output, "m3_vs_m2_per_session_elpd")
        context = comparison.context
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(context.timing + "\n" + context.congruence, context.mean_delta_bits_per_trial_m3_vs_m2)
        axis.axhline(0, color="black"); axis.set(ylabel="Δ bits/trial", title="M3−M2: early/late × common/rare")
        files["m3_m2_context"] = _save(figure, output, "m3_vs_m2_context")
        surprising = comparison.surprising_category_aligned
        if not surprising.empty:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(surprising.relative_trial, surprising.mean_probability_change_m3_minus_m2)
            axes[1].plot(surprising.relative_trial, surprising.mean_delta_bits_per_trial_m3_vs_m2)
            for axis in axes:
                axis.axvline(0, color="black", ls="--"); axis.set(xlabel="trials from surprising category")
            axes[0].set(ylabel="P(long) M3−M2"); axes[1].set(ylabel="Δ bits/trial M3−M2")
            files["m3_m2_surprising_categories"] = _save(
                figure, output, "m3_vs_m2_surprising_category_aligned"
            )
        keys = [name for name in ("mouse_id", "split_role", "session_id", "trial_index")
                if name in data.columns and name in m2_predictions.columns]
        overlay = data[keys + ["model_prior_long"]].merge(
            m2_predictions[keys + ["model_prior_long"]], on=keys,
            suffixes=("_m3", "_m2"), validate="one_to_one"
        )
        overlay = overlay.loc[overlay.session_id.astype(str).eq(str(overlay.session_id.iloc[0]))]
        figure, axis = plt.subplots(figsize=(13, 4))
        axis.plot(overlay.trial_index, overlay.model_prior_long_m2, label="M2")
        axis.plot(overlay.trial_index, overlay.model_prior_long_m3, label="M3", ls="--")
        axis.set(xlabel="trial", ylabel="P(long)", title="M2/M3 causal prior overlay"); axis.legend()
        files["m2_m3_prior_overlay"] = _save(figure, output, "m2_m3_prior_overlay")
        psychometric = comparison.psychometric_by_block
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for axis, block in zip(axes, ("N", "S", "L")):
            selected = psychometric.loc[psychometric.true_block.eq(block)]
            for model, frame in selected.groupby("model", observed=False):
                axis.plot(frame.duration_rounded, frame.predicted_long, marker="o", label=model)
            axis.set(title=f"block {block}", xlabel="duration (s)", ylim=(0, 1)); axis.legend()
        axes[0].set(ylabel="predicted P(long)")
        files["m2_m3_psychometric"] = _save(figure, output, "m2_m3_psychometric_by_block")

    manifest = {
        "analysis": "complete M3 dynamic-prototype diagnostic set",
        "files": files,
        "figure_families_requested": [
            "all_model_validation_nll_brier", "m3_m2_per_session_elpd", "fitted_alpha_theta",
            "delta_trajectory", "prototype_trajectory", "update_signal", "responsibilities",
            "projection_audit", "switch_aligned", "rare_trial_aligned", "early_late_context",
            "m2_m3_prior_overlay", "m2_m3_psychometric", "identifiability_surfaces",
        ],
        "m2_comparison_included": m2_predictions is not None,
        "m1_comparison_included": m1_predictions is not None,
        "m0_comparison_included": m0_predictions is not None,
    }
    (output / "m3_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
