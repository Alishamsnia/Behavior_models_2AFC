"""Focused duration-state diagnostics for M4."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import build_m4_m3_validation_comparison


def generate_m4_duration_figures(
    predictions: pd.DataFrame,
    hazard_curve: pd.DataFrame,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Plot the fitted hazard and inferred duration summaries by session."""

    required = {
        "session_id", "trial_index", "model_expected_capped_run_length_pre",
        "model_expected_hazard_pre", "model_tail_probability_pre",
    }
    if missing := required.difference(predictions.columns):
        raise KeyError(f"M4 predictions missing duration diagnostics: {sorted(missing)}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(
        hazard_curve.represented_run_length,
        hazard_curve.biased_hazard,
        color="black",
        linewidth=2,
        label="biased $h_B(r)$",
    )
    axis.plot(
        hazard_curve.represented_run_length,
        hazard_curve.neutral_hazard,
        color="tab:green",
        linewidth=2,
        label="neutral $h_N(r)=h_B(r)/2$",
    )
    center = hazard_curve.loc[hazard_curve.is_center].iloc[0]
    axis.axvline(float(center.represented_run_length), color="tab:blue", linestyle="--", label="center = 23")
    tail = hazard_curve.loc[hazard_curve.is_capped_tail].iloc[0]
    tail_run_length = float(tail.represented_run_length)
    axis.scatter([tail_run_length], [float(tail.biased_hazard)], color="tab:red", label="capped tail")
    axis.set(xlabel="Represented run length (trials)", ylabel="Switch hazard", ylim=(0, 1))
    axis.legend(frameon=False)
    figure.tight_layout()
    path = output / "m4_fitted_hazard_curve.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    for session_id, frame in predictions.groupby("session_id", sort=False, observed=False):
        ordered = frame.sort_values("trial_index")
        figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(ordered.trial_index, ordered.model_expected_capped_run_length_pre, color="tab:blue")
        axes[0].set_ylabel("Expected capped run")
        axes[1].plot(ordered.trial_index, ordered.model_expected_hazard_pre, color="tab:orange")
        axes[1].set(ylabel="Expected hazard", ylim=(0, 1))
        axes[2].plot(ordered.trial_index, ordered.model_tail_probability_pre, color="tab:red")
        axes[2].set(xlabel="Trial", ylabel="P(R >= 65)", ylim=(0, 1))
        figure.suptitle(f"M4 duration belief: {session_id}")
        figure.tight_layout()
        safe_id = str(session_id).replace("/", "_")
        path = output / f"m4_duration_trajectory_{safe_id}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def _save_both(figure, output: Path, stem: str) -> list[str]:
    files = []
    for suffix in ("png", "pdf"):
        destination = output / f"{stem}.{suffix}"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        files.append(destination.name)
    plt.close(figure)
    return files


def generate_m4_diagnostic_figures(
    predictions: pd.DataFrame,
    hazard_curve: pd.DataFrame,
    output_directory: str | Path,
    *,
    m3_predictions: pd.DataFrame | None = None,
    identifiability_surfaces: tuple[pd.DataFrame, ...] = (),
) -> dict[str, object]:
    """Generate M4 psychometric, calibration, duration, transition, and fit diagnostics."""

    required = {
        "session_id", "trial_index", "split_role", "duration_s", "choice_long",
        "category_long", "true_block", "model_evaluated", "model_probability_long",
        "model_prior_long", "model_delta_pre", "model_expected_capped_run_length_pre",
        "model_expected_hazard_pre", "model_tail_probability_pre",
    }
    if missing := required.difference(predictions.columns):
        raise KeyError(f"M4 diagnostics missing columns: {sorted(missing)}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {}
    data = predictions.copy()
    valid = data.loc[data.model_evaluated.astype(bool)].copy()

    valid["duration_bin"] = valid.duration_s.round(3)
    psychometric = valid.groupby(["split_role", "duration_bin"], observed=False).agg(
        n=("choice_long", "size"), observed_long=("choice_long", "mean"),
        predicted_long=("model_probability_long", "mean"),
    ).reset_index()
    psychometric.to_csv(output / "m4_psychometric.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 5))
    for role, frame in psychometric.groupby("split_role", observed=False):
        axis.plot(frame.duration_bin, frame.observed_long, "o", alpha=.55, label=f"mouse {role}")
        axis.plot(frame.duration_bin, frame.predicted_long, "-", label=f"M4 {role}")
    axis.set(xlabel="duration (s)", ylabel="P(long)", ylim=(0, 1), title="M4 psychometric fit")
    axis.legend(frameon=False)
    files["psychometric"] = _save_both(figure, output, "m4_psychometric")

    valid["calibration_bin"] = pd.cut(valid.model_probability_long, np.linspace(0, 1, 11), include_lowest=True)
    calibration = valid.groupby(["split_role", "calibration_bin"], observed=False).agg(
        mean_predicted=("model_probability_long", "mean"),
        observed_long=("choice_long", "mean"), n=("choice_long", "size"),
    ).dropna().reset_index()
    calibration.to_csv(output / "m4_calibration.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 6)); axis.plot([0, 1], [0, 1], "k--")
    for role, frame in calibration.groupby("split_role", observed=False):
        axis.plot(frame.mean_predicted, frame.observed_long, "o-", label=role)
    axis.set(xlabel="Predicted P(long)", ylabel="Observed long rate", xlim=(0, 1), ylim=(0, 1), title="M4 calibration")
    axis.legend(frameon=False)
    files["calibration"] = _save_both(figure, output, "m4_calibration")

    example_id = str(data.session_id.iloc[0])
    example = data.loc[data.session_id.astype(str).eq(example_id)].sort_values("trial_index")
    trial = example.trial_index.to_numpy()
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(trial, example.model_prior_long, color="black")
    axes[0].set_ylabel("prior long")
    axes[1].plot(trial, example.model_delta_pre, color="tab:purple")
    axes[1].set_ylabel("delta")
    axes[2].plot(trial, example.model_expected_capped_run_length_pre, label="E[R]")
    if "model_map_capped_run_length_pre" in example:
        axes[2].plot(trial, example.model_map_capped_run_length_pre, alpha=.5, label="MAP R (diagnostic)")
    axes[2].set_ylabel("run length"); axes[2].legend(frameon=False)
    axes[3].plot(trial, example.model_expected_hazard_pre, label="E[h(R)]")
    axes[3].plot(trial, example.model_tail_probability_pre, label="P(R≥65)")
    axes[3].set(xlabel="trial", ylabel="probability", ylim=(0, 1)); axes[3].legend(frameon=False)
    figure.suptitle(f"M4 causal trajectory: {example_id}")
    files["causal_trajectory"] = _save_both(figure, output, "m4_causal_trajectory")

    records = []
    rare_records = []
    for session_id, frame in data.groupby("session_id", sort=False, observed=False):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        switches = np.flatnonzero(frame.true_block.ne(frame.true_block.shift()).to_numpy())[1:]
        rare = np.flatnonzero((
            (frame.true_block.eq("S") & frame.category_long.eq(1))
            | (frame.true_block.eq("L") & frame.category_long.eq(0))
        ).to_numpy())
        for destination, events, window in ((records, switches, 10), (rare_records, rare, 3)):
            for event, center in enumerate(events):
                for index in range(max(0, center-window), min(len(frame), center+window+1)):
                    destination.append({
                        "session_id": session_id, "event": event, "relative_trial": index-center,
                        "expected_run_length": frame.loc[index, "model_expected_capped_run_length_pre"],
                        "expected_hazard": frame.loc[index, "model_expected_hazard_pre"],
                        "tail_probability": frame.loc[index, "model_tail_probability_pre"],
                        "delta": frame.loc[index, "model_delta_pre"],
                    })
    for label, event_records in (("block_transition", records), ("rare_trial", rare_records)):
        table = pd.DataFrame(event_records)
        if table.empty:
            continue
        summary = table.groupby("relative_trial", observed=False).agg(
            expected_run_length=("expected_run_length", "mean"),
            expected_hazard=("expected_hazard", "mean"),
            tail_probability=("tail_probability", "mean"), delta=("delta", "mean"),
            n=("event", "size"),
        ).reset_index()
        summary.to_csv(output / f"m4_{label}_aligned.csv", index=False)
        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        for axis, column in zip(axes, ("expected_run_length", "expected_hazard", "tail_probability", "delta"), strict=True):
            axis.plot(summary.relative_trial, summary[column]); axis.axvline(0, color="black", ls="--")
            axis.set(xlabel="relative trial", ylabel=column)
        files[f"{label}_aligned"] = _save_both(figure, output, f"m4_{label}_aligned")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        hazard_curve.represented_run_length,
        hazard_curve.biased_hazard,
        linewidth=2,
        label="biased $h_B(r)$",
    )
    axis.plot(
        hazard_curve.represented_run_length,
        hazard_curve.neutral_hazard,
        linewidth=2,
        label="neutral $h_N(r)$",
    )
    axis.axvline(23, color="black", ls="--", label="registered center 23")
    axis.axvline(65, color="tab:red", ls=":", label="capped tail")
    axis.set(xlabel="represented run length", ylabel="hazard", ylim=(0, 1), title="Fitted M4 hazard")
    axis.legend(frameon=False)
    files["hazard"] = _save_both(figure, output, "m4_hazard")

    for index, surface in enumerate(identifiability_surfaces, start=1):
        axes_names = [name for name in ("ah", "delta0", "alpha_theta", "bh") if name in surface.columns]
        if len(axes_names) != 2:
            raise KeyError("Each M4 identifiability surface must have exactly two dynamic axes")
        horizontal, vertical = axes_names
        table = surface.pivot(index=vertical, columns=horizontal, values="delta_mean_nll_from_grid_minimum")
        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(table.to_numpy(), origin="lower", aspect="auto",
            extent=[table.columns.min(), table.columns.max(), table.index.min(), table.index.max()])
        axis.set(xlabel=horizontal, ylabel=vertical, title="M4 post-fit identifiability")
        figure.colorbar(image, ax=axis, label="mean NLL above grid minimum")
        files[f"identifiability_{index}"] = _save_both(figure, output, f"m4_identifiability_{horizontal}_{vertical}")

    if m3_predictions is not None:
        comparison = build_m4_m3_validation_comparison(data, m3_predictions)
        for name in ("overall", "per_session", "calibration", "context", "transition_aligned", "rare_common"):
            getattr(comparison, name).to_csv(output / f"m4_vs_m3_{name}.csv", index=False)
        session = comparison.per_session
        figure, axis = plt.subplots(figsize=(max(7, len(session)*.3), 4))
        values = session.delta_elpd_m4_vs_m3.to_numpy(float)
        axis.bar(np.arange(len(session)), values, color=np.where(values >= 0, "#2e7d32", "#c62828"))
        axis.axhline(0, color="black"); axis.set(xlabel="session", ylabel="ΔELPD M4−M3")
        files["m4_m3_comparison"] = _save_both(figure, output, "m4_vs_m3_per_session")

    manifest = {"model_id": "M4_hsmm_delta_v1", "files": files}
    (output / "m4_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
