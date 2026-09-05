"""M2 belief, prototype, calibration, switch, and comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        names.append(path.name)
    plt.close(fig)
    return names


def _paired_improvement(
    m2: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    reference_label: str,
    output: Path,
    files: dict[str, list[str]],
) -> None:
    keys = ["mouse_id", "session_id", "trial_index", "split_role"]
    reference_nll = reference[keys + ["per_trial_negative_log_likelihood"]].rename(
        columns={"per_trial_negative_log_likelihood": "reference_nll"}
    )
    paired = m2.merge(reference_nll, on=keys, validate="one_to_one")
    paired = paired[paired["model_evaluated"].astype(bool)].copy()
    paired["delta_elpd"] = (
        paired["reference_nll"] - paired["per_trial_negative_log_likelihood"]
    )
    stem_label = reference_label.lower()
    session = paired.groupby("session_id").delta_elpd.sum().sort_values()
    session.to_csv(output / f"m2_vs_{stem_label}_per_session_delta_elpd.csv")
    fig, ax = plt.subplots(figsize=(max(7, len(session) * 0.3), 4))
    ax.bar(np.arange(len(session)), session, color=np.where(session >= 0, "#2e7d32", "#c62828"))
    ax.axhline(0, color="black")
    ax.set(
        xlabel="Session",
        ylabel=f"ΔELPD M2−{reference_label}",
        title=f"Paired M2 versus {reference_label} improvement by session",
    )
    files[f"per_session_vs_{stem_label}"] = _save(
        fig, output, f"m2_vs_{stem_label}_per_session_delta_elpd"
    )

    paired["block_run_position"] = paired.groupby("session_id")["true_block"].transform(
        lambda values: values.groupby(values.ne(values.shift()).cumsum()).cumcount() + 1
    )
    paired["timing"] = np.where(paired.block_run_position <= 10, "early_1_10", "late_11_plus")
    paired["congruence"] = np.where(
        ((paired.true_block == "L") & (paired.category_long == 1))
        | ((paired.true_block == "S") & (paired.category_long == 0)),
        "common",
        "rare",
    )
    context = paired.groupby(["timing", "congruence"]).agg(
        delta_elpd=("delta_elpd", "sum"), n=("delta_elpd", "size")
    ).reset_index()
    context["delta_bits_per_trial"] = context.delta_elpd / (context.n * np.log(2))
    context.to_csv(output / f"m2_vs_{stem_label}_context_improvement.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(context.timing + "\n" + context.congruence, context.delta_bits_per_trial, color="#6a1b9a")
    ax.axhline(0, color="black")
    ax.set(ylabel="Δ bits/trial", title=f"M2 improvement over {reference_label}: early/late and rare/common")
    files[f"context_vs_{stem_label}"] = _save(
        fig, output, f"m2_vs_{stem_label}_context_improvement"
    )


def generate_m2_identifiability_figure(
    surface: pd.DataFrame,
    output_directory: str | Path,
) -> dict:
    """Save the post-fit fixed-other-parameters hazard/contrast loss contour."""

    required = {"ah", "delta0", "delta_mean_nll_from_grid_minimum", "fitted_ah", "fitted_delta0"}
    missing = sorted(required.difference(surface.columns))
    if missing:
        raise KeyError(f"M2 identifiability surface is missing columns: {missing}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface.to_csv(output / "m2_ah_delta0_loss_surface.csv", index=False)
    pivot = surface.pivot(index="delta0", columns="ah", values="delta_mean_nll_from_grid_minimum")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x, y, pivot.to_numpy(dtype=float), levels=20, cmap="viridis")
    ax.scatter(
        [float(surface["fitted_ah"].iloc[0])],
        [float(surface["fitted_delta0"].iloc[0])],
        marker="*",
        s=180,
        color="red",
        edgecolor="white",
        label="Fitted checkpoint",
    )
    ax.axhline(np.log(4.0), color="white", ls="--", lw=1.2, label="δ₀=log(4)")
    ax.set(
        xlabel="Hazard logit aₕ",
        ylabel="Static contrast δ₀",
        title="M2 post-fit hazard–contrast objective slice",
    )
    ax.legend(loc="best")
    fig.colorbar(contour, ax=ax, label="Mean NLL above grid minimum")
    files = _save(fig, output, "m2_ah_delta0_loss_surface")
    manifest = {
        "analysis": "post_fit_identifiability_fixed_other_four",
        "changes_fitted_loss_or_parameters": False,
        "surface_csv": "m2_ah_delta0_loss_surface.csv",
        "files": files,
    }
    (output / "m2_identifiability_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def generate_m2_diagnostic_figures(
    m2_predictions: pd.DataFrame,
    output_directory: str | Path,
    *,
    m1_predictions: pd.DataFrame | None = None,
    m0_predictions: pd.DataFrame | None = None,
) -> dict:
    """Create M2 diagnostics and optional paired M1/M0 comparisons."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = m2_predictions.copy()
    files: dict[str, list[str]] = {}
    evaluated = data["model_evaluated"].astype(bool)

    session_id = str(data["session_id"].iloc[0])
    example = data[data["session_id"].astype(str).eq(session_id)].sort_values("trial_index")
    x = np.arange(len(example))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(x, example["model_prior_long"], color="black", label="π(long)")
    axes[0].axhline(0.5, color="gray", ls="--")
    axes[0].set(ylabel="Prior", ylim=(0, 1), title=f"M2 causal prior: {session_id}")
    for state, color in zip(("N", "S", "L"), ("#777777", "#1565c0", "#c62828")):
        axes[1].plot(x, example[f"model_pre_choice_probability_{state}"], label=state, color=color)
    axes[1].set(xlabel="Task trial", ylabel="Pre-choice belief", ylim=(0, 1))
    axes[1].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=0.2)
    files["belief_trajectory"] = _save(fig, output, "m2_belief_trajectory")

    valid = data[evaluated].copy()
    valid["calibration_bin"] = pd.cut(
        valid["model_probability_long"], np.linspace(0, 1, 11), include_lowest=True
    )
    calibration = valid.groupby("calibration_bin", observed=False).agg(
        mean_predicted=("model_probability_long", "mean"),
        observed_long=("choice_long", "mean"),
        n=("choice_long", "size"),
    ).dropna().reset_index()
    calibration.to_csv(output / "m2_calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(calibration["mean_predicted"], calibration["observed_long"], "o-", color="#7b1fa2")
    ax.set(xlabel="Predicted P(long)", ylabel="Observed long rate", title="M2 calibration", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    files["calibration"] = _save(fig, output, "m2_calibration")

    aligned = []
    for sid, frame in data.groupby("session_id", sort=False):
        frame = frame.sort_values("trial_index").reset_index(drop=True)
        switches = np.flatnonzero(frame["true_block"].astype(str).ne(frame["true_block"].shift()).to_numpy())[1:]
        for event, center in enumerate(switches):
            for index in range(max(0, center - 10), min(len(frame), center + 11)):
                aligned.append({
                    "session_id": sid,
                    "event": event,
                    "relative_trial": index - center,
                    "prior_long": frame.loc[index, "model_prior_long"],
                    "probability_correct": frame.loc[index, "model_probability_correct"],
                })
    aligned_frame = pd.DataFrame(aligned)
    if not aligned_frame.empty:
        switch_summary = aligned_frame.groupby("relative_trial").agg(
            prior_long=("prior_long", "mean"),
            probability_correct=("probability_correct", "mean"),
            n=("event", "size"),
        ).reset_index()
        switch_summary.to_csv(output / "m2_switch_aligned.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(switch_summary.relative_trial, switch_summary.prior_long, color="#7b1fa2")
        axes[1].plot(switch_summary.relative_trial, switch_summary.probability_correct, color="#1565c0")
        axes[0].set(ylabel="Mean π(long)")
        axes[1].set(ylabel="Mean expected P(correct)")
        for axis in axes:
            axis.axvline(0, color="black", ls="--")
            axis.set(xlabel="Trials from true switch")
        files["switch_aligned"] = _save(fig, output, "m2_switch_aligned")

    hazard = float(data["fitted_hazard"].iloc[0])
    delta0 = float(data["fitted_delta0"].iloc[0])
    theta = [float(data[f"model_subjective_prototype_{state}"].iloc[0]) for state in ("N", "S", "L")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["N", "S", "L"], theta, color=["#777777", "#1565c0", "#c62828"])
    axes[0].scatter(["N", "S", "L"], [0.5, 0.2, 0.8], marker="x", s=80, color="black", label="M1 true rates")
    axes[0].set(ylabel="P(long category)", ylim=(0, 1), title=f"Static prototypes; δ₀={delta0:.4f}")
    axes[0].legend()
    axes[1].bar(
        ["h_B", "h_N", "1/h_B (/100)", "1/h_N (/100)"],
        [hazard, hazard / 2, (1 / hazard) / 100, (2 / hazard) / 100],
        color=["#7b1fa2", "#2e7d32", "#607d8b", "#90a4ae"],
    )
    axes[1].set(
        ylabel="Value",
        title=f"Biased mean={1/hazard:.2f}; neutral mean={2/hazard:.2f}",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    files["prototypes_and_hazard"] = _save(fig, output, "m2_fitted_prototypes_and_hazard")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Fitted δ₀"], [delta0], color="#7b1fa2", width=0.5)
    ax.axhline(np.log(4.0), color="black", ls="--", label="Task-like δ₀=log(4)")
    ax.set(ylabel="Static subjective contrast", title="M2 fitted contrast versus M1 restriction")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    files["delta0_reference"] = _save(fig, output, "m2_fitted_delta0_reference")

    if m1_predictions is not None:
        _paired_improvement(data, m1_predictions, reference_label="M1", output=output, files=files)
        keys = ["mouse_id", "session_id", "trial_index", "split_role"]
        trajectory_columns = keys + ["model_prior_long"] + [
            f"model_pre_choice_probability_{state}" for state in ("N", "S", "L")
        ]
        aligned = data[trajectory_columns].merge(
            m1_predictions[trajectory_columns],
            on=keys,
            suffixes=("_m2", "_m1"),
            validate="one_to_one",
        )
        example_id = str(aligned["session_id"].iloc[0])
        example = aligned[aligned["session_id"].astype(str).eq(example_id)].sort_values("trial_index")
        x = np.arange(len(example))
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(x, example["model_prior_long_m1"], label="M1", color="#c62828")
        ax.plot(x, example["model_prior_long_m2"], label="M2", color="#7b1fa2", ls="--")
        ax.set(xlabel="Task trial", ylabel="π(long)", ylim=(0, 1), title=f"M1/M2 pre-choice prior: {example_id}")
        ax.legend(); ax.grid(alpha=.2)
        files["m1_m2_prior_overlay"] = _save(fig, output, "m1_m2_prior_trajectory_overlay")
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for axis, state, color in zip(axes, ("N", "S", "L"), ("#777777", "#1565c0", "#c62828")):
            axis.plot(x, example[f"model_pre_choice_probability_{state}_m1"], label="M1", color=color)
            axis.plot(x, example[f"model_pre_choice_probability_{state}_m2"], label="M2", color=color, ls="--")
            axis.set(ylabel=f"P({state})", ylim=(0, 1)); axis.grid(alpha=.2)
        axes[0].set(title=f"M1/M2 pre-choice block beliefs: {example_id}")
        axes[-1].set(xlabel="Task trial"); axes[0].legend()
        files["m1_m2_belief_overlay"] = _save(fig, output, "m1_m2_block_belief_overlay")
    if m0_predictions is not None:
        _paired_improvement(data, m0_predictions, reference_label="M0", output=output, files=files)
    if m1_predictions is not None:
        comparison_frames = [("M1", m1_predictions), ("M2", data)]
        if m0_predictions is not None:
            comparison_frames.insert(0, ("M0", m0_predictions))
        rows = []
        for label, frame in comparison_frames:
            selected = frame.loc[
                frame["split_role"].astype(str).eq("validation")
                & frame["model_evaluated"].astype(bool)
            ].copy()
            if selected.empty:
                continue
            observed = pd.to_numeric(selected["choice_long"], errors="coerce").to_numpy(float)
            probability = selected["model_probability_long"].to_numpy(float)
            nll = selected["per_trial_negative_log_likelihood"].to_numpy(float)
            rows.append({
                "model": label,
                "n_valid": len(selected),
                "mean_nll": float(np.mean(nll)),
                "elpd": float(-np.sum(nll)),
                "brier_score": float(np.mean((probability - observed) ** 2)),
            })
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary.to_csv(output / "m0_m1_m2_validation_metrics.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].bar(summary.model, summary.mean_nll, color="#607d8b")
            axes[0].set(ylabel="Validation mean NLL", title="M0/M1/M2 validation NLL")
            axes[1].bar(summary.model, summary.brier_score, color="#00897b")
            axes[1].set(ylabel="Validation Brier score", title="M0/M1/M2 validation Brier")
            for axis in axes: axis.grid(axis="y", alpha=.25)
            files["validation_nll_brier_comparison"] = _save(fig, output, "m0_m1_m2_validation_nll_brier")
            m1_row = summary.loc[summary.model.eq("M1")]
            m2_row = summary.loc[summary.model.eq("M2")]
            if not m1_row.empty and not m2_row.empty:
                m1_elpd = float(m1_row.elpd.iloc[0]); m2_elpd = float(m2_row.elpd.iloc[0])
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["M1", "M2"], [m1_elpd, m2_elpd], color=["#c62828", "#7b1fa2"])
                ax.set(ylabel="Validation ELPD", title=f"M2−M1 ΔELPD = {m2_elpd-m1_elpd:.3f}")
                ax.grid(axis="y", alpha=.25)
                files["m1_m2_validation_elpd"] = _save(fig, output, "m1_m2_validation_elpd")
    manifest = {
        "analysis": "M2 static-prototype causal diagnostics",
        "files": files,
        "m1_comparison_included": m1_predictions is not None,
        "m0_comparison_included": m0_predictions is not None,
    }
    (output / "m2_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
