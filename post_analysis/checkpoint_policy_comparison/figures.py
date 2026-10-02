"""Figures and short report for checkpoint-policy comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MODEL_BY_ID, MODEL_SPECS


def _save(fig, root: Path, name: str):
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_outputs(output: Path) -> None:
    tables, figures = output / "tables", output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(tables / "model_replay_status.csv")
    selections = pd.read_csv(tables / "checkpoint_selection.csv")
    deltas = pd.read_csv(tables / "policy_deltas.csv")
    summary = pd.read_csv(tables / "bootstrap_summary.csv")
    ranking = pd.read_csv(tables / "model_rankings.csv")
    outcomes = pd.read_csv(tables / "test_win_tie_loss.csv")
    order = [s.model_id for s in MODEL_SPECS if s.model_id in set(status.loc[status.status == "replayed", "model_id"])]
    labels = [MODEL_BY_ID[m].label for m in order]
    colors = [MODEL_BY_ID[m].color for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.4 * len(order))), sharey=True)
    for ax, policy in zip(axes, ["validation_min", "validation_conservative"]):
        frame = summary[(summary.policy == policy) & (summary.split_role == "test")].set_index("model_id").reindex(order)
        y = np.arange(len(order)); means = frame.mean_delta_nll.to_numpy(float)
        err = np.vstack([means - frame.bootstrap_ci_low.to_numpy(float), frame.bootstrap_ci_high.to_numpy(float) - means])
        ax.errorbar(means, y, xerr=err, fmt="o", color="#222222", capsize=3)
        ax.axvline(0, color="0.5", lw=1); ax.set_title(policy.replace("_", " "))
        ax.set_xlabel("Test mean NLL delta vs current (lower is better)")
    axes[0].set_yticks(np.arange(len(order)), labels)
    _save(fig, figures, "fig01_test_nll_delta_by_model")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, policy in zip(axes, ["validation_min", "validation_conservative"]):
        for model, color in zip(order, colors):
            frame = deltas[(deltas.policy == policy) & (deltas.model_id == model)]
            means = frame.groupby("split_role").delta_mean_nll.mean().reindex(["train", "validation", "test"])
            ax.plot([0, 1, 2], means, marker="o", color=color, alpha=.8, label=MODEL_BY_ID[model].label)
        ax.axhline(0, color="0.5", lw=1); ax.set_xticks([0, 1, 2], ["Train", "Validation*", "Test"])
        ax.set_title(policy.replace("_", " ")); ax.set_ylabel("Mean NLL delta vs current")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, figures, "fig02_split_nll_delta_by_model")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .4 * len(order))), sharey=True)
    good = selections[selections.status == "selected"]
    for ax, policy in zip(axes, ["validation_min", "validation_conservative"]):
        data = [good[(good.model_id == model) & (good.policy == policy)].checkpoint_fraction.dropna() for model in order]
        ax.boxplot(data, vert=False, labels=labels, showfliers=False)
        ax.set_xlim(-.02, 1.02); ax.set_xlabel("Checkpoint position in saved history")
        ax.set_title(policy.replace("_", " "))
    _save(fig, figures, "fig03_checkpoint_fraction")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .4 * len(order))), sharey=True)
    for ax, policy in zip(axes, ["validation_min", "validation_conservative"]):
        counts = outcomes[outcomes.policy == policy].groupby(["model_id", "outcome"]).size().unstack(fill_value=0).reindex(order).fillna(0)
        total = counts.sum(axis=1).replace(0, np.nan)
        left = np.zeros(len(order))
        for outcome, color in [("win", "#2CA02C"), ("tie", "#BDBDBD"), ("loss", "#D62728")]:
            values = counts.get(outcome, pd.Series(0, index=order)).to_numpy() / total.to_numpy()
            ax.barh(np.arange(len(order)), values, left=left, label=outcome, color=color)
            left += np.nan_to_num(values)
        ax.set_xlim(0, 1); ax.set_xlabel("Fraction of windows"); ax.set_title(policy.replace("_", " "))
    axes[0].set_yticks(np.arange(len(order)), labels); axes[1].legend(loc="lower right")
    _save(fig, figures, "fig04_test_win_tie_loss")

    fig, ax = plt.subplots(figsize=(10, max(5, .4 * len(order))))
    for policy, marker in [("current_train_mle", "o"), ("validation_min", "s"), ("validation_conservative", "^")]:
        frame = ranking[ranking.policy == policy].set_index("model_id").reindex(order)
        ax.scatter(frame.mean_test_nll, np.arange(len(order)), label=policy.replace("_", " "), marker=marker, s=45)
    ax.set_yticks(np.arange(len(order)), labels); ax.set_xlabel("Mean test NLL on strict common windows")
    ax.legend(); _save(fig, figures, "fig05_test_model_ranking_by_policy")

    test = summary[summary.split_role == "test"]
    report = [
        "# MC08 checkpoint-policy comparison", "",
        "This is a retrospective, exploratory replay of saved L-BFGS histories. No fitted artifacts or model code were changed.", "",
        "## Coverage", "",
    ]
    for row in status.itertuples():
        report.append(f"- {row.model_id}: {row.status}; {row.windows_scored}/{row.windows_found} windows scored" + (f" — {row.reason}" if isinstance(row.reason, str) and row.reason else ""))
    report.extend(["", "## Test-set summary", ""])
    for policy in ["validation_min", "validation_conservative"]:
        frame = test[test.policy == policy]
        report.append(f"### {policy}")
        report.append("")
        if frame.empty:
            report.append("No eligible results.")
        else:
            improved = int((frame.mean_delta_nll < 0).sum())
            report.append(f"Across {len(frame)} replayed model families, {improved} had a lower mean test NLL than the current policy.")
            best = frame.sort_values("mean_delta_nll").iloc[0]
            worst = frame.sort_values("mean_delta_nll").iloc[-1]
            report.append(f"Largest mean improvement: {best.model_id} ({best.mean_delta_nll:+.6f}). Largest mean worsening: {worst.model_id} ({worst.mean_delta_nll:+.6f}).")
        report.append("")
    report.extend([
        "## Interpretation limits", "",
        "Validation is the selection split for the two new policies, so its improvement is not independent evidence. Test comparisons were unlocked only after checkpoint selection and validation replay checks. Rolling windows overlap strongly; block-bootstrap intervals (block length 12) are exploratory, not independent-window inference. The 0.001 policy threshold was fixed before test replay.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n")
