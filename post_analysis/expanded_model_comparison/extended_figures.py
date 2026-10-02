"""Sliding-only Figures 7--24 for the expanded comparison."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import rankdata

from .config import MODEL_BY_ID, MODEL_ORDER, SPLIT_ROLES, group_style
from .extended_collect import DURATION_BIN_DEFINITIONS

FIGURE_IDS = tuple(f"fig{number:02d}" for number in range(7, 25))


def _eligible(status: pd.DataFrame) -> tuple[str, ...]:
    ids = set(status.loc[status.eligible_for_comparison.astype(str).str.lower().eq("true"), "model_id"])
    return tuple(model for model in MODEL_ORDER if model in ids)


def _state(status: pd.DataFrame) -> str:
    n = len(_eligible(status))
    return f"PROVISIONAL: {n}/{len(status)} models complete" if n < len(status) else "FINAL CONFIGURED SET"


def _label(model: str) -> str:
    return MODEL_BY_ID[model].short_label


def _finish(fig, output: Path, stem: str, *, rect=None) -> None:
    fig.tight_layout(rect=rect)
    temporary = []
    try:
        for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
            target = output / f"{stem}.{suffix}"
            pending = output / f".{stem}.pending.{suffix}"
            temporary.append(pending)
            fig.savefig(pending, bbox_inches="tight", **options)
            pending.replace(target)
    finally:
        plt.close(fig)
        for pending in temporary:
            pending.unlink(missing_ok=True)


def _line_style(model: str) -> str:
    return group_style(model, "line")


def _legend(ax, models, ncol=4, *, include_mouse=False):
    handles = ([plt.Line2D([], [], color="black", marker="o", label="Mouse")] if include_mouse else [])
    handles += [plt.Line2D([], [], color=MODEL_BY_ID[m].color, linestyle=_line_style(m), label=_label(m)) for m in models]
    ax.legend(handles=handles, ncol=ncol, frameon=False, fontsize=7)


def figure07_subsets(table, status, output):
    models = _eligible(status); subsets = table.trial_subset.drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    for ax, split in zip(axes, SPLIT_ROLES):
        part = table[table.split_role.eq(split)]
        image = np.full((len(models), len(subsets)), np.nan)
        counts = np.zeros_like(image)
        for j, subset in enumerate(subsets):
            p = part[part.trial_subset.eq(subset)].pivot(index="window_id", columns="model_id", values="mean_nll").reindex(columns=models).dropna()
            if len(p): image[:, j] = p.mean().to_numpy(); counts[:, j] = len(p)
        im = ax.imshow(image, aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(models)), [_label(m) for m in models]); ax.set_xticks(range(len(subsets)), subsets, rotation=30, ha="right")
        ax.set_title(f"{split.capitalize()} — color=mean NLL; cell support is common complete windows")
        for i in range(len(models)):
            for j in range(len(subsets)):
                if np.isfinite(image[i, j]): ax.text(j, i, f"{image[i,j]:.3f}\nN={int(counts[i,j])}", ha="center", va="center", fontsize=5.5)
        fig.colorbar(im, ax=ax, fraction=.015, pad=.01, label="Mean NLL")
    fig.suptitle(f"Figure 7 — NLL by trial subset — {_state(status)}")
    _finish(fig, output, "fig07_sliding_window_trial_subset_nll")


def figure08_rank_stability(metrics, status, output):
    models = _eligible(status); fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, split in zip(axes, SPLIT_ROLES):
        matrix = metrics[metrics.split_role.eq(split)].pivot(index="window_id", columns="model_id", values="mean_nll").reindex(columns=models).dropna()
        ranks = matrix.apply(lambda row: rankdata(row, method="average"), axis=1, result_type="expand"); ranks.columns=models
        for model in models: ax.plot(matrix.index + 12, ranks[model], color=MODEL_BY_ID[model].color, linestyle=_line_style(model), lw=1, marker="o", ms=2)
        ax.invert_yaxis(); ax.set_ylabel("Rank"); ax.set_title(split.capitalize()); ax.grid(alpha=.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Held-out session chronological rank")
    handles = [plt.Line2D([], [], color=MODEL_BY_ID[m].color, linestyle=_line_style(m), label=_label(m)) for m in models]
    axes[0].legend(
        handles=handles, ncol=4, frameon=False, fontsize=6.5, loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.3), mode="expand",
    )
    fig.suptitle(f"Figure 8 — Model-rank stability across windows — {_state(status)}", y=0.995)
    _finish(fig, output, "fig08_sliding_window_rank_stability", rect=(0, 0, 1, 0.96))


def _parameter_panels(parameters, status, names, output, stem, title, mode="trajectory"):
    models = _eligible(status); present = [n for n in names if n in set(parameters.param_name)]
    cols=2; rows=max(1, int(np.ceil(len(present)/cols))); fig, axes=plt.subplots(rows, cols, figsize=(14, 3.7*rows), squeeze=False)
    for ax, name in zip(axes.flat, present):
        part=parameters[parameters.param_name.eq(name)]
        applicable=[m for m in models if m in set(part.model_id)]
        for model in applicable:
            values=part[part.model_id.eq(model)].sort_values("test_rank")
            if mode=="trajectory": ax.plot(values.test_rank, values.param_value, color=MODEL_BY_ID[model].color, linestyle=_line_style(model), marker="o", ms=2, lw=1, label=_label(model))
            else: ax.boxplot(values.param_value, positions=[applicable.index(model)], widths=.55, showfliers=False); ax.scatter(np.full(len(values), applicable.index(model)), values.param_value, s=8, alpha=.3, color=MODEL_BY_ID[model].color)
        ax.set_title(name); ax.grid(axis="y", alpha=.2)
        if mode=="distribution": ax.set_xticks(range(len(applicable)), [_label(m) for m in applicable], rotation=40, ha="right")
    for ax in axes.flat[len(present):]: ax.axis("off")
    if present and mode=="trajectory": _legend(axes.flat[0], models)
    fig.suptitle(f"{title} — {_state(status)}")
    _finish(fig, output, stem)


def figure09(parameters,status,output): _parameter_panels(parameters,status,["w","s0","ell_short","ell_long","kappa"],output=output,stem="fig09_shared_parameter_trajectories",title="Figure 9 — Shared fitted-parameter trajectories")


def figure10(status, output):
    fig, ax=plt.subplots(figsize=(13,5)); x=np.arange(len(status)); complete=status.complete_window_count.astype(int)
    bars=ax.bar(x,complete,color=[MODEL_BY_ID[m].color for m in status.model_id]);
    for bar,oracle in zip(bars,status.oracle_benchmark):
        if bool(oracle): bar.set_hatch("//")
    ax.axhline(status.expected_windows.iloc[0],color="black",ls="--",lw=1); ax.set_ylim(0,status.expected_windows.max()+3)
    ax.set_xticks(x,[_label(m) for m in status.model_id],rotation=45,ha="right"); ax.set_ylabel("Complete windows")
    ax.set_title(f"Figure 10 — Sliding-window fit availability — {_state(status)}\nIncomplete models are shown here only and excluded from every scientific panel")
    _finish(fig,output,"fig10_sliding_window_availability")


def figure11(parameters,status,output): _parameter_panels(parameters,status,["w","s0","ell_short","ell_long","kappa"],output=output,stem="fig11_shared_parameter_distributions",title="Figure 11 — Shared fitted-parameter distributions",mode="distribution")


# Registered hazard-kernel constants shared by every ah/bh-bearing model
# (m1_hmm_true, m1_dynamic, m2_static, m2_dynamic, and the two dynamic-bias
# models): hB(r) = sigmoid(ah + bh*(r-center)), frozen at the r=cap tail --
# see src/m0_sensory_v1/model/capped_tail_hsmm.py and each model's own
# registry, which hard-fails fit if center/cap ever drift from these.
_HAZARD_CENTER = 23
_HAZARD_CAP = 65


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _expected_block_length(ah: float, bh: float) -> float:
    """Exact mean of the capped-tail duration law for one fitted (ah, bh).

    Reduces to the geometric-distribution mean 1/sigmoid(ah) when bh=0, so
    this single closed form is correct for both constant-hazard models
    (bh absent, treated as 0) and run-length-dependent ones -- unlike
    1/sigmoid(ah) alone, which is only the reciprocal hazard AT the r=23
    center once bh != 0.
    """
    if not np.isfinite(ah) or not np.isfinite(bh):
        return float("nan")
    r = np.arange(1, _HAZARD_CAP, dtype=float)
    hazards = _sigmoid_np(ah + bh * (r - _HAZARD_CENTER))
    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - hazards)[:-1]))
    pmf = survival_before * hazards
    mean_before_tail = float(np.sum(r * pmf))
    survival_at_tail = float(np.prod(1.0 - hazards))
    tail_hazard = float(_sigmoid_np(np.array([ah + bh * (_HAZARD_CAP - _HAZARD_CENTER)]))[0])
    return mean_before_tail + survival_at_tail * ((_HAZARD_CAP - 1) + 1.0 / tail_hazard)


def _model_param_series(parameters: pd.DataFrame, model: str, name: str) -> pd.Series:
    part = parameters[parameters.model_id.eq(model) & parameters.param_name.eq(name)]
    return part.sort_values("test_rank").set_index("test_rank").param_value


def _linear_trend_xy(series: pd.Series) -> pd.Series | None:
    valid = series.dropna()
    if len(valid) < 2:
        return None
    slope, intercept = np.polyfit(valid.index.to_numpy(float), valid.to_numpy(float), 1)
    x = series.index.to_numpy(float)
    return pd.Series(slope * x + intercept, index=series.index)


# (heading, explanation) cards, one per raw fitted coordinate that gets its
# own row in Figure 12 -- mirrors post_analysis/model_comparison/figures.py's
# MODEL_SPECIFIC_PARAMETER_CARDS / fig09 for the (differently parameterized)
# 13-model expanded registry.
_PARAMETER_CARDS = {
    "ah": (
        "Switch hazard and expected block length",
        "hB(r) = sigmoid(ah + bh×(r−23)); r is the represented run length, 23 the registered "
        "hazard center, 65 the cap. Column 3 plots the exact mean of the full discrete duration law "
        "implied by each window's fitted (ah, bh) -- this reduces to 1/sigmoid(ah) for the rows below "
        "with no bh (treated as bh=0). See Figure 22 for the full per-window hazard-curve family.",
    ),
    "delta0": (
        "Initial subjective category-rate contrast",
        "At session start: θS = sigmoid(−delta0), θL = sigmoid(delta0) -- solid/dashed in "
        "column 3. Larger |delta0| means a stronger expected short/long category bias in the "
        "corresponding block.",
    ),
    "bh": (
        "Run-length dependence of switch hazard",
        "hB(r) = sigmoid(ah + bh×(r−23)); bh=0 is the constant-hazard boundary nested inside "
        "every dynamic-hazard model here. Column 3 plots the excess hazard hB(r=65)−hB(r=23) implied "
        "by each window's fitted (ah, bh); positivity alone is not strong evidence of a real "
        "run-length effect.",
    ),
    "a": (
        "History-kernel decay weight",
        "alpha = sigmoid(a) is the constrained logistic-kernel decay in (0,1) the model actually uses "
        "(near 1: recent trials dominate; near 0: history barely matters). Column 1 is the raw, "
        "unconstrained fit coordinate a, which can be poorly identified and take extreme values when "
        "the kernel has little influence on behavior -- column 3's alpha is the quantity to read.",
    ),
    "eta_logit": (
        "History/action bias weight",
        "eta = sigmoid(eta_logit) is the constrained bias weight in (0,1) placed on the model's "
        "history/action-kernel signal. Column 1 is the raw, unconstrained fit coordinate; column 3's "
        "eta is the quantity to read.",
    ),
}
_PARAMETER_ROWS = ("ah", "delta0", "bh", "a", "eta_logit")


def figure12(parameters, status, output):
    """Model-specific parameter trajectories, styled after model_comparison's
    fig09: raw fitted coordinate (col 1), an explanatory card (col 2), and
    the closed-form quantity the card defines, with a linear-trend overlay
    (col 3) -- instead of plotting the raw and sigmoid-constrained
    coordinates as separate undifferentiated rows with a legend listing
    every eligible model regardless of whether it fits that parameter.
    """
    models = _eligible(status)
    present = [name for name in _PARAMETER_ROWS if name in set(parameters.param_name)]
    if not present:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No model-specific parameters available yet", ha="center", va="center")
        _finish(fig, output, "fig12_model_specific_parameter_trajectories")
        return

    fig, axes = plt.subplots(
        len(present), 3, figsize=(19.5, 4.3 * len(present)),
        gridspec_kw={"width_ratios": [2.1, 1.0, 2.1]}, squeeze=False,
    )
    for row_index, name in enumerate(present):
        raw_ax, card_ax, derived_ax = axes[row_index]
        applicable = [m for m in models if name in MODEL_BY_ID[m].parameter_names]

        for model in applicable:
            series = _model_param_series(parameters, model, name)
            raw_ax.plot(
                series.index, series.to_numpy(float), color=MODEL_BY_ID[model].color,
                linestyle=_line_style(model), marker="o", ms=2.6, lw=1.1,
                label=f"{_label(model)} ({len(series)})",
            )
        if name == "bh":
            raw_ax.axhline(0, color="0.5", ls="--", lw=0.8, zorder=0)
        raw_ax.set_ylabel(name)
        raw_ax.grid(axis="y", alpha=.2)
        if applicable:
            raw_ax.legend(fontsize=6.5, ncol=min(len(applicable), 2), frameon=False)
        else:
            raw_ax.text(0.5, 0.5, "No eligible model fits this parameter", ha="center", va="center",
                         transform=raw_ax.transAxes, fontsize=8, color="0.5", style="italic")

        card_ax.axis("off")
        heading, explanation = _PARAMETER_CARDS[name]
        card_ax.text(0.0, 0.92, heading, transform=card_ax.transAxes, ha="left", va="top",
                     fontsize=9.5, fontweight="bold")
        card_ax.text(0.0, 0.74, textwrap.fill(explanation, width=58), transform=card_ax.transAxes,
                     ha="left", va="top", fontsize=7.8, linespacing=1.35,
                     bbox={"boxstyle": "round,pad=0.5", "facecolor": "0.97", "edgecolor": "0.82"})

        if name == "ah":
            for model in applicable:
                ah_series = _model_param_series(parameters, model, "ah")
                if "bh" in MODEL_BY_ID[model].parameter_names:
                    bh_series = _model_param_series(parameters, model, "bh").reindex(ah_series.index)
                else:
                    bh_series = pd.Series(0.0, index=ah_series.index)
                derived = pd.Series(
                    {rank: _expected_block_length(ah, bh) for rank, ah, bh in
                     zip(ah_series.index, ah_series.to_numpy(float), bh_series.fillna(0.0).to_numpy(float))}
                )
                derived_ax.plot(derived.index, derived.to_numpy(float), color=MODEL_BY_ID[model].color,
                                 linestyle=_line_style(model), marker="o", ms=2.2, lw=1.0, alpha=.9)
                trend = _linear_trend_xy(derived)
                if trend is not None:
                    derived_ax.plot(trend.index, trend.to_numpy(float), color=MODEL_BY_ID[model].color,
                                     linestyle=_line_style(model), lw=2.6, alpha=.28)
            derived_ax.set_ylabel("Expected block length (trials)")
        elif name == "delta0":
            for model in applicable:
                delta0 = _model_param_series(parameters, model, "delta0")
                theta_s = 1.0 / (1.0 + np.exp(delta0.to_numpy(float)))
                theta_l = 1.0 / (1.0 + np.exp(-delta0.to_numpy(float)))
                derived_ax.plot(delta0.index, theta_s, color=MODEL_BY_ID[model].color,
                                 linestyle="-", marker="o", ms=2.2, lw=1.0, alpha=.9)
                derived_ax.plot(delta0.index, theta_l, color=MODEL_BY_ID[model].color,
                                 linestyle="--", marker="o", ms=2.2, lw=1.0, alpha=.9)
            derived_ax.set_ylabel("Subjective category rate")
            derived_ax.set_ylim(0, 1)
        elif name == "bh":
            for model in applicable:
                ah_series = _model_param_series(parameters, model, "ah")
                bh_series = _model_param_series(parameters, model, "bh").reindex(ah_series.index)
                hazard_23 = _sigmoid_np(ah_series.to_numpy(float))
                hazard_65 = _sigmoid_np(
                    ah_series.to_numpy(float) + bh_series.to_numpy(float) * (_HAZARD_CAP - _HAZARD_CENTER)
                )
                excess = pd.Series(hazard_65 - hazard_23, index=ah_series.index)
                derived_ax.plot(excess.index, excess.to_numpy(float), color=MODEL_BY_ID[model].color,
                                 linestyle=_line_style(model), marker="o", ms=2.2, lw=1.0, alpha=.9)
                trend = _linear_trend_xy(excess)
                if trend is not None:
                    derived_ax.plot(trend.index, trend.to_numpy(float), color=MODEL_BY_ID[model].color,
                                     linestyle=_line_style(model), lw=2.6, alpha=.28)
            derived_ax.axhline(0, color="0.5", ls="--", lw=0.8, zorder=0)
            derived_ax.set_ylabel("Excess switch hazard hB(r=65)−hB(r=23)")
        else:
            derived_name = "alpha" if name == "a" else "eta"
            for model in applicable:
                derived = _model_param_series(parameters, model, derived_name)
                derived_ax.plot(derived.index, derived.to_numpy(float), color=MODEL_BY_ID[model].color,
                                 linestyle=_line_style(model), marker="o", ms=2.2, lw=1.0, alpha=.9)
                trend = _linear_trend_xy(derived)
                if trend is not None:
                    derived_ax.plot(trend.index, trend.to_numpy(float), color=MODEL_BY_ID[model].color,
                                     linestyle=_line_style(model), lw=2.6, alpha=.28)
            derived_ax.set_ylabel(derived_name)
            derived_ax.set_ylim(-0.02, 1.02)
        derived_ax.grid(axis="y", alpha=.2)
        if row_index == 0:
            trend_handles = [
                plt.Line2D([0], [0], color="0.25", marker="o", ms=3, lw=1.0, label="Derived value (per window)"),
                plt.Line2D([0], [0], color="0.25", lw=2.6, alpha=.4, label="Linear trend across windows"),
            ]
            derived_ax.legend(handles=trend_handles, fontsize=6.2, loc="best", framealpha=.85)

    axes[-1, 0].set_xlabel("Held-out session chronological rank")
    axes[-1, 2].set_xlabel("Held-out session chronological rank")
    fig.suptitle(f"Figure 12 — Model-specific parameter trajectories — {_state(status)}", y=0.995)
    fig.text(
        0.5, 0.006,
        "Single mouse; point estimates from overlapping rolling windows, not independent replications. "
        "A model missing from a row does not fit that parameter. Column 1 is the raw fitted coordinate; "
        "column 3 is the closed-form quantity column 2 defines, with a per-model linear-trend overlay. "
        "Figure 21 gives mean±SD across windows for every fitted parameter; Figure 22 gives the full "
        "per-window hazard-curve family for every ah-bearing model.",
        ha="center", fontsize=7.6, style="italic",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.975))
    _finish(fig, output, "fig12_model_specific_parameter_trajectories")


def figure13(wins,status,output):
    models=_eligible(status); data=wins[wins.split_role.eq("train")].sort_values("window_id"); fig,ax=plt.subplots(figsize=(14,4))
    for _,row in data.iterrows():
        for k,m in enumerate(str(row.winning_models).split("|")): ax.scatter(row.window_id+12,k*.08,color=MODEL_BY_ID[m].color,marker="s",s=30)
    ax.set_yticks([]); ax.set_xlabel("Held-out session chronological rank"); ax.set_title(f"Figure 13 — Training-fit winner timeline — {_state(status)}"); ax.grid(axis="x",alpha=.2); _legend(ax,models)
    _finish(fig,output,"fig13_training_winner_timeline")


def _winner_agreement_matrix(wins: pd.DataFrame, models: tuple[str, ...]) -> np.ndarray:
    train=wins[wins.split_role.eq("train")].set_index("window_id"); test=wins[wins.split_role.eq("test")].set_index("window_id"); matrix=np.zeros((len(models),len(models)),float)
    for window in train.index.intersection(test.index):
        train_winners=str(train.loc[window,"winning_models"]).split("|"); test_winners=str(test.loc[window,"winning_models"]).split("|"); credit=1/(len(train_winners)*len(test_winners))
        for a in train_winners:
            for b in test_winners: matrix[models.index(a),models.index(b)]+=credit
    return matrix


def figure14(wins,status,output):
    models=_eligible(status); matrix=_winner_agreement_matrix(wins, models)
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(matrix,cmap="Blues"); ax.set_xticks(range(len(models)),[_label(m) for m in models],rotation=45,ha="right"); ax.set_yticks(range(len(models)),[_label(m) for m in models]); ax.set_xlabel("Test winner"); ax.set_ylabel("Train winner")
    for i,j in np.ndindex(matrix.shape):
        if matrix[i,j]: ax.text(j,i,f"{matrix[i,j]:g}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="Fractional window count"); ax.set_title(f"Figure 14 — Train-versus-test winner agreement — {_state(status)}")
    _finish(fig,output,"fig14_train_test_winner_agreement")


def figure15(ic,status,output):
    models=_eligible(status); data=ic[ic.ranking_scope.eq("all_models")]; fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,field,title in zip(axes,["aic_weight","bic_weight"],["AIC","BIC"]):
        means=data.groupby("model_id")[field].mean().reindex(models); bars=ax.bar(range(len(models)),means,color=[MODEL_BY_ID[m].color for m in models]);
        for bar,m in zip(bars,models):
            if MODEL_BY_ID[m].oracle: bar.set_hatch("//")
        ax.set_xticks(range(len(models)),[_label(m) for m in models],rotation=45,ha="right"); ax.set_ylabel("Mean model weight"); ax.set_title(title); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"Figure 15 — Training information-criterion weights — {_state(status)}")
    _finish(fig,output,"fig15_information_criterion_weights")


def _window_weighted(events: pd.DataFrame, value: str) -> pd.DataFrame:
    """Collapse trial types with their event counts, then retain windows as replicates."""
    data = events.dropna(subset=[value, "n_events"]).copy()
    data["weighted_value"] = data[value] * data.n_events
    grouped = data.groupby(["model_id", "window_id", "relative_position"], as_index=False).agg(
        weighted_value=("weighted_value", "sum"), n_events=("n_events", "sum"))
    grouped[value] = grouped.weighted_value / grouped.n_events
    return grouped


def _event_grid(events,status,output,stem,title,value,events_order):
    models=_eligible(status); fig,axes=plt.subplots(3,len(events_order),figsize=(4.5*len(events_order),10),squeeze=False,sharex="col")
    for i,split in enumerate(SPLIT_ROLES):
        for j,event in enumerate(events_order):
            ax=axes[i,j]; part=events[events.split_role.eq(split)&events.event.eq(event)]
            for model in models:
                window_values = _window_weighted(part[part.model_id.eq(model)], value)
                p=window_values.groupby("relative_position")[value].agg(["mean","sem"])
                if len(p): ax.plot(p.index,p["mean"],color=MODEL_BY_ID[model].color,ls=_line_style(model),lw=1.2); ax.fill_between(p.index,p["mean"]-p["sem"],p["mean"]+p["sem"],color=MODEL_BY_ID[model].color,alpha=.06)
            pretty={"short_to_long":"short → long","long_to_short":"long → short","rare_short":"rare short","rare_long":"rare long"}[event]
            ax.axvline(0,color="black",ls="--",lw=.7); ax.set_title(f"{split}: {pretty}"); ax.grid(alpha=.15)
            if i == 2: ax.set_xlabel("Trials relative to event (0 = event trial)")
            if j == 0: ax.set_ylabel("Mean per-trial NLL")
    handles=[plt.Line2D([],[],color=MODEL_BY_ID[m].color,ls=_line_style(m),label=_label(m)) for m in models]
    fig.legend(handles=handles,ncol=len(models),loc="upper center",bbox_to_anchor=(.5,.945),frameon=False,fontsize=7)
    fig.suptitle(f"{title} — lower is better — {_state(status)}",y=.995)
    fig.text(.5,.012,"Natural-log NLL; independent y-scales. Ribbons are descriptive SEM across correlated, overlapping windows.",ha="center",fontsize=8)
    _finish(fig,output,stem,rect=(0,.035,1,.90))


def figure16(e,s,o):
    focused=e[e.event_span.eq("focus")]
    _event_grid(focused,s,o,"fig16_event_aligned_nll","Figure 16 — Event-aligned NLL","mean_nll",["short_to_long","long_to_short","rare_short","rare_long"])


def figure17(events,status,output):
    data=events[events.event_span.eq("focus")].copy(); data["correctness_gap"]=data.model_probability_correct-data.mouse_correct
    _event_grid(data,status,output,"fig17_event_aligned_correctness_gap","Figure 17 — Event-aligned model-minus-mouse correctness","correctness_gap",["short_to_long","long_to_short","rare_short","rare_long"])


def figure18(psych,status,output):
    models=_eligible(status); blocks=[b for b in ("N","S","L") if b in set(psych.block)]; side=5.5; fig,axes=plt.subplots(1,len(blocks),figsize=(side*len(blocks),side),squeeze=False)
    for ax,block in zip(axes.flat,blocks):
        ax.set_box_aspect(1); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        p=psych[psych.block.eq(block)].copy(); mouse_source=p[p.model_id.eq(models[0])].copy(); mouse_source["mouse_weighted"]=mouse_source.mouse_long_rate*mouse_source.n_trials; mouse=mouse_source.groupby("duration_s").agg(weighted=("mouse_weighted","sum"),n=("n_trials","sum")); mouse["rate"]=mouse.weighted/mouse.n; ax.plot(mouse.index,mouse.rate,"ko-",lw=2,ms=3)
        for model in models:
            q=p[p.model_id.eq(model)].copy(); q["weighted"]=q.model_long_rate*q.n_trials; g=q.groupby("duration_s").agg(weighted=("weighted","sum"),n=("n_trials","sum")); ax.plot(g.index,g.weighted/g.n,color=MODEL_BY_ID[model].color,linestyle=_line_style(model),marker="o",ms=2.5,lw=1)
        ax.set_ylim(0,1); ax.set_title(f"True block {block}"); ax.set_xlabel("Duration (s)"); ax.set_ylabel("P(long)")
    handles=[plt.Line2D([],[],color="black",marker="o",ms=3,lw=2,label="Mouse")]+[plt.Line2D([],[],color=MODEL_BY_ID[m].color,linestyle=_line_style(m),marker="o",ms=2.5,lw=1,label=_label(m)) for m in models]
    fig.suptitle(f"Figure 18 — Held-out sliding-window psychometrics — {_state(status)}",y=0.98)
    fig.legend(handles=handles,ncol=min(len(handles),4),loc="upper center",bbox_to_anchor=(.5,0.92),frameon=False,fontsize=7)
    _finish(fig,output,"fig18_heldout_psychometrics",rect=(0,0,1,0.72))


def _performance_grid(events,status,output,stem,title,event_names,*,auto_ylim=False,show_grid=True,show_markers=False):
    models=_eligible(status); combinations=[(e,t) for e in event_names for t in ("short","long")]; fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,(event,trial_type) in zip(axes.flat,combinations):
        part=events[events.event.eq(event)&events.trial_type.eq(trial_type)&events.split_role.eq("test")]
        mouse_marker="o" if show_markers else None
        mouse_source=part[part.model_id.eq(models[0])]; mouse=mouse_source.groupby("relative_position").mouse_correct.agg(["mean","sem"]); ax.plot(mouse.index,mouse["mean"],"k-",marker=mouse_marker,ms=3,lw=2,label="Mouse"); ax.fill_between(mouse.index,mouse["mean"]-mouse["sem"],mouse["mean"]+mouse["sem"],color="black",alpha=.10)
        for model in models:
            p=part[part.model_id.eq(model)].groupby("relative_position").model_probability_correct.agg(["mean","sem"]); ax.plot(p.index,p["mean"],color=MODEL_BY_ID[model].color,ls=_line_style(model),marker="o" if show_markers else None,ms=2.5,lw=1.1); ax.fill_between(p.index,p["mean"]-p["sem"],p["mean"]+p["sem"],color=MODEL_BY_ID[model].color,alpha=.04)
        support=mouse_source.groupby("relative_position").agg(n_windows=("window_id","nunique"),n_events=("n_events","sum"))
        window_range=f"{int(support.n_windows.min())}–{int(support.n_windows.max())}"; event_range=f"{int(support.n_events.min())}–{int(support.n_events.max())}"
        pretty={"short_to_long":"short → long","long_to_short":"long → short","rare_short":"rare short","rare_long":"rare long"}[event]
        ax.axvline(0,color="black",ls="--",lw=.7)
        if not auto_ylim: ax.set_ylim(0,1)
        ax.set_title(f"{pretty}; {trial_type} trials\nsupport by offset: {window_range} windows, {event_range} window-event instances"); ax.set_ylabel("P(correct)"); ax.set_xlabel("Trials relative to event (0 = event trial)")
        if show_grid: ax.grid(alpha=.2)
    legend_marker="o" if show_markers else None
    handles=[plt.Line2D([],[],color="black",marker="o",label="Mouse")]+[plt.Line2D([],[],color=MODEL_BY_ID[m].color,ls=_line_style(m),marker=legend_marker,ms=2.5,label=_label(m)) for m in models]
    fig.legend(handles=handles,ncol=5,loc="upper center",bbox_to_anchor=(.5,.94),frameon=False,fontsize=7)
    fig.suptitle(f"{title} — held-out test sessions — {_state(status)}",y=.995)
    fig.text(.5,.012,"Lines are window-level means ± descriptive SEM; overlapping windows are correlated, not independent replicates.",ha="center",fontsize=8)
    _finish(fig,output,stem,rect=(0,.035,1,.89))


def figure19(e,s,o): _performance_grid(e[e.event_span.eq("wide")],s,o,"fig19_block_transition_performance_by_trial_type","Figure 19 — Block-transition performance",["short_to_long","long_to_short"],auto_ylim=True,show_grid=False,show_markers=True)


# Shading for Figure 20's ground-truth block background: 'N' is the
# pre-first-block neutral warm-up, 'S'/'L' the true short/long blocks.
_TRUE_BLOCK_SHADE = {"S": "#4E79A7", "L": "#E15759", "N": "0.6"}
_TRUE_BLOCK_LABEL = {"S": "True short block", "L": "True long block", "N": "Pre-block / neutral"}


def _shade_true_blocks(ax, trial_index: pd.Series, true_block: pd.Series) -> None:
    """Shade contiguous ground-truth block spans behind the trajectories."""
    order = np.argsort(trial_index.to_numpy())
    index = trial_index.to_numpy()[order]
    block = true_block.to_numpy()[order]
    start = 0
    for position in range(1, len(index) + 1):
        if position == len(index) or block[position] != block[start]:
            label = block[start]
            ax.axvspan(
                index[start] - 0.5, index[position - 1] + 0.5,
                color=_TRUE_BLOCK_SHADE[label], alpha=0.16 if label != "N" else 0.25,
                zorder=0, linewidth=0,
            )
            start = position


def _first_n_block_window(truth: pd.DataFrame, n_blocks: int) -> tuple[float, float]:
    """Trial-index bounds spanning the session's first ``n_blocks`` true
    S/L blocks (the pre-first-block 'N' warm-up doesn't count as a block)."""
    ordered = truth.sort_values("trial_index")
    trial_index = ordered.trial_index.to_numpy()
    block = ordered.true_block.to_numpy()
    spans = []
    start = 0
    for position in range(1, len(block) + 1):
        if position == len(block) or block[position] != block[start]:
            spans.append((block[start], trial_index[start], trial_index[position - 1]))
            start = position
    real_blocks = [span for span in spans if span[0] != "N"][:n_blocks]
    if not real_blocks:
        return float(trial_index.min()), float(trial_index.max())
    return real_blocks[0][1] - 0.5, real_blocks[-1][2] + 0.5


def _finish_pages(pages: list[plt.Figure], output: Path, stem: str) -> None:
    """Write page 1 as the PNG and every page as a multi-page PDF -- the
    same convention Figure 0 uses for its ranking/CI/TOST pages."""
    png_path, pdf_path = output / f"{stem}.png", output / f"{stem}.pdf"
    pending_png, pending_pdf = output / f".{stem}.pending.png", output / f".{stem}.pending.pdf"
    try:
        pages[0].savefig(pending_png, dpi=300, bbox_inches="tight")
        with PdfPages(pending_pdf) as pdf:
            for page in pages:
                pdf.savefig(page, bbox_inches="tight")
        pending_png.replace(png_path)
        pending_pdf.replace(pdf_path)
    finally:
        for page in pages:
            plt.close(page)
        pending_png.unlink(missing_ok=True)
        pending_pdf.unlink(missing_ok=True)



_FIGURE20_COLUMNS = (
    ("sensory_logit", "Sensory logit  f0(x_t)"),
    ("belief_logit", "Belief logit  logit(π_t)"),
    ("summation_logit", "Summation (approx.)  f0(x_t) + logit(π_t)"),
)
_FIGURE20_LOGIT_CHANNELS = (
    ("mean_sensory_logit", "Sensory  f0(x_t)"),
    ("mean_belief_logit", "Belief  logit(π_t)"),
    ("mean_summation_logit", "Summation (approx.)"),
)
_FIGURE20_PROBABILITY_CHANNELS = (
    ("sensory_probability", "Sensory  P(long | sensory only)"),
    ("belief_probability", "Belief  π(t)"),
    ("summation_probability", "Summation  P(long)"),
)
# Sensory is the model's own Gaussian decision-region integral with neutral
# prior/kappa (see extended_collect._sensory_only_probability), not a
# sigmoid; belief is model_prior_long as-is; summation is
# model_probability_long as-is -- the real full probability, unlike the
# logit-sum approximation pages 3-4 use.
_FIGURE20_BLOCK_FILTERS = (("All blocks", None), ("Short block", "S"), ("Long block", "L"))
_FIGURE20_EVENT_PRETTY = {
    "short_to_long": "short → long", "long_to_short": "long → short",
    "rare_short": "rare short", "rare_long": "rare long",
}
# Row 0 pools every window's train+validation+test rows together ("all
# sessions"); rows 1-3 filter to one split role each, same as _event_grid.
_FIGURE20_ROW_SPLITS = (("all", None),) + tuple((split, split) for split in SPLIT_ROLES)


def _figure20_event_page(events, models, legend_handles, event_names, event_span, page_title):
    """4 rows (all/train/validation/test) x 6 columns (2 events x
    sensory/belief/summation), each model's window-mean-of-means +- SEM
    across windows, same aggregation _event_grid uses for NLL/correctness.
    """
    columns = [
        (event, channel_col, f"{_FIGURE20_EVENT_PRETTY[event]}\n{channel_title}")
        for event in event_names for channel_col, channel_title in _FIGURE20_LOGIT_CHANNELS
    ]
    scoped = events[events.event_span.eq(event_span)]
    fig, axes = plt.subplots(4, 6, figsize=(33, 15), squeeze=False, sharex="col")
    for r, (row_label, split_filter) in enumerate(_FIGURE20_ROW_SPLITS):
        row_data = scoped if split_filter is None else scoped[scoped.split_role.eq(split_filter)]
        for c, (event, channel_col, column_title) in enumerate(columns):
            ax = axes[r, c]
            part = row_data[row_data.event.eq(event)]
            non_wsls_bounds = []
            for model in models:
                p = part[part.model_id.eq(model)].groupby("relative_position")[channel_col].agg(["mean", "sem"])
                if not len(p):
                    continue
                sem = p["sem"].fillna(0)
                ax.plot(p.index, p["mean"], color=MODEL_BY_ID[model].color, linestyle=_line_style(model),
                        lw=1.2, zorder=3)
                ax.fill_between(p.index, p["mean"] - sem, p["mean"] + sem,
                                color=MODEL_BY_ID[model].color, alpha=0.05, zorder=2)
                if MODEL_BY_ID[model].group != "wsls":
                    lower, upper = (p["mean"] - sem).min(), (p["mean"] + sem).max()
                    if pd.notna(lower) and pd.notna(upper):
                        non_wsls_bounds.append((lower, upper))
            ax.axvline(0, color="black", ls="--", lw=0.7)
            if channel_col != "mean_sensory_logit" and non_wsls_bounds:
                # Same rationale as pages 1-2: WSLS's near-deterministic
                # belief swings the visible range far past every other
                # model's -- scale to the non-WSLS models instead.
                lo, hi = min(b[0] for b in non_wsls_bounds), max(b[1] for b in non_wsls_bounds)
                pad = 0.1 * max(hi - lo, 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            if r == 0:
                ax.set_title(column_title, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(row_label.capitalize(), fontsize=8)
            if r == 3:
                ax.set_xlabel("Trials relative to event (0 = event trial)", fontsize=7)
    fig.suptitle(page_title, y=0.99, fontsize=11)
    fig.legend(handles=legend_handles, ncol=min(len(legend_handles), 8), loc="upper center",
              bbox_to_anchor=(.5, 0.965), frameon=False, fontsize=7)
    fig.text(.5, .005,
             "Lines are per-window means ± descriptive SEM across windows; overlapping windows are correlated, "
             "not independent replicates. Row 0 pools train+validation+test windows together.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .02, 1, .93))
    return fig


def _figure20_psychometric_page(psych, models, legend_handles, page_title):
    """4 rows (all/train/validation/test) x 9 columns (all/short/long block
    x sensory/belief/summation), each model's n-trial-weighted mean
    probability by presented duration, pooled across windows (same
    weighting figure18 uses)."""
    columns = [
        (block_filter, channel_col, f"{block_title}\n{channel_title}")
        for block_title, block_filter in _FIGURE20_BLOCK_FILTERS
        for channel_col, channel_title in _FIGURE20_PROBABILITY_CHANNELS
    ]
    side = 3.0
    fig, axes = plt.subplots(
        4, len(columns), figsize=(side * len(columns), side * 4), squeeze=False, sharex="col",
    )
    for r, (row_label, split_filter) in enumerate(_FIGURE20_ROW_SPLITS):
        row_data = psych if split_filter is None else psych[psych.split_role.eq(split_filter)]
        for c, (block_filter, channel_col, column_title) in enumerate(columns):
            ax = axes[r, c]
            ax.set_box_aspect(1)
            block_data = row_data if block_filter is None else row_data[row_data.true_block.eq(block_filter)]
            for model in models:
                q = block_data[block_data.model_id.eq(model)].dropna(subset=[channel_col])
                if q.empty:
                    continue
                weighted = q[channel_col] * q.n_trials
                g = pd.DataFrame({"weighted": weighted, "n": q.n_trials, "duration_s": q.duration_s}).groupby(
                    "duration_s"
                ).sum()
                ax.plot(g.index, g.weighted / g.n, color=MODEL_BY_ID[model].color, linestyle=_line_style(model),
                        lw=1.2, marker="o", ms=2.5, zorder=3)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(column_title, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{row_label.capitalize()}\nP", fontsize=8)
            if r == 3:
                ax.set_xlabel("Duration (s)", fontsize=7)
    fig.suptitle(page_title, y=0.99, fontsize=11)
    fig.legend(handles=legend_handles, ncol=min(len(legend_handles), 8), loc="upper center",
              bbox_to_anchor=(.5, 0.965), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def figure20(logit_samples, logit_events, logit_psychometrics, status, output):
    """5 pages of sensory/belief/summation decision-logit views (see
    _per_trial_logits for the sensory/belief split and its accuracy
    caveat), all superimposing every eligible model:

    Page 1: 3 columns x 15 rows -- sensory/belief/summation logit
    trajectories for 5 randomly drawn windows x one randomly drawn
    train/validation/test session each, true short/long block boundaries
    shaded behind each trajectory.
    Page 2: same as page 1, zoomed to each row's first four true blocks.
    Page 3: 6 columns (short->long, long->short x sensory/belief/summation)
    x 4 rows (all/train/validation/test sessions pooled across every
    window) -- mean +- SEM aligned -15..+20 trials around block transitions.
    Page 4: same layout as page 3, aligned -2..+2 trials around rare trials
    (rare_short, rare_long) instead of block transitions.
    Page 5: 9 columns (all/short/long block x sensory/belief/summation) x 4
    rows (all/train/validation/test) -- psychometric curves (P(long) by
    duration). No sigmoid: sensory is the model's own Gaussian
    decision-region integral with a neutral prior/kappa, belief is
    model_prior_long as-is, summation is model_probability_long as-is.
    """
    models = [
        m for m in _eligible(status)
        if m in set(logit_samples.model_id) | set(logit_events.model_id) | set(logit_psychometrics.model_id)
    ]
    model_legend_handles = [
        plt.Line2D([], [], color=MODEL_BY_ID[m].color, linestyle=_line_style(m), label=_label(m)) for m in models
    ]
    legend_handles = model_legend_handles + [
        plt.Rectangle((0, 0), 1, 1, color=_TRUE_BLOCK_SHADE[label], alpha=0.35, label=_TRUE_BLOCK_LABEL[label])
        for label in ("S", "L", "N")
    ]
    window_ids = sorted(logit_samples.window_id.unique().tolist())
    rows = [(window_id, split) for window_id in window_ids for split in SPLIT_ROLES]

    def render_page(*, zoomed: bool, page_title: str) -> plt.Figure:
        fig, axes = plt.subplots(len(rows), 3, figsize=(30, 2.6 * len(rows)), squeeze=False)
        for r, (window_id, split) in enumerate(rows):
            row_data = logit_samples[logit_samples.window_id.eq(window_id) & logit_samples.split_role.eq(split)]
            if row_data.empty:
                for c in range(3):
                    axes[r, c].axis("off")
                axes[r, 0].text(0.5, 0.5, "No sampled data", ha="center", va="center", transform=axes[r, 0].transAxes)
                continue
            session_id = str(row_data.session_id.iloc[0])
            short_session_id = "_".join(session_id.split("_")[-2:])
            truth = row_data.drop_duplicates("trial_index").sort_values("trial_index")[["trial_index", "true_block"]]
            xlim = _first_n_block_window(truth, 4) if zoomed else (
                truth.trial_index.min() - 0.5, truth.trial_index.max() + 0.5,
            )
            row_label = f"Win {window_id}\n{split.capitalize()}\n{short_session_id}"
            for c, (column, column_title) in enumerate(_FIGURE20_COLUMNS):
                ax = axes[r, c]
                _shade_true_blocks(ax, truth.trial_index, truth.true_block)
                for model in models:
                    p = row_data[row_data.model_id.eq(model)].sort_values("trial_index")
                    if p.empty or p[column].isna().all():
                        continue
                    ax.plot(
                        p.trial_index, p[column], color=MODEL_BY_ID[model].color,
                        linestyle=_line_style(model), lw=1.1, zorder=3,
                    )
                ax.set_xlim(*xlim)
                if column != "sensory_logit":
                    # WSLS's near-deterministic P(long) sends logit(pi_t) to +-20-30,
                    # dwarfing the O(1-10) belief swings of every other model -- scale
                    # the visible range to the non-WSLS models so those stay legible;
                    # WSLS is still drawn, just clipped where it exceeds this range.
                    non_wsls = row_data.loc[
                        row_data.model_id.map(lambda m: MODEL_BY_ID[m].group != "wsls"), column,
                    ].dropna()
                    if len(non_wsls):
                        lo, hi = float(non_wsls.min()), float(non_wsls.max())
                        pad = 0.1 * max(hi - lo, 1e-6)
                        ax.set_ylim(lo - pad, hi + pad)
                if r == 0:
                    ax.set_title(column_title, fontsize=9)
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=7)
                if r == len(rows) - 1:
                    ax.set_xlabel("Trial index within session")
        fig.suptitle(page_title, y=0.997, fontsize=11)
        fig.legend(
            handles=legend_handles, ncol=min(len(legend_handles), 8), loc="upper center",
            bbox_to_anchor=(.5, 0.988), frameon=False, fontsize=7,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    page1 = render_page(
        zoomed=False,
        page_title=f"Figure 20 — Sensory/belief/summation decision-logit trajectories, 5 random windows "
                   f"x train/validation/test — {_state(status)}",
    )
    page2 = render_page(
        zoomed=True,
        page_title=f"Figure 20 (page 2) — Same windows/sessions/models, zoomed to the first four true "
                   f"blocks — {_state(status)}",
    )
    page3 = _figure20_event_page(
        logit_events, models, model_legend_handles, ["short_to_long", "long_to_short"], "wide",
        f"Figure 20 (page 3) — Block-transition-aligned decision logits, -15..+20 trials — {_state(status)}",
    )
    page4 = _figure20_event_page(
        logit_events, models, model_legend_handles, ["rare_short", "rare_long"], "focus",
        f"Figure 20 (page 4) — Rare-trial-aligned decision logits, -2..+2 trials — {_state(status)}",
    )
    page5 = _figure20_psychometric_page(
        logit_psychometrics, models, model_legend_handles,
        f"Figure 20 (page 5) — Sensory/belief/summation psychometrics by duration — {_state(status)}",
    )
    _finish_pages([page1, page2, page3, page4, page5], output, "fig20_prior_trajectory_samples")


def figure21(parameters,status,output):
    models=_eligible(status); names=parameters.param_name.drop_duplicates().tolist(); cols=3; rows=int(np.ceil(len(names)/cols)); fig,axes=plt.subplots(rows,cols,figsize=(16,3.3*rows),squeeze=False)
    for ax,name in zip(axes.flat,names):
        p=parameters[parameters.param_name.eq(name)]; applicable=[m for m in models if m in set(p.model_id)]
        for x,m in enumerate(applicable):
            v=p[p.model_id.eq(m)].param_value; ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt="o",color=MODEL_BY_ID[m].color,capsize=3)
        ax.set_xticks(range(len(applicable)),[_label(m) for m in applicable],rotation=40,ha="right",fontsize=7); ax.set_title(name); ax.grid(axis="y",alpha=.2)
    for ax in axes.flat[len(names):]: ax.axis("off")
    fig.suptitle(f"Figure 21 — Parameter mean ± SD across windows — {_state(status)}")
    _finish(fig,output,"fig21_parameter_mean_sd")


def figure22(parameters,status,output):
    models=[m for m in _eligible(status) if "ah" in MODEL_BY_ID[m].parameter_names]; r=np.arange(1,66); fig,axes=plt.subplots(len(models),1,figsize=(11,max(4,2.3*len(models))),squeeze=False,sharex=True)
    for ax,m in zip(axes.flat,models):
        p=parameters[parameters.model_id.eq(m)].pivot(index="window_id",columns="param_name",values="param_value")
        for k,(_,row) in enumerate(p.iterrows()):
            bh=row.get("bh",0.0); hazard=1/(1+np.exp(-(row["ah"]+bh*(r-23)))); ax.plot(r,hazard,color=MODEL_BY_ID[m].color,alpha=.15+.75*(k+1)/len(p),lw=.8)
        ax.set_ylabel(_label(m)); ax.set_ylim(0,1); ax.grid(alpha=.15)
    axes[-1,0].set_xlabel("Run length (trials)"); fig.suptitle(f"Figure 22 — Fitted switch-hazard curves across windows — {_state(status)}")
    _finish(fig,output,"fig22_dynamic_hazard_curves")


def figure23(e,s,o): _performance_grid(e[e.event_span.eq("focus")],s,o,"fig23_rare_trial_performance_by_type","Figure 23 — Rare-trial performance",["rare_short","rare_long"],auto_ylim=True,show_grid=False,show_markers=True)


_BIN_LABELS = tuple(meta["label"] for meta in DURATION_BIN_DEFINITIONS)
_DIFFICULTY_CONDITIONS = (
    ("hard_own", "Hard trial\n(near boundary)"), ("easy_own", "Easy trial\n(far from boundary)"),
    ("hard_next", "Trial after\na hard trial"), ("easy_next", "Trial after\nan easy trial"),
)
_CONDITION_COLORS = {"hard_own": "#E15759", "easy_own": "#4E79A7", "hard_next": "#F1CE63", "easy_next": "#A0CBE8"}


def _sensory_family(status: pd.DataFrame) -> tuple[str, ...]:
    """Eligible models with a Bayesian sensory/prior decomposition at all
    -- excludes Pure/Extended WSLS, which have no sensory readout."""
    return tuple(m for m in _eligible(status) if "w" in MODEL_BY_ID[m].parameter_names)


def _figure24_page1(bins_table: pd.DataFrame, models: tuple[str, ...], state_text: str) -> plt.Figure:
    """One row per model, one column per ISI bin: color = the share of mean
    decision strength (|prior_logit| vs |sensory_logit|) coming from the
    prior, 0=all sensory, 1=all prior, 0.5=even split."""
    matrix = np.full((len(models), len(_BIN_LABELS)), np.nan)
    prior_mag = np.full_like(matrix, np.nan)
    sensory_mag = np.full_like(matrix, np.nan)
    for row, model in enumerate(models):
        part = bins_table[bins_table.model_id.eq(model)]
        for col, label in enumerate(_BIN_LABELS):
            cell = part[part.bin_label.eq(label)]
            if cell.empty:
                continue
            prior_mean = cell.mean_abs_prior_logit.mean()
            sensory_mean = cell.mean_abs_sensory_logit.mean()
            prior_mag[row, col] = prior_mean
            sensory_mag[row, col] = sensory_mean
            total = prior_mean + sensory_mean
            matrix[row, col] = prior_mean / total if total > 0 else np.nan

    fig, ax = plt.subplots(figsize=(13.5, 1.0 + 0.55 * len(models)))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(_BIN_LABELS)), _BIN_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), [_label(m) for m in models])
    for row in range(len(models)):
        for col in range(len(_BIN_LABELS)):
            if np.isfinite(matrix[row, col]):
                ax.text(
                    col, row, f"{matrix[row, col]:.2f}\n|π|={prior_mag[row, col]:.2f}\n|sen|={sensory_mag[row, col]:.2f}",
                    ha="center", va="center", fontsize=6.2,
                )
    ax.axvline(3.5, color="black", lw=1.2)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="Share of decision strength from prior (πₜ)")
    ax.set_title(
        f"Figure 24 (page 1) — Prior vs. sensory contribution to the pre-lapse decision, by ISI bin — {state_text}\n"
        "Λ(x) = sensory_logit(x) + logit(πₜ) − log(κ); cell = mean|logit(πₜ)| / (mean|logit(πₜ)| + mean|sensory_logit|) across windows; "
        "vertical line separates short (left) from long (right) categories",
        fontsize=9.5,
    )
    fig.tight_layout()
    return fig


def _figure24_page2(difficulty_table: pd.DataFrame, models: tuple[str, ...], state_text: str) -> plt.Figure:
    """Per model: mean |prior_logit| (± SEM across windows) on hard vs.
    easy trials, and on the trial immediately following each."""
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(models) + 2), 6.5))
    group_width = 0.8
    bar_width = group_width / len(_DIFFICULTY_CONDITIONS)
    x = np.arange(len(models))
    for index, (condition, condition_label) in enumerate(_DIFFICULTY_CONDITIONS):
        part = difficulty_table[difficulty_table.condition.eq(condition)]
        means, sems = [], []
        for model in models:
            values = part.loc[part.model_id.eq(model), "mean_abs_prior_logit"].dropna()
            means.append(values.mean() if len(values) else np.nan)
            sems.append(values.sem() if len(values) > 1 else 0.0)
        offset = (index - (len(_DIFFICULTY_CONDITIONS) - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, width=bar_width * 0.92, yerr=sems, capsize=2.5,
            color=_CONDITION_COLORS[condition], label=condition_label,
        )
    ax.set_xticks(x, [_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("Mean |logit(πₜ)| across windows (± SEM)")
    ax.grid(axis="y", alpha=0.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_CONDITION_COLORS[condition], label=condition_label)
        for condition, condition_label in _DIFFICULTY_CONDITIONS
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, .93))
    fig.suptitle(
        f"Figure 24 (page 2) — Is the prior stronger for hard trials, and does it carry to the next trial? — {state_text}\n"
        "Hard = ISI closest to the 1.25s category boundary (0.80s / 1.70s); Easy = ISI farthest from it (0.20s / 2.30s); "
        "'next' = the trial immediately following, same session",
        y=.995, fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


def figure24(bins_table: pd.DataFrame, difficulty_table: pd.DataFrame, status: pd.DataFrame, output: Path) -> None:
    """Page 1: for each ISI bin, how much of the pre-lapse decision comes
    from the prior vs. the sensory evidence, per Bayesian-family model.
    Page 2: whether that prior is stronger on hard (near-boundary) trials
    than easy (far-from-boundary) trials, own-trial and on the very next
    trial. WSLS models have no sensory/prior decomposition and are excluded.
    """
    models = tuple(m for m in _sensory_family(status) if m in set(bins_table.model_id))
    state_text = _state(status)
    if not models:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No eligible Bayesian-sensory model available", ha="center", va="center")
        _finish(fig, output, "fig24_prior_vs_sensory_decision_strength")
        return
    page1 = _figure24_page1(bins_table, models, state_text)
    page2 = _figure24_page2(difficulty_table, models, state_text)
    _finish_pages([page1, page2], output, "fig24_prior_vs_sensory_decision_strength")


def run(tables: dict[str,pd.DataFrame], status: pd.DataFrame, output: Path) -> tuple[Path,...]:
    metrics=tables["sliding_window_metrics.csv"]; wins=tables["win_counts_per_window.csv"]; ic=tables["information_criteria.csv"]
    subsets=tables["extended_trial_subset_metrics.csv"]; events=tables["extended_event_aligned.csv"]; parameters=tables["extended_parameters.csv"]; psych=tables["extended_psychometrics.csv"]; logit_samples=tables["extended_prior_logit_samples.csv"]; logit_events=tables["extended_logit_event_aligned.csv"]; logit_psychometrics=tables["extended_logit_psychometrics.csv"]
    prior_sensory_bins=tables["extended_prior_sensory_bins.csv"]; prior_sensory_difficulty=tables["extended_prior_sensory_difficulty.csv"]
    renderers=(
        lambda:figure07_subsets(subsets,status,output), lambda:figure08_rank_stability(metrics,status,output),
        lambda:figure09(parameters,status,output), lambda:figure10(status,output), lambda:figure11(parameters,status,output),
        lambda:figure12(parameters,status,output), lambda:figure13(wins,status,output), lambda:figure14(wins,status,output),
        lambda:figure15(ic,status,output), lambda:figure16(events,status,output), lambda:figure17(events,status,output),
        lambda:figure18(psych,status,output), lambda:figure19(events,status,output), lambda:figure20(logit_samples,logit_events,logit_psychometrics,status,output),
        lambda:figure21(parameters,status,output), lambda:figure22(parameters,status,output), lambda:figure23(events,status,output),
        lambda:figure24(prior_sensory_bins,prior_sensory_difficulty,status,output),
    )
    if len(renderers) != len(FIGURE_IDS):
        raise RuntimeError("Extended figure registry must contain exactly Figures 7 through 24")
    for figure_id,render in zip(FIGURE_IDS,renderers,strict=True): render(); print(f"wrote {figure_id} to {output}")
    return tuple(sorted(output.glob("fig*.png")))+tuple(sorted(output.glob("fig*.pdf")))
