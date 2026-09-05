"""Session-to-session ("meta learning") analysis of block-transition plateaus.

For every short->long (S2L) and long->short (L2S) block transition, trial 0 is
the adjusted transition point used elsewhere in this codebase: the formal
block-type switch, walked backward over any terminal run of rare trials from
the preceding block, so trial 0 is never preceded by a rare trial
(see ``adjusted_block_start`` in block_sequence_epoch_analysis.py).

Performance after the transition is tracked separately for Long trials
(trial_type == 2) and Short trials (trial_type == 1), indexed by their own
occurrence count since the transition (the 1st Long trial after the
transition, the 2nd Long trial after the transition, ...) rather than by raw
trial position, since one trial type is common and the other rare within a
block.

Two plateau metrics are computed per transition x trial-type-class:
  - trial-to-criterion: first occurrence where a causal rolling-mean of
    ``correct`` (window = 2 or 3 trials) reaches a criterion (0.6-0.8).
  - exponential time constant tau, fit to the occurrence-indexed accuracy
    curve pooled across a session's transitions of that condition.

Both metrics are then examined across sessions (chronological
subject_session_idx) to test for meta-learning: does the animal reach
plateau faster in later sessions than in earlier ones?
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import linregress, mannwhitneyu, wilcoxon
from scipy.optimize import curve_fit


DATA_PATH = Path("all_subjects_final_data.csv")
OUTPUT_PATH = Path("meta_learning_transition_plateau.pdf")

CORRECT_OUTCOMES = {"Reward", "RewardNaive"}
INCORRECT_OUTCOMES = {"Punish", "PunishNaive"}
VALID_OUTCOMES = CORRECT_OUTCOMES | INCORRECT_OUTCOMES

TRIAL_RANGE = 20        # raw post-transition trials searched within the entered block
OCCURRENCE_RANGE = 10   # cap on per-trial-type occurrence index kept for analysis
CRITERION_LEVELS = [0.6, 0.7, 0.75, 0.8]
CRITERION_MAIN = 0.7
WINDOW_LEVELS = [2, 3]
WINDOW_MAIN = 3
SESSION_TERTILE_LABELS = ["Early", "Middle", "Late"]
MIN_TAU_R2 = 0.15  # per-session exponential fits below this R^2 are excluded from tau figures

CONDITIONS = [
    ("S2L", "Long trial"), ("S2L", "Short trial"),
    ("L2S", "Long trial"), ("L2S", "Short trial"),
]
CONDITION_LABELS = {
    ("S2L", "Long trial"): "Short→Long, Long trials",
    ("S2L", "Short trial"): "Short→Long, Short trials",
    ("L2S", "Long trial"): "Long→Short, Long trials",
    ("L2S", "Short trial"): "Long→Short, Short trials",
}
TRANS_COLORS = {"S2L": "#0072B2", "L2S": "#D55E00"}
SUBJECT_COLORS = {"YH24LG": "#0072B2", "MC06": "#E69F00", "MC07": "#009E73", "MC08": "#CC79A7"}
TERTILE_COLORS = {"Early": "#377eb8", "Middle": "#6a3d9a", "Late": "#984ea3"}

SUBJECT_ROWS = [
    ("All subjects", None),
    ("YH24LG", "YH24LG"),
    ("MC06", "MC06"),
    ("MC07", "MC07"),
    ("MC08", "MC08"),
]


# ===============================
# Data preparation
# ===============================

def prepare_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"subject", "subject_session_idx", "session_uid", "block_type", "trial_type", "outcome"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["_source_order"] = np.arange(len(df))
    for column in ["block_type", "trial_type", "subject_session_idx", "warm_up"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "warm_up" in df.columns:
        df = df[df["warm_up"].fillna(0).eq(0)].copy()
    df = df[df["outcome"].isin(VALID_OUTCOMES)].copy()
    df["correct"] = df["outcome"].isin(CORRECT_OUTCOMES).astype(float)
    df = df[df["block_type"].isin([1, 2])].copy()
    df["is_rare"] = (
        (df["block_type"].eq(1) & df["trial_type"].eq(2))
        | (df["block_type"].eq(2) & df["trial_type"].eq(1))
    )
    df["trial_type_class"] = np.where(df["trial_type"].eq(2), "Long trial", "Short trial")
    df["session_key"] = df["session_uid"].astype(str)
    return df.sort_values(["subject", "session_key", "_source_order"]).reset_index(drop=True)


# ===============================
# Adjusted block-transition alignment
# (matches adjusted_block_start in block_sequence_epoch_analysis.py)
# ===============================

def adjusted_block_start(sdf, formal_switch_idx):
    last_previous = formal_switch_idx - 1
    if last_previous < 0 or not bool(sdf.loc[last_previous, "is_rare"]):
        return formal_switch_idx
    previous_block = sdf.loc[last_previous, "block_type"]
    start = last_previous
    while start - 1 >= 0:
        previous_row = sdf.loc[start - 1]
        if previous_row["block_type"] != previous_block or not bool(previous_row["is_rare"]):
            break
        start -= 1
    return start


def _session_adjusted_starts(sdf):
    formal_switches = sdf.index[
        sdf["block_type"].ne(sdf["block_type"].shift())
        & sdf["block_type"].shift().notna()
    ].tolist()
    starts = [0] + [adjusted_block_start(sdf, idx) for idx in formal_switches]
    duplicate_starts = sorted({start for start in starts if starts.count(start) > 1})
    if duplicate_starts:
        session = (
            str(sdf["session_key"].iloc[0])
            if "session_key" in sdf.columns and not sdf.empty
            else "unknown"
        )
        raise ValueError(
            "Adjusted block boundaries collapse in session "
            f"{session} at trial position(s) {duplicate_starts}. "
            "An intervening block may contain only rare trials; refusing "
            "to delete that block or let the transition window cross it."
        )
    if any(stop <= start for start, stop in zip(starts[:-1], starts[1:])):
        raise ValueError(f"Adjusted block boundaries are not strictly increasing: {starts}")
    invalid_starts = [
        start for start in starts[1:]
        if start > 0 and bool(sdf.loc[start - 1, "is_rare"])
    ]
    if invalid_starts:
        raise AssertionError(f"Adjusted transition preceded by a rare trial: {invalid_starts}")
    return starts, formal_switches


def extract_transition_trials(data):
    """Build a trial-level table of post-transition trials, indexed by each
    trial's own occurrence count (1st, 2nd, ... of that trial type) since the
    adjusted transition point."""
    rows = []
    trans_id = 0
    for (subject, session_key), session_df in data.groupby(["subject", "session_key"], sort=False):
        sdf = session_df.sort_values("_source_order").reset_index(drop=True)
        if sdf.empty:
            continue
        subject_session_idx = int(sdf["subject_session_idx"].iloc[0])
        starts, formal_switches = _session_adjusted_starts(sdf)
        if len(starts) < 2:
            continue
        for transition_number, align_idx in enumerate(starts[1:], start=1):
            later_starts = [s for s in starts if s > align_idx]
            stop = min(later_starts) if later_starts else len(sdf)
            formal_switch_idx = next(
                idx for idx in formal_switches if adjusted_block_start(sdf, idx) == align_idx
            )
            from_block = int(sdf.loc[formal_switch_idx - 1, "block_type"])
            to_block = int(sdf.loc[formal_switch_idx, "block_type"])
            if (from_block, to_block) == (1, 2):
                trans_type = "S2L"
            elif (from_block, to_block) == (2, 1):
                trans_type = "L2S"
            else:
                continue
            window_stop = min(stop, align_idx + TRIAL_RANGE)
            trans_id += 1
            occurrence_counter = {"Long trial": 0, "Short trial": 0}
            for position in range(align_idx, window_stop):
                row = sdf.loc[position]
                cls = row["trial_type_class"]
                occurrence_counter[cls] += 1
                if occurrence_counter[cls] > OCCURRENCE_RANGE:
                    continue
                rows.append({
                    "subject": subject,
                    "session": session_key,
                    "subject_session_idx": subject_session_idx,
                    "trans_id": trans_id,
                    "trans_type": trans_type,
                    "block_id": f"{session_key}_T{transition_number}",
                    "rel_trial": position - align_idx,
                    "trial_type_class": cls,
                    "occurrence": occurrence_counter[cls],
                    "correct": float(row["correct"]),
                })
    return pd.DataFrame(rows)


# ===============================
# Plateau metric 1: trial (occurrence)-to-criterion
# ===============================

def add_smoothed_correct(events, windows=WINDOW_LEVELS):
    """Causal rolling mean of ``correct`` requiring a *full* window of the
    given size (min_periods=window), so criterion cannot be declared "met"
    from a single lucky trial padded by a partially filled window."""
    events = events.sort_values(["trans_id", "trial_type_class", "occurrence"]).copy()
    for window in windows:
        events[f"smooth_w{window}"] = (
            events.groupby(["trans_id", "trial_type_class"])["correct"]
            .transform(lambda s: s.rolling(window, min_periods=window).mean())
        )
    return events


def compute_trial_to_criterion(events, window, criterion):
    """One row per (transition, trial_type_class): first occurrence where the
    causal window-trial rolling mean reaches criterion, else NaN (censored)."""
    smooth_col = f"smooth_w{window}"
    group_cols = ["subject", "session", "subject_session_idx", "trans_id", "trans_type", "trial_type_class"]
    rows = []
    for keys, group in events.groupby(group_cols, sort=False):
        group = group.sort_values("occurrence")
        hit = group.loc[group[smooth_col] >= criterion]
        reached = not hit.empty
        ttc = float(hit["occurrence"].iloc[0]) if reached else np.nan
        row = dict(zip(group_cols, keys))
        row.update({
            "window": window, "criterion": criterion,
            "trial_to_criterion": ttc, "reached": reached,
            "n_occurrences": int(group["occurrence"].max()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_session_ttc(ttc_events):
    """One row per (subject, session, trans_type, trial_type_class, window,
    criterion): median trial-to-criterion across that session's transitions,
    plus the fraction of transitions that reached criterion at all."""
    group_cols = [
        "subject", "session", "subject_session_idx",
        "trans_type", "trial_type_class", "window", "criterion",
    ]
    summary = ttc_events.groupby(group_cols).agg(
        median_ttc=("trial_to_criterion", "median"),
        mean_ttc=("trial_to_criterion", "mean"),
        n_transitions=("trans_id", "nunique"),
        n_reached=("reached", "sum"),
    ).reset_index()
    summary["reached_fraction"] = summary["n_reached"] / summary["n_transitions"]
    return summary


# ===============================
# Plateau metric 2: saturating-exponential tau
# ===============================

def exp_func(x, a, tau, b):
    return a * (1 - np.exp(-x / tau)) + b


def fit_exponential_curve(sub_events):
    curve = (
        sub_events.groupby("occurrence")["correct"].mean().reset_index().sort_values("occurrence")
    )
    x = curve["occurrence"].to_numpy(dtype=float)
    y = curve["correct"].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return None
    try:
        p0 = [0.3, 3.0, float(np.clip(y[0], 0, 1))]
        bounds = ([0, 0.1, 0], [1, 50, 1])
        popt, _ = curve_fit(exp_func, x, y, p0=p0, bounds=bounds, maxfev=8000)
        a, tau, b = popt
        y_pred = exp_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"a": float(a), "tau": float(tau), "b": float(b), "r2": float(r2), "n_points": int(len(x))}
    except (RuntimeError, ValueError):
        return None


def compute_session_exponential(events):
    """One saturating-exponential fit per (subject, session, trans_type,
    trial_type_class), pooling that session's transitions of the condition."""
    group_cols = ["subject", "session", "subject_session_idx", "trans_type", "trial_type_class"]
    rows = []
    for keys, group in events.groupby(group_cols, sort=False):
        fit = fit_exponential_curve(group)
        row = dict(zip(group_cols, keys))
        if fit is None:
            row.update({"tau": np.nan, "a": np.nan, "b": np.nan, "r2": np.nan, "n_points": 0})
        else:
            row.update(fit)
        rows.append(row)
    return pd.DataFrame(rows)


# ===============================
# Session tertiles (early/middle/late training)
# ===============================

def add_session_tertile(summary):
    summary = summary.copy()
    summary["session_tertile"] = summary.groupby("subject")["subject_session_idx"].transform(
        lambda s: pd.qcut(s, 3, labels=SESSION_TERTILE_LABELS, duplicates="drop")
    )
    return summary


# ===============================
# Shared stats helpers
# ===============================

def _sig_label(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _bracket(ax, x1, x2, y, p, height=None):
    if height is None:
        height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="0.15", lw=0.9)
    ax.text((x1 + x2) / 2, y + height + height * 0.4, _sig_label(p), ha="center", fontsize=7)


def _style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===============================
# Figure 1/2/3: metric vs session (trend)
# ===============================

def plot_session_trend_panel(ax, summary, subject, trans_type, trial_type_class, value_col, ylabel):
    selected = summary[
        summary["trans_type"].eq(trans_type) & summary["trial_type_class"].eq(trial_type_class)
    ]
    if subject is not None:
        selected = selected[selected["subject"].eq(subject)]
    selected = selected.dropna(subset=[value_col, "subject_session_idx"])

    if subject is None:
        for subj, sdata in selected.groupby("subject"):
            ax.scatter(
                sdata["subject_session_idx"], sdata[value_col], s=14, alpha=0.55,
                color=SUBJECT_COLORS.get(subj, "0.5"), label=subj, linewidths=0,
            )
    else:
        ax.scatter(
            selected["subject_session_idx"], selected[value_col], s=18, alpha=0.6,
            color=TRANS_COLORS[trans_type], linewidths=0,
        )

    x = selected["subject_session_idx"].to_numpy(dtype=float)
    y = selected[value_col].to_numpy(dtype=float)
    n = len(x)
    if n >= 4 and np.unique(x).size >= 2:
        fit = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, fit.intercept + fit.slope * x_line, color="black", lw=1.6, zorder=5)
        stats_text = f"slope={fit.slope:+.3f}\np={fit.pvalue:.3g}, R²={fit.rvalue**2:.2f}, n={n}"
    else:
        stats_text = f"n={n} (insufficient for trend)"
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", fontsize=6.5)

    ax.set_xlabel("Session (chronological order)")
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    if subject is None:
        ax.legend(fontsize=6, frameon=False, loc="lower right")


def build_trend_page(summary, value_col, ylabel, title):
    fig = plt.figure(figsize=(30, 27))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.42)
    for row, (label, subject) in enumerate(SUBJECT_ROWS):
        for col, (trans_type, trial_type_class) in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[row, col])
            plot_session_trend_panel(ax, summary, subject, trans_type, trial_type_class, value_col, ylabel)
            ax.set_title(f"{label} — {CONDITION_LABELS[(trans_type, trial_type_class)]}", fontsize=9)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.965, bottom=0.045)
    return fig


# ===============================
# Figure 4/5: early/middle/late tertile summary
# ===============================

def _tertile_groups(summary, subject, trans_type, trial_type_class, value_col):
    selected = summary[
        summary["trans_type"].eq(trans_type) & summary["trial_type_class"].eq(trial_type_class)
    ]
    if subject is not None:
        selected = selected[selected["subject"].eq(subject)]
        return {
            tertile: selected.loc[selected["session_tertile"].eq(tertile), value_col].dropna()
            for tertile in SESSION_TERTILE_LABELS
        }, None
    subject_means = (
        selected.groupby(["subject", "session_tertile"], observed=True)[value_col]
        .mean()
        .reset_index()
    )
    groups = {
        tertile: subject_means.loc[subject_means["session_tertile"].eq(tertile), value_col].dropna()
        for tertile in SESSION_TERTILE_LABELS
    }
    paired = subject_means.pivot(index="subject", columns="session_tertile", values=value_col)
    return groups, paired


def _early_late_pvalue(groups, paired, subject):
    early, late = groups["Early"], groups["Late"]
    if subject is None:
        if paired is None or "Early" not in paired or "Late" not in paired:
            return np.nan
        pair = paired[["Early", "Late"]].dropna()
        if len(pair) < 2:
            return np.nan
        if np.allclose(pair["Early"] - pair["Late"], 0):
            return 1.0
        try:
            return float(wilcoxon(pair["Early"], pair["Late"]).pvalue)
        except ValueError:
            return np.nan
    if len(early) < 2 or len(late) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(early, late, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def plot_tertile_panel(ax, summary, subject, trans_type, trial_type_class, value_col, ylabel):
    groups, paired = _tertile_groups(summary, subject, trans_type, trial_type_class, value_col)
    means, sems, ns = [], [], []
    for tertile in SESSION_TERTILE_LABELS:
        values = groups[tertile]
        means.append(values.mean() if len(values) else np.nan)
        sems.append(values.sem() if len(values) > 1 else 0.0)
        ns.append(len(values))
    x = np.arange(3)
    ax.errorbar(
        x, means, yerr=sems, fmt="o-", color=TRANS_COLORS[trans_type],
        markersize=7, capsize=4, elinewidth=1.4, zorder=3,
    )
    for xi, tertile in zip(x, SESSION_TERTILE_LABELS):
        ax.scatter(
            [xi] * len(groups[tertile]) + np.random.default_rng(0).uniform(-0.06, 0.06, len(groups[tertile])),
            groups[tertile], s=10, alpha=0.3, color=TERTILE_COLORS[tertile], zorder=1,
        )
    finite_means = [m for m in means if np.isfinite(m)]
    if finite_means:
        top = max(m + s for m, s in zip(means, sems) if np.isfinite(m))
        p = _early_late_pvalue(groups, paired, subject)
        _bracket(ax, x[0], x[2], top * 1.06 if top > 0 else 0.05, p)
    ax.set_xticks(x, [f"{t}\n(n={n})" for t, n in zip(SESSION_TERTILE_LABELS, ns)])
    ax.set_ylabel(ylabel)
    _style_axis(ax)


def build_tertile_page(summary, value_col, ylabel, title):
    fig = plt.figure(figsize=(24, 27))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.45)
    for row, (label, subject) in enumerate(SUBJECT_ROWS):
        for col, (trans_type, trial_type_class) in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[row, col])
            plot_tertile_panel(ax, summary, subject, trans_type, trial_type_class, value_col, ylabel)
            ax.set_title(f"{label} — {CONDITION_LABELS[(trans_type, trial_type_class)]}", fontsize=9)
    fig.suptitle(f"{title}\nBracket: Early vs Late (paired Wilcoxon across subjects / Mann-Whitney within subject)",
                 fontsize=15, y=0.997)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.045)
    return fig


# ===============================
# Figure 6: window x criterion robustness (pooled)
# ===============================

def build_robustness_page(ttc_all, title):
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)
    for row, window in enumerate(WINDOW_LEVELS):
        for col, (trans_type, trial_type_class) in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[row, col])
            selected = ttc_all[
                ttc_all["window"].eq(window)
                & ttc_all["trans_type"].eq(trans_type)
                & ttc_all["trial_type_class"].eq(trial_type_class)
            ]
            means, sems, reached_fracs = [], [], []
            for criterion in CRITERION_LEVELS:
                crit_rows = selected[selected["criterion"].eq(criterion)]
                values = crit_rows["trial_to_criterion"].dropna()
                means.append(values.mean() if len(values) else np.nan)
                sems.append(values.sem() if len(values) > 1 else 0.0)
                reached_fracs.append(crit_rows["reached"].mean() if len(crit_rows) else np.nan)
            x = np.arange(len(CRITERION_LEVELS))
            ax.bar(x, means, color=TRANS_COLORS[trans_type], alpha=0.75, width=0.6)
            ax.errorbar(x, means, yerr=sems, fmt="none", color="0.2", capsize=4, lw=1.3)
            for xi, frac in zip(x, reached_fracs):
                ax.text(xi, 0.02, f"{frac:.0%}\nreached", ha="center", va="bottom",
                        fontsize=6.5, transform=ax.get_xaxis_transform())
            ax.set_xticks(x, [f"{c:.2f}" for c in CRITERION_LEVELS])
            ax.set_xlabel("Criterion threshold")
            ax.set_ylabel("Mean occurrences to criterion")
            ax.set_title(
                f"window={window} trials — {CONDITION_LABELS[(trans_type, trial_type_class)]}", fontsize=8.5
            )
            _style_axis(ax)
    fig.suptitle(title, fontsize=15, y=1.01)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.90, bottom=0.09)
    return fig


# ===============================
# Figure 7: occurrence-indexed accuracy curves, Early vs Late tertile
# ===============================

def build_curve_page(events_with_tertile, title):
    fig = plt.figure(figsize=(24, 27))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.42)
    for row, (label, subject) in enumerate(SUBJECT_ROWS):
        for col, (trans_type, trial_type_class) in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[row, col])
            selected = events_with_tertile[
                events_with_tertile["trans_type"].eq(trans_type)
                & events_with_tertile["trial_type_class"].eq(trial_type_class)
            ]
            if subject is not None:
                selected = selected[selected["subject"].eq(subject)]
            for tertile in ["Early", "Late"]:
                curve = (
                    selected[selected["session_tertile"].eq(tertile)]
                    .groupby("occurrence")["correct"]
                    .agg(["mean", "sem", "count"])
                    .reset_index()
                )
                if curve.empty:
                    continue
                ax.errorbar(
                    curve["occurrence"], curve["mean"], yerr=curve["sem"].fillna(0),
                    marker="o", markersize=4, capsize=2, linewidth=1.6,
                    color=TERTILE_COLORS[tertile], alpha=0.85, label=tertile,
                )
            ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(f"Occurrence of this trial type since transition")
            ax.set_ylabel("P(correct)")
            ax.set_title(f"{label} — {CONDITION_LABELS[(trans_type, trial_type_class)]}", fontsize=9)
            _style_axis(ax)
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.965, bottom=0.045)
    return fig


# ===============================
# Orchestration
# ===============================

def run(data_path=DATA_PATH, output_path=OUTPUT_PATH, show=True):
    data = prepare_data(data_path)
    events = extract_transition_trials(data)
    if events.empty:
        raise ValueError("No qualifying block transitions were extracted.")
    events = add_smoothed_correct(events, windows=WINDOW_LEVELS)

    ttc_all = pd.concat(
        [
            compute_trial_to_criterion(events, window=window, criterion=criterion)
            for window in WINDOW_LEVELS
            for criterion in CRITERION_LEVELS
        ],
        ignore_index=True,
    )
    session_ttc = summarize_session_ttc(ttc_all)
    session_ttc = add_session_tertile(session_ttc)

    session_tau = compute_session_exponential(events)
    session_tau = add_session_tertile(session_tau)
    session_tau_reliable = session_tau[session_tau["r2"].ge(MIN_TAU_R2)].copy()
    excluded_fraction = 1 - len(session_tau_reliable) / session_tau["tau"].notna().sum()
    print(
        f"Exponential tau: excluded {excluded_fraction:.0%} of session-condition fits "
        f"with R^2 < {MIN_TAU_R2} (poor/degenerate fits, common with sparse per-session data)."
    )

    events_tertile = add_session_tertile(
        events.assign(subject_session_idx=events["subject_session_idx"])
    )

    main_window_ttc = session_ttc[session_ttc["window"].eq(WINDOW_MAIN) & session_ttc["criterion"].eq(CRITERION_MAIN)]
    secondary_window_ttc = session_ttc[
        session_ttc["window"].eq(2 if WINDOW_MAIN != 2 else 3) & session_ttc["criterion"].eq(CRITERION_MAIN)
    ]

    pages = [
        build_trend_page(
            main_window_ttc, "median_ttc",
            f"Occurrences to criterion (P≥{CRITERION_MAIN}, window={WINDOW_MAIN})",
            f"Trial-to-criterion vs. session — window={WINDOW_MAIN} trials, criterion={CRITERION_MAIN}",
        ),
        build_trend_page(
            secondary_window_ttc, "median_ttc",
            f"Occurrences to criterion (P≥{CRITERION_MAIN}, window=2)",
            f"Trial-to-criterion vs. session — window=2 trials, criterion={CRITERION_MAIN} (robustness)",
        ),
        build_trend_page(
            session_tau_reliable, "tau",
            "Exponential time constant τ (occurrences)",
            f"Exponential adaptation time constant τ vs. session (fits with R²≥{MIN_TAU_R2} only)",
        ),
        build_tertile_page(
            main_window_ttc, "median_ttc",
            "Occurrences to criterion",
            f"Early / Middle / Late training-session tertiles — trial-to-criterion "
            f"(window={WINDOW_MAIN}, criterion={CRITERION_MAIN})",
        ),
        build_tertile_page(
            session_tau_reliable, "tau",
            "Exponential τ (occurrences)",
            f"Early / Middle / Late training-session tertiles — exponential τ (fits with R²≥{MIN_TAU_R2} only)",
        ),
        build_robustness_page(
            ttc_all,
            "Trial-to-criterion robustness: rolling window (2 vs 3 trials) × criterion threshold, pooled across sessions",
        ),
        build_curve_page(
            events_tertile,
            "Occurrence-indexed accuracy curves: Early vs Late training-session tertile",
        ),
    ]

    output_path = Path(output_path)
    with PdfPages(output_path) as pdf:
        for page in pages:
            pdf.savefig(page, bbox_inches="tight")
    if show:
        plt.show()
    else:
        for page in pages:
            plt.close(page)

    return {
        "output_path": output_path,
        "data": data,
        "events": events,
        "ttc_all": ttc_all,
        "session_ttc": session_ttc,
        "session_tau": session_tau,
        "session_tau_reliable": session_tau_reliable,
    }


if __name__ == "__main__":
    result = run(show=False)
    print(f"Saved: {result['output_path'].resolve()}")
