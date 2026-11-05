"""Disentangled sensory / action-history analysis of m1_action_kernel on the TEST sessions.

Four pages (MC06, MC07, MC08, pooled), one PDF, written to
results_post_analysis/expanded_model_comparison/pooled/action_history_parameters/.

Traces (all evaluated on the held-out test session of every sliding window):
  mouse            observed choices
  full model       sensory_actionHistory_lapse, its own P(long)
  sensory part     the same model's fitted sensory readout with the action-history prior set to
                   0.5, kappa to 1 and the lapse removed (closed-form decision-region integral,
                   same code path as extended_collect._sensory_only_probability)
  sensory model    M0 sensory_lapse fit in the same window (its full prediction)
  action history   the model's own prior pi_t (probability of choosing long on action history alone)

Psychometrics follow Behavior/New_version/psychometric_transition_analysis.py (4 rank-bins per
trial type, logist fit with curve_fit and the same bounds); transition windows follow
learning_curve.ipynb (adjusted alignment on the terminal rare run, -20..+19, first four
trials of a block never count as rare); rare windows follow rare_trials_analysis.ipynb
(-2..+2 inside the block).

Run from the repo root:  python -m post_analysis.expanded_model_comparison.action_history_disentangled
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.special import expit

from post_analysis.expanded_model_comparison.extended_collect import _sensory_only_probability, _task_moments

SUBJECTS = ["MC06", "MC07", "MC08"]
SLIDING = ROOT / "results_sliding_window"
OUT = ROOT / "results_post_analysis" / "expanded_model_comparison" / "pooled" / "action_history_parameters"
OUT_PDF = OUT / "action_history_disentangled_test_sessions.pdf"

TRIAL_RANGE = range(-20, 20)
RARE_RANGE = range(-2, 3)
FIRST_BLOCK_TRIALS_FORCED_COMMON = 4
EPS = 1e-6

TRACES = {  # key: (legend label, color, linestyle)
    "mouse": ("Mouse", "black", "-"),
    "full": ("Sensory + action history", "#4477AA", "-"),
    "sens_part": ("Sensory part of that model", "#EE7733", "-"),
    "m0": ("Sensory-only model (M0)", "#228833", "--"),
    "prior": ("Action history part", "#AA3377", "-"),
}
PROB_COLS = {"full": "p_full", "sens_part": "p_sens_part", "m0": "p_m0", "prior": "p_prior"}


# ----------------------------------------------------------------------------- psychometrics
def logist(x, alpha, beta, lam, gamma):
    return lam + (1 - lam - gamma) * expit(beta * (x - alpha))


def fit_data(x, y, boundary=1.25, min_trials=10):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_trials or np.unique(y).size < 2 or x.min() == x.max():
        return np.full(4, np.nan)
    p0 = [float(np.clip(boundary, x.min(), x.max())), 10.0, 0.05, 0.05]
    bounds = ([x.min(), 0.0, 0.0, 0.0], [x.max(), 100.0, 0.5, 0.5])
    try:
        return curve_fit(logist, x, y, p0=p0, bounds=bounds, maxfev=10000)[0]
    except (RuntimeError, ValueError, FloatingPointError):
        return np.full(4, np.nan)


def assign_bins(df, n_bins_per_type=4):
    """Rank-based quantile bins within each trial type, exactly as get_binned_stats does."""
    out = pd.Series(-1, index=df.index)
    for cat in (0, 1):
        sub = df[df.category_long == cat]
        if sub.empty:
            continue
        n = min(n_bins_per_type, len(sub))
        codes = pd.qcut(sub.duration_s.rank(method="first"), q=n, labels=False, duplicates="drop")
        out.loc[sub.index] = codes.astype(int) + cat * n_bins_per_type
    return out


def bin_stats(df, value):
    rows = []
    for _, g in df.groupby("_bin"):
        v = g[value]
        rows.append((g.duration_s.mean(), v.mean(), v.sem() if len(v) > 1 else 0.0))
    return np.array(sorted(rows))


# ----------------------------------------------------------------------------- data
def _window_files(model_dir: str, subject: str) -> list[Path]:
    files = {}
    for p in (SLIDING / model_dir / subject).rglob("trial_predictions.csv"):
        parts = p.parts
        if "full_session_matched_v1" not in parts or not p.parent.name.startswith("window_"):
            continue
        files[p.parent.name] = p
    return [files[k] for k in sorted(files)]


def _nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_subject(subject: str) -> pd.DataFrame:
    moments = _task_moments(subject)
    m0_files = {p.parent.name: p for p in _window_files("M0_sensory_v1", subject)}
    frames = []
    for path in _window_files("M1_action_kernel_v1", subject):
        t = pd.read_csv(path)
        t = t[t.split_role.eq("test")].copy()
        m0 = pd.read_csv(m0_files[path.parent.name], usecols=["session_id", "trial_index", "model_probability_long"])
        m0 = m0.rename(columns={"model_probability_long": "p_m0"})
        t = t.merge(m0, on=["session_id", "trial_index"], how="left", validate="one_to_one")
        ev = t.model_evaluated.astype(bool)
        t["p_full"] = t.model_probability_long
        t["p_prior"] = t.model_prior_long
        t["p_sens_part"] = np.nan
        sens_in = t.loc[ev].assign(fitted_ell_short=0.0, fitted_ell_long=0.0)
        t.loc[ev, "p_sens_part"] = _sensory_only_probability("m1_action_kernel", sens_in, moments)
        t["subject"] = subject
        t["window"] = path.parent.name
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["window", "session_id", "trial_index"]).reset_index(drop=True)
    return df


def valid_mask(df):
    return df.model_evaluated.astype(bool) & df.choice_long.isin([0, 1])


# ----------------------------------------------------------------------------- events
def adjusted_start(is_rare, block, formal):
    last_prev = formal - 1
    if last_prev < 0 or not is_rare[last_prev]:
        return formal
    start = last_prev
    while start - 1 >= 0 and block[start - 1] == block[last_prev] and is_rare[start - 1]:
        start -= 1
    return start


def collect_events(df: pd.DataFrame):
    """Returns (transition rows, rare rows). One row per (event, relative position) with the
    trial's own category, observed correctness and each trace's probability of being correct."""
    trans, rare = [], []
    ev_id = 0
    for (subject, session), s in df.groupby(["subject", "session_id"], sort=False):
        s = s.sort_values("trial_index").reset_index(drop=True)
        sl = s[s.true_block.isin(["S", "L"])].reset_index(drop=True)
        if sl.empty:
            continue
        # NaN-out the category of "rare" trials inside the first trials of a block (raw-behavior rule)
        block_run = sl.true_block.ne(sl.true_block.shift()).cumsum()
        in_block = sl.groupby(block_run).cumcount() + 1
        is_rare = ((sl.true_block.eq("S") & sl.category_long.eq(1)) | (sl.true_block.eq("L") & sl.category_long.eq(0))).to_numpy()
        early_rare = is_rare & (in_block.to_numpy() <= FIRST_BLOCK_TRIALS_FORCED_COMMON)
        blk = sl.true_block.to_numpy()
        change = sl.true_block.ne(sl.true_block.shift()) & sl.true_block.shift().notna()
        # a formal switch counts only between consecutive trials of the same session
        for formal in sl.index[change]:
            prev, cur = blk[formal - 1], blk[formal]
            kind = "S2L" if (prev, cur) == ("S", "L") else "L2S"
            align = adjusted_start(is_rare, blk, formal)
            pb_start = formal - 1
            while pb_start - 1 >= 0 and blk[pb_start - 1] == prev:
                pb_start -= 1
            cb_end = formal
            while cb_end + 1 < len(sl) and blk[cb_end + 1] == cur:
                cb_end += 1
            ev_id += 1
            for rel in TRIAL_RANGE:
                i = align + rel
                if i < pb_start or i > cb_end or early_rare[i]:
                    continue
                trans.append(_row(sl.loc[i], subject, ev_id, kind, rel, float(is_rare[i])))
        # rare-trial anchored windows (inside the anchor's own block)
        run_id = block_run.to_numpy()
        for a in np.flatnonzero(is_rare):
            ev_id += 1
            for rel in RARE_RANGE:
                i = a + rel
                if i < 0 or i >= len(sl) or run_id[i] != run_id[a]:
                    continue
                rare.append(_row(sl.loc[i], subject, ev_id, "rare_long" if sl.category_long[a] == 1 else "rare_short", rel))
    return pd.DataFrame(trans), pd.DataFrame(rare)


def _row(r, subject, ev_id, kind, rel, rare=np.nan):
    ok = bool(r.model_evaluated) and r.choice_long in (0, 1)
    row = dict(subject=subject, event_id=ev_id, kind=kind, rel=rel, category=int(r.category_long), valid=ok, rare=rare, bridge=False)
    if ok:
        y = float(r.choice_long)
        row["mouse"] = float(y == r.category_long)
        for key, col in PROB_COLS.items():
            p = float(np.clip(r[col], EPS, 1 - EPS)) if np.isfinite(r[col]) else np.nan
            row[key] = p if r.category_long == 1 else 1 - p
            row["nll_" + key] = float(_nll(y, p)) if np.isfinite(p) else np.nan
    return row


# ----------------------------------------------------------------------------- plotting
def style(ax):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def legend_nll(label, nll):
    return f"{label}  (NLL {nll:.3f})" if np.isfinite(nll) else label


def psychometric_block(fig, gs, col0, width, df_all, keys, title, row_main, row_diff):
    """One psychometric panel (+ difference panel below). df_all: valid test trials of the block subset."""
    ax = fig.add_subplot(gs[row_main, col0:col0 + width])
    axd = fig.add_subplot(gs[row_diff, col0:col0 + width], sharex=ax)
    df = df_all.copy()
    df["_bin"] = assign_bins(df)
    y = df.choice_long.to_numpy(float)
    x = df.duration_s.to_numpy(float)
    xs = np.linspace(0.15, 2.35, 200)
    m = bin_stats(df.assign(v=y), "v")
    mfit = fit_data(x, y)
    ax.errorbar(m[:, 0], m[:, 1], yerr=m[:, 2], fmt="o", color="black", ms=5, capsize=2.5, lw=1.2, zorder=5,
                label=f"Mouse  (n={len(df)})")
    if np.isfinite(mfit).all():
        ax.plot(xs, logist(xs, *mfit), color="black", lw=1.6)
    for k in keys:
        lab, col, ls = TRACES[k]
        p = df[PROB_COLS[k]].to_numpy(float)
        nll = _nll(y, p).mean()
        b = bin_stats(df.assign(v=p), "v")
        ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt="s", color=col, ms=4.5, capsize=2.5, lw=1.1, alpha=.9)
        f = fit_data(x, p)
        if np.isfinite(f).all():
            ax.plot(xs, logist(xs, *f), color=col, lw=1.6, ls=ls, label=legend_nll(lab, nll))
        else:
            ax.plot(b[:, 0], b[:, 1], color=col, lw=1.4, ls=ls, label=legend_nll(lab, nll))
        # difference mouse - model, paired per trial inside each bin
        d = df.assign(v=y - p)
        db = bin_stats(d, "v")
        axd.errorbar(db[:, 0], db[:, 1], yerr=db[:, 2], fmt="o", color=col, ms=4, capsize=2.5, lw=1.1)
        if np.isfinite(mfit).all() and np.isfinite(f).all():
            axd.plot(xs, logist(xs, *mfit) - logist(xs, *f), color=col, lw=1.2, ls=ls)
    ax.axvline(1.25, color="0.6", lw=.8, ls=":")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("P(choose long)", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    o = [i for i, t in enumerate(l) if t.startswith("Mouse")] + [i for i, t in enumerate(l) if not t.startswith("Mouse")]
    ax.legend([h[i] for i in o], [l[i] for i in o], fontsize=8.5, frameon=False, loc="upper left")
    plt.setp(ax.get_xticklabels(), visible=False)
    axd.axhline(0, color="0.5", lw=.8)
    axd.set_ylabel("mouse − model", fontsize=9)
    axd.set_xlabel("Stimulus duration (s)", fontsize=10)
    style(ax)
    style(axd)


def trace_summary(rows, key):
    g = rows.groupby("rel")
    return g[key].mean(), g[key].sem(), g[key].count()


def add_bridge(rows):
    """Raw-behavior rule: if a trial type never occurs at -1 (e.g. long trials just before a
    short->long switch), copy each event's last valid earlier trial of that type to -1."""
    ok = rows[rows.valid]
    if (ok.rel == -1).any():
        return rows
    last = ok[ok.rel < 0].sort_values("rel").groupby("event_id").tail(1).copy()
    last["rel"] = -1
    last["bridge"] = True
    return pd.concat([rows, last], ignore_index=True)


def mean_sem(rows, key):
    g = rows.groupby("rel")[key]
    return g.mean().reindex(list(TRIAL_RANGE)), g.sem().reindex(list(TRIAL_RANGE))


def event_block(fig, gs, c0, width, rows, keys, title, offsets, row_main, row_diff, xlabel, xticks=None,
                all_rows=None, transition=False):
    ax = fig.add_subplot(gs[row_main, c0:c0 + width])
    axd = fig.add_subplot(gs[row_diff, c0:c0 + width], sharex=ax)
    rows = rows[rows.valid]
    n_ev = rows.event_id.nunique()
    if transition:
        rows = add_bridge(rows)
    full_x = np.array(offsets)

    def series(frame, key):
        g = frame.groupby("rel")[key]
        return g.mean().reindex(full_x), g.sem().reindex(full_x)

    def draw(a, mu, se, col, ls, label=None, marker=None, **kw):
        vis = mu.notna()
        a.plot(full_x[vis.to_numpy()], mu[vis], color=col, ls=ls, lw=kw.get("lw", 1.6), marker=marker, ms=2.5, label=label,
               zorder=kw.get("zorder", 3))
        bm, bs = mu.interpolate(limit_area="inside"), se.interpolate(limit_area="inside")
        band = (bm.notna() & bs.notna()).to_numpy()
        a.fill_between(full_x, bm - bs, bm + bs, where=band, color=col, alpha=.15, lw=0)

    mu, se = series(rows, "mouse")
    draw(ax, mu, se, "black", "-", f"Mouse  (events={n_ev})", marker="o", lw=1.4, zorder=5)
    if transition and rows.bridge.any() and np.isfinite(mu.get(-1, np.nan)):
        ax.scatter(-1, mu.loc[-1], s=40, facecolors="white", edgecolors="black", linewidths=1.5, zorder=6,
                   label="Previous matching trial")
    real = rows[~rows.bridge]
    for k in keys:
        lab, col, ls = TRACES[k]
        m2, s2 = series(rows, k)
        draw(ax, m2, s2, col, ls, legend_nll(lab, real["nll_" + k].mean()), marker="o")
        diff = rows.assign(d=rows["mouse"] - rows[k])
        dm, ds = series(diff, "d")
        draw(axd, dm, ds, col, ls, marker="o")
    for a in (ax, axd):
        a.axvline(0, color="black", ls="--", lw=1.1, alpha=.8)
        style(a)
        if transition:
            a.axvspan(0, offsets[-1] + .5, color="gray", alpha=.08, lw=0)
            a.set_xlim(offsets[0] - .5, offsets[-1] + .5)
    ax.set_title(title, fontsize=10)
    if transition and all_rows is not None:
        rr = all_rows.groupby("rel")["rare"]
        rm, rs = rr.mean().reindex(full_x), rr.sem().reindex(full_x)
        tw = ax.twinx()
        tw.plot(full_x, rm, color="gray", lw=1.6, alpha=.85, label="Rare-trial rate")
        ok = (rm.notna() & rs.notna()).to_numpy()
        tw.fill_between(full_x, rm - rs, rm + rs, where=ok, color="gray", alpha=.12, lw=0)
        tw.set_ylim(-0.05, 1.05)
        tw.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
        tw.tick_params(axis="y", colors="gray", labelsize=8)
        tw.spines["top"].set_visible(False)
        tw.spines["right"].set_color("gray")
        h2, l2 = tw.get_legend_handles_labels()
    else:
        h2, l2 = [], []
    h, l = ax.get_legend_handles_labels()
    h, l = h + h2, l + l2
    o = [i for i, t in enumerate(l) if t.startswith("Mouse")] + [i for i, t in enumerate(l) if not t.startswith("Mouse")]
    ax.legend([h[i] for i in o], [l[i] for i in o], fontsize=7.5, frameon=False,
              loc="lower left" if transition else "lower right", handlelength=1.6)
    axd.axhline(0, color="0.5", lw=.8)
    plt.setp(ax.get_xticklabels(), visible=False)
    axd.set_xlabel(xlabel, fontsize=9)
    if xticks is not None:
        axd.set_xticks(xticks[0], xticks[1])
    return ax, axd


def group_header(fig, axes, text, color):
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    y = max(a.get_position().y1 for a in axes)
    fig.text((x0 + x1) / 2, y + 0.018, text, ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=color)


def make_page(df: pd.DataFrame, trans: pd.DataFrame, rare: pd.DataFrame, name: str) -> plt.Figure:
    fig = plt.figure(figsize=(58, 21))
    gs = GridSpec(6, 36, figure=fig, height_ratios=[3.4, 1.2, 3.0, 1.2, 2.3, 0.9], hspace=0.55, wspace=1.4,
                  left=0.035, right=0.985, top=0.93, bottom=0.04)
    d = df[valid_mask(df) & df.warm_up.eq(0)]
    subsets = [("All blocks", d), ("Short block", d[d.true_block.eq("S")]), ("Long block", d[d.true_block.eq("L")])]
    groups = [
        ("A. Sensory + action-history model (full)", ["full"], "#4477AA"),
        ("B. Sensory, disentangled: model's sensory part vs sensory-only model", ["sens_part", "m0"], "#EE7733"),
        ("C. Action history only (prior)", ["prior"], "#AA3377"),
    ]
    # row 0/1: psychometrics
    for g, (gtitle, keys, color) in enumerate(groups):
        axes = []
        for j, (sname, sub) in enumerate(subsets):
            psychometric_block(fig, gs, g * 12 + j * 4, 4, sub, keys, sname, 0, 1)
            axes.append(fig.axes[-2])
        group_header(fig, axes, gtitle, color)
    # rows 2/3: block transitions;   rows 4/5: rare trials
    t = trans
    for g, (gtitle, keys, color) in enumerate(groups):
        axes = []
        for j, (kind, cat) in enumerate([("S2L", 0), ("S2L", 1), ("L2S", 0), ("L2S", 1)]):
            rows = t[(t.kind == kind) & (t.category == cat)]
            ttl = f"{'Short→Long' if kind == 'S2L' else 'Long→Short'} · {'short' if cat == 0 else 'long'} trials"
            ax, _ = event_block(fig, gs, g * 12 + j * 3, 3, rows, keys, ttl, list(TRIAL_RANGE), 2, 3,
                                "Trial relative to adjusted block transition", all_rows=t[t.kind == kind],
                                transition=True)
            axes.append(ax)
            if j == 0 and g == 0:
                ax.set_ylabel("P(correct)", fontsize=10)
        group_header(fig, axes, gtitle, color)
    for g, (gtitle, keys, color) in enumerate(groups):
        axes = []
        for j, kind in enumerate(["rare_long", "rare_short"]):
            rows = rare[rare.kind == kind]
            ttl = "Rare long trial (in short block)" if kind == "rare_long" else "Rare short trial (in long block)"
            ax, _ = event_block(fig, gs, g * 12 + 1 + j * 6, 4, rows, keys, ttl, list(RARE_RANGE), 4, 5,
                                "Trial relative to rare trial", xticks=([-2, -1, 0, 1, 2], ["−2", "−1", "rare", "+1", "+2"]))
            axes.append(ax)
            if j == 0 and g == 0:
                ax.set_ylabel("P(correct)", fontsize=10)
        group_header(fig, axes, gtitle, color)
    n_sess = df.session_id.nunique()
    fig.suptitle(f"{name} — action-history model, disentangled, held-out test sessions (n={n_sess})", fontsize=17,
                 fontweight="bold", y=0.985)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, trans, rare = {}, {}, {}
    for s in SUBJECTS:
        data[s] = load_subject(s)
        trans[s], rare[s] = collect_events(data[s])
        print(s, "test sessions", data[s].session_id.nunique(), "trans rows", len(trans[s]), "rare rows", len(rare[s]),
              flush=True)
    pooled = pd.concat(data.values(), ignore_index=True)
    pt, pr = pd.concat(trans.values(), ignore_index=True), pd.concat(rare.values(), ignore_index=True)
    # event ids are unique within a subject only; offset them so pooled counts stay correct
    for sub, frame in (("t", pt), ("r", pr)):
        frame["event_id"] = frame.subject + "_" + frame.event_id.astype(str)
    with PdfPages(OUT_PDF) as pdf:
        for s in SUBJECTS:
            for frame in (trans[s], rare[s]):
                frame["event_id"] = frame.subject + "_" + frame.event_id.astype(str)
            fig = make_page(data[s], trans[s], rare[s], s)
            pdf.savefig(fig)
            plt.close(fig)
        fig = make_page(pooled, pt, pr, "Pooled (MC06 + MC07 + MC08)")
        pdf.savefig(fig)
        plt.close(fig)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
