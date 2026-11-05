"""Parameter means and chronology for m2_dynamic_repetition_bias
(sensory + learned category ratio + learned dynamic block switch + single previous-action bias + lapse).

Writes two PDFs under results_post_analysis/expanded_model_comparison/pooled/m2_dynamic_repetition_bias_parameters/:
  *_by_subject.pdf  page 1 raw names, page 2 descriptive names + hints (mean ± SD across windows)
  *_chronology.pdf  one page per subject: points, LOWESS smooth (colour), linear trend (grey)
Run from the repo root: python post_analysis/expanded_model_comparison/m2_repetition_param_pages.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess

from post_analysis.expanded_model_comparison.extended_figures import _expected_block_length, _sigmoid_np

BASE = REPO / "results_post_analysis" / "expanded_model_comparison"
OUT = BASE / "pooled" / "m2_dynamic_repetition_bias_parameters"
MODEL = "m2_dynamic_repetition_bias"
SUBJECTS = ["MC06", "MC07", "MC08"]
COLORS = {"MC06": "#1b9e77", "MC07": "#d95f02", "MC08": "#7570b3"}  # same as the action-history PDFs
# name: (lower bound, upper bound, raw label, title, y label, neutral line, (hint high, hint low))
SPEC = {
    "w": (0, None, "w", "Duration-dependent noise\n(Weber-like coefficient)", "w  (variance = w·s² + s0²)", None,
          ("more noise at long durations", "noise ~ constant")),
    "s0": (0, None, "s0", "Baseline sensory noise", "SD of timing noise (s)", None,
           ("noisier perception", "more precise")),
    "ell_short": (0, 1, "ell_short", "Lapse rate, short trials", "P(random guess | short stimulus)", None,
                  ("more random guesses", "fewer guesses")),
    "ell_long": (0, 1, "ell_long", "Lapse rate, long trials", "P(random guess | long stimulus)", None,
                 ("more random guesses", "fewer guesses")),
    "kappa": (0, None, "kappa", "Internal fixed bias", "cost ratio  (1 = unbiased)", 1.0,
              ("short-biased  (>1)", "long-biased  (<1)")),
    "delta0": (0, None, "delta0", "Assumed block contrast\n(learned category ratio)", "δ0  (θS=σ(−δ0), θL=σ(δ0))", None,
               ("blocks assumed more lopsided", "blocks assumed more balanced")),
    "block_len": (1, None, "expected block length", "Expected block length\n(dynamic block switch)", "trials per block", None,
                  ("long blocks (rare switches)", "short blocks (frequent switches)")),
    "excess_hazard": (-1, 1, "hazard(65) − hazard(23)", "Run-length dependence of switching", "extra switch hazard late in a block", 0.0,
                      ("switch more likely late in block", "switch less likely late in block")),
    "eta": (0, 1, "eta", "Previous-action bias weight\n(single action bias)", "η  (0 = ignore, 1 = repeat last choice)", None,
            ("repeats previous choice more", "ignores previous choice")),
}
NAMES = list(SPEC)


def load() -> pd.DataFrame:
    frames = []
    for s in SUBJECTS:
        d = pd.read_csv(BASE / s / "tables" / "extended_parameters.csv")
        d = d[d.model_id.eq(MODEL)]
        w = d.pivot(index="window_id", columns="param_name", values="param_value").sort_index()
        ah, bh = w["ah"].to_numpy(), w["bh"].to_numpy()
        w["block_len"] = [_expected_block_length(a, b) for a, b in zip(ah, bh)]
        w["excess_hazard"] = _sigmoid_np(ah + bh * (65 - 23)) - _sigmoid_np(ah + bh * (23 - 23))
        w["subject"] = s
        frames.append(w.reset_index())
    return pd.concat(frames, ignore_index=True)


def ylim(ax, name, vals, ref):
    lo, hi = SPEC[name][0], SPEC[name][1]
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if ref is not None:
        vmin, vmax = min(vmin, ref), max(vmax, ref)
    pad = 0.1 * (vmax - vmin or 1)
    ax.set_ylim(vmin - pad if lo is None else max(lo if name != "excess_hazard" else -1, vmin - pad),
                vmax + pad if hi is None else min(hi, vmax + pad))


def decorate(ax, name, described):
    lo, hi, raw, title, ylab, ref, (hi_txt, lo_txt) = SPEC[name]
    ax.grid(False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title if described else raw, fontsize=10)
    ax.set_ylabel(ylab if described else raw, fontsize=8)
    if described:
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=.8)
            ax.text(.98, ref, "neutral", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)
        kw = dict(transform=ax.transAxes, fontsize=7, style="italic", color="0.35", ha="left")
        ax.text(0.02, 0.98, "▲ " + hi_txt, va="top", **kw)
        ax.text(0.02, 0.02, "▼ " + lo_txt, va="bottom", **kw)


def means_page(df, described):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, name in zip(axes.flat, NAMES):
        allv = []
        for x, s in enumerate(SUBJECTS):
            v = df[df.subject.eq(s)][name].dropna().to_numpy()
            ax.scatter(x + np.random.default_rng(0).uniform(-.12, .12, len(v)), v, s=10, color=COLORS[s], alpha=.35)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", color=COLORS[s], capsize=5, ms=8, mec="k", lw=1.8)
            allv.append(v)
        ylim(ax, name, np.concatenate(allv), SPEC[name][5] if described else None)
        ax.set_xticks(range(3), SUBJECTS); ax.set_xlim(-.5, 2.5)
        decorate(ax, name, described)
    fig.suptitle(f"{MODEL}: mean ± SD across windows; dots = windows" if not described
                 else "Same data, with interpretation (arrows show what higher / lower values mean)")
    fig.tight_layout()
    return fig


def chronology_page(df, s):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    d = df[df.subject.eq(s)]
    for ax, name in zip(axes.flat, NAMES):
        x, y = d.window_id.to_numpy(float), d[name].to_numpy()
        ok = np.isfinite(y); x, y = x[ok], y[ok]
        ax.scatter(x, y, s=18, color=COLORS[s], alpha=.55)
        sm = lowess(y, x, frac=0.4, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1], color=COLORS[s], lw=2.2)
        slope, icpt = np.polyfit(x, y, 1)
        ax.plot(x, slope * x + icpt, color="0.5", lw=1.6)
        ylim(ax, name, np.concatenate([y, sm[:, 1]]), SPEC[name][5])
        ax.set_xlabel("sliding window (chronological order)", fontsize=8)
        decorate(ax, name, True)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[s], alpha=.55, label="window fit"),
                        Line2D([], [], color=COLORS[s], lw=2.2, label="smoothed (LOWESS)"),
                        Line2D([], [], color="0.5", lw=1.6, label="linear trend")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{s} — {MODEL}: parameters across windows (arrows show what higher / lower values mean)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    rows = [dict(parameter=n, subject=s, n_windows=int(df[df.subject.eq(s)][n].notna().sum()),
                 mean=df[df.subject.eq(s)][n].mean(), sd=df[df.subject.eq(s)][n].std(ddof=1))
            for n in NAMES for s in SUBJECTS]
    pd.DataFrame(rows).to_csv(OUT / "m2_dynamic_repetition_bias_parameter_means.csv", index=False)
    with PdfPages(OUT / "m2_dynamic_repetition_bias_parameters_by_subject.pdf") as pdf:
        for described in (False, True):
            fig = means_page(df, described); pdf.savefig(fig); plt.close(fig)
    with PdfPages(OUT / "m2_dynamic_repetition_bias_parameters_chronology.pdf") as pdf:
        for s in SUBJECTS:
            fig = chronology_page(df, s); pdf.savefig(fig); plt.close(fig)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
