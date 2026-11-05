"""Mean fitted parameters of m1_action_kernel (action-history model) across windows, MC06-MC08.

Page 1: raw parameter names. Page 2: same layout with descriptive names and interpretation hints.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results_post_analysis" / "expanded_model_comparison"
OUT = BASE / "pooled" / "action_history_parameters"
SUBJECTS = ["MC06", "MC07", "MC08"]
COLORS = {"MC06": "#1b9e77", "MC07": "#d95f02", "MC08": "#7570b3"}
# natural bounds: probabilities in [0,1]; kappa and w are non-negative scales
PARAMS = {"w": (0, None), "s0": (None, None), "ell_short": (0, 1), "ell_long": (0, 1),
          "kappa": (0, None), "alpha": (0, 1)}
LABEL = {"alpha": "alpha (learning rate)", "ell_short": "ell_short (lapse)", "ell_long": "ell_long (lapse)"}
# page 2: (title, y label, neutral reference line or None, (top hint, bottom hint) shown at the axis edges)
DESC = {
    "w": ("Duration-dependent noise\n(Weber-like coefficient)", "w  (variance = w·s² + s0²)", None,
          ("more noise at long durations", "noise ~ constant")),
    "s0": ("Baseline sensory noise", "SD of timing noise (s)", None,
           ("noisier perception", "more precise")),
    "ell_short": ("Lapse rate, short trials", "P(random guess | short stimulus)", None,
                  ("more random guesses", "fewer guesses")),
    "ell_long": ("Lapse rate, long trials", "P(random guess | long stimulus)", None,
                 ("more random guesses", "fewer guesses")),
    "kappa": ("Internal fixed bias", "cost ratio  (1 = unbiased)", 1.0,
              ("short-biased  (>1)", "long-biased  (<1)")),
    "alpha": ("Learning rate of action-history prior", "learning rate  (0 = never update, 1 = copy last choice)", None,
              ("fast updating, ~1/rate trials memory", "slow updating, long memory")),
}

df = pd.concat([pd.read_csv(BASE / s / "tables" / "extended_parameters.csv").assign(subject=s) for s in SUBJECTS])
df = df[df.model_id.eq("m1_action_kernel")]
OUT.mkdir(parents=True, exist_ok=True)
rows = []


def page(described: bool):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (name, (lo, hi)) in zip(axes.flat, PARAMS.items()):
        vals_all = []
        for x, s in enumerate(SUBJECTS):
            v = df[(df.subject == s) & (df.param_name == name)].param_value.to_numpy()
            m, sd = v.mean(), v.std(ddof=1)
            if not described:
                rows.append(dict(parameter=name, subject=s, n_windows=len(v), mean=m, sd=sd, sem=sd / np.sqrt(len(v))))
            ax.scatter(x + np.random.default_rng(0).uniform(-.12, .12, len(v)), v, s=10, color=COLORS[s], alpha=.35)
            ax.errorbar(x, m, yerr=sd, fmt="o", color=COLORS[s], capsize=5, ms=8, mec="k", lw=1.8)
            vals_all.append(v)
        allv = np.concatenate(vals_all)
        ref = DESC[name][2] if described else None
        ymin, ymax = allv.min(), allv.max()
        if ref is not None:
            ymin, ymax = min(ymin, ref), max(ymax, ref)
        pad = 0.1 * (ymax - ymin or 1)
        bot = ymin - pad if lo is None else max(lo, ymin - pad)
        top = ymax + pad if hi is None else min(hi, ymax + pad)
        ax.set_ylim(bot, top)
        ax.set_xticks(range(3), SUBJECTS); ax.set_xlim(-.5, 2.5)
        ax.grid(False); ax.spines[["top", "right"]].set_visible(False)
        if not described:
            ax.set_ylabel(LABEL.get(name, name)); ax.set_title(name, fontsize=10)
            continue
        title, ylab, ref, (hi_txt, lo_txt) = DESC[name]
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab, fontsize=8)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=.8)
            ax.text(2.45, ref, "neutral", ha="right", va="bottom", fontsize=7)
        kw = dict(transform=ax.transAxes, fontsize=7, style="italic", color="0.35", ha="left")
        ax.text(0.02, 0.98, "▲ " + hi_txt, va="top", **kw)
        ax.text(0.02, 0.02, "▼ " + lo_txt, va="bottom", **kw)
    fig.suptitle("Action-history model (m1_action_kernel): mean ± SD across windows; dots = windows" if not described
                 else "Same data, with interpretation (arrows show what higher / lower values mean)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    with PdfPages(OUT / "action_history_parameters_by_subject.pdf") as pdf:
        for d in (False, True):
            fig = page(d)
            pdf.savefig(fig); plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "action_history_parameter_means.csv", index=False)
