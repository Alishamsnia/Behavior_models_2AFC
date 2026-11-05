"""Chronological (window order) trajectories of m1_action_kernel parameters, one page per subject.

Same colours, names and hints as page 2 of action_history_parameters_by_subject.pdf.
Coloured curve = LOWESS smooth, grey straight line = least-squares linear trend.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess

from action_history_param_means import COLORS, DESC, OUT, PARAMS, SUBJECTS, df

with PdfPages(OUT / "action_history_parameters_chronology.pdf") as pdf:
    for s in SUBJECTS:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
        d = df[df.subject == s]
        for ax, (name, (lo, hi)) in zip(axes.flat, PARAMS.items()):
            g = d[d.param_name == name].sort_values("window_id")
            x, y = g.window_id.to_numpy(float), g.param_value.to_numpy()
            ax.scatter(x, y, s=18, color=COLORS[s], alpha=.55)
            sm = lowess(y, x, frac=0.4, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=COLORS[s], lw=2.2)
            slope, icpt = np.polyfit(x, y, 1)
            ax.plot(x, slope * x + icpt, color="0.5", lw=1.6)
            title, ylab, ref, (hi_txt, lo_txt) = DESC[name]
            lo_v = min(y.min(), sm[:, 1].min(), ref if ref is not None else np.inf)
            hi_v = max(y.max(), sm[:, 1].max(), ref if ref is not None else -np.inf)
            pad = 0.1 * (hi_v - lo_v or 1)
            ax.set_ylim(lo_v - pad if lo is None else max(lo, lo_v - pad),
                        hi_v + pad if hi is None else min(hi, hi_v + pad))
            ax.set_title(title, fontsize=10); ax.set_ylabel(ylab, fontsize=8)
            ax.set_xlabel("sliding window (chronological order)", fontsize=8)
            ax.grid(False); ax.spines[["top", "right"]].set_visible(False)
            if ref is not None:
                ax.axhline(ref, color="k", ls="--", lw=.8)
                ax.text(0.98, ref, "neutral", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)
            kw = dict(transform=ax.transAxes, fontsize=7, style="italic", color="0.35", ha="left")
            ax.text(0.02, 0.98, "▲ " + hi_txt, va="top", **kw)
            ax.text(0.02, 0.02, "▼ " + lo_txt, va="bottom", **kw)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[s], alpha=.55, label="window fit"),
                            Line2D([], [], color=COLORS[s], lw=2.2, label="smoothed (LOWESS)"),
                            Line2D([], [], color="0.5", lw=1.6, label="linear trend")],
                   loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"{s} — action-history model: parameters across windows (arrows show what higher / lower values mean)")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        pdf.savefig(fig); plt.close(fig)
