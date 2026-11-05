#!/usr/bin/env python3
"""Figure 5 variant with NLL on the y axis.

The original fig05_chronological_winner_timeline plots each window's winning
model at a fixed height (the y axis carries no information). Here each window's
winner is drawn at the winning model's own mean NLL for that split, so the y
axis shows how well the best model fit, window by window. Same colors, marker
shapes, legend and x axis as the original. Reads only the existing
win_counts_per_window.csv; the original figure and code are untouched.

  python extra_tools/fig05_nll_for_mouse.py --mouse MC06      # writes ..._nll.png/.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from post_analysis.expanded_model_comparison import figures as F  # noqa: E402
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, SPLIT_ROLES, group_style, validate_mouse_id  # noqa: E402

STEM = "fig05_chronological_winner_timeline_nll"


def render(mouse: str) -> None:
    root = F.common.POST_ANALYSIS_ROOT / "expanded_model_comparison" / mouse
    tables, figures_dir = root / "tables", root / "figures"
    status = pd.read_csv(tables / "model_completion_status.csv")
    per_window = pd.read_csv(tables / "win_counts_per_window.csv")
    eligible = F._eligible_order(status)
    offset = F.EXPECTED_OFFSET  # test rank = window id + 12 (9 train + 3 validation)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.4), sharex=True)
    for ax, split_role in zip(axes, SPLIT_ROLES):
        split = per_window[per_window["split_role"].eq(split_role)].sort_values("window_id")
        ranks = split["window_id"].astype(int) + offset
        ax.plot(ranks, split["best_mean_nll"], color="0.75", lw=0.9, zorder=1)
        for rank, (_, row) in zip(ranks, split.iterrows()):
            winners = str(row["winning_models"]).split("|")
            for winner in winners:  # ties share the same NLL, so they overlap exactly
                ax.scatter(rank, row["best_mean_nll"], color=MODEL_BY_ID[winner].color,
                           s=46 / len(winners), marker=group_style(winner, "marker"), zorder=3)
        ax.set_ylabel(f"{split_role.capitalize()} NLL\n(winning model, nats/choice)")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Chronological rank of held-out session")
    handles = [plt.Line2D([], [], marker=group_style(m, "marker"), linestyle="", color=MODEL_BY_ID[m].color,
                          label=MODEL_BY_ID[m].short_label) for m in eligible]
    axes[0].legend(handles=handles, ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.65), loc="upper center")
    fig.suptitle(f"{mouse} — chronological best-fit model by split, plotted at its NLL — {F._provisional_text(status)}",
                 y=1.06)
    fig.tight_layout()
    F._save(fig, figures_dir, STEM)
    print(f"wrote {figures_dir / (STEM + '.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", nargs="+", required=True)
    for mouse in parser.parse_args().mouse:
        render(validate_mouse_id(mouse))


if __name__ == "__main__":
    main()
