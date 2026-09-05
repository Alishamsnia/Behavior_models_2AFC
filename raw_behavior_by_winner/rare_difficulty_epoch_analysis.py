from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, wilcoxon


OUTPUT_PATH = Path("rare_difficulty_early_late_performance_changes.pdf")
EARLY_N, LATE_N = 10, 7
DIFFICULTY_COLORS = {"Easy": "#1b9e77", "Hard": "#d95f02"}
EPOCH_COLORS = {"Early": "#377eb8", "Late": "#984ea3"}


def session_extremes(sdf):
    isi = pd.to_numeric(sdf["trial_isi"], errors="coerce")
    short = isi[sdf["trial_type"].eq(1)].dropna()
    long = isi[sdf["trial_type"].eq(2)].dropna()
    return {
        "short_easy": short.min() if len(short) else np.nan,
        "short_hard": short.max() if len(short) else np.nan,
        "long_easy": long.max() if len(long) else np.nan,
        "long_hard": long.min() if len(long) else np.nan,
    }


def difficulty(row, limits):
    isi = float(row["trial_isi"])
    prefix = "short" if row["trial_type"] == 1 else "long"
    easy = np.isclose(isi, limits[f"{prefix}_easy"], rtol=0, atol=1e-9)
    hard = np.isclose(isi, limits[f"{prefix}_hard"], rtol=0, atol=1e-9)
    if easy == hard:  # Neither extreme, or only one ISI level in the session.
        return None
    return "Easy" if easy else "Hard"


def collect_events(data, adjusted_block_start, qualified_change):
    columns = [
        "subject", "session", "block_id", "block_context",
        "epoch", "difficulty", "change",
    ]
    rows = []
    for session, session_df in data.groupby("change_session", sort=False):
        sdf = session_df[session_df["block_type"].isin([1, 2])].copy()
        sdf = sdf.sort_values("_source_order").reset_index(drop=True)
        if sdf.empty:
            continue
        limits = session_extremes(sdf)
        switches = sdf.index[
            sdf["block_type"].ne(sdf["block_type"].shift())
            & sdf["block_type"].shift().notna()
        ].tolist()
        starts = [0] + [adjusted_block_start(sdf, idx) for idx in switches]
        duplicate_starts = sorted({start for start in starts if starts.count(start) > 1})
        if duplicate_starts:
            raise ValueError(
                "Adjusted block boundaries collapse in session "
                f"{session} at trial position(s) {duplicate_starts}. "
                "An intervening block may contain only rare trials; refusing "
                "to compare epochs across a deleted block."
            )
        if any(stop <= start for start, stop in zip(starts[:-1], starts[1:])):
            raise ValueError(
                f"Adjusted block boundaries are not strictly increasing in {session}: "
                f"{starts}"
            )

        # Intervals ending at the next adjusted start are complete blocks;
        # the final session block has no endpoint and is never included.
        for block_number, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
            positions = np.arange(start, stop, dtype=int)
            # Fixed first-10 and last-7 windows are only comparable when they
            # are disjoint. Shorter blocks cannot satisfy that design and are
            # excluded rather than contributing the same trial to both epochs.
            if len(positions) < EARLY_N + LATE_N:
                continue
            epochs = {"Early": positions[:EARLY_N], "Late": positions[-LATE_N:]}
            for epoch, anchors in epochs.items():
                for idx in anchors:
                    anchor = sdf.loc[idx]
                    if pd.isna(anchor["is_rare"]) or not bool(anchor["is_rare"]):
                        continue
                    change = qualified_change(sdf, int(idx))
                    if not np.isfinite(change):
                        continue
                    level = difficulty(anchor, limits)
                    if level is None:
                        continue
                    if anchor["block_type"] == 1 and anchor["trial_type"] == 2:
                        context = "Short block"
                    elif anchor["block_type"] == 2 and anchor["trial_type"] == 1:
                        context = "Long block"
                    else:
                        continue
                    rows.append({
                        "subject": anchor["subject"], "session": session,
                        "block_id": f"{session}_B{block_number}",
                        "block_context": context, "epoch": epoch,
                        "difficulty": level, "change": change,
                    })
    # Keep a stable schema when no trial satisfies this deliberately strict
    # rare/difficulty/epoch selection, so the analysis can report an empty
    # result and render its no-data panels instead of failing on column lookup.
    return pd.DataFrame(rows, columns=columns)


def select(events, spec):
    block, epoch, level = spec
    return events[
        events["block_context"].eq(block)
        & events["epoch"].eq(epoch)
        & events["difficulty"].eq(level)
    ]


def values(events, spec, mode):
    chosen = select(events, spec)
    if mode == "pooled":
        return chosen["change"].dropna(), len(chosen)
    return chosen.groupby("session")["change"].mean().dropna(), len(chosen)


def mean_sem(series):
    series = pd.Series(series).dropna()
    if series.empty:
        return np.nan, np.nan
    return series.mean(), series.sem() if len(series) > 1 else 0.0


def pvalue(events, left_spec, right_spec, mode):
    left, right = select(events, left_spec), select(events, right_spec)
    if mode == "pooled":
        if left.empty or right.empty:
            return np.nan
        return mannwhitneyu(left["change"], right["change"], alternative="two-sided").pvalue
    paired = pd.concat([
        left.groupby("session")["change"].mean().rename("left"),
        right.groupby("session")["change"].mean().rename("right"),
    ], axis=1, join="inner").dropna()
    if len(paired) < 2:
        return np.nan
    if np.allclose(paired["left"] - paired["right"], 0):
        return 1.0
    try:
        return wilcoxon(paired["left"], paired["right"]).pvalue
    except ValueError:
        return np.nan


def stars(p):
    if not np.isfinite(p): return "n/a"
    if p < 0.0001: return "****"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


PAIR_SPECS = [
    (("Short block", "Early", "Hard"), ("Short block", "Early", "Easy"), "Short block early\nHard vs Easy"),
    (("Short block", "Late", "Hard"), ("Short block", "Late", "Easy"), "Short block late\nHard vs Easy"),
    (("Long block", "Early", "Hard"), ("Long block", "Early", "Easy"), "Long block early\nHard vs Easy"),
    (("Long block", "Late", "Hard"), ("Long block", "Late", "Easy"), "Long block late\nHard vs Easy"),
    (("Short block", "Early", "Easy"), ("Short block", "Late", "Easy"), "Short block easy\nEarly vs Late"),
    (("Short block", "Early", "Hard"), ("Short block", "Late", "Hard"), "Short block hard\nEarly vs Late"),
    (("Long block", "Early", "Easy"), ("Long block", "Late", "Easy"), "Long block easy\nEarly vs Late"),
    (("Long block", "Early", "Hard"), ("Long block", "Late", "Hard"), "Long block hard\nEarly vs Late"),
]


def plot_panel(ax, events, mode, title):
    handles = []
    seen = set()
    for x, (left_spec, right_spec, tick) in enumerate(PAIR_SPECS):
        compare_difficulty = left_spec[1] == right_spec[1]
        colors = (DIFFICULTY_COLORS[left_spec[2]], DIFFICULTY_COLORS[right_spec[2]]) if compare_difficulty else (
            EPOCH_COLORS[left_spec[1]], EPOCH_COLORS[right_spec[1]])
        for offset, spec, color in zip((-0.14, 0.14), (left_spec, right_spec), colors):
            vals, n = values(events, spec, mode)
            mean, sem = mean_sem(vals)
            ax.errorbar(x + offset, mean, yerr=sem, fmt="o", color=color,
                        markersize=5.5, capsize=3.5, elinewidth=1.3)
            label = f"{spec[0]}, {spec[1]}, {spec[2]} (n={n:,} trials)"
            if label not in seen:
                handles.append(Line2D([0], [0], marker="o", linestyle="none", color=color, label=label))
                seen.add(label)
        y = 0.92 if x < 4 else 1.13
        ax.plot([x-.14, x-.14, x+.14, x+.14], [y-.025, y, y, y-.025], color="black", lw=.9)
        ax.text(x, y+.015, stars(pvalue(events, left_spec, right_spec, mode)), ha="center", fontsize=8)
    ax.axhline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set(xlim=(-.5, 7.5), ylim=(-1.12, 1.38),
           ylabel="Performance change\n(trial 0 − trial −1)", title=title)
    ax.set_xticks(range(8), [item[2] for item in PAIR_SPECS], rotation=22, ha="right", fontsize=7)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=True, fontsize=5.5, borderaxespad=0)


def make_figure(events, subject_rows, output_path=OUTPUT_PATH, show=True):
    fig = plt.figure(figsize=(26, 29))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=.63, wspace=.82)
    for row, (label, subject) in enumerate(subject_rows):
        row_events = events if subject is None else events[events["subject"].eq(subject)]
        plot_panel(fig.add_subplot(gs[row, 0]), row_events, "pooled", f"{label} — pooled trials")
        plot_panel(fig.add_subplot(gs[row, 1]), row_events, "grand", f"{label} — grand average across sessions")
    fig.suptitle(
        "Rare-trial performance changes by difficulty and epoch "
        f"(early first {EARLY_N}; late last {LATE_N}; "
        f"blocks shorter than {EARLY_N + LATE_N} excluded)",
        fontsize=16,
        y=.995,
    )
    fig.subplots_adjust(left=.055, right=.78, top=.975, bottom=.07)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return Path(output_path)


def run(data, adjusted_block_start, qualified_change, subject_rows, show=True):
    events = collect_events(data, adjusted_block_start, qualified_change)
    output = make_figure(events, subject_rows, show=show)
    print(f"Saved: {output.resolve()}")
    print(events.groupby(["subject", "block_context", "epoch", "difficulty"]).size())
    return output, events
