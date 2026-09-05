from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon


EPOCHS = ["Epoch 1", "Epoch 2", "Epoch 3"]
PHASES = ["pre", "rare", "post"]
PHASE_LABELS = {"pre": "Pre rare", "rare": "Rare", "post": "Post rare"}

# Okabe-Ito colorblind-safe colors. Rare is visually dominant; pre/post remain distinct.
PHASE_COLORS = {"pre": "#0072B2", "rare": "#D55E00", "post": "#009E73"}
EPOCH_COLORS = {"Epoch 1": "#56B4E9", "Epoch 2": "#E69F00", "Epoch 3": "#CC79A7"}
CONTRAST_COLORS = {"drop": "#D55E00", "recover": "#009E73"}
CONTRAST_LABELS = {"drop": "Drop: rare − pre", "recover": "Recovery: rare − post"}

WINDOW_SIZE = 7
WINDOW_STEP = 3
MAX_WINDOW_CENTER = 25


def _adjusted_blocks(data, adjusted_block_start):
    """Yield complete adjusted short/long blocks; omit each session's final block."""
    for session, session_df in data.groupby("change_session", sort=False):
        sdf = session_df[session_df["block_type"].isin([1, 2])].copy()
        sdf = sdf.sort_values("_source_order").reset_index(drop=True)
        if sdf.empty:
            continue
        formal_switches = sdf.index[
            sdf["block_type"].ne(sdf["block_type"].shift())
            & sdf["block_type"].shift().notna()
        ].tolist()
        starts = [0] + [adjusted_block_start(sdf, idx) for idx in formal_switches]
        duplicate_starts = sorted({start for start in starts if starts.count(start) > 1})
        if duplicate_starts:
            raise ValueError(
                "Adjusted block boundaries collapse in session "
                f"{session} at trial position(s) {duplicate_starts}. "
                "An intervening block may contain only rare trials; refusing "
                "to delete that block or let an analysis window cross it."
            )
        if any(stop <= start for start, stop in zip(starts[:-1], starts[1:])):
            raise ValueError(
                f"Adjusted block boundaries are not strictly increasing in {session}: "
                f"{starts}"
            )
        invalid_starts = [
            start for start in starts[1:]
            if start > 0 and bool(sdf.loc[start - 1, "is_rare"])
        ]
        if invalid_starts:
            raise AssertionError(
                f"Adjusted transition preceded by a rare trial in {session}: {invalid_starts}"
            )
        for block_number, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
            yield session, sdf, block_number, int(start), int(stop)


def collect_sequence_events(data, adjusted_block_start):
    """Collect common/common/rare/common trains and pre/rare/post performance."""
    rows = []
    for session, sdf, block_number, start, stop in _adjusted_blocks(data, adjusted_block_start):
        positions = np.arange(start, stop, dtype=int)
        epoch_lookup = {}
        for epoch_number, epoch_positions in enumerate(np.array_split(positions, 3), start=1):
            for position in epoch_positions:
                epoch_lookup[int(position)] = f"Epoch {epoch_number}"

        for anchor in range(start + 2, stop - 1):
            train = sdf.loc[[anchor - 2, anchor - 1, anchor, anchor + 1]]
            rare_pattern = train["is_rare"].astype(bool).tolist()
            if rare_pattern != [False, False, True, False]:
                continue
            phase_values = {
                "pre": sdf.loc[anchor - 1, "correct"],
                "rare": sdf.loc[anchor, "correct"],
                "post": sdf.loc[anchor + 1, "correct"],
            }
            if any(pd.isna(value) for value in phase_values.values()):
                continue
            block_type = int(sdf.loc[anchor, "block_type"])
            rows.append(
                {
                    "subject": sdf.loc[anchor, "subject"],
                    "session": session,
                    "block_id": f"{session}_B{block_number}",
                    "sequence_id": f"{session}_B{block_number}_T{anchor - start}",
                    "block_context": "Short block" if block_type == 1 else "Long block",
                    "epoch": epoch_lookup[anchor],
                    "anchor_position": anchor - start,
                    "train_start_position": anchor - start - 2,
                    "train_end_position": anchor - start + 1,
                    "block_length": stop - start,
                    **{key: float(value) for key, value in phase_values.items()},
                }
            )
    events = pd.DataFrame(rows)
    if not events.empty:
        events["drop"] = events["rare"] - events["pre"]
        # This follows the requested sign convention exactly.
        events["recover"] = events["rare"] - events["post"]
    return events


def collect_sliding_window_events(sequence_events, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """Place complete qualifying trains into every complete seven-trial sliding window."""
    rows = []
    if sequence_events.empty:
        return pd.DataFrame(rows)
    for _, event in sequence_events.iterrows():
        block_length = int(event["block_length"])
        anchor = int(event["anchor_position"])
        for window_start in range(0, block_length - window_size + 1, step):
            window_stop = window_start + window_size
            # One-based trial number of the middle trial in this seven-trial window.
            window_center = window_start + (window_size + 1) / 2
            if window_center > MAX_WINDOW_CENTER:
                continue
            # Require all four train trials (-2, -1, 0, +1) inside the window.
            if anchor - 2 < window_start or anchor + 1 >= window_stop:
                continue
            rows.append(
                {
                    **event.to_dict(),
                    "window_start": window_start,
                    "window_stop": window_stop,
                    "window_center": window_center,
                    "window_number": window_start // step + 1,
                }
            )
    return pd.DataFrame(rows)


def collect_pre_rare_sequence_events(data, adjusted_block_start):
    """Collect strict common/common/rare trains with rare anchored at trial zero."""
    rows = []
    for session, sdf, block_number, start, stop in _adjusted_blocks(data, adjusted_block_start):
        for anchor in range(start + 2, stop):
            train = sdf.loc[[anchor - 2, anchor - 1, anchor]]
            if train["is_rare"].astype(bool).tolist() != [False, False, True]:
                continue
            pre = sdf.loc[anchor - 1, "correct"]
            rare = sdf.loc[anchor, "correct"]
            if pd.isna(pre) or pd.isna(rare):
                continue
            block_type = int(sdf.loc[anchor, "block_type"])
            rows.append(
                {
                    "subject": sdf.loc[anchor, "subject"],
                    "session": session,
                    "block_id": f"{session}_B{block_number}",
                    "sequence_id": f"{session}_B{block_number}_T{anchor - start}",
                    "block_context": "Short block" if block_type == 1 else "Long block",
                    "anchor_position": anchor - start,
                    "train_start_position": anchor - start - 2,
                    "train_end_position": anchor - start,
                    "block_length": stop - start,
                    "pre": float(pre),
                    "rare": float(rare),
                }
            )
    return pd.DataFrame(rows)


def collect_pre_rare_sliding_window_events(
    sequence_events, window_size=WINDOW_SIZE, step=WINDOW_STEP
):
    """Place complete common/common/rare trains into complete sliding windows."""
    rows = []
    if sequence_events.empty:
        return pd.DataFrame(rows)
    for _, event in sequence_events.iterrows():
        block_length = int(event["block_length"])
        anchor = int(event["anchor_position"])
        for window_start in range(0, block_length - window_size + 1, step):
            window_stop = window_start + window_size
            # One-based trial number of the middle trial in this seven-trial window.
            window_center = window_start + (window_size + 1) / 2
            if window_center > MAX_WINDOW_CENTER:
                continue
            if anchor - 2 < window_start or anchor >= window_stop:
                continue
            rows.append(
                {
                    **event.to_dict(),
                    "window_start": window_start,
                    "window_stop": window_stop,
                    "window_center": window_center,
                    "window_number": window_start // step + 1,
                }
            )
    result = pd.DataFrame(rows)
    validate_sliding_window_boundaries(result)
    return result


def collect_common_common_rare_rare_common_events(data, adjusted_block_start):
    """Collect common/common/rare/rare/common trains with the first rare at trial zero."""
    rows = []
    for session, sdf, block_number, start, stop in _adjusted_blocks(data, adjusted_block_start):
        for anchor in range(start + 2, stop - 2):
            train = sdf.loc[[anchor - 2, anchor - 1, anchor, anchor + 1, anchor + 2]]
            if train["is_rare"].astype(bool).tolist() != [False, False, True, True, False]:
                continue
            values = {
                "pre_common": sdf.loc[anchor - 1, "correct"],
                "rare_1": sdf.loc[anchor, "correct"],
                "rare_2": sdf.loc[anchor + 1, "correct"],
                "post_common": sdf.loc[anchor + 2, "correct"],
            }
            if any(pd.isna(value) for value in values.values()):
                continue
            block_type = int(sdf.loc[anchor, "block_type"])
            rows.append(
                {
                    "subject": sdf.loc[anchor, "subject"],
                    "session": session,
                    "block_id": f"{session}_B{block_number}",
                    "sequence_id": f"{session}_B{block_number}_T{anchor - start}",
                    "block_context": "Short block" if block_type == 1 else "Long block",
                    "anchor_position": anchor - start,
                    "train_start_position": anchor - start - 2,
                    "train_end_position": anchor - start + 2,
                    "block_length": stop - start,
                    **{key: float(value) for key, value in values.items()},
                }
            )
    return pd.DataFrame(rows)


def collect_consecutive_common_events(data, adjusted_block_start):
    """Collect target performance for the second and third consecutive common trials."""
    rows = []
    for session, sdf, block_number, start, stop in _adjusted_blocks(data, adjusted_block_start):
        common_run_position = 0
        for anchor in range(start, stop):
            block_type = int(sdf.loc[anchor, "block_type"])
            block_context = "Short block" if block_type == 1 else "Long block"
            if bool(sdf.loc[anchor, "is_rare"]):
                common_run_position = 0
                continue
            common_run_position += 1
            if common_run_position not in [2, 3]:
                continue
            value = sdf.loc[anchor, "correct"]
            if pd.isna(value):
                continue
            trace = "two_common" if common_run_position == 2 else "three_common"
            rows.append(
                {
                    "subject": sdf.loc[anchor, "subject"], "session": session,
                    "block_id": f"{session}_B{block_number}",
                    "sequence_id": f"{session}_B{block_number}_{trace}_{anchor - start}",
                    "block_context": block_context, "trace": trace,
                    "performance": float(value), "anchor_position": anchor - start,
                    "train_start_position": anchor - start - (common_run_position - 1),
                    "train_end_position": anchor - start, "block_length": stop - start,
                    "common_run_position": common_run_position,
                }
            )
    return pd.DataFrame(rows)


def collect_previous_outcome_common_events(data, adjusted_block_start):
    """Collect a common trial conditional on rewarded/punished preceding common trial."""
    rows = []
    for session, sdf, block_number, start, stop in _adjusted_blocks(data, adjusted_block_start):
        for anchor in range(start + 1, stop):
            block_type = int(sdf.loc[anchor, "block_type"])
            block_context = "Short block" if block_type == 1 else "Long block"
            pair = sdf.loc[[anchor - 1, anchor]]
            if pair["is_rare"].astype(bool).tolist() != [False, False]:
                continue
            previous_correct = sdf.loc[anchor - 1, "correct"]
            target_correct = sdf.loc[anchor, "correct"]
            if pd.isna(previous_correct) or pd.isna(target_correct):
                continue
            trace = "rewarded_common" if float(previous_correct) == 1.0 else "punished_common"
            rows.append(
                {
                    "subject": sdf.loc[anchor, "subject"], "session": session,
                    "block_id": f"{session}_B{block_number}",
                    "sequence_id": f"{session}_B{block_number}_{trace}_{anchor - start}",
                    "block_context": block_context, "trace": trace,
                    "performance": float(target_correct), "anchor_position": anchor - start,
                    "train_start_position": anchor - start - 1,
                    "train_end_position": anchor - start, "block_length": stop - start,
                }
            )
    return pd.DataFrame(rows)


def collect_complete_train_sliding_events(events, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """Place events in windows only when both window and full train stay in one block."""
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for _, event in events.iterrows():
        block_length = int(event["block_length"])
        train_start = int(event["train_start_position"])
        train_end = int(event["train_end_position"])
        for window_start in range(0, block_length - window_size + 1, step):
            window_stop = window_start + window_size
            window_center = window_start + (window_size + 1) / 2
            if window_center > MAX_WINDOW_CENTER:
                continue
            if train_start < window_start or train_end >= window_stop:
                continue
            rows.append(
                {
                    **event.to_dict(), "window_start": window_start, "window_stop": window_stop,
                    "window_center": window_center, "window_number": window_start // step + 1,
                }
            )
    result = pd.DataFrame(rows)
    validate_sliding_window_boundaries(result)
    return result


def validate_sliding_window_boundaries(events):
    """Raise if any window or qualifying train crosses its adjusted block boundary."""
    if events.empty:
        return
    bad = events[
        events["window_start"].lt(0)
        | events["window_stop"].gt(events["block_length"])
        | events["train_start_position"].lt(events["window_start"])
        | events["train_end_position"].ge(events["window_stop"])
    ]
    if not bad.empty:
        raise AssertionError(
            "Sliding window or qualifying sequence crossed an adjusted block boundary: "
            f"{bad[['block_id', 'window_start', 'window_stop']].head().to_dict('records')}"
        )


def _mean_sem(values):
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return np.nan, np.nan, 0
    return float(values.mean()), float(values.sem()) if len(values) > 1 else 0.0, len(values)


def _unit_values(events, value_column, mode):
    if events.empty:
        return pd.Series(dtype=float)
    if mode == "pooled":
        return events[value_column].dropna().astype(float)
    return events.groupby("session")[value_column].mean().dropna().astype(float)


def _safe_wilcoxon(left, right):
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2:
        return np.nan
    if np.allclose(paired["left"] - paired["right"], 0):
        return 1.0
    try:
        return float(wilcoxon(paired["left"], paired["right"]).pvalue)
    except ValueError:
        return np.nan


def _within_event_pvalue(events, left_column, right_column, mode):
    if events.empty:
        return np.nan
    if mode == "pooled":
        paired = events[[left_column, right_column]].dropna()
    else:
        paired = events.groupby("session")[[left_column, right_column]].mean().dropna()
    return _safe_wilcoxon(paired[left_column], paired[right_column])


def _paired_binary_pvalue(events, left_column, right_column, mode):
    """Exact McNemar/binomial test for events; paired Wilcoxon for session means."""
    if events.empty:
        return np.nan
    if mode != "pooled":
        paired = events.groupby("session")[[left_column, right_column]].mean().dropna()
        return _safe_wilcoxon(paired[left_column], paired[right_column])
    paired = events[[left_column, right_column]].dropna()
    left_only = int(((paired[left_column] == 1) & (paired[right_column] == 0)).sum())
    right_only = int(((paired[left_column] == 0) & (paired[right_column] == 1)).sum())
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    return float(binomtest(left_only, n=discordant, p=0.5, alternative="two-sided").pvalue)


def _between_epoch_pvalue(events, value_column, left_epoch, right_epoch, mode, binary=False):
    left = events[events["epoch"].eq(left_epoch)]
    right = events[events["epoch"].eq(right_epoch)]
    if mode == "pooled":
        left_values = left[value_column].dropna()
        right_values = right[value_column].dropna()
        if left_values.empty or right_values.empty:
            return np.nan
        if binary:
            table = [
                [int(left_values.sum()), int(len(left_values) - left_values.sum())],
                [int(right_values.sum()), int(len(right_values) - right_values.sum())],
            ]
            return float(fisher_exact(table, alternative="two-sided").pvalue)
        return float(mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue)
    paired = pd.concat(
        [
            left.groupby("session")[value_column].mean().rename("left"),
            right.groupby("session")[value_column].mean().rename("right"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    return _safe_wilcoxon(paired["left"], paired["right"])


def _sig_label(p):
    if not np.isfinite(p):
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


def _bracket(ax, x1, x2, y, p, height=0.025, fontsize=6):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="0.15", lw=0.75)
    ax.text((x1 + x2) / 2, y + height + 0.008, _sig_label(p), ha="center", fontsize=fontsize)


def _style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_epoch_phase_panel(ax, events, mode, title):
    phase_offsets = {"pre": -0.18, "rare": 0.0, "post": 0.18}
    epoch_offsets = {"Epoch 1": -0.09, "Epoch 2": 0.0, "Epoch 3": 0.09}

    # First three ticks: paired pre/rare/post performance within each epoch.
    for epoch_index, epoch in enumerate(EPOCHS):
        epoch_events = events[events["epoch"].eq(epoch)]
        for phase in PHASES:
            mean, sem, _ = _mean_sem(_unit_values(epoch_events, phase, mode))
            ax.errorbar(
                epoch_index + phase_offsets[phase], mean, yerr=sem, fmt="o",
                color=PHASE_COLORS[phase], markersize=4.5, capsize=2.5, elinewidth=1.0,
            )
        comparisons = [("pre", "rare"), ("rare", "post"), ("pre", "post")]
        for level, (left, right) in enumerate(comparisons):
            _bracket(
                ax, epoch_index + phase_offsets[left], epoch_index + phase_offsets[right],
                1.03 + level * 0.12,
                _paired_binary_pvalue(epoch_events, left, right, mode),
            )

    # Fourth tick: three mini-clusters (Pre, Rare, Post), each comparing all epochs.
    cluster_centers = {"pre": 3.62, "rare": 4.0, "post": 4.38}
    epoch_pairs = [("Epoch 1", "Epoch 2"), ("Epoch 2", "Epoch 3"), ("Epoch 1", "Epoch 3")]
    for phase in PHASES:
        center = cluster_centers[phase]
        for epoch in EPOCHS:
            epoch_events = events[events["epoch"].eq(epoch)]
            mean, sem, _ = _mean_sem(_unit_values(epoch_events, phase, mode))
            ax.errorbar(
                center + epoch_offsets[epoch], mean, yerr=sem, fmt="o",
                color=EPOCH_COLORS[epoch], markersize=4.2, capsize=2.2, elinewidth=1.0,
            )
        for level, (left_epoch, right_epoch) in enumerate(epoch_pairs):
            _bracket(
                ax, center + epoch_offsets[left_epoch], center + epoch_offsets[right_epoch],
                1.03 + level * 0.12,
                _between_epoch_pvalue(events, phase, left_epoch, right_epoch, mode, binary=True),
                height=0.02, fontsize=5.5,
            )
        ax.text(center, -0.09, PHASE_LABELS[phase], ha="center", va="top", fontsize=6)

    ax.set_xlim(-0.45, 4.65)
    ax.set_xticks([0, 1, 2, 4], ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch comparison"])
    ax.set_ylabel("Performance (fraction correct)")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=PHASE_COLORS[p], label=PHASE_LABELS[p])
        for p in PHASES
    ] + [
        Line2D([0], [0], marker="o", linestyle="none", color=EPOCH_COLORS[e], label=e)
        for e in EPOCHS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.5, ncol=2, loc="lower left")


def plot_epoch_contrast_panel(ax, events, mode, title):
    contrast_offsets = {"drop": -0.13, "recover": 0.13}
    epoch_offsets = {"Epoch 1": -0.10, "Epoch 2": 0.0, "Epoch 3": 0.10}

    # Epoch ticks: compare the two requested contrasts within the same sequences.
    for epoch_index, epoch in enumerate(EPOCHS):
        epoch_events = events[events["epoch"].eq(epoch)]
        for contrast in ["drop", "recover"]:
            mean, sem, _ = _mean_sem(_unit_values(epoch_events, contrast, mode))
            ax.errorbar(
                epoch_index + contrast_offsets[contrast], mean, yerr=sem, fmt="o",
                color=CONTRAST_COLORS[contrast], markersize=5, capsize=3, elinewidth=1.1,
            )
        _bracket(
            ax, epoch_index + contrast_offsets["drop"], epoch_index + contrast_offsets["recover"],
            0.79, _within_event_pvalue(epoch_events, "drop", "recover", mode),
        )

    # Final two ticks: compare all three epochs for each contrast.
    epoch_pairs = [("Epoch 1", "Epoch 2"), ("Epoch 2", "Epoch 3"), ("Epoch 1", "Epoch 3")]
    for center, contrast in [(3.35, "drop"), (4.35, "recover")]:
        for epoch in EPOCHS:
            selected = events[events["epoch"].eq(epoch)]
            mean, sem, _ = _mean_sem(_unit_values(selected, contrast, mode))
            ax.errorbar(
                center + epoch_offsets[epoch], mean, yerr=sem, fmt="o",
                color=EPOCH_COLORS[epoch], markersize=4.5, capsize=2.5, elinewidth=1.0,
            )
        for level, (left_epoch, right_epoch) in enumerate(epoch_pairs):
            _bracket(
                ax, center + epoch_offsets[left_epoch], center + epoch_offsets[right_epoch],
                0.72 + level * 0.18,
                _between_epoch_pvalue(events, contrast, left_epoch, right_epoch, mode),
                height=0.025, fontsize=5.5,
            )

    ax.axhline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set_xlim(-0.45, 4.75)
    ax.set_xticks(
        [0, 1, 2, 3.35, 4.35],
        ["Epoch 1", "Epoch 2", "Epoch 3", "Drop across\nepochs", "Recovery across\nepochs"],
    )
    ax.set_ylabel("Performance contrast")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=CONTRAST_COLORS[c], label=CONTRAST_LABELS[c])
        for c in ["drop", "recover"]
    ] + [
        Line2D([0], [0], marker="o", linestyle="none", color=EPOCH_COLORS[e], label=e)
        for e in EPOCHS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.5, ncol=2, loc="lower left")


def summarize_sliding_windows(events, mode):
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for window_center in sorted(events["window_center"].unique()):
        window_events = events[events["window_center"].eq(window_center)]
        for phase in PHASES:
            mean, sem, n = _mean_sem(_unit_values(window_events, phase, mode))
            rows.append(
                {"window_center": window_center, "phase": phase, "mean": mean, "sem": sem, "n": n}
            )
    return pd.DataFrame(rows)


def plot_sliding_window_panel(ax, events, mode, title):
    summary = summarize_sliding_windows(events, mode)
    for phase in PHASES:
        trace = summary[summary["phase"].eq(phase)] if not summary.empty else pd.DataFrame()
        if trace.empty:
            continue
        ax.errorbar(
            trace["window_center"], trace["mean"], yerr=trace["sem"],
            marker="o", markersize=3.8, capsize=2, linewidth=1.5, elinewidth=0.9,
            color=PHASE_COLORS[phase], alpha=0.52, label=PHASE_LABELS[phase],
        )
    ax.set_xlim(2.5, MAX_WINDOW_CENTER + 0.5)
    ax.set_xlabel("Seven-trial window center within adjusted block\n(step = 3 trials)")
    ax.set_ylabel("Performance (fraction correct)")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=6, loc="best")


def summarize_pre_rare_windows(events, mode):
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for window_center in sorted(events["window_center"].unique()):
        window_events = events[events["window_center"].eq(window_center)]
        for phase in ["pre", "rare"]:
            mean, sem, n = _mean_sem(_unit_values(window_events, phase, mode))
            rows.append(
                {"window_center": window_center, "phase": phase, "mean": mean, "sem": sem, "n": n}
            )
    return pd.DataFrame(rows)


def plot_pre_rare_sliding_panel(ax, events, mode, title):
    summary = summarize_pre_rare_windows(events, mode)
    phase_offsets = {"pre": -0.22, "rare": 0.22}

    # Connect the paired pre-rare and rare means inside every window.
    if not summary.empty:
        paired = summary.pivot(index="window_center", columns="phase", values="mean")
        for window_center, row in paired.dropna(subset=["pre", "rare"]).iterrows():
            ax.plot(
                [window_center + phase_offsets["pre"], window_center + phase_offsets["rare"]],
                [row["pre"], row["rare"]],
                color="0.35", linewidth=1.0, alpha=0.28, zorder=1,
            )

    # Light phase traces make changes across successive windows readable as well.
    for phase in ["pre", "rare"]:
        trace = summary[summary["phase"].eq(phase)].sort_values("window_center") \
            if not summary.empty else pd.DataFrame()
        if trace.empty:
            continue
        x = trace["window_center"] + phase_offsets[phase]
        ax.errorbar(
            x, trace["mean"], yerr=trace["sem"], marker="o", markersize=4.2,
            capsize=2.2, linewidth=1.3, elinewidth=0.9,
            color=PHASE_COLORS[phase], alpha=0.68, label=PHASE_LABELS[phase], zorder=2,
        )

    ax.set_xlim(2.5, MAX_WINDOW_CENTER + 0.5)
    ax.set_xlabel("Seven-trial window center within adjusted block\n(step = 3 trials; rare = trial 0)")
    ax.set_ylabel("Performance (fraction correct)")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=6, loc="best")


def _summarize_wide_window_traces(events, value_columns, mode):
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for window_center in sorted(events["window_center"].unique()):
        window_events = events[events["window_center"].eq(window_center)]
        for value_column in value_columns:
            mean, sem, n = _mean_sem(_unit_values(window_events, value_column, mode))
            rows.append(
                {"window_center": window_center, "trace": value_column, "mean": mean, "sem": sem, "n": n}
            )
    return pd.DataFrame(rows)


def plot_double_rare_sliding_panel(ax, events, mode, title):
    trace_order = ["pre_common", "rare_1", "rare_2", "post_common"]
    labels = {
        "pre_common": "Pre common (−1)", "rare_1": "Rare 1 (0)",
        "rare_2": "Rare 2 (+1)", "post_common": "Post common (+2)",
    }
    colors = {
        "pre_common": "#0072B2", "rare_1": "#D55E00",
        "rare_2": "#CC79A7", "post_common": "#009E73",
    }
    summary = _summarize_wide_window_traces(events, trace_order, mode)
    for trace_name in trace_order:
        trace = summary[summary["trace"].eq(trace_name)] if not summary.empty else pd.DataFrame()
        if trace.empty:
            continue
        ax.errorbar(
            trace["window_center"], trace["mean"], yerr=trace["sem"], marker="o",
            markersize=3.8, capsize=2, linewidth=1.35, elinewidth=0.85,
            color=colors[trace_name], alpha=0.58, label=labels[trace_name],
        )
    ax.set_xlim(2.5, MAX_WINDOW_CENTER + 0.5)
    ax.set_xlabel("Seven-trial window center within adjusted block\n(step = 3; first rare = trial 0)")
    ax.set_ylabel("Performance (fraction correct)")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=5.5, loc="best")


def _summarize_long_window_traces(events, mode):
    rows = []
    if events.empty:
        return pd.DataFrame(rows)
    for (window_center, trace_name), selected in events.groupby(
        ["window_center", "trace"], sort=True
    ):
        mean, sem, n = _mean_sem(_unit_values(selected, "performance", mode))
        rows.append(
            {"window_center": window_center, "trace": trace_name, "mean": mean, "sem": sem, "n": n}
        )
    return pd.DataFrame(rows)


def _plot_long_trace_panel(ax, events, mode, title, trace_specs, page_label):
    summary = _summarize_long_window_traces(events, mode)
    for trace_name, label, color, marker in trace_specs:
        trace = summary[summary["trace"].eq(trace_name)] if not summary.empty else pd.DataFrame()
        if trace.empty:
            continue
        ax.errorbar(
            trace["window_center"], trace["mean"], yerr=trace["sem"], marker=marker,
            markersize=4, capsize=2.2, linewidth=1.4, elinewidth=0.9,
            color=color, alpha=0.62, label=label,
        )
    ax.set_xlim(2.5, MAX_WINDOW_CENTER + 0.5)
    ax.set_xlabel(f"Seven-trial window center within adjusted block\n(step = 3; {page_label})")
    ax.set_ylabel("Performance (fraction correct)")
    ax.set_title(title, fontsize=9)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=6, loc="best")


def plot_consecutive_common_sliding_panel(ax, events, mode, title):
    specs = [
        ("two_common", "2 consecutive: second common", "#0072B2", "o"),
        ("three_common", "3 consecutive: third common", "#CC79A7", "s"),
    ]
    _plot_long_trace_panel(ax, events, mode, title, specs, "target = final common")


def plot_previous_outcome_common_sliding_panel(ax, events, mode, title):
    specs = [
        ("rewarded_common", "Previous common rewarded", "#009E73", "o"),
        ("punished_common", "Previous common punished", "#D55E00", "s"),
    ]
    _plot_long_trace_panel(ax, events, mode, title, specs, "performance = second common")


def _six_panel_specs(row_events, plot_function, fig, gs, row):
    short_events = row_events[row_events["block_context"].eq("Short block")]
    long_events = row_events[row_events["block_context"].eq("Long block")]
    return [
        (row_events, "pooled", "pooled events"),
        (row_events, "grand", "grand average across sessions"),
        (short_events, "pooled", "short block — pooled events"),
        (short_events, "grand", "short block — grand average"),
        (long_events, "pooled", "long block — pooled events"),
        (long_events, "grand", "long block — grand average"),
    ]


def _build_page(events, subject_rows, plot_function, title):
    fig = plt.figure(figsize=(66, 27))
    gs = gridspec.GridSpec(5, 6, figure=fig, hspace=0.60, wspace=0.52)
    for row, (row_label, subject) in enumerate(subject_rows):
        row_events = events if subject is None else events[events["subject"].eq(subject)].copy()
        for column, (panel_events, mode, suffix) in enumerate(
            _six_panel_specs(row_events, plot_function, fig, gs, row)
        ):
            plot_function(fig.add_subplot(gs[row, column]), panel_events, mode, f"{row_label} — {suffix}")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.972, bottom=0.055)
    return fig


def make_extended_figure(
    original_page_builder,
    change_events,
    data,
    adjusted_block_start,
    subject_rows,
    output_path="block_segment_performance_changes.pdf",
    show=True,
):
    sequence_events = collect_sequence_events(data, adjusted_block_start)
    window_events = collect_sliding_window_events(sequence_events)
    pre_rare_events = collect_pre_rare_sequence_events(data, adjusted_block_start)
    pre_rare_window_events = collect_pre_rare_sliding_window_events(pre_rare_events)
    double_rare_events = collect_common_common_rare_rare_common_events(data, adjusted_block_start)
    double_rare_window_events = collect_complete_train_sliding_events(double_rare_events)
    consecutive_common_events = collect_consecutive_common_events(data, adjusted_block_start)
    consecutive_common_window_events = collect_complete_train_sliding_events(consecutive_common_events)
    previous_outcome_events = collect_previous_outcome_common_events(data, adjusted_block_start)
    previous_outcome_window_events = collect_complete_train_sliding_events(previous_outcome_events)

    # Revalidate every sliding dataset together before any page is rendered.
    for sliding_events in [
        window_events, pre_rare_window_events, double_rare_window_events,
        consecutive_common_window_events, previous_outcome_window_events,
    ]:
        validate_sliding_window_boundaries(sliding_events)

    original_page = original_page_builder(change_events)
    pre_rare_page = _build_page(
        pre_rare_window_events,
        subject_rows,
        plot_pre_rare_sliding_panel,
        "Common/common/rare sequences: pre-rare and rare performance in seven-trial sliding windows",
    )
    epoch_page = _build_page(
        sequence_events,
        subject_rows,
        plot_epoch_phase_panel,
        "Common/common/rare/common sequences: pre, rare, and post performance by adjusted-block epoch",
    )
    contrast_page = _build_page(
        sequence_events,
        subject_rows,
        plot_epoch_contrast_panel,
        "Sequence performance contrasts by adjusted-block epoch",
    )
    sliding_page = _build_page(
        window_events,
        subject_rows,
        plot_sliding_window_panel,
        "Common/common/rare/common performance in seven-trial sliding windows",
    )
    double_rare_page = _build_page(
        double_rare_window_events,
        subject_rows,
        plot_double_rare_sliding_panel,
        "Common/common/rare/rare/common sequences in seven-trial sliding windows",
    )
    consecutive_common_page = _build_page(
        consecutive_common_window_events,
        subject_rows,
        plot_consecutive_common_sliding_panel,
        "Second and third consecutive-common performance in seven-trial sliding windows",
    )
    previous_outcome_page = _build_page(
        previous_outcome_window_events,
        subject_rows,
        plot_previous_outcome_common_sliding_panel,
        "Common-trial performance after rewarded versus punished common trials",
    )

    output_path = Path(output_path)
    pages = [
        original_page, pre_rare_page, epoch_page, contrast_page, sliding_page,
        double_rare_page, consecutive_common_page, previous_outcome_page,
    ]
    with PdfPages(output_path) as pdf:
        for page in pages:
            pdf.savefig(page, bbox_inches="tight")
    if show:
        plt.show()
    else:
        for page in pages:
            plt.close(page)
    return (
        output_path, sequence_events, window_events, pre_rare_events, pre_rare_window_events,
        double_rare_events, double_rare_window_events,
        consecutive_common_events, consecutive_common_window_events,
        previous_outcome_events, previous_outcome_window_events,
    )
