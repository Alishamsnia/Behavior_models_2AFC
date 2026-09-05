"""Write a short Markdown summary tying the model-comparison tables together."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import numpy as np

from post_analysis import common
from post_analysis.common import MODEL_ORDER, MODEL_SPECS
from post_analysis.model_comparison import stats as mc_stats
from post_analysis.model_comparison.collect import (
    common_event_session_ids,
    common_event_window_ids,
    common_session_ids,
    PSYCHOMETRIC_BLOCK_LABELS,
    PSYCHOMETRIC_BLOCK_ORDER,
    TRIAL_SUBSET_LABELS,
    TRIAL_SUBSET_ORDER,
)


def _fmt(x, digits=4):
    if pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_p(p, digits=4):
    """Format a p-value, switching to scientific notation below the display
    precision so a tiny p (e.g. 1.9e-21) reads as tiny rather than rounding
    to "0.0000" and looking like exactly zero.
    """
    if pd.isna(p):
        return "n/a"
    if p != 0 and abs(p) < 10 ** (-digits):
        return f"{p:.2e}"
    return f"{p:.{digits}f}"


def _append_adjacent_results(
    lines: list[str], table: pd.DataFrame, filter_col: str,
    groups: list[tuple[str, str]],
) -> None:
    """Append compact Holm-Wilcoxon results for declared adjacent pairs."""
    lines.append("| Population | Pair | N | Median Δ (first−second) | Holm p | Result |")
    lines.append("|---|---|---:|---:|---:|---|")
    for filter_value, label in groups:
        subset = table[table[filter_col] == filter_value]
        for _, row in subset.iterrows():
            p = row["wilcoxon_p_holm"]
            if not np.isfinite(p):
                result = "unavailable"
            else:
                difference = float(row["median_diff_a_minus_b"])
                if difference > 0:
                    favored = row["model_b"]
                elif difference < 0:
                    favored = row["model_a"]
                else:
                    favored = "neither (paired median tie)"
                result = (
                    f"{favored} lower NLL" if p < 0.05
                    else f"not significant (paired median favors {favored})"
                )
            lines.append(
                f"| {label} | {row['model_a']}–{row['model_b']} | {int(row['n_pairs'])} | "
                f"{_fmt(row['median_diff_a_minus_b'])} | {_fmt_p(p)} | {result} |"
            )
    lines.append("")


def _append_trial_subset_matrix(
    lines: list[str], table: pd.DataFrame, row_filter_col: str, row_groups: list[tuple[str, str]],
) -> None:
    """Compact rows=adjacent-pair x columns=trial-subset table (significance
    stars + Holm p) for the Figure 01/03 trial-subset grid, one table per
    ``row_groups`` entry. Looping ``_append_adjacent_results`` once per
    subset instead would produce 7 subsets x 3-or-2 panels x 4 pairs = 84/56
    rows, too dense to read; this collapses each panel to one small table.
    """
    for filter_value, label in row_groups:
        lines.append(f"**{label}:**\n")
        lines.append("| Pair | " + " | ".join(TRIAL_SUBSET_LABELS[s] for s in TRIAL_SUBSET_ORDER) + " |")
        lines.append("|---|" + "---:|" * len(TRIAL_SUBSET_ORDER))
        panel = table[table[row_filter_col] == filter_value]
        for model_a, model_b in mc_stats.ADJACENT_MODEL_PAIRS:
            cells = []
            for trial_subset in TRIAL_SUBSET_ORDER:
                row = panel[
                    (panel["trial_subset"] == trial_subset)
                    & (panel["model_a"] == model_a) & (panel["model_b"] == model_b)
                ]
                if row.empty:
                    cells.append("n/a")
                    continue
                p = row.iloc[0]["wilcoxon_p_holm"]
                cells.append(f"{mc_stats.significance_stars(p)} ({_fmt_p(p, 3)})")
            lines.append(f"| {model_a}–{model_b} | " + " | ".join(cells) + " |")
        lines.append("")


def build_report(mouse_id: str) -> str:
    tables_dir = common.POST_ANALYSIS_ROOT / "model_comparison" / mouse_id / "tables"

    sw_global = pd.read_csv(tables_dir / "sliding_window_global_pooled_metrics.csv")
    ss_avail = pd.read_csv(tables_dir / "single_session_availability.csv")
    win_summary = pd.read_csv(tables_dir / "single_session_win_counts_summary.csv")
    sw_win_summary = pd.read_csv(tables_dir / "sliding_window_win_counts_summary.csv")
    ss_metrics = pd.read_csv(tables_dir / "single_session_metrics.csv")
    phase_summary = pd.read_csv(tables_dir / "chronological_winner_phase_summary.csv")
    sw_train_win_summary = pd.read_csv(tables_dir / "sliding_window_train_win_counts_summary.csv")
    ss_train_win_summary = pd.read_csv(tables_dir / "single_session_train_win_counts_summary.csv")
    agreement_summary = pd.read_csv(tables_dir / "chronological_train_vs_test_winner_agreement_summary.csv")
    sw_paired_pairwise = pd.read_csv(tables_dir / "sliding_window_paired_pairwise.csv")
    ss_paired_pairwise = pd.read_csv(tables_dir / "single_session_paired_pairwise.csv")
    ic_summary = pd.read_csv(tables_dir / "information_criteria_summary.csv")
    ss_ic_summary = pd.read_csv(tables_dir / "information_criteria_single_session_summary.csv")
    ic_paired_pairwise = pd.read_csv(tables_dir / "information_criteria_paired_pairwise.csv")
    sw_params = pd.read_csv(tables_dir / "sliding_window_parameters.csv")
    sw_trial_subset_paired = pd.read_csv(tables_dir / "sliding_window_trial_subset_paired_pairwise.csv")
    ss_trial_subset_paired = pd.read_csv(tables_dir / "single_session_trial_subset_paired_pairwise.csv")
    sw_event_aligned = pd.read_csv(tables_dir / "sliding_window_event_aligned_nll.csv")
    ss_event_aligned = pd.read_csv(tables_dir / "single_session_event_aligned_nll.csv")
    sw_event_paired = pd.read_csv(tables_dir / "sliding_window_event_aligned_paired_pairwise.csv")
    sw_event_trial0_paired = pd.read_csv(
        tables_dir / "sliding_window_event_aligned_trial0_paired_pairwise.csv"
    )
    ss_event_paired = pd.read_csv(tables_dir / "single_session_event_aligned_paired_pairwise.csv")
    psychometric_summary = pd.read_csv(tables_dir / "psychometric_comparison_summary.csv")

    lines = []
    lines.append(f"# Model comparison (M0-M4) — {mouse_id}\n")
    lines.append(
        "Generated by `post_analysis/model_comparison`. Reads only files under "
        "`results_sliding_window/` and `results_single_session/`; writes nothing "
        "outside `results_post_analysis/`.\n"
    )

    n_common = int(sw_global["common_support_window_count"].iloc[0]) if len(sw_global) else 0
    lines.append("## 1. Sliding-window rolling evaluation (32 windows, 9 train / 3 val / 1 test sessions)\n")
    lines.append(
        f"**All five models are pooled over the same {n_common} common windows here** "
        "(the intersection of windows every model reached; bottlenecked by M4's 5 "
        "non-converged windows). This matters: pooling each model over its own "
        "available windows instead (32 for M0-M3, 27 for M4) makes M4 look like the "
        "best test-split model, but that is an artifact of M4 only being evaluated on "
        "an easier/smaller window subset -- on the common 27 windows M1 is (marginally) "
        "best, not M4. `source_window_count` in the underlying table records each "
        "model's own full reach for reference; it is not what these numbers use.\n"
    )
    lines.append(
        "`test` is a true walk-forward evaluation: each rolling window contributes "
        "exactly one held-out session, so these numbers have no session duplication. "
        "`validation` pools 3 sessions per window and `train` pools 9 sessions per "
        "window, so both are pooled over repeated window-session analysis units "
        "(each session enters once per window it belonged to that split), per the "
        "pipeline's own stated pooling policy.\n"
    )
    lines.append("| Model | Split | N trials | Mean NLL | ELPD | Brier | Threshold acc. | Common windows pooled |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model_id in MODEL_ORDER:
        subset = sw_global[sw_global["model_id"] == model_id]
        for split_role in ["train", "validation", "test"]:
            row = subset[subset["split_role"] == split_role]
            if row.empty:
                lines.append(f"| {model_id} | {split_role} | n/a | n/a | n/a | n/a | n/a | 0 |")
                continue
            row = row.iloc[0]
            lines.append(
                f"| {model_id} | {split_role} | {int(row['valid_trial_count'])} | "
                f"{_fmt(row['mean_nll'])} | {_fmt(row['elpd'], 1)} | {_fmt(row['brier_score'])} | "
                f"{_fmt(row['model_threshold_accuracy'])} | {int(row['common_support_window_count'])} |"
            )
    lines.append("")
    lines.append(
        "Per-window detailed and global (ALL_SESSIONS) numbers, including the "
        "`train` split, are in `tables/sliding_window_split_session_metrics.csv`; "
        "see `figures/fig01_sliding_window_global_and_detailed_nll.png` (a 3x7 grid: "
        "train/validation/test rows by all-trials/short-block/long-block/common-short/"
        "common-long/rare-short/rare-long columns -- `tables/sliding_window_trial_subset_metrics.csv` "
        "is the underlying per-subset table) and `figures/fig02_sliding_window_nll_trajectory.png` "
        "(three rows, same order).\n"
    )

    lines.append("## 2. Single-session fine-tuning: session availability\n")
    lines.append(
        "Each model attempts fine-tuning on the sessions matched to the "
        "sliding-window test sessions (chronological ranks 13-44; M0-M3 reach "
        "all 32, M4 only reaches 27 because 5 of its rolling windows failed to "
        "converge and so produced no matched initialization -- see Section 1). "
        "Every attempted session has a real fitted parameter vector and a "
        "computed held-out NLL and is counted as valid here. The two flagged "
        "columns are diagnostics from the pipeline's own per-run bookkeeping "
        "(a plain L-BFGS convergence check for every model, plus M4's stricter "
        "gate -- projection activity and run-length tail occupancy -- used to "
        "decide what M4 itself folds into its pooled psychometric figures); "
        "they are **not** subtracted from `n_sessions_valid` here, since flagged "
        "sessions still carry usable fitted parameters and NLL.\n"
    )
    lines.append(
        "| Model | Matched windows | Attempted | **Valid (used below)** | "
        "Flagged not converged | Flagged ineligible by pipeline gate | Never attempted |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in ss_avail.iterrows():
        lines.append(
            f"| {row['model_id']} | {int(row['n_matched_windows_expected'])} | "
            f"{int(row['n_sessions_attempted'])} | **{int(row['n_sessions_valid'])}** | "
            f"{int(row['n_sessions_flagged_not_converged'])} | "
            f"{int(row['n_sessions_flagged_ineligible_by_pipeline_gate'])} | "
            f"{int(row['n_sessions_never_attempted'])} |"
        )
    lines.append("")
    lines.append("See `figures/fig10_single_session_availability.png`.\n")

    lines.append("## 3. Single-session held-out (last-two-block) NLL by model\n")
    common_ss_ids = common_session_ids(ss_metrics)
    n_ss_total = ss_metrics["session_id"].nunique()
    lines.append(
        f"**Restricted to the {len(common_ss_ids)} of {n_ss_total} matched sessions every "
        "model reached** (bottlenecked by M4 -- see Section 2), so every model in the table "
        "below and in `figures/fig03_single_session_heldout_nll.png` (a 2x7 grid: "
        "fine-tune-prefix/held-out rows by the same 7 trial-subset columns as Figure 01 -- "
        "`tables/single_session_trial_subset_metrics.csv` is the underlying per-subset table) "
        "is compared on the identical session population per (role, subset). Previously this "
        "table (and fig03) used each model's own available N (32 for M0-M3, 27 for M4); a "
        "wider M0-M3 sample was not itself evidence of a fairer comparison, so it has been "
        "fixed to match.\n"
    )
    heldout = ss_metrics[
        (ss_metrics["analysis_role"] == "heldout") & (ss_metrics["mean_nll"].notna())
        & (ss_metrics["session_id"].isin(common_ss_ids))
    ]
    lines.append("| Model | N sessions | Mean of session mean-NLL | Median | Std |")
    lines.append("|---|---:|---:|---:|---:|")
    for model_id in MODEL_ORDER:
        values = heldout.loc[heldout["model_id"] == model_id, "mean_nll"]
        if values.empty:
            lines.append(f"| {model_id} | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {model_id} | {len(values)} | {_fmt(values.mean())} | "
            f"{_fmt(values.median())} | {_fmt(values.std())} |"
        )
    lines.append("")

    lines.append("### 3b. Paired significance tests across models\n")
    lines.append(
        "Every comparison is matched by exact `window_id` or `session_id`. The four "
        "scientifically pre-specified nested-model contrasts are **M0–M1, M1–M2, "
        "M2–M3, and M3–M4**. Two-sided Wilcoxon signed-rank p-values are Holm-corrected "
        "over exactly those four contrasts separately within each panel; they are not "
        "gated by a data-dependent omnibus test. Paired t-tests remain in the CSVs as a "
        "parametric cross-check but do not drive the figure stars.\n"
    )
    lines.append(
        "Figure 01 now plots only the same complete common windows for every model in all "
        "three panels. Rolling fits overlap in their training history, and train/validation "
        "also reuse evaluated sessions; those p-values are therefore exploratory within-mouse "
        "summaries, not independent-replicate population inference. Figure 03 uses the same "
        "common session IDs in both panels.\n"
    )
    _append_adjacent_results(
        lines, sw_paired_pairwise, "split_role",
        [("train", "Sliding train"), ("validation", "Sliding validation"), ("test", "Sliding test")],
    )
    _append_adjacent_results(
        lines, ss_paired_pairwise, "analysis_role",
        [("fine_tune_prefix", "Single-session training"), ("heldout", "Single-session held-out")],
    )

    lines.append(
        "**Trial-subset breakdown (Figures 01/03's 7 columns).** Same four adjacent contrasts, "
        "Holm-corrected within each (split/role, subset) subplot independently -- one set of 4 "
        "tests per subplot, not one shared correction across all 7 subsets. Each subplot's own "
        "matched-N population is recomputed per subset (a window/session can have zero "
        "qualifying trials of a given subset, e.g. no rare-long trials in a short test "
        "session); exact N per cell is in the source CSVs and on the figures themselves.\n"
    )
    _append_trial_subset_matrix(
        lines, sw_trial_subset_paired, "split_role",
        [("train", "Sliding train"), ("validation", "Sliding validation"), ("test", "Sliding test")],
    )
    _append_trial_subset_matrix(
        lines, ss_trial_subset_paired, "analysis_role",
        [("fine_tune_prefix", "Single-session training"), ("heldout", "Single-session held-out")],
    )

    lines.append("## 4. Which model is best? Win counts, single-session vs. sliding-window\n")
    n_ss_compared = int(win_summary["n_sessions_compared"].iloc[0]) if len(win_summary) else 0
    n_sw_compared = int(sw_win_summary["n_windows_compared"].iloc[0]) if len(sw_win_summary) else 0
    lines.append(
        "\"Wins\" = lowest held-out mean NLL among the 5 models, counted at two "
        "different levels of analysis:\n"
    )
    lines.append(
        f"- **Single-session** ({n_ss_compared} sessions where all five models have "
        "a computed held-out fit; bottlenecked by M4 reaching only 27 of the 32 "
        "matched sessions -- see Section 2, not an extra quality filter). "
        "See `tables/single_session_win_counts_per_session.csv` and "
        "`figures/fig04_single_session_win_counts.png`."
    )
    lines.append(
        f"- **Sliding-window** ({n_sw_compared} rolling windows where all five "
        "models produced a converged winner; bottlenecked by M4's 5 "
        "non-converged windows -- see Section 1). See "
        "`tables/sliding_window_win_counts_per_window.csv` and "
        "`figures/fig06_sliding_window_win_counts.png`."
    )
    lines.append(
        "\nBoth are shown together in "
        "`figures/fig07_win_counts_single_session_vs_sliding_window.png`. The two "
        "levels disagree in this dataset: sliding-window fits pool many sessions "
        "per parameter vector, favoring the richer M1-M3 models, while "
        "single-session fits estimate a full parameter vector from one short "
        "session, where M0's fewer parameters are harder to overfit with.\n"
    )
    ss_margin = win_summary["median_winning_margin_nll"].iloc[0] if len(win_summary) else float("nan")
    sw_margin = sw_win_summary["median_winning_margin_nll"].iloc[0] if len(sw_win_summary) else float("nan")
    lines.append(
        "**These are descriptive point-estimate argmins, not a significance test.** "
        "Every session/window is won by exactly one model regardless of how close "
        "the runner-up was; the per-unit CSVs (`winning_margin_nll`) record that gap "
        f"so a near-tie is visible rather than hidden -- median margin over the "
        f"runner-up is {_fmt(ss_margin)} NLL at the single-session level and "
        f"{_fmt(sw_margin)} NLL at the sliding-window level (some individual margins "
        "are far smaller, down to ~1e-6 NLL between nested models). Treat the win "
        "counts as descriptive, not as evidence one model is significantly better on "
        "any given unit.\n"
    )
    lines.append("| Model | Sessions won (single-session) | Windows won (sliding-window) |")
    lines.append("|---|---:|---:|")
    for model_id in MODEL_ORDER:
        ss_row = win_summary[win_summary["model_id"] == model_id].iloc[0]
        sw_row = sw_win_summary[sw_win_summary["model_id"] == model_id].iloc[0]
        lines.append(
            f"| {model_id} | {int(ss_row['n_sessions_won'])} | {int(sw_row['n_windows_won'])} |"
        )
    lines.append("")

    lines.append("## 5. Chronological pattern: which model wins over time\n")
    lines.append(
        "Every matched chronological rank (13-44), for both analysis levels, "
        "is plotted as a timeline strip in `figures/fig08_chronological_winner_timeline.png` "
        "(tile opacity = winning margin; hatched = fewer than 5 models comparable "
        "at that rank -- these are ranks 13, 14, 15, 17, 21, all early, so they are "
        "shown explicitly rather than silently dropped). The full per-rank table is "
        "`tables/chronological_winner_timeline.csv`.\n"
    )
    lines.append(
        "**The two analysis levels do not agree on a single \"early vs. late\" "
        "story.** Phases are fixed chronological thirds (not equal-count, since "
        "the missing ranks are concentrated early); `n` is the number of "
        "comparable units in that phase and differs by level and phase, so read "
        "win share alongside it, not alone.\n"
    )
    for level, level_label in [
        ("sliding_window", "Sliding-window (walk-forward test)"),
        ("single_session", "Single-session (fine-tune held-out)"),
    ]:
        lines.append(f"**{level_label}:**\n")
        lines.append("| Phase | n comparable | " + " | ".join(MODEL_ORDER) + " |")
        lines.append("|---|---:|" + "---:|" * len(MODEL_ORDER))
        level_rows = phase_summary[phase_summary["analysis_level"] == level]
        for phase in level_rows["phase"].unique():
            phase_rows = level_rows[level_rows["phase"] == phase].set_index("model_id")
            n_units = int(phase_rows["n_comparable_units_in_phase"].iloc[0])
            cells = " | ".join(
                f"{int(phase_rows.loc[m, 'n_wins'])} ({phase_rows.loc[m, 'win_share']:.0%})"
                for m in MODEL_ORDER
            )
            lines.append(f"| {phase} | {n_units} | {cells} |")
        lines.append("")
    lines.append(
        "Single-session fine-tuning shows M0 rising steadily (33% to 40% to 64% "
        "win share from early to late) -- consistent with a \"simple model wins "
        "once the mouse is well-trained\" story. Sliding-window shows the "
        "opposite for M0 (0% win share early) and instead has M4 (and M1) "
        "pick up wins late. These are genuinely different signals, not two "
        "views of the same fact, because the two analysis levels evaluate "
        "different things (see the caption on fig08) -- report both, not a "
        "single merged narrative.\n"
    )

    lines.append("### 5b. Training-set counterpart: does the best-fitting model generalize?\n")
    lines.append(
        "Fig08/the tables above use held-out NLL (sliding-window test split; "
        "single-session final-two-block heldout). `figures/fig13_chronological_winner_timeline_train.png` "
        "is the exact same layout but using **in-sample training NLL** instead "
        "(sliding-window pooled train split; single-session fine-tune prefix). "
        "This is a fit-quality diagnostic, not a generalization one: a model "
        "with more free parameters can typically fit its own training data at "
        "least as well as a simpler nested model, so training-NLL \"wins\" "
        "skew toward the richest model almost by construction.\n"
    )
    lines.append("| Model | Windows won (train) | Sessions won (train) |")
    lines.append("|---|---:|---:|")
    for model_id in MODEL_ORDER:
        sw_row = sw_train_win_summary[sw_train_win_summary["model_id"] == model_id].iloc[0]
        ss_row = ss_train_win_summary[ss_train_win_summary["model_id"] == model_id].iloc[0]
        lines.append(
            f"| {model_id} | {int(sw_row['n_windows_won'])} | {int(ss_row['n_sessions_won'])} |"
        )
    lines.append("")
    lines.append(
        "As expected, M4 (the richest model, 9 parameters) wins nearly every "
        "unit on training fit alone (sliding-window 23/27, single-session "
        "17/27) -- yet M4 does not dominate the held-out comparison in "
        "Section 4 (7/27 both levels). **The interesting number is how often "
        "the training-fit winner and the held-out winner actually agree** -- "
        "see `figures/fig14_train_vs_test_winner_agreement.png` and "
        "`tables/chronological_train_vs_test_winner_agreement_per_rank.csv`:\n"
    )
    lines.append("| Analysis level | Ranks compared | Agree | Agreement rate |")
    lines.append("|---|---:|---:|---:|")
    for _, row in agreement_summary.iterrows():
        lines.append(
            f"| {row['analysis_level']} | {int(row['n_ranks_compared'])} | "
            f"{int(row['n_agree'])} | {row['agreement_rate']:.0%} |"
        )
    lines.append("")
    lines.append(
        "Both rates (~22-26%) sit close to the 20% chance level for a random "
        "pick among 5 models. In this dataset, **knowing which model best fits "
        "the training data is close to useless for predicting which model "
        "generalizes best** -- consistent with the richer nested models "
        "overfitting the training prefix without a proportional held-out "
        "benefit. This is exactly the kind of result the held-out/test-based "
        "comparisons in Sections 1-5 are designed to protect against, and why "
        "training-set \"wins\" should never be reported as evidence of a "
        "better model on their own.\n"
    )

    lines.append("## 6. Fitted parameters\n")
    lines.append(
        "Full per-window and per-session parameter values (long format: one "
        "row per model/window(or session)/parameter) are in "
        "`tables/sliding_window_parameters.csv` and "
        "`tables/single_session_parameters.csv`. Parameter sets are nested "
        "across models:\n"
    )
    for model_id in MODEL_ORDER:
        spec = MODEL_SPECS[model_id]
        lines.append(f"- **{model_id}** (`{spec.label}`): `{', '.join(spec.param_names)}`")
    lines.append("")
    lines.append(
        "Shared sensory/lapse parameter trajectories across the 32 rolling "
        "windows are in `figures/fig05_shared_parameter_trajectories.png`; the "
        "model-specific coordinates each nested model adds (`ah`, `delta0`, "
        "`alpha_theta`, `bh`) are in "
        "`figures/fig09_sliding_window_model_specific_parameters.png` -- alongside each raw "
        "trajectory and its plain-language explanation, that figure's third column plots the "
        "closed-form quantity the explanation defines (e.g. expected block length instead of raw "
        "`ah`) with a per-model linear-trend overlay. The "
        "single-session (fine-tuned per session) counterparts are "
        "`figures/fig11_single_session_shared_parameter_trajectories.png` and "
        "`figures/fig12_single_session_model_specific_parameters.png` -- both mark "
        "non-converged sessions at reduced opacity rather than dropping them, and "
        "clip their y-axis to a robust range around the converged points so a "
        "handful of wild non-converged outliers (unconstrained parameters can "
        "reach values in the hundreds of thousands from a single short session) "
        "do not flatten the informative range; see the source CSVs for exact "
        "values of anything clipped from view.\n"
    )

    lines.append("## 7. Model comparison via information criteria (AIC/BIC)\n")
    n_sw_ic = int(ic_summary["n_windows_total"].iloc[0]) if len(ic_summary) else 0
    n_ss_ic = int(ss_ic_summary["n_sessions_total"].iloc[0]) if len(ss_ic_summary) else 0
    lines.append(
        f"Figure 15 now contains two analysis levels. Sliding-window AIC/BIC use {n_sw_ic} "
        "converged fits drawn from Figure 01's common rolling population after additionally "
        "requiring the same positive training-trial count across models. Single-session AIC/BIC use "
        "sessions for which **all five** fine-tunes are numerically sound -- finite objective/"
        "parameters/gradients, a stable training-NLL trajectory, and no parameter-transform/"
        "probability-normalization/optimizer-runtime error -- and every model shared the same "
        f"valid training trials. On {mouse_id} this is a {n_ss_ic}-session sensitivity subset.\n"
        "\n"
        "**Reliability note:** this deliberately is *not* each model's own binary `converged` "
        "flag. That flag additionally requires an absolute gradient-norm tolerance (and, for M4, "
        "tail-occupancy/projection thresholds) calibrated against the pooled rolling-window "
        "training set (thousands of trials); applied to a single session's far smaller trial "
        f"count it collapses the jointly-successful population on {mouse_id} to just 5 sessions, "
        "even though most of the excluded fits were only a few-fold over that absolute tolerance "
        "rather than genuinely diverged. Sessions with a real numerical breakdown (an exception "
        "raised mid-optimization, e.g. an invalid posterior/pre-choice delta) still fail the "
        "numerically-sound check here and are excluded either way. In both rows, "
        "`nll_total = -elpd`, `k` is the model's number of free parameters, and `n` is the "
        "number of valid training choices.\n"
    )
    lines.append(
        "**BIC is the primary criterion highlighted here, AIC is secondary.** This pipeline "
        "already has a genuine cross-validated held-out NLL (Sections 1/3 above), which is a "
        "better estimate of what AIC approximates (out-of-sample predictive risk) than AIC's "
        "asymptotic in-sample correction would be, so an AIC-led comparison would mostly "
        "duplicate an existing, stronger result. BIC's `k*ln(n)` penalty targets a different "
        "question instead -- which model structure this TRAINING data supports once "
        "complexity is penalized on a log(n) scale. Treat BIC here as a **penalized in-sample "
        "fit comparison, not decisive evidence for a single \"true\" model**: M3 and M4 nest "
        "simpler models at a parameter boundary (`alpha_theta=0`, `bh=0` respectively; see "
        "`docs/models/M3_hmm_delta_v1.md` / `M4_hsmm_delta_v1.md`), which weakens BIC's "
        "standard asymptotic justification.\n"
    )
    lines.append(
        "The rolling IC contrasts are exploratory because neighboring windows reuse most of "
        f"their training sessions. The single-session IC contrasts have only N={n_ss_ic} (reduced "
        "from the 27-session common population above); at that size an exact two-sided Wilcoxon "
        "test still has meaningfully less resolution than the sliding-window row, so this row "
        "should be read descriptively even though adjusted p-values are reported.\n"
    )
    lines.append("### Sliding-window IC summary\n")
    lines.append(f"| Model | k (params) | Mean BIC | Mean ΔBIC vs. best | Mean BIC weight | Windows best by BIC (of {n_sw_ic}) | Mean AIC | Windows best by AIC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_id in MODEL_ORDER:
        row = ic_summary[ic_summary["model_id"] == model_id]
        if row.empty:
            continue
        row = row.iloc[0]
        lines.append(
            f"| {model_id} | {int(row['n_params'])} | {_fmt(row['mean_bic'], 1)} | "
            f"{_fmt(row['mean_delta_bic'], 2)} | {_fmt(row['mean_bic_weight'], 3)} | "
            f"{int(row['n_windows_best_by_bic'])} | {_fmt(row['mean_aic'], 1)} | "
            f"{int(row['n_windows_best_by_aic'])} |"
        )
    lines.append("")
    lines.append("### Single-session IC sensitivity summary\n")
    lines.append(
        f"| Model | k (params) | Mean BIC | Mean ΔBIC vs. best | Mean BIC weight | "
        f"Sessions best by BIC (of {n_ss_ic}) | Mean AIC | Sessions best by AIC |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_id in MODEL_ORDER:
        row = ss_ic_summary[ss_ic_summary["model_id"] == model_id]
        if row.empty:
            continue
        row = row.iloc[0]
        lines.append(
            f"| {model_id} | {int(row['n_params'])} | {_fmt(row['mean_bic'], 1)} | "
            f"{_fmt(row['mean_delta_bic'], 2)} | {_fmt(row['mean_bic_weight'], 3)} | "
            f"{int(row['n_sessions_best_by_bic'])} | {_fmt(row['mean_aic'], 1)} | "
            f"{int(row['n_sessions_best_by_aic'])} |"
        )
    lines.append("")

    ic_results = ic_paired_pairwise.copy()
    ic_results["panel"] = ic_results["analysis_level"] + "_" + ic_results["metric"]
    _append_adjacent_results(
        lines, ic_results, "panel",
        [
            ("sliding_window_aic", "Sliding AIC"),
            ("sliding_window_bic", "Sliding BIC"),
            ("single_session_aic", "Single-session AIC"),
            ("single_session_bic", "Single-session BIC"),
        ],
    )

    lines.append("## 8. Average fitted parameters and what they mean\n")
    lines.append(
        "Per-model, per-parameter summaries below use every CONVERGED rolling-window fit "
        "(`tables/sliding_window_parameters.csv`, `converged=True`) -- each model's own full "
        "reach (up to 32 windows), not the 27-window common-support population used for "
        "comparisons elsewhere in this report, since this section describes what value each "
        "model independently found for its own parameters rather than comparing models "
        "against each other. **N differs by model** (richer models converge less reliably); "
        "it is reported alongside every row so a reader can judge that. Derived quantities "
        "(hazard, block length, subjective category rates) are computed **per window first, "
        "then averaged** -- never by transforming the already-averaged raw parameter -- since "
        "these are nonlinear (sigmoid/softplus) transforms and averaging afterward would be "
        "systematically biased (Jensen's inequality).\n"
    )

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _param_values(model_id: str, param_name: str) -> np.ndarray:
        subset = sw_params[
            (sw_params["model_id"] == model_id) & (sw_params["param_name"] == param_name)
            & (sw_params["converged"] == True)  # noqa: E712
        ]
        return pd.to_numeric(subset["param_value"], errors="coerce").dropna().to_numpy()

    def _stat_row(label: str, values: np.ndarray, digits: int = 4) -> str:
        if len(values) == 0:
            return f"| {label} | 0 | n/a | n/a | n/a |"
        return (
            f"| {label} | {len(values)} | {_fmt(np.mean(values), digits)} | "
            f"{_fmt(np.median(values), digits)} | {_fmt(np.std(values), digits)} |"
        )

    param_meanings = {
        "w": "Weber/duration-scaled sensory variance coefficient: measurement noise grows with "
             "the square of stimulus duration (`sigma² = w·s² + s0²`); larger `w` = relatively "
             "noisier perception of longer durations.",
        "s0": "Baseline sensory standard deviation, in seconds: the duration-independent noise "
              "floor in the sensory measurement (present even for very short durations).",
        "ell_short": "Lapse rate on short-category trials: probability of a random (stimulus-"
                     "independent) guess response when the true stimulus is short.",
        "ell_long": "Lapse rate on long-category trials: probability of a random guess response "
                    "when the true stimulus is long.",
        "kappa": "Decision cost-ratio bias: internal asymmetry in the cost of the two choices. "
                 "`kappa=1` is unbiased; `kappa>1` biases toward \"short\" choices; `kappa<1` "
                 "biases toward \"long\" choices.",
        "ah": "Raw biased-block hazard coordinate. For the constant-hazard models (M1-M3), "
              "`h_B = sigmoid(ah)` is the per-trial probability of a block switch while in a "
              "biased block, and this is exactly \"the constant block length\" parameter: mean "
              "block length is `1/h_B` trials in a biased block and `2/h_B` trials in a neutral "
              "block (the neutral hazard is fixed at half the biased hazard by task design; "
              "neutral is never re-entered). **For M4**, whose hazard also depends on elapsed "
              "run length (see `bh`), `ah` instead sets the hazard specifically AT the "
              "reference run length of 23 trials -- M4 has no single constant block length.",
        "delta0": "Session-start subjective category-rate contrast: the mouse's believed "
                  "probability of the short vs. long category before any within-session "
                  "learning, `theta_S = sigmoid(-delta0)`, `theta_L = sigmoid(delta0)` (M2-M4; "
                  "M2 holds this fixed for the whole session, M3/M4 let it adapt via "
                  "`alpha_theta`).",
        "alpha_theta": "Belief-learning rate: how fast the subjective category-rate belief "
                       "(`delta`) updates trial-to-trial from feedback (M3/M4 only). Larger = "
                       "faster within-session adaptation of the subjective category rates.",
        "bh": "HSMM run-length hazard slope (M4 only): the increase in the biased-block hazard, "
              "in logit units, per extra trial of elapsed run length beyond the 23-trial "
              "reference (`h_B(r) = sigmoid(ah + bh·(r-23))`). `bh=0` exactly recovers M3's "
              "constant hazard; a positive `bh` means a block becomes progressively more "
              "likely to switch the longer it has already run.",
    }

    for model_id in MODEL_ORDER:
        spec = MODEL_SPECS[model_id]
        lines.append(f"### {model_id} (`{spec.label}`)\n")
        lines.append("| Parameter | N converged windows | Mean | Median | Std |")
        lines.append("|---|---:|---:|---:|---:|")
        for param_name in spec.param_names:
            values = _param_values(model_id, param_name)
            lines.append(_stat_row(f"`{param_name}`", values))
        lines.append("")
        for param_name in spec.param_names:
            lines.append(f"- **`{param_name}`**: {param_meanings[param_name]}")
        lines.append("")

        ah_values = _param_values(model_id, "ah")
        if len(ah_values) and model_id != "M0":
            h_b = _sigmoid(ah_values)
            if model_id == "M4":
                lines.append(
                    f"  Derived: mean hazard AT run length 23, `h_B(23) = sigmoid(ah)` "
                    f"(per-window then averaged) = {_fmt(np.mean(h_b), 4)} "
                    f"(median {_fmt(np.median(h_b), 4)}, N={len(h_b)}). No single mean block "
                    "length is reported for M4 -- see the `ah` meaning above.\n"
                )
            else:
                block_biased = 1.0 / h_b
                block_neutral = 2.0 / h_b
                lines.append(
                    f"  Derived: mean biased-block hazard `h_B = sigmoid(ah)` "
                    f"(per-window then averaged) = {_fmt(np.mean(h_b), 4)} -> mean block length "
                    f"≈ {_fmt(np.mean(block_biased), 1)} trials (biased), "
                    f"{_fmt(np.mean(block_neutral), 1)} trials (neutral), N={len(h_b)}.\n"
                )

        delta0_values = _param_values(model_id, "delta0")
        if len(delta0_values):
            theta_s = _sigmoid(-delta0_values)
            theta_l = _sigmoid(delta0_values)
            lines.append(
                f"  Derived: mean subjective category rate (per-window then averaged) "
                f"`theta_S = sigmoid(-delta0)` = {_fmt(np.mean(theta_s), 3)}, "
                f"`theta_L = sigmoid(delta0)` = {_fmt(np.mean(theta_l), 3)}, N={len(delta0_values)}.\n"
            )

    lines.append("## 9. Event-aligned NLL: block transitions and rare trials\n")
    lines.append(
        "`figures/fig16_sliding_window_event_aligned_nll.png` (sliding-window) and "
        "`figures/fig17_single_session_event_aligned_nll.png` (single-session) show mean NLL "
        "per model at each relative trial position around two kinds of events: a block "
        "transition (relative trial -5..+5; trial 0 = the first trial of the new block) and a "
        "rare trial (relative trial -2..+2; trial 0 = the rare trial itself, i.e. a short "
        "stimulus in a long block or a long stimulus in a short block). Multiple qualifying "
        "events within one window/session are averaged together FIRST, so a window/session "
        "contributes one number per relative position regardless of how many events it "
        "contains -- trials are never treated as independent samples. Events whose displayed "
        "window would reach into a third block, leave the rare trial's own block, or contain a "
        "trial absent from the evaluated sequence are excluded rather than allowing the event "
        "sample to vary by relative position. The saved table retains both the nominal event "
        "count and the per-position contributing count, and collection requires them to match.\n"
    )
    lines.append(
        "**Figure 16 is now 4 pages** (one multi-page PDF, plus a separate flat PNG per page): "
        "page 1 pools every session in a window regardless of split (**all** -- the numbers "
        "tabulated below), page 2 restricts to that window's 9 **train** sessions, page 3 to its "
        "3 **validation** sessions, and page 4 to its 1 held-out **test** session. Earlier versions "
        "of this report pooled all splits together with no way to separate them; `train`/`validation` "
        "NLL reflects in-sample (or tuning-set) fit, not held-out generalization, so page 4 (test) is "
        "the only page comparable in kind to Figures 01/03/15's held-out NLL. The tables below cover "
        "only page 1 (`all`) for brevity; the full per-page breakdown is in "
        "`sliding_window_event_aligned_nll.csv` and the two "
        "`sliding_window_event_aligned_*paired_pairwise.csv` tables (`split_role` column).\n"
    )
    lines.append(
        "The four adjacent-model paired tests per subplot use each window/session's mean NLL "
        "over the WHOLE displayed relative-position range, collapsed to one scalar before "
        "testing -- one set of 4 Holm-corrected tests per subplot, not one test per relative "
        "position (which would reopen an 11-way or 5-way multiplicity problem). This tests "
        "which model has lower NLL averaged across the whole displayed window, not whether the "
        "event-locked shape/peak itself differs between models. Sliding-window inference is "
        "exploratory (overlapping training history).\n"
    )
    lines.append(
        "Figure 16 additionally shows eight paired-window distributions, laid out as two rows of "
        "four: row 3 covers block transitions (Short→Long and Long→Short at trial 0, then the "
        "same two directions averaged over the complete -5..+5 range) and row 4 mirrors that same "
        "layout for rare trials (rare-long-in-a-short-block and rare-short-in-a-long-block at "
        "trial 0, i.e. the rare trial itself, then the same two categories averaged over the "
        "complete -2..+2 range). Trial-0 tests use only the anchor trial; whole-range tests weight "
        "each displayed relative position equally. All eight panels use the same four adjacent "
        "two-sided Wilcoxon contrasts with Holm correction within each panel. Figure arrows show "
        "the lower-NLL model and `|Δmed|` gives the paired-median magnitude; the tables below "
        "retain the signed difference `model_a - model_b`. These descriptive directions do not "
        "remove the exploratory overlapping-window limitation.\n"
    )
    event_labels = [
        ("block_transition_short_to_long", "Short→Long transition"),
        ("block_transition_long_to_short", "Long→Short transition"),
        ("rare_short", "Rare short (in a long block)"),
        ("rare_long", "Rare long (in a short block)"),
    ]
    for level, event_aligned, level_label, unit_label, unit_col, common_ids_fn, split_role_filter in [
        ("sliding_window", sw_event_aligned, "Sliding-window (page 1: all)", "windows", "window_id",
         lambda cat: common_event_window_ids(sw_event_aligned, mouse_id, cat, split_role="all"),
         "all"),
        ("single_session", ss_event_aligned, "Single-session", "sessions", "session_id",
         lambda cat: common_event_session_ids(ss_event_aligned, cat), None),
    ]:
        lines.append(f"**{level_label}:**\n")
        lines.append(f"| Event category | Matched {unit_label} | Total events (matched population) |")
        lines.append("|---|---:|---:|")
        for event_category, label in event_labels:
            common_ids = common_ids_fn(event_category)
            subset = event_aligned[
                (event_aligned["event_category"] == event_category)
                & (event_aligned[unit_col].isin(common_ids))
            ]
            if split_role_filter is not None:
                subset = subset[subset["split_role"] == split_role_filter]
            n_events = int(
                subset.drop_duplicates([unit_col, "model_id"]).groupby(unit_col)["n_events_in_unit"].first().sum()
            ) if not subset.empty else 0
            lines.append(f"| {label} | {len(common_ids)} | {n_events} |")
        lines.append("")

    _append_adjacent_results(
        lines, sw_event_paired[sw_event_paired["split_role"] == "all"], "event_category",
        [(cat, f"Sliding (page 1: all) {label}") for cat, label in event_labels],
    )
    _append_adjacent_results(
        lines, sw_event_trial0_paired[sw_event_trial0_paired["split_role"] == "all"], "event_category",
        [(cat, f"Sliding (page 1: all) trial 0: {label}") for cat, label in event_labels],
    )
    _append_adjacent_results(
        lines, ss_event_paired, "event_category",
        [(cat, f"Single-session {label}") for cat, label in event_labels],
    )

    lines.append("## 10. Sliding-window vs. single-session psychometrics (matched trial population)\n")
    lines.append(
        "`figures/fig18_psychometric_comparison.png` overlays, for each model (rows) across the same "
        "4 block-type columns used throughout this report, the mouse's actual psychometric curve "
        "against two model-predicted curves on an **identical trial population**: the sliding-window "
        "model's zero-shot prediction on that session's blocks before the literal final two (a model "
        "that never trained on this session) versus the single-session model's fine-tuned prediction "
        "(trained directly on those same trials). Per session, the sliding-window side is trimmed to "
        "the same block boundary the single-session fine-tune's own training prefix uses, and the "
        "`trial_index` set actually used from each source is intersected across all 5 models -- so "
        "every model's row, and both curves within a row, are built from the exact same trials. Mouse "
        "choices are identical between the two sources by construction and are drawn once per panel.\n"
    )
    if psychometric_summary.empty:
        lines.append("No matched psychometric-comparison population was available.\n")
    else:
        n_sessions = int(psychometric_summary["n_matched_sessions"].iloc[0])
        lines.append(f"Matched population: **N={n_sessions} sessions**, identical for every model.\n")
        header = " | ".join(PSYCHOMETRIC_BLOCK_LABELS[key] for key in PSYCHOMETRIC_BLOCK_ORDER)
        lines.append(f"| Model | {header} |")
        lines.append("|---|" + "---:|" * len(PSYCHOMETRIC_BLOCK_ORDER))
        one_source = psychometric_summary[psychometric_summary["source"] == "sliding_window"]
        for model_id in MODEL_ORDER:
            model_rows = one_source[one_source["model_id"] == model_id]
            cells = []
            for block_key in PSYCHOMETRIC_BLOCK_ORDER:
                block_rows = model_rows[model_rows["block_subset"] == block_key]
                n_trials = int(block_rows["n_trials"].sum()) if not block_rows.empty else 0
                cells.append(str(n_trials))
            lines.append(f"| {model_id} | " + " | ".join(cells) + " |")
        lines.append(
            "\nTrial counts (per block-type column) are identical across every model row and both "
            "sources -- shown once above rather than duplicated per source.\n"
        )

    return "\n".join(lines) + "\n"


def run(mouse_id: str) -> Path:
    base = common.POST_ANALYSIS_ROOT / "model_comparison" / mouse_id
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / "summary.md"
    out_path.write_text(build_report(mouse_id))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC06")
    args = parser.parse_args()
    path = run(args.mouse)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
