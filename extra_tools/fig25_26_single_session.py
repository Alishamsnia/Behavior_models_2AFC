#!/usr/bin/env python3
"""Figures 25 and 26 on the matched single-session fits, for the train and the
test part of every session, per mouse and pooled across mice.

A single-session fit starts from the winner of the rolling window whose test
session it is, is fine-tuned on the session prefix (``train``: every block but
the last two) and scored on the final two blocks (``test``: never used for
fitting). Here the unit is one session instead of one rolling window:

* Figure 25: per session, each model's mean NLL on that split; sessions in
  chronological order with equal weight; the per-mouse procedure is exactly the
  rolling-window one (block_bootstrap_figures: moving blocks L=5, B=50,000,
  percentile / basic CIs, Holm, edge shift, sensitivity, TOST); the pooled one
  is the stratified version of extra_tools/fig25_pooled.py.
* Figure 26: event-conditioned NLL (Figure 26 event windows) found separately
  inside each split -- an event whose window would cross the train/test
  boundary is dropped from both -- with the same bootstrap.

Train is the data each model was fine-tuned on: an in-sample fit that favours
models with more parameters, shown as a diagnostic. Test is the held-out part.

  python extra_tools/fig25_26_single_session.py collect --mouse MC06
  python extra_tools/fig25_26_single_session.py figures --mouse MC06      # per mouse
  python extra_tools/fig25_26_single_session.py figures --pooled          # MC06+MC07+MC08
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "extra_tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from post_analysis.expanded_model_comparison import block_bootstrap_figures as bbf  # noqa: E402
from post_analysis.expanded_model_comparison import figure26_event_pairwise as f26  # noqa: E402
from post_analysis.expanded_model_comparison.config import MODEL_BY_ID, MODEL_ORDER, MODEL_SPECS  # noqa: E402
from post_analysis.expanded_model_comparison.extended_collect import FIGURE26_EVENT_RANGES, _event_rows  # noqa: E402
import fig25_pooled as pooled_panels  # noqa: E402

MICE = ("MC06", "MC07", "MC08")
SPLITS = ("train", "test")
SPLIT_TEXT = {
    "train": "TRAIN (fine-tuned session prefix; in-sample)",
    "test": "TEST (final two blocks; held out)",
}
SPLIT_NOTE = {
    "train": "Train = the session prefix each model was fine-tuned on: an in-sample fit that favours models with "
             "more parameters; a diagnostic, not a held-out comparison.",
    "test": "Test = each session's final two blocks, never used for fitting.",
}
OUTPUT_ROOT = REPOSITORY_ROOT / "results_post_analysis" / "expanded_model_comparison" / "single_session"
TRIAL_KEY = ["split_role", "session_id", "source_trial_index"]
OBSERVED = ["choice_long", "category_long", "duration_s", "choice_valid", "feedback_valid", "source_sha256"]
COLUMNS = set(TRIAL_KEY + OBSERVED) | {
    "trial_index", "block_index", "true_block", "true_run_length", "model_evaluated",
    "model_probability_long", "per_trial_negative_log_likelihood",
}


# ----------------------------------------------------------------------------- collection

def _single_root(model_dir: str, mouse: str) -> Path:
    standard = REPOSITORY_ROOT / "results_single_session" / model_dir / mouse / "full_session_matched_v1"
    nested = REPOSITORY_ROOT / "results_single_session" / model_dir / mouse / model_dir / mouse / "full_session_matched_v1"
    return standard if any(standard.glob("session_*")) else nested


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _session_table(directory: Path) -> pd.DataFrame:
    trials = pd.read_csv(directory / "trial_predictions.csv", usecols=lambda c: c in COLUMNS)
    missing = COLUMNS - set(trials.columns)
    if missing:
        raise ValueError(f"{directory}: missing columns {sorted(missing)}")
    evaluated = _bool(trials["model_evaluated"])
    p = pd.to_numeric(trials.loc[evaluated, "model_probability_long"], errors="raise").to_numpy(float)
    y = pd.to_numeric(trials.loc[evaluated, "choice_long"], errors="raise").to_numpy(float)
    nll = pd.to_numeric(trials.loc[evaluated, "per_trial_negative_log_likelihood"], errors="raise").to_numpy(float)
    if not np.isin(y, (0.0, 1.0)).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{directory}: invalid choices or probabilities")
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    if not np.allclose(nll, -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)), rtol=1e-8, atol=1e-10):
        raise ValueError(f"{directory}: per-trial NLL does not match probability/choice")
    if set(trials.loc[evaluated, "split_role"]) != set(SPLITS):
        raise ValueError(f"{directory}: evaluated trials are not exactly train + test")
    return trials


def collect(mouse: str) -> tuple[Path, Path]:
    metric_rows, event_rows, reference = [], [], {}
    for spec in MODEL_SPECS:
        root = _single_root(spec.directory, mouse)
        for directory in sorted(root.glob("session_*")):
            manifest = json.loads((directory / "single_session_manifest.json").read_text())
            if manifest.get("workflow") != "full_session_matched" or manifest.get("status") not in ("complete", "ineligible"):
                raise ValueError(f"{directory}: not a fitted matched single-session result")
            rank = int(re.match(r"session_(\d+)_", directory.name).group(1))
            trials = _session_table(directory)
            evaluated = trials[_bool(trials["model_evaluated"])]
            identity = evaluated[TRIAL_KEY + OBSERVED].sort_values(TRIAL_KEY).reset_index(drop=True)
            if rank in reference:
                if not identity.astype(str).equals(reference[rank]):
                    raise ValueError(f"{spec.label} session {rank}: evaluated trials differ from the other models")
            else:
                reference[rank] = identity.astype(str)
            saved_path = next(p for p in (directory / "prefix_and_heldout_metrics.csv",
                                          directory / "split_and_session_metrics.csv") if p.is_file())
            saved = pd.read_csv(saved_path).query("session_id == 'ALL_SESSIONS'").set_index("split_role")
            completed_by = manifest.get("completed_by") or {}
            for split in SPLITS:
                values = pd.to_numeric(evaluated.loc[evaluated.split_role.eq(split), "per_trial_negative_log_likelihood"])
                if int(saved.loc[split, "valid_trial_count"]) != len(values) or \
                        not np.isclose(float(saved.loc[split, "mean_nll"]), values.mean(), rtol=1e-9, atol=1e-12):
                    raise ValueError(f"{directory}: {split} metrics disagree with trial predictions")
                metric_rows.append({
                    "mouse_id": mouse, "model_id": spec.model_id, "model_label": spec.label, "rank": rank,
                    "session_id": manifest["session_id"], "split_role": split, "valid_trial_count": len(values),
                    "mean_nll": values.mean(), "status": manifest.get("status"),
                    "converged": manifest.get("converged") is True,
                    # Before today's completion run, forced windows were never used to start a session.
                    "from_forced_window": bool(completed_by.get("source_window_forced_best_training_nll")),
                    "scored_at_failed_fit_best": "evaluated_at_failed_fit_best" in completed_by,
                })
            # Events are found inside one split at a time: trials of the other split are
            # marked not-evaluated, so an event window crossing the boundary is dropped.
            for split in SPLITS:
                masked = trials.copy()
                masked.loc[~masked.split_role.eq(split), "model_evaluated"] = False
                event_rows.extend(_event_rows(masked, spec.model_id, rank, range_overrides=FIGURE26_EVENT_RANGES))
    metrics = pd.DataFrame(metric_rows)
    counts = metrics.groupby(["rank", "split_role"]).model_id.nunique()
    if not counts.eq(len(MODEL_SPECS)).all():
        raise ValueError(f"{mouse}: sessions without all {len(MODEL_SPECS)} models: {counts[counts.ne(len(MODEL_SPECS))]}")
    out = OUTPUT_ROOT / mouse / "tables"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "single_session_metrics.csv", index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(out / "single_session_events.csv", index=False)
    print(f"{mouse}: {metrics['rank'].nunique()} sessions x {metrics.model_id.nunique()} models collected")
    return out / "single_session_metrics.csv", out / "single_session_events.csv"


# ----------------------------------------------------------------------------- Figure 25

def _scores(metrics: pd.DataFrame, split: str) -> bbf.LoadedScores:
    part = metrics[metrics.split_role.eq(split)]
    wide = part.pivot(index="rank", columns="model_id", values="mean_nll").reindex(columns=list(MODEL_ORDER)).sort_index()
    if wide.isna().any().any():
        raise ValueError("incomplete session x model matrix")
    ranks = wide.index.to_numpy(int)
    if not np.array_equal(ranks, np.arange(ranks[0], ranks[0] + len(ranks))):
        raise ValueError("sessions are not chronologically contiguous")
    counts = part.drop_duplicates("rank").set_index("rank").reindex(wide.index).valid_trial_count.to_numpy(int)
    return bbf.LoadedScores(matrix=wide.to_numpy(float), model_ids=tuple(MODEL_ORDER), test_ranks=ranks,
                            trial_counts=counts)


def _inclusion_note(metrics: pd.DataFrame) -> str:
    sessions = metrics[metrics.split_role.eq("test")]
    n = sessions["rank"].nunique() if "mouse_id" not in sessions else sessions.groupby("mouse_id")["rank"].nunique().sum()
    return (f"All {n} sessions included for every model, regardless of fine-tune convergence "
            f"(converged: {int(sessions.converged.sum())} of {len(sessions)} session fits); "
            f"{int(sessions.from_forced_window.sum())} fits started from a forced rolling-window winner; "
            f"{int(sessions.scored_at_failed_fit_best.sum())} scored at the best point before a numerical failure.")


def _document_page(title, scores, per_mouse, split, replicates, length, note):
    n_models = len(scores.model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    results, _ = bbf._document_pair_results(scores.matrix, scores.model_ids, length, replicates, scores.strata)
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(5, 2, width_ratios=(0.86, 1.34), height_ratios=(1.0, 0.72, 0.42, 0.72, 0.15))
    ranking_ax = figure.add_subplot(grid[0, 0])
    if per_mouse is None:
        leader, runner_up = bbf._ranking_panel(ranking_ax, scores)
    else:
        leader, runner_up = pooled_panels._ranking_panel(ranking_ax, scores, per_mouse)
    ranking_ax.set_xlabel(f"Mean {split} NLL (nats/choice; lower is better)", fontsize=8)
    ranking_ax.set_title(f"A  Observed {split} ranking" + (" (pooled; grey = each mouse)" if per_mouse else ""),
                         loc="left", fontweight="bold", fontsize=9)
    top_two = bbf._oriented_result(bbf._result_lookup(results), leader, runner_up)
    verdict = "superiority supported" if top_two.holm_p < 0.05 else "superiority not established"
    ranking_ax.text(0.99, 0.96, f"Top two: Holm p={top_two.holm_p:.3g} (raw p={top_two.raw_p:.3g}); {verdict}",
                    transform=ranking_ax.transAxes, ha="right", va="top", fontsize=6.3)
    bbf._pair_matrix_panel(figure.add_subplot(grid[:4, 1]), scores.model_ids, results,
                           f"B  Guide method: all {n_pairs} paired comparisons", label_fontsize=4.0)
    trajectory_ax = figure.add_subplot(grid[1, 0])
    if per_mouse is None:
        bbf._trajectory_panel(trajectory_ax, scores, leader, runner_up)
        bbf._acf_panel(figure.add_subplot(grid[2, 0]), scores, leader, runner_up)
        trajectory_ax.set_xlabel(f"Chronological session rank {scores.test_ranks.min()}-{scores.test_ranks.max()} "
                                 f"({len(scores.matrix)} sessions)", fontsize=8)
    else:
        pooled_panels._trajectory_panel(trajectory_ax, per_mouse, leader, runner_up)
        pooled_panels._acf_panel(figure.add_subplot(grid[2, 0]), per_mouse, leader, runner_up)
        trajectory_ax.set_xlabel("Chronological session rank within each mouse", fontsize=8)
    trajectory_ax.set_title("C  Top-two session difference", loc="left", fontweight="bold", fontsize=9)
    bbf._document_sensitivity_panel(figure.add_subplot(grid[3, 0]), scores, leader, runner_up, replicates)
    footer = figure.add_subplot(grid[4, :])
    footer.axis("off")
    stratified = "; moving blocks drawn within each mouse (never across mice)" if scores.strata else ""
    footer.text(0.5, 0.5,
                f"W={len(scores.matrix)} sessions; {n_models} models; B={replicates:,}; primary L={length}; equal-session "
                f"weighting{stratified}; pointwise percentile 95% CIs; uncentred 2x-min tails; Holm family={n_pairs}.\n"
                f"{SPLIT_NOTE[split]}\n{note}\nEdge shift = exact expected bootstrap effect minus observed effect. "
                "Exploratory, within these mice; no model was refitted for this figure.",
                ha="center", va="center", fontsize=7.0, wrap=True)
    figure.suptitle(f"{title} - {SPLIT_TEXT[split]} - paired moving-block bootstrap (guide)", fontsize=12, fontweight="bold")
    return figure


def _hybrid_page(title, scores, split, replicates, length, note):
    n_models = len(scores.model_ids)
    n_pairs = n_models * (n_models - 1) // 2
    order = np.argsort(scores.matrix.mean(axis=0))
    leader, runner_up = scores.model_ids[order[0]], scores.model_ids[order[1]]
    results, _ = bbf._hybrid_pair_results(scores.matrix, scores.model_ids, length, replicates, scores.strata)
    sampled = bbf._sampled_means(scores.matrix, bbf.EQUIVALENCE_BLOCK_LENGTH, replicates, scores.strata)
    equivalence = bbf._equivalence_rows(scores, sampled, leader, replicates)
    figure = plt.figure(figsize=(16.54, 11.69), layout="constrained")
    grid = figure.add_gridspec(4, 2, width_ratios=(1.3, 0.7), height_ratios=(1.0, 0.92, 0.72, 0.14))
    bbf._pair_matrix_panel(figure.add_subplot(grid[:3, 0]), scores.model_ids, results,
                           f"A  Hybrid null-centred test: all {n_pairs} model pairs", label_fontsize=3.8)
    bbf._reference_forest_panel(figure.add_subplot(grid[0, 1]), scores, results, leader)
    bbf._equivalence_panel(figure.add_subplot(grid[1, 1]), equivalence, leader)
    bbf._hybrid_sensitivity_panel(figure.add_subplot(grid[2, 1]), scores, leader, runner_up, replicates)
    footer = figure.add_subplot(grid[3, :])
    footer.axis("off")
    stratified = "; moving blocks drawn within each mouse" if scores.strata else ""
    footer.text(0.5, 0.5,
                f"W={len(scores.matrix)} sessions; equal-session weighting{stratified}; {n_models} models; B={replicates:,}; "
                f"pairwise primary L={length}; basic 95% CIs; two-sided null-centred tails; Holm family={n_pairs}.\n"
                f"TOST L={bbf.EQUIVALENCE_BLOCK_LENGTH}, margin +/-{bbf.EQUIVALENCE_MARGIN}, 90% basic CIs, "
                f"Holm family={n_models - 1} post-selection leader contrasts.\n{SPLIT_NOTE[split]}\n{note}",
                ha="center", va="center", fontsize=6.9, wrap=True)
    figure.suptitle(f"{title} - {SPLIT_TEXT[split]} - hybrid centred bootstrap and practical equivalence",
                    fontsize=12, fontweight="bold")
    return figure


# ----------------------------------------------------------------------------- Figure 26

def _contiguous(index) -> bool:
    ids = sorted(int(i) for i in index)
    return bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))


def _domain_inputs(events_by_mouse: dict[str, pd.DataFrame], split: str, domain: str):
    parts, strata, used, skipped, counts = [], [], [], [], []
    for mouse, events in events_by_mouse.items():
        # _domain_matrix reads the split_role == "test" rows; hand it the requested split.
        chosen = events[events.split_role.eq(split)].assign(split_role="test")
        matrix = f26._domain_matrix(chosen, domain, MODEL_ORDER)
        if len(matrix) < f26.MIN_USABLE_WINDOWS or not _contiguous(matrix.index):
            reason = "too few" if len(matrix) < f26.MIN_USABLE_WINDOWS else "non-contiguous"
            skipped.append(f"{mouse} ({len(matrix)} sessions, {reason})")
            continue
        parts.append(matrix.set_index(pd.Index([f"{mouse}:{r}" for r in matrix.index])))
        strata.append(len(matrix))
        used.append(mouse)
        counts.append(f26._domain_event_counts(chosen, domain, MODEL_ORDER, matrix.index))
    if not parts:
        return None, skipped
    return (pd.concat(parts), tuple(strata), used, pd.concat(counts, axis=1).sum(axis=1)), skipped


def _event_page(title, split, domain, matrix, strata, used, skipped, event_counts, replicates):
    model_ids = tuple(MODEL_ORDER)
    n_pairs = len(model_ids) * (len(model_ids) - 1) // 2
    strata_arg = strata if len(used) > 1 else None
    length = min(f26.DEFAULT_PRIMARY_LENGTH, max(1, min(strata) - 1))
    results, sampled = f26._document_pair_results(matrix.to_numpy(float), model_ids, length, replicates, strata_arg)
    figure = plt.figure(figsize=(19.5, 12.4), layout="constrained")
    grid = figure.add_gridspec(3, 2, width_ratios=(0.85, 1.35), height_ratios=(0.09, 1.0, 0.16))
    banner = figure.add_subplot(grid[0, :])
    banner.axis("off")
    primary = domain == "combined"
    banner.text(0.5, 0.5, "PRIMARY EVENT-CONDITIONED COMPARISON — pools all four event types below" if primary
                else "DESCRIPTIVE BREAKDOWN of the combined (primary) page — not an independently confirmatory test",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1b5e20" if primary else "#8a6d00", edgecolor="none"))
    f26._event_ranking_panel(figure.add_subplot(grid[1, 0]), model_ids, matrix, sampled)
    f26._pair_matrix_panel(figure.add_subplot(grid[1, 1]), model_ids, results,
                           f"B  {f26.DOMAIN_TITLES[domain]}: all {n_pairs} paired comparisons", label_fontsize=5.3)
    footer = figure.add_subplot(grid[2, :])
    footer.axis("off")
    lo, hi = int(event_counts.min()), int(event_counts.max())
    sessions = ", ".join(f"{m} {s}" for m, s in zip(used, strata))
    left_out = f" Left out (too few or non-contiguous sessions with events): {', '.join(skipped)}." if skipped else ""
    footer.text(0.5, 0.5,
                f"{SPLIT_NOTE[split]} Sessions with a qualifying event for every model: {sessions}.{left_out}\n"
                f"Event window: {f26.DOMAIN_WINDOW_DESCRIPTIONS[domain]}; events found within this split only (an event "
                "crossing the train/test boundary is dropped). Per-session score = event-count-weighted mean NLL; "
                f"sessions equally weighted. Events per model: {lo if lo == hi else f'{lo}-{hi}'}.\n"
                f"Moving-block bootstrap as in Figure 25's guide method{', blocks drawn within each mouse' if strata_arg else ''} "
                f"(L={length}; {replicates:,} resamples); uncentred two-minimum tails; Holm across {n_pairs} pairs. Exploratory.",
                ha="center", va="center", fontsize=8.4, linespacing=1.4, wrap=True)
    figure.suptitle(f"Figure 26 — {title} — {SPLIT_TEXT[split]}\n{f26.DOMAIN_TITLES[domain]}", fontsize=14, fontweight="bold")
    return figure


def _text_page(text):
    figure = plt.figure(figsize=(19.5, 12.4))
    figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    return figure


# ----------------------------------------------------------------------------- driver

def _write_pdf(path: Path, pages) -> None:
    pending = path.with_name(f".{path.name}.pending")
    try:
        with PdfPages(pending) as pdf:
            for page in pages:
                pdf.savefig(page)
                plt.close(page)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def figures(mice: tuple[str, ...], replicates: int) -> list[Path]:
    tables = {m: (pd.read_csv(OUTPUT_ROOT / m / "tables" / "single_session_metrics.csv"),
                  pd.read_csv(OUTPUT_ROOT / m / "tables" / "single_session_events.csv")) for m in mice}
    label = mice[0] if len(mice) == 1 else "pooled"
    title = f"{mice[0]} single-session fits" if len(mice) == 1 else f"Pooled single-session fits ({' + '.join(mice)})"
    out = OUTPUT_ROOT / label / "figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([t[0] for t in tables.values()])
    note = _inclusion_note(metrics)
    style = {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    written = []
    with plt.rc_context(style):
        for split in SPLITS:
            per_mouse = {m: _scores(tables[m][0], split) for m in mice}
            if len(mice) == 1:
                scores, panels = per_mouse[mice[0]], None
            else:
                scores = bbf.LoadedScores(
                    matrix=np.vstack([s.matrix for s in per_mouse.values()]), model_ids=tuple(MODEL_ORDER),
                    test_ranks=np.concatenate([s.test_ranks for s in per_mouse.values()]),
                    trial_counts=np.concatenate([s.trial_counts for s in per_mouse.values()]),
                    strata=tuple(len(s.matrix) for s in per_mouse.values()))
                panels = per_mouse
            pages = [_document_page(title, scores, panels, split, replicates, bbf.DEFAULT_PRIMARY_LENGTH, note),
                     _hybrid_page(title, scores, split, replicates, bbf.DEFAULT_PRIMARY_LENGTH, note)]
            path = out / f"fig25_single_session_{split}.pdf"
            _write_pdf(path, pages)
            written.append(path)
            means = pd.DataFrame({m: s.matrix.mean(axis=0) for m, s in per_mouse.items()}, index=list(MODEL_ORDER))
            means.insert(0, "equal_session_mean", scores.matrix.mean(axis=0))
            means.insert(0, "model_label", [MODEL_BY_ID[m].label for m in MODEL_ORDER])
            means.sort_values("equal_session_mean").to_csv(out / f"fig25_single_session_{split}_means.csv", index_label="model_id")

            events_by_mouse = {m: tables[m][1] for m in mice}
            pages = []
            for domain in f26.DOMAIN_ORDER:
                inputs, skipped = _domain_inputs(events_by_mouse, split, domain)
                if inputs is None:
                    pages.append(_text_page(f"Figure 26 — {title} — {SPLIT_TEXT[split]}\n{f26.DOMAIN_TITLES[domain]}\n\n"
                                            f"Not reported: no mouse has at least {f26.MIN_USABLE_WINDOWS} chronologically "
                                            f"contiguous sessions with a qualifying event for every model.\n{'; '.join(skipped)}"))
                    continue
                pages.append(_event_page(title, split, domain, *inputs[:3], skipped, inputs[3], replicates))
            path = out / f"fig26_single_session_{split}.pdf"
            _write_pdf(path, pages)
            written.append(path)
    for path in written:
        print(f"wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("collect", "figures"))
    parser.add_argument("--mouse", choices=MICE)
    parser.add_argument("--pooled", action="store_true", help="figures: pool MC06+MC07+MC08")
    parser.add_argument("--replicates", type=int, default=bbf.DEFAULT_REPLICATES)
    args = parser.parse_args()
    if args.action == "collect":
        for mouse in ([args.mouse] if args.mouse else MICE):
            collect(mouse)
    else:
        if args.pooled == bool(args.mouse):
            parser.error("figures needs exactly one of --mouse or --pooled")
        figures(MICE if args.pooled else (args.mouse,), args.replicates)


if __name__ == "__main__":
    main()
