"""Build tidy CSV tables comparing M0-M4 for one mouse.

Every function here only reads files under ``results_sliding_window/`` and
``results_single_session/``; nothing under the source tree or the raw result
trees is modified. Outputs are written under
``results_post_analysis/model_comparison/<mouse_id>/tables/``.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from post_analysis import common
from post_analysis.common import MODEL_ORDER, MODEL_SPECS
from post_analysis.model_comparison import stats as mc_stats

_WINDOW_ID_FROM_RUN_ID = re.compile(r"^window_(\d+)_")


def _window_id_from_run_id(run_id: str):
    match = _WINDOW_ID_FROM_RUN_ID.match(str(run_id))
    return int(match.group(1)) if match else None

SPLIT_METRIC_COLUMNS = [
    "split_role", "valid_trial_count", "elpd", "mean_nll", "brier_score",
    "observed_long_rate", "predicted_long_rate", "mouse_accuracy",
    "model_expected_accuracy", "session_id",
]


# ---------------------------------------------------------------------------
# Sliding-window: per-window detailed (per session) + global (ALL_SESSIONS)
# ---------------------------------------------------------------------------

def collect_sliding_window_split_session_metrics(mouse_id: str) -> pd.DataFrame:
    """One row per (model, window, split_role, session).

    ``session_id == "ALL_SESSIONS"`` rows are the pooled/global metric for
    that split within that window; all other rows are the per-session
    (detailed) breakdown. This directly mirrors the "global and detailed
    performance for each window" request.
    """
    rows = []
    for model_id in MODEL_ORDER:
        for window_dir in common.list_window_dirs(model_id, mouse_id):
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            window_id = manifest.get("window_id")
            metrics = common.read_split_session_metrics(window_dir)
            winner = common.find_multistart_winner(window_dir)
            converged = winner.get("converged") if winner else False
            if metrics is None:
                # Window fit did not converge / no eligible winner: record a
                # placeholder row so the window is still visible in the table.
                rows.append({
                    "model_id": model_id,
                    "window_id": window_id,
                    "window_dir": window_dir.name,
                    "test_rank": manifest.get("test_rank"),
                    "optimizer": manifest.get("optimizer"),
                    "converged": converged,
                    "split_role": None,
                    "session_id": None,
                    "valid_trial_count": None,
                    "elpd": None,
                    "mean_nll": None,
                    "brier_score": None,
                    "observed_long_rate": None,
                    "predicted_long_rate": None,
                    "mouse_accuracy": None,
                    "model_expected_accuracy": None,
                })
                continue
            metrics = metrics.copy()
            metrics["model_id"] = model_id
            metrics["window_id"] = window_id
            metrics["window_dir"] = window_dir.name
            metrics["test_rank"] = manifest.get("test_rank")
            metrics["optimizer"] = manifest.get("optimizer")
            metrics["converged"] = converged
            rows.append(metrics)

    if not rows:
        raise RuntimeError(
            f"No sliding-window results found for mouse={mouse_id!r} under any of "
            f"{[str(common.sliding_window_dir(m, mouse_id)) for m in MODEL_ORDER]}. "
            "Check --mouse and that results_sliding_window/ has been populated."
        )
    frames = [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows]
    table = pd.concat(frames, ignore_index=True, sort=False)
    prefix_cols = ["model_id", "window_id", "window_dir", "test_rank", "optimizer", "converged"]
    ordered_cols = prefix_cols + [c for c in SPLIT_METRIC_COLUMNS if c not in prefix_cols]
    remaining = [c for c in table.columns if c not in ordered_cols]
    return table[ordered_cols + remaining]


def _per_model_window_ids(mouse_id: str) -> dict:
    """Each model's own set of window IDs it produced a multistart winner
    for, read from its ``global_figures/global_figure_manifest.json``. A
    winner is only recorded there when a start actually converged (see
    ``docs/models/M0_sensory_v1.md`` Part VIII: "Only converged runs are
    eligible"), so these sets are already convergence-filtered.
    """
    model_window_ids: dict[str, set] = {}
    for model_id in MODEL_ORDER:
        window_root = common.sliding_window_dir(model_id, mouse_id)
        manifest = common.read_json(window_root / "global_figures" / "global_figure_manifest.json") or {}
        ids = {
            wid for run_id in manifest.get("source_run_ids", [])
            for wid in [_window_id_from_run_id(run_id)] if wid is not None
        }
        model_window_ids[model_id] = ids
    return model_window_ids


def common_window_ids(mouse_id: str) -> set:
    """Windows every one of the 5 models reached with a converged winner --
    the authoritative "common support" population, already used for fig01's
    diamond markers (``collect_sliding_window_global_pooled_metrics``) and
    now reused for the paired significance tests and the AIC/BIC table so
    "N common windows" means the same population everywhere in this report.

    Raises rather than silently intersecting a smaller set if any model's
    manifest is missing/empty -- an absent model must be an error here, not
    a set that happens to still produce non-empty output.
    """
    model_window_ids = _per_model_window_ids(mouse_id)
    empty = [m for m, ids in model_window_ids.items() if not ids]
    if empty:
        raise RuntimeError(
            f"No windows found for model(s) {empty} for mouse={mouse_id!r}; "
            "cannot compute a common-support window set across all 5 models."
        )
    return set.intersection(*model_window_ids.values())


def common_metric_window_ids(sw_split_session: pd.DataFrame, mouse_id: str) -> set:
    """Windows with one finite global NLL for every model in every split.

    The manifest intersection establishes that every model produced an
    eligible rolling fit.  This second validation prevents a missing or
    duplicate split metric from silently giving Figure 01 unequal samples.
    """
    common_ids = common_window_ids(mouse_id)
    for split_role in ("train", "validation", "test"):
        subset = sw_split_session[
            (sw_split_session["split_role"] == split_role)
            & (sw_split_session["session_id"] == "ALL_SESSIONS")
            & (sw_split_session["window_id"].isin(common_ids))
        ]
        matrix = mc_stats.complete_case_matrix(subset, "window_id", "mean_nll")
        common_ids &= set(matrix.index)
    return common_ids


def collect_sliding_window_global_pooled_metrics(
    mouse_id: str, sw_split_session: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (model, split_role): metrics pooled over every rolling
    window's saved trial-level predictions (``global_figures/pooled_trial_predictions.csv``).

    All five models are restricted to the SAME common set of rolling windows
    before pooling -- the intersection of windows every model produced a
    winner for (27/32 here, bottlenecked by M4's 5 non-converged windows).
    Without this, a model that only reached fewer windows would be compared
    against a different, not-directly-comparable population; on this
    dataset that reversed the apparent test-split ranking (M4 only looked
    best when pooled over its own easier 27-window subset -- on the common
    27 windows M1 is marginally best). ``source_window_count`` keeps each
    model's own full window count for transparency; ``common_support_window_count``
    is what these pooled numbers actually use.

    For the ``test`` split this is a true walk-forward evaluation: every
    rolling window contributes exactly one held-out session, so there is no
    session duplication. ``train``/``validation`` pool across the (repeated)
    window-session analysis units by the project's own stated pooling policy.
    """
    model_window_ids = _per_model_window_ids(mouse_id)
    common_ids = common_metric_window_ids(sw_split_session, mouse_id)

    rows = []
    for model_id in MODEL_ORDER:
        window_root = common.sliding_window_dir(model_id, mouse_id)
        pooled_path = window_root / "global_figures" / "pooled_trial_predictions.csv"
        if not pooled_path.is_file():
            raise FileNotFoundError(
                f"Missing pooled trial predictions for {model_id}: {pooled_path}"
            )
        df = pd.read_csv(pooled_path)
        evaluated = df["model_evaluated"].astype(str).str.lower().eq("true")
        df["_window_id"] = df["source_run_id"].map(_window_id_from_run_id)
        in_common_support = df["_window_id"].isin(common_ids)
        df = df.loc[evaluated & in_common_support].copy()
        for split_role in ("train", "validation", "test"):
            subset = df[df["split_role"] == split_role]
            observed_ids = set(subset["_window_id"].dropna().astype(int))
            if observed_ids != common_ids:
                missing_ids = sorted(common_ids - observed_ids)
                extra_ids = sorted(observed_ids - common_ids)
                raise RuntimeError(
                    f"Incomplete pooled predictions for {model_id}/{split_role}: "
                    f"missing common windows={missing_ids}, unexpected windows={extra_ids}"
                )
            y = pd.to_numeric(subset["choice_long"]).to_numpy(float)
            p = pd.to_numeric(subset["model_probability_long"]).to_numpy(float)
            nll = pd.to_numeric(subset["per_trial_negative_log_likelihood"]).to_numpy(float)
            rows.append({
                "model_id": model_id,
                "split_role": split_role,
                "valid_trial_count": len(subset),
                "elpd": -np.sum(nll),
                "mean_nll": np.mean(nll),
                "brier_score": np.mean((p - y) ** 2),
                "observed_long_rate": np.mean(y),
                "predicted_long_rate": np.mean(p),
                "mouse_accuracy": np.mean(pd.to_numeric(subset["mouse_correct"])),
                "model_expected_accuracy": np.mean(pd.to_numeric(subset["model_probability_correct"])),
                # Recomputed directly from p/y rather than relying on a
                # precomputed column, since one model (M3) does not export
                # "model_choice_correct" in its pooled predictions table.
                "model_threshold_accuracy": np.mean((p >= 0.5) == y),
                "source_window_count": len(model_window_ids[model_id]),
                "common_support_window_count": len(common_ids),
                "prediction_unit_count": subset["source_run_id"].nunique(),
            })
    return pd.DataFrame(rows)


def collect_sliding_window_parameters(mouse_id: str) -> pd.DataFrame:
    """Long-format table: one row per (model, window, parameter)."""
    rows = []
    for model_id in MODEL_ORDER:
        param_names = MODEL_SPECS[model_id].param_names
        for window_dir in common.list_window_dirs(model_id, mouse_id):
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            winner = common.find_multistart_winner(window_dir)
            base = {
                "model_id": model_id,
                "window_id": manifest.get("window_id"),
                "test_rank": manifest.get("test_rank"),
                "window_dir": window_dir.name,
                "optimizer": manifest.get("optimizer"),
            }
            if winner is None:
                rows.append({
                    **base, "param_name": None, "param_value": None,
                    "converged": False, "training_mean_nll": None,
                    "validation_mean_nll": None,
                })
                continue
            values = winner.get("best_constrained_parameters") or []
            for name, value in zip(param_names, values):
                rows.append({
                    **base,
                    "param_name": name,
                    "param_value": value,
                    "converged": winner.get("converged"),
                    "training_mean_nll": winner.get("recalculated_best_training_mean_nll"),
                    "validation_mean_nll": winner.get("final_validation_mean_nll"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-session fine-tuning
# ---------------------------------------------------------------------------

def _single_session_pipeline_gate_status(model_id: str, mouse_id: str) -> dict:
    """Map session_id -> the pipeline's own diagnostic gate status, using
    ``single_session_global_selection.json`` when present (only written for
    models with an extra acceptance gate beyond plain optimizer convergence;
    currently only M4). This is metadata only -- see ``collect_single_session_metrics``,
    which does NOT use it to drop sessions. Every model's attempted single-session
    fits, converged or not, carry a real fitted parameter vector and computed
    heldout NLL, so every model is counted the same way: a session is "valid"
    for comparison once it has that computed metric, full stop.
    """
    selection = common.read_json(
        common.single_session_dir(model_id, mouse_id) / "single_session_global_selection.json"
    )
    gate_status: dict[str, dict] = {}
    if selection is None:
        return gate_status
    for run_id in selection.get("eligible_source_run_ids", []):
        session_id = run_id.split("_", 2)[-1] if run_id.count("_") >= 2 else run_id
        gate_status[session_id] = {"pipeline_gate_pass": True, "ineligibility_reasons": []}
    for excluded in selection.get("excluded_sources", []):
        gate_status[excluded["session_id"]] = {
            "pipeline_gate_pass": False,
            "ineligibility_reasons": excluded.get("ineligibility_reasons", []),
        }
    return gate_status



# ``convergence_checks`` entries that compare a fitted run's final gradient
# norm (or, for M4, its HSMM projection/tail-occupancy fractions) against an
# ABSOLUTE tolerance carried over unchanged from the pooled rolling-window
# training objective (thousands of pooled trials). A single-session
# fine-tune trains on roughly an order of magnitude fewer trials, so its
# gradient estimate near a genuinely good optimum is intrinsically noisier;
# empirically (see Figure 15's reliability note in the report) the great
# majority of single-session fits that fail only these checks are a few-fold
# over tolerance, not diverged, while runs with a genuine numerical
# breakdown instead fail one of the checks below and raised an actual
# exception mid-optimization. Excluded from ``_numerically_sound_fit``;
# still recorded verbatim in each run's own ``fine_tune_fit_result.json``.
#
# REVISIT AFTER THE CHECKPOINT-SELECTION FIX: every fitter's checkpoint used
# to be selected only when training NLL improved by more than the same
# meaningful-improvement tolerance used for early-stopping patience, so a
# later point with a much smaller gradient norm but a sub-tolerance NLL
# improvement could be discarded in favor of an earlier, noisier checkpoint
# (see fit.py's checkpoint-selection fix). That bug inflated how often
# ``small_gradient_norm`` failed independent of sample size, which was one
# of the observations motivating this carve-out. Once single-session fits
# are redone under the corrected checkpoint logic, recount how many still
# fail only ``small_gradient_norm`` before deciding whether this carve-out
# is still needed at its current scope, narrower, or not at all.
_SCALE_SENSITIVE_CONVERGENCE_CHECKS = {
    "small_gradient_norm", "projection_activation_acceptable", "tail_occupancy_acceptable",
}


def _numerically_sound_fit(session_dir: Path) -> Optional[bool]:
    """True iff a single-session fine-tune hit no genuine numerical
    breakdown: finite objective/parameters/gradients, a stable fresh-training
    NLL trajectory across the run, and no parameter-transform,
    probability-normalization, or optimizer-runtime error.

    Deliberately a looser bar than the model's own ``converged`` flag -- see
    ``_SCALE_SENSITIVE_CONVERGENCE_CHECKS`` -- because that flag's remaining
    checks are calibrated for the much larger rolling-window training set.
    Returns ``None`` if no fit-result file exists or it carries no
    convergence-check record (fit never ran / older schema).
    """
    result = common.read_json(session_dir / "fine_tune_fit_result.json")
    if result is None:
        return None
    checks = result.get("convergence_checks")
    if not checks:
        return None
    return all(
        bool(value) for name, value in checks if name not in _SCALE_SENSITIVE_CONVERGENCE_CHECKS
    )


def collect_single_session_metrics(mouse_id: str) -> pd.DataFrame:
    """One row per (model, session, analysis_role in {fine_tune_prefix, heldout}).

    A session counts as valid/usable for cross-model comparison as soon as it
    has a computed heldout metric -- the same rule for every model. Some
    models (M0-M3) additionally expose a plain ``converged`` flag per session,
    and M4 additionally exposes a stricter pipeline gate (``pipeline_gate_pass``,
    used by *its own* pooled global-figure generation to decide which runs to
    fold into psychometric plots etc.). A third column, ``numerically_sound``
    (see ``_numerically_sound_fit``), is a looser fit-quality signal used only
    by the single-session information-criteria table. None of the three drop
    rows here; every model is counted on equal footing (see
    ``collect_single_session_availability``).
    """
    rows = []
    for model_id in MODEL_ORDER:
        gate_status = _single_session_pipeline_gate_status(model_id, mouse_id)
        for session_dir in common.list_session_dirs(model_id, mouse_id):
            manifest = common.read_json(session_dir / "single_session_manifest.json") or {}
            session_id = manifest.get("session_id")
            metrics = common.read_split_session_metrics(session_dir)
            gate = gate_status.get(session_id, {"pipeline_gate_pass": None, "ineligibility_reasons": []})
            if metrics is None:
                continue
            all_sessions = metrics.loc[metrics["session_id"] == "ALL_SESSIONS"].copy()
            all_sessions["analysis_role"] = all_sessions["split_role"].map(
                {"train": "fine_tune_prefix", "test": "heldout"}
            )
            all_sessions["model_id"] = model_id
            all_sessions["session_id"] = session_id
            all_sessions["chronological_rank"] = manifest.get("chronological_rank")
            all_sessions["session_dir"] = session_dir.name
            all_sessions["converged"] = manifest.get("converged")
            all_sessions["numerically_sound"] = _numerically_sound_fit(session_dir)
            all_sessions["pipeline_gate_pass"] = gate["pipeline_gate_pass"]
            all_sessions["ineligibility_reasons"] = "; ".join(gate["ineligibility_reasons"])
            rows.append(all_sessions)
    prefix_cols = [
        "model_id", "session_id", "chronological_rank", "session_dir",
        "converged", "numerically_sound", "pipeline_gate_pass", "ineligibility_reasons",
        "analysis_role",
    ]
    if not rows:
        raise RuntimeError(
            f"No single-session results found for mouse={mouse_id!r} under any of "
            f"{[str(common.single_session_dir(m, mouse_id)) for m in MODEL_ORDER]}. "
            "Check --mouse and that results_single_session/ has been populated."
        )
    table = pd.concat(rows, ignore_index=True, sort=False)
    ordered_cols = prefix_cols + [c for c in SPLIT_METRIC_COLUMNS if c not in prefix_cols]
    remaining = [c for c in table.columns if c not in ordered_cols]
    return table[ordered_cols + remaining]


def common_session_ids(ss_metrics: pd.DataFrame) -> set:
    """Sessions where every model has BOTH a finite fine-tune training NLL
    AND a finite held-out NLL -- the fixed comparison population used
    everywhere a single-session figure or table needs a fair (equal-N)
    cross-model comparison: fig03, the single-session paired tests, and the
    fitted-parameter summary's model-vs-model framing.

    This is derived independently from ``common_window_ids`` (a different
    fitting stage: whole-session fine-tuning, not the rolling-window fits),
    but in this pipeline the two are causally linked, not coincidentally
    equal: ``collect_single_session_availability`` documents that a model's
    missing single-session fits trace back to the SAME rolling windows that
    failed to converge (a non-converged window produces no matched
    initialization for its single-session fine-tune), so the two
    intersections track the same underlying failures by construction.
    """
    common_ids: set | None = None
    for role in ("fine_tune_prefix", "heldout"):
        subset = ss_metrics[ss_metrics["analysis_role"] == role]
        matrix = mc_stats.complete_case_matrix(subset, "session_id", "mean_nll")
        role_ids = set(matrix.index)
        common_ids = role_ids if common_ids is None else common_ids & role_ids
    return common_ids or set()


def _has_finite_heldout_metric(session_dir: Path) -> bool:
    """True iff this session directory has a heldout (test-split,
    ALL_SESSIONS) row with a finite ``mean_nll`` -- the actual requirement
    for a session to be usable downstream (win counts, NLL summaries), not
    merely that a metrics CSV exists (a metrics file with only training rows,
    or a NaN heldout NLL, must not count as valid).
    """
    metrics = common.read_split_session_metrics(session_dir)
    if metrics is None:
        return False
    heldout = metrics.loc[(metrics["session_id"] == "ALL_SESSIONS") & (metrics["split_role"] == "test")]
    if heldout.empty:
        return False
    return bool(np.isfinite(pd.to_numeric(heldout["mean_nll"].iloc[0], errors="coerce")))


def _all_matched_ranks(mouse_id: str) -> list:
    """Union of sliding-window test ranks across every model -- the
    authoritative set of chronological ranks (sessions) this whole analysis
    is matched against, independent of which models attempted or converged.
    """
    all_ranks = set()
    for model_id in MODEL_ORDER:
        for window_dir in common.list_window_dirs(model_id, mouse_id):
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            rank = manifest.get("test_rank")
            if rank is not None:
                all_ranks.add(rank)
    return sorted(all_ranks)


def collect_single_session_availability(mouse_id: str) -> pd.DataFrame:
    """Per-model session counts answering "how many sessions do we have".

    ``n_sessions_valid`` is every attempted session with a finite heldout
    metric -- the count actually used downstream (win counts, NLL summaries).
    The two "flagged" columns are diagnostics only and are already included
    in ``n_sessions_valid``; they are not subtracted from it.

    ``n_matched_windows_expected`` comes from the sliding-window test ranks
    (chronological ranks 13-44 that some model produced a rolling-window
    winner for), not from which single-session directories happen to exist --
    if every model failed to even attempt a matched session, that session
    must still count as "never attempted" rather than silently vanish from
    the denominator.
    """
    n_expected = len(_all_matched_ranks(mouse_id))

    rows = []
    for model_id in MODEL_ORDER:
        session_dirs = common.list_session_dirs(model_id, mouse_id)
        gate_status = _single_session_pipeline_gate_status(model_id, mouse_id)
        n_valid = sum(1 for session_dir in session_dirs if _has_finite_heldout_metric(session_dir))
        n_not_converged = sum(
            1 for session_dir in session_dirs
            if (common.read_json(session_dir / "single_session_manifest.json") or {}).get("converged") is False
        )
        n_gate_failed = sum(1 for v in gate_status.values() if v["pipeline_gate_pass"] is False)
        rows.append({
            "model_id": model_id,
            "n_matched_windows_expected": n_expected,
            "n_sessions_attempted": len(session_dirs),
            "n_sessions_valid": n_valid,
            "n_sessions_flagged_not_converged": n_not_converged,
            "n_sessions_flagged_ineligible_by_pipeline_gate": n_gate_failed,
            "n_sessions_never_attempted": max(n_expected - len(session_dirs), 0),
        })
    return pd.DataFrame(rows)


def collect_single_session_parameters(mouse_id: str) -> pd.DataFrame:
    """Long-format table: one row per (model, session, parameter), fitted
    (fine-tuned) value plus the window-derived initial value it started from.
    """
    rows = []
    for model_id in MODEL_ORDER:
        param_names = MODEL_SPECS[model_id].param_names
        for session_dir in common.list_session_dirs(model_id, mouse_id):
            manifest = common.read_json(session_dir / "single_session_manifest.json") or {}
            fit_result = common.read_json(session_dir / "fine_tune_fit_result.json") or {}
            session_id = manifest.get("session_id")
            rank = manifest.get("chronological_rank")
            fitted_values = manifest.get("best_constrained_parameters") or fit_result.get("best_constrained_parameters") or []
            initial_values = manifest.get("initial_constrained_parameters") or fit_result.get("initial_constrained_parameters") or []
            for idx, name in enumerate(param_names):
                rows.append({
                    "model_id": model_id,
                    "session_id": session_id,
                    "chronological_rank": rank,
                    "param_name": name,
                    "initial_value": initial_values[idx] if idx < len(initial_values) else None,
                    "fitted_value": fitted_values[idx] if idx < len(fitted_values) else None,
                    "converged": manifest.get("converged"),
                    "training_mean_nll": fit_result.get("recalculated_best_training_mean_nll"),
                })
    return pd.DataFrame(rows)


def _compute_win_counts(
    long_df: pd.DataFrame, unit_col: str, rank_col: str, unit_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared "which model has the lowest NLL" logic for one row per
    (unit, model) -> mean_nll, restricted to units where all five models
    have a value. Used for both single-session heldout NLL (``unit_col``
    = session_id) and sliding-window test NLL (``unit_col`` = window_id).

    This is a descriptive point-estimate argmin only: no significance test,
    no correction for the small held-out trial counts some units have, and
    no special handling of ties beyond reporting the runner-up and the
    margin between them. ``winning_margin_nll`` (best minus second-best) is
    included precisely so a reader can see when a "win" was a near-tie
    (margins as small as ~1e-6 nats occur in this dataset, e.g. between
    nested models whose fitted parameters landed almost exactly on the
    parent model's boundary) rather than treating every win count equally.
    """
    wide = long_df.pivot_table(index=unit_col, columns="model_id", values="mean_nll", aggfunc="first")
    wide = wide.reindex(columns=MODEL_ORDER)
    complete = wide.dropna(axis=0, how="any")

    rank_lookup = (
        long_df[[unit_col, rank_col]]
        .drop_duplicates()
        .set_index(unit_col)[rank_col]
    )

    per_unit = complete.copy()
    per_unit[rank_col] = per_unit.index.map(rank_lookup)
    values_sorted = np.sort(complete.to_numpy(), axis=1)
    per_unit["winning_model"] = complete.idxmin(axis=1)
    per_unit["winning_mean_nll"] = values_sorted[:, 0]
    per_unit["runner_up_model"] = complete.apply(
        lambda row: row.drop(labels=[row.idxmin()]).idxmin(), axis=1,
    )
    per_unit["runner_up_mean_nll"] = values_sorted[:, 1] if values_sorted.shape[1] > 1 else np.nan
    per_unit["winning_margin_nll"] = per_unit["runner_up_mean_nll"] - per_unit["winning_mean_nll"]
    per_unit = per_unit.sort_values(rank_col).reset_index()

    counts = per_unit["winning_model"].value_counts().reindex(MODEL_ORDER, fill_value=0)
    summary = pd.DataFrame({
        "model_id": MODEL_ORDER,
        f"n_{unit_label}_won": [int(counts[m]) for m in MODEL_ORDER],
        f"n_{unit_label}_compared": len(per_unit),
        f"n_{unit_label}_available_per_model": [int(wide[m].notna().sum()) for m in MODEL_ORDER],
        "median_winning_margin_nll": per_unit["winning_margin_nll"].median(),
    })
    return per_unit, summary


def compute_single_session_win_counts(
    single_session_metrics: pd.DataFrame, analysis_role: str = "heldout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Which model has the lowest NLL, per session, restricted to sessions
    where all five models have a value.

    ``analysis_role="heldout"`` (default) is the fine-tuned last-two-block
    held-out NLL -- a generalization signal. ``analysis_role="fine_tune_prefix"``
    is the in-sample training NLL instead: since a model with more free
    parameters can typically fit its own training prefix at least as well as
    a simpler nested model, "wins" here are expected to skew toward the
    richer models almost by construction and are NOT evidence of better
    generalization -- see the held-out variant (or their disagreement) for
    that. Every attempted session with a computed metric counts, regardless
    of the ``converged``/``pipeline_gate_pass`` diagnostic flags (see
    ``collect_single_session_metrics``).
    """
    subset = single_session_metrics.loc[
        (single_session_metrics["analysis_role"] == analysis_role)
        & (single_session_metrics["mean_nll"].notna())
    ]
    return _compute_win_counts(subset, "session_id", "chronological_rank", "sessions")


def compute_sliding_window_win_counts(
    sw_split_session: pd.DataFrame, split_role: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Which model has the lowest rolling-window NLL, per window, restricted
    to windows where all five models have a winner (bottlenecked by M4's 5
    non-converged windows -- see ``collect_sliding_window_split_session_metrics``).

    ``split_role="test"`` (default) is the true walk-forward held-out
    evaluation. ``split_role="train"`` is the in-sample pooled training NLL
    instead -- a fit-quality diagnostic, not a generalization one; richer
    nested models are expected to win more often here almost by
    construction. Compare against the test-split winner (or their
    disagreement) to see where added flexibility actually helps versus
    merely overfits.
    """
    test_all_sessions = sw_split_session.loc[
        (sw_split_session["split_role"] == split_role)
        & (sw_split_session["session_id"] == "ALL_SESSIONS")
        & (sw_split_session["mean_nll"].notna())
    ]
    return _compute_win_counts(test_all_sessions, "window_id", "test_rank", "windows")


# ---------------------------------------------------------------------------
# Paired significance tests (four pre-specified, Holm-corrected adjacent
# Wilcoxon signed-rank / paired t-test contrasts), matched by session/window -- see
# ``model_comparison/stats.py`` for the test implementations and the
# multiplicity-handling rationale.
# ---------------------------------------------------------------------------

def compute_single_session_paired_tests(ss_metrics: pd.DataFrame) -> pd.DataFrame:
    """Four adjacent-model paired tests for each Figure 03 panel.

    Every contrast is matched by exact ``session_id`` and uses the same fixed
    five-model complete-case population in both panels.  Holm correction is
    performed over the four pre-specified nested-model comparisons within
    each panel; no data-dependent omnibus gate is applied.
    """
    common_ids = common_session_ids(ss_metrics)
    pairwise_frames = []
    for role in ("fine_tune_prefix", "heldout"):
        subset = ss_metrics[
            (ss_metrics["analysis_role"] == role) & (ss_metrics["session_id"].isin(common_ids))
        ]
        matrix = mc_stats.complete_case_matrix(subset, "session_id", "mean_nll")
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        pairwise.insert(0, "analysis_role", role)
        pairwise["units_independent"] = True
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


def compute_sliding_window_paired_tests(
    sw_split_session: pd.DataFrame, mouse_id: str,
) -> pd.DataFrame:
    """Four adjacent-model paired tests for each Figure 01 split panel.

    Pairing is by exact rolling ``window_id`` on the fixed complete population
    used by the plot.  Train/validation windows reuse sessions, and even test
    windows share much of their fitted history; ``units_independent`` is
    therefore false for every split and the resulting p-values are explicitly
    exploratory within-mouse summaries.
    """
    common_ids = common_metric_window_ids(sw_split_session, mouse_id)
    pairwise_frames = []
    for split_role in ("train", "validation", "test"):
        subset = sw_split_session[
            (sw_split_session["split_role"] == split_role)
            & (sw_split_session["session_id"] == "ALL_SESSIONS")
            & (sw_split_session["window_id"].isin(common_ids))
        ]
        matrix = mc_stats.complete_case_matrix(subset, "window_id", "mean_nll")
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        pairwise.insert(0, "split_role", split_role)
        pairwise["units_independent"] = False
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Trial-subset breakdowns (Figures 01/03) and event-aligned NLL (Figures
# 16/17): both read the per-trial ``trial_predictions.csv`` written for every
# converged window/session, which no function above touches (they all read
# pre-aggregated split/session metrics, or -- in one case -- the separately
# pooled ``global_figures/pooled_trial_predictions.csv``). Reproducibility
# check: mean(per_trial_negative_log_likelihood) grouped by (model, window,
# split_role), restricted to model_evaluated==True, reproduces
# ``collect_sliding_window_split_session_metrics``'s own ``mean_nll`` to full
# float precision -- see the "all" trial_subset below and its regression
# test. Each window/session's file is read exactly once (shared between the
# trial-subset and event-aligned tables via the ``*_trial_level_tables``
# functions) since these files total ~700 MB across the mouse.
# ---------------------------------------------------------------------------

TRIAL_PREDICTION_COLUMNS = [
    "session_id", "session_date", "trial_index", "block_index", "true_run_length", "split_role",
    "per_trial_negative_log_likelihood", "model_evaluated", "category_long", "true_block",
    "duration_s", "choice_long", "model_probability_long",
    "mouse_correct", "model_probability_correct",
]

# The 7 trial subsets requested for Figures 01/03, derived purely from
# ``category_long`` (0=short/1=long stimulus) and ``true_block`` (N/L/S block
# bias) -- no new upstream data needed. These two columns are used (rather
# than the friendlier ``trial_type``/``block_type`` aliases some models also
# export) because M3's trial_predictions.csv omits the aliases entirely;
# category_long/true_block are present and give identical values across all
# 5 models (confirmed: category_long==0 <-> trial_type=="short", true_block
# =="L" <-> block_type=="long_block", etc.). Neutral-block ("N") trials are
# only ever included in "all".
TRIAL_SUBSETS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all": lambda df: pd.Series(True, index=df.index),
    "short_block": lambda df: df["true_block"] == "S",
    "long_block": lambda df: df["true_block"] == "L",
    "common_short": lambda df: (df["category_long"] == 0) & (df["true_block"] == "S"),
    "common_long": lambda df: (df["category_long"] == 1) & (df["true_block"] == "L"),
    "rare_short": lambda df: (df["category_long"] == 0) & (df["true_block"] == "L"),
    "rare_long": lambda df: (df["category_long"] == 1) & (df["true_block"] == "S"),
}
TRIAL_SUBSET_ORDER = list(TRIAL_SUBSETS)
TRIAL_SUBSET_LABELS = {
    "all": "All trials", "short_block": "Short block", "long_block": "Long block",
    "common_short": "Common short", "common_long": "Common long",
    "rare_short": "Rare short", "rare_long": "Rare long",
}

_SINGLE_SESSION_ROLE_MAP = {"train": "fine_tune_prefix", "test": "heldout"}

# Block-transition event window (trial 0 = first trial of the new block) and
# rare-trial event window (trial 0 = the rare trial itself), per the
# scientifically specified analysis.
BLOCK_TRANSITION_RELATIVE_RANGE = range(-5, 6)
RARE_TRIAL_RELATIVE_RANGE = range(-2, 3)
EVENT_CATEGORIES = (
    "block_transition_short_to_long", "block_transition_long_to_short",
    "rare_short", "rare_long",
)
EVENT_RELATIVE_POSITIONS = {
    "block_transition_short_to_long": tuple(BLOCK_TRANSITION_RELATIVE_RANGE),
    "block_transition_long_to_short": tuple(BLOCK_TRANSITION_RELATIVE_RANGE),
    "rare_short": tuple(RARE_TRIAL_RELATIVE_RANGE),
    "rare_long": tuple(RARE_TRIAL_RELATIVE_RANGE),
}


def _read_trial_predictions(run_dir: Path) -> Optional[pd.DataFrame]:
    """Read one window/session's per-trial predictions, or ``None`` if the
    file is absent -- a non-converged run (e.g. M4's 5 non-converged sliding
    windows) never writes ``trial_predictions.csv``; same silent-skip
    contract as ``common.read_split_session_metrics``. Restricted to
    ``model_evaluated`` trials, matching the convention already used in
    ``collect_sliding_window_global_pooled_metrics``.
    """
    path = run_dir / "trial_predictions.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, usecols=TRIAL_PREDICTION_COLUMNS)
    evaluated = df["model_evaluated"].astype(str).str.lower().eq("true")
    return df.loc[evaluated].copy()


def _trial_subset_rows(trials: pd.DataFrame, split_role_values: tuple) -> list:
    """(split_role, trial_subset) -> mean_nll/n_trials rows for one already-
    read, already-evaluated-filtered trials frame."""
    rows = []
    for split_role in split_role_values:
        split_trials = trials[trials["split_role"] == split_role]
        for subset_name, mask_fn in TRIAL_SUBSETS.items():
            subset_trials = split_trials[mask_fn(split_trials)]
            nll = pd.to_numeric(
                subset_trials["per_trial_negative_log_likelihood"], errors="coerce",
            ).dropna()
            rows.append({
                "split_role": split_role,
                "trial_subset": subset_name,
                "mean_nll": float(nll.mean()) if len(nll) else float("nan"),
                "n_trials": int(len(nll)),
            })
    return rows


def _detect_block_transition_events(trials: pd.DataFrame, relative_range: range) -> pd.DataFrame:
    """One row per surviving block-transition event: (session_id,
    event_category, anchor_trial_index, block_index). ``true_run_length==1``
    marks the first trial of a new block (trial 0). Drops the post-neutral
    transition (the first non-neutral block has no short/long "previous"
    block to compare against) and any transition whose displayed window
    would reach into a third block -- confirmed on real data this excludes
    <1% of transitions, so it is not over-strict. All boundary logic is
    scoped to one ``session_id`` at a time since ``trial_index`` resets per
    session within one window's pooled file.
    """
    lo, hi = relative_range.start, relative_range.stop - 1
    events = []
    for session_id, session_trials in trials.groupby("session_id"):
        available_trial_indices = set(session_trials["trial_index"])
        block_types = session_trials.drop_duplicates("block_index").set_index("block_index")["true_block"]
        block_bounds = session_trials.groupby("block_index")["trial_index"].agg(["min", "max"])
        starts = session_trials[session_trials["true_run_length"] == 1]
        for _, start_row in starts.iterrows():
            block_index = start_row["block_index"]
            if block_index not in block_types.index or (block_index - 1) not in block_types.index:
                continue
            new_type, prev_type = block_types[block_index], block_types[block_index - 1]
            if new_type == "L" and prev_type == "S":
                category = "block_transition_short_to_long"
            elif new_type == "S" and prev_type == "L":
                category = "block_transition_long_to_short"
            else:
                continue
            anchor = start_row["trial_index"]
            prev_lo, _ = block_bounds.loc[block_index - 1]
            _, new_hi = block_bounds.loc[block_index]
            if anchor + lo < prev_lo or anchor + hi > new_hi:
                continue
            if not all(anchor + offset in available_trial_indices for offset in relative_range):
                continue
            events.append({
                "session_id": session_id, "event_category": category,
                "anchor_trial_index": anchor,
            })
    return pd.DataFrame(events, columns=["session_id", "event_category", "anchor_trial_index"])


def _detect_rare_trial_events(trials: pd.DataFrame, relative_range: range) -> pd.DataFrame:
    """One row per surviving rare-trial event: (session_id, event_category,
    anchor_trial_index). Trial 0 = the rare trial itself. Drops any event
    whose displayed window would leave its own block -- avoiding conflating
    a rare-trial effect with an adjacent block-transition effect; confirmed
    on real data this excludes ~10% of rare-trial events, a modest discard,
    not an over-strict rule.
    """
    lo, hi = relative_range.start, relative_range.stop - 1
    events = []
    for session_id, session_trials in trials.groupby("session_id"):
        available_trial_indices = set(session_trials["trial_index"])
        block_bounds = session_trials.groupby("block_index")["trial_index"].agg(["min", "max"])
        for subset_name in ("rare_short", "rare_long"):
            mask = TRIAL_SUBSETS[subset_name](session_trials)
            for _, row in session_trials.loc[mask].iterrows():
                anchor, block_index = row["trial_index"], row["block_index"]
                blk_lo, blk_hi = block_bounds.loc[block_index]
                if anchor + lo < blk_lo or anchor + hi > blk_hi:
                    continue
                if not all(anchor + offset in available_trial_indices for offset in relative_range):
                    continue
                events.append({
                    "session_id": session_id, "event_category": subset_name,
                    "anchor_trial_index": anchor,
                })
    return pd.DataFrame(events, columns=["session_id", "event_category", "anchor_trial_index"])


def _aggregate_event_aligned_nll(
    trials: pd.DataFrame, events: pd.DataFrame, relative_range: range,
) -> pd.DataFrame:
    """For each surviving event, pull its window of relative-position NLL,
    then average across events sharing the same (event_category,
    relative_position) WITHIN this one window/session -- events are
    aggregated before being combined across windows/sessions, so individual
    trials are never treated as independent samples by the downstream paired
    tests.
    """
    if events.empty:
        return pd.DataFrame(columns=["event_category", "relative_position", "mean_nll", "n_events"])
    nll_by_session = {
        session_id: session_trials.set_index("trial_index")["per_trial_negative_log_likelihood"]
        for session_id, session_trials in trials.groupby("session_id")
    }
    rows = []
    for _, event in events.iterrows():
        nll_by_index = nll_by_session[event["session_id"]]
        for relative_position in relative_range:
            rows.append({
                "event_category": event["event_category"],
                "relative_position": relative_position,
                "nll": nll_by_index.get(event["anchor_trial_index"] + relative_position, float("nan")),
            })
    long_df = pd.DataFrame(rows)
    return long_df.groupby(["event_category", "relative_position"]).agg(
        mean_nll=("nll", "mean"), n_events=("nll", "count"),
    ).reset_index()


def _event_aligned_rows(trials: pd.DataFrame) -> list:
    """Event-aligned rows (event_category, relative_position, mean_nll,
    n_events_in_unit, n_events_at_position) for one already-read trials
    frame, covering both block transitions and rare trials. The two counts
    are retained separately so downstream tests can prove that every event
    contributes at every requested relative position."""
    transition_events = _detect_block_transition_events(trials, BLOCK_TRANSITION_RELATIVE_RANGE)
    rare_events = _detect_rare_trial_events(trials, RARE_TRIAL_RELATIVE_RANGE)
    transition_agg = _aggregate_event_aligned_nll(trials, transition_events, BLOCK_TRANSITION_RELATIVE_RANGE)
    rare_agg = _aggregate_event_aligned_nll(trials, rare_events, RARE_TRIAL_RELATIVE_RANGE)
    all_events = pd.concat([transition_events, rare_events], sort=False)
    n_events_by_category = (
        all_events.groupby("event_category").size() if not all_events.empty else pd.Series(dtype=int)
    )
    rows = []
    for agg in (transition_agg, rare_agg):
        for _, row in agg.iterrows():
            rows.append({
                "event_category": row["event_category"],
                "relative_position": int(row["relative_position"]),
                "mean_nll": row["mean_nll"],
                "n_events_in_unit": int(n_events_by_category.get(row["event_category"], 0)),
                "n_events_at_position": int(row["n_events"]),
            })
    return rows


# Figure 16 pages: "all" pools every session in a window regardless of split
# (the original, unrestricted behavior -- page 1); train/validation/test
# restrict block-transition/rare-trial detection and NLL aggregation to just
# that window's own sessions in that split (pages 2-4). A session belongs to
# exactly one split_role, so this is a simple filter, not a re-derivation.
EVENT_ALIGNED_SPLIT_ROLES = ("all", "train", "validation", "test")
EVENT_ALIGNED_SPLIT_LABELS = {
    "all": "All sessions (train + validation + test pooled)",
    "train": "Train sessions only",
    "validation": "Validation sessions only",
    "test": "Test session only",
}


def _event_aligned_rows_by_split(trials: pd.DataFrame) -> list:
    """``_event_aligned_rows`` computed once per ``EVENT_ALIGNED_SPLIT_ROLES``
    value, each row tagged with which. Figure 16's four pages read this
    table filtered to one ``split_role`` at a time."""
    rows = []
    for split_role in EVENT_ALIGNED_SPLIT_ROLES:
        subset = trials if split_role == "all" else trials.loc[trials["split_role"] == split_role]
        for row in _event_aligned_rows(subset):
            rows.append({"split_role": split_role, **row})
    return rows


# Figure 19: a wider +/-10 trial window around block transitions than Figure
# 16's NLL analysis (which uses BLOCK_TRANSITION_RELATIVE_RANGE, +/-5), and
# a "performance" (accuracy) metric rather than NLL, further split by the
# trial's own stimulus category (short/long) at each relative position.
BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE = range(-10, 11)
BLOCK_TRANSITION_PERFORMANCE_COLUMNS = (
    ("block_transition_short_to_long", "short", "Short trials, Short→Long transition"),
    ("block_transition_short_to_long", "long", "Long trials, Short→Long transition"),
    ("block_transition_long_to_short", "short", "Short trials, Long→Short transition"),
    ("block_transition_long_to_short", "long", "Long trials, Long→Short transition"),
)


def _aggregate_event_aligned_performance(
    trials: pd.DataFrame, events: pd.DataFrame, relative_range: range,
) -> pd.DataFrame:
    """For each surviving block-transition event, pull its window of
    relative-position mouse correctness / model expected correctness AND
    that position's own trial_type (short/long stimulus), then average
    across events sharing the same (event_category, trial_type,
    relative_position) WITHIN this one window -- mirrors
    ``_aggregate_event_aligned_nll``'s aggregate-within-window-first
    discipline, extended with the trial_type split Figure 19 needs.
    """
    columns = [
        "event_category", "trial_type", "relative_position",
        "mean_mouse_correct", "mean_model_probability_correct", "n_trials",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    by_session = {
        session_id: session_trials.set_index("trial_index")[
            ["category_long", "mouse_correct", "model_probability_correct"]
        ]
        for session_id, session_trials in trials.groupby("session_id")
    }
    rows = []
    for _, event in events.iterrows():
        lookup = by_session[event["session_id"]]
        for relative_position in relative_range:
            trial = lookup.loc[event["anchor_trial_index"] + relative_position]
            category = trial["category_long"]
            trial_type = "short" if category == 0 else ("long" if category == 1 else None)
            if trial_type is None:
                continue
            rows.append({
                "event_category": event["event_category"],
                "trial_type": trial_type,
                "relative_position": relative_position,
                "mouse_correct": trial["mouse_correct"],
                "model_probability_correct": trial["model_probability_correct"],
            })
    if not rows:
        return pd.DataFrame(columns=columns)
    long_df = pd.DataFrame(rows)
    return long_df.groupby(["event_category", "trial_type", "relative_position"]).agg(
        mean_mouse_correct=("mouse_correct", "mean"),
        mean_model_probability_correct=("model_probability_correct", "mean"),
        n_trials=("mouse_correct", "count"),
    ).reset_index()


def _block_transition_performance_rows_by_split(trials: pd.DataFrame) -> list:
    """Figure 19's rows for one already-read window's trials: one
    ``_aggregate_event_aligned_performance`` pass per
    ``EVENT_ALIGNED_SPLIT_ROLES`` value, mirroring
    ``_event_aligned_rows_by_split``."""
    rows = []
    for split_role in EVENT_ALIGNED_SPLIT_ROLES:
        subset = trials if split_role == "all" else trials.loc[trials["split_role"] == split_role]
        events = _detect_block_transition_events(subset, BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE)
        agg = _aggregate_event_aligned_performance(
            subset, events, BLOCK_TRANSITION_PERFORMANCE_RELATIVE_RANGE,
        )
        for _, row in agg.iterrows():
            rows.append({
                "split_role": split_role,
                "event_category": row["event_category"],
                "trial_type": row["trial_type"],
                "relative_position": int(row["relative_position"]),
                "mean_mouse_correct": row["mean_mouse_correct"],
                "mean_model_probability_correct": row["mean_model_probability_correct"],
                "n_trials": int(row["n_trials"]),
            })
    return rows


def collect_sliding_window_trial_level_tables(mouse_id: str) -> tuple:
    """Single pass over every window's ``trial_predictions.csv`` (read once
    per model/window) producing the trial-subset metrics table (Figure 01:
    one row per (model, window, split_role, trial_subset)), the event-aligned
    NLL table (Figure 16: one row per (model, window, split_role,
    event_category, relative_position), split_role in
    ``EVENT_ALIGNED_SPLIT_ROLES``), and the block-transition performance-by-
    trial-type table (Figure 19: one row per (model, window, split_role,
    event_category, trial_type, relative_position)) -- reading each file
    three times for three separate tables would triple this step's I/O for
    no benefit.

    Returns ``(trial_subset_metrics, event_aligned_nll, block_transition_performance)``.
    """
    subset_rows, event_rows, block_transition_rows = [], [], []
    for model_id in MODEL_ORDER:
        for window_dir in common.list_window_dirs(model_id, mouse_id):
            manifest = common.read_json(window_dir / "window_manifest.json") or {}
            window_id = manifest.get("window_id")
            trials = _read_trial_predictions(window_dir)
            if trials is None:
                continue
            for row in _trial_subset_rows(trials, ("train", "validation", "test")):
                subset_rows.append({"model_id": model_id, "window_id": window_id, **row})
            for row in _event_aligned_rows_by_split(trials):
                event_rows.append({"model_id": model_id, "window_id": window_id, **row})
            for row in _block_transition_performance_rows_by_split(trials):
                block_transition_rows.append({"model_id": model_id, "window_id": window_id, **row})
    # Explicit columns so an entirely empty result (e.g. no window anywhere
    # produced a qualifying event) still has a readable schema instead of a
    # columnless DataFrame that breaks downstream groupby/column access.
    subset_columns = ["model_id", "window_id", "split_role", "trial_subset", "mean_nll", "n_trials"]
    event_columns = [
        "model_id", "window_id", "split_role", "event_category", "relative_position",
        "mean_nll", "n_events_in_unit", "n_events_at_position",
    ]
    block_transition_columns = [
        "model_id", "window_id", "split_role", "event_category", "trial_type", "relative_position",
        "mean_mouse_correct", "mean_model_probability_correct", "n_trials",
    ]
    return (
        pd.DataFrame(subset_rows, columns=subset_columns),
        pd.DataFrame(event_rows, columns=event_columns),
        pd.DataFrame(block_transition_rows, columns=block_transition_columns),
    )


def collect_single_session_trial_level_tables(mouse_id: str) -> tuple:
    """Mirrors ``collect_sliding_window_trial_level_tables`` for the
    single-session fine-tuning directories (Figures 03/17). ``split_role``
    in ``trial_predictions.csv`` uses train/test; mapped to
    fine_tune_prefix/heldout exactly like ``collect_single_session_metrics``.
    """
    subset_rows, event_rows = [], []
    for model_id in MODEL_ORDER:
        for session_dir in common.list_session_dirs(model_id, mouse_id):
            manifest = common.read_json(session_dir / "single_session_manifest.json") or {}
            session_id = manifest.get("session_id")
            trials = _read_trial_predictions(session_dir)
            if trials is None:
                continue
            for row in _trial_subset_rows(trials, tuple(_SINGLE_SESSION_ROLE_MAP)):
                row = {**row, "analysis_role": _SINGLE_SESSION_ROLE_MAP[row.pop("split_role")]}
                subset_rows.append({"model_id": model_id, "session_id": session_id, **row})
            for row in _event_aligned_rows(trials):
                event_rows.append({"model_id": model_id, "session_id": session_id, **row})
    subset_columns = ["model_id", "session_id", "analysis_role", "trial_subset", "mean_nll", "n_trials"]
    event_columns = [
        "model_id", "session_id", "event_category", "relative_position",
        "mean_nll", "n_events_in_unit", "n_events_at_position",
    ]
    return pd.DataFrame(subset_rows, columns=subset_columns), pd.DataFrame(event_rows, columns=event_columns)


# ---------------------------------------------------------------------------
# Psychometric comparison (Figure 18): sliding-window zero-shot test
# predictions vs. single-session fine-tuned train predictions, on the
# IDENTICAL trial population for every model.
# ---------------------------------------------------------------------------

def _add_trial_type_and_block_type(trials: pd.DataFrame) -> pd.DataFrame:
    """Derive ``trial_type``/``block_type`` from ``category_long``/``true_block``
    -- the two raw columns present in every model's trial_predictions.csv
    (unlike the friendlier aliases, which M3/M4 omit; see
    ``TRIAL_SUBSETS``'s module comment)."""
    result = trials.copy()
    category = pd.to_numeric(result["category_long"], errors="coerce")
    result["trial_type"] = np.select(
        [category.eq(0), category.eq(1)], ["short", "long"], default="undefined",
    )
    result["block_type"] = result["true_block"].map(
        {"N": "neutral", "S": "short_block", "L": "long_block"}
    ).fillna("undefined")
    return result


def collect_psychometric_comparison_trials(mouse_id: str) -> pd.DataFrame:
    """Trial-level rows for Figure 18, on an IDENTICAL trial population for
    every model and both sources.

    For a session to enter this table, EVERY model must have both a
    sliding-window test window and a numerically sound single-session
    fine-tune for it (see ``_numerically_sound_fit``; the sliding-window
    side needs no analogous check -- a rolling window's winner-selection
    already only ever keeps a converged run, see ``_per_model_window_ids``).
    Within a shared session, the actual ``trial_index`` set used from each
    source is the intersection across all five models, protecting against
    one model silently evaluating a slightly different trial subset within
    an otherwise-shared session (e.g. a warm-up or omission-handling
    difference) -- without this, "same trials for every model" would only
    be true at the session level, not the trial level.

    The sliding-window side is additionally trimmed to exclude that
    session's final two blocks (by ``block_index``), matching exactly what
    the single-session fine-tune's training prefix already excludes by
    construction (``fine_tune_scope: "all blocks before the literal final
    two blocks"``), so both sources describe the same trial range: this is
    what makes "sliding-window test" and "single-session fine-tuning" a fair,
    like-for-like overlay instead of comparing different amounts of data.
    """
    sw_by_model: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in MODEL_ORDER}
    ss_by_model: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in MODEL_ORDER}

    # Single-session fine-tuning first: its own train/heldout split already
    # encodes "all blocks before the literal final two blocks" correctly and
    # consistently across every model (verified: all 5 models agree on the
    # exact same block set for a shared session). The sliding-window side's
    # OWN "test" split cannot be used to rediscover that boundary -- a whole
    # trailing block can have zero ``model_evaluated`` rows there for reasons
    # unrelated to the block structure (observed on MC06/M0: block 12 was
    # entirely unevaluated in the sliding-window test split alone), which
    # would silently shift "last two blocks" and drop a real block. So the
    # highest block_index kept by single-session fine-tuning is the
    # authoritative boundary for trimming the sliding-window side too.
    max_train_block_by_session: dict[str, list[int]] = {}
    for model_id in MODEL_ORDER:
        for session_dir in common.list_session_dirs(model_id, mouse_id):
            if _numerically_sound_fit(session_dir) is not True:
                continue
            trials = _read_trial_predictions(session_dir)
            if trials is None:
                continue
            train_trials = trials.loc[trials["split_role"] == "train"]
            if train_trials.empty:
                continue
            session_ids = train_trials["session_id"].unique()
            if len(session_ids) != 1:
                raise ValueError(
                    f"Expected exactly one session in {session_dir}, got {session_ids}"
                )
            session_id = session_ids[0]
            if session_id in ss_by_model[model_id]:
                raise ValueError(
                    f"Duplicate single-session fine-tune session {session_id!r} for {model_id}"
                )
            ss_by_model[model_id][session_id] = train_trials
            max_train_block_by_session.setdefault(session_id, []).append(
                int(train_trials["block_index"].max())
            )
    # The conservative (minimum) boundary across models that share this
    # session, so trimming never keeps a block some model's fine-tune prefix
    # excluded.
    keep_up_to_block = {
        session_id: min(values) for session_id, values in max_train_block_by_session.items()
    }

    for model_id in MODEL_ORDER:
        for window_dir in common.list_window_dirs(model_id, mouse_id):
            trials = _read_trial_predictions(window_dir)
            if trials is None:
                continue
            test_trials = trials.loc[trials["split_role"] == "test"]
            if test_trials.empty:
                continue
            session_ids = test_trials["session_id"].unique()
            if len(session_ids) != 1:
                raise ValueError(
                    f"Expected exactly one test session in {window_dir}, got {session_ids}"
                )
            session_id = session_ids[0]
            if session_id not in keep_up_to_block:
                continue
            if session_id in sw_by_model[model_id]:
                raise ValueError(
                    f"Duplicate sliding-window test session {session_id!r} for {model_id}"
                )
            sw_by_model[model_id][session_id] = test_trials.loc[
                test_trials["block_index"] <= keep_up_to_block[session_id]
            ]

    sessions_per_model = [set(sw_by_model[m]) & set(ss_by_model[m]) for m in MODEL_ORDER]
    common_sessions = set.intersection(*sessions_per_model) if sessions_per_model else set()

    rows = []
    for session_id in sorted(common_sessions):
        sw_common_index = set.intersection(
            *(set(sw_by_model[m][session_id]["trial_index"]) for m in MODEL_ORDER)
        )
        ss_common_index = set.intersection(
            *(set(ss_by_model[m][session_id]["trial_index"]) for m in MODEL_ORDER)
        )
        # Each of the two lines above only guarantees "same trials across
        # models WITHIN one source" -- it does not by itself guarantee the
        # two sources agree with each other, even though both are meant to
        # describe the same session's same (block-trimmed) trial range. A
        # trial some model's zero-shot sliding-window evaluation happened to
        # skip (for a reason unrelated to the block structure -- the same
        # kind of gap already documented above for block_index) but its own
        # single-session fine-tune did not (or vice versa) would otherwise
        # let the two plotted sources silently diverge at the trial level
        # despite this function's own "IDENTICAL trial population ... for
        # every model and both sources" docstring. Intersecting the two
        # index sets here is what actually enforces that claim.
        common_index = sw_common_index & ss_common_index
        if not common_index:
            continue
        for model_id in MODEL_ORDER:
            sw_rows = sw_by_model[model_id][session_id]
            sw_rows = sw_rows.loc[sw_rows["trial_index"].isin(common_index)].copy()
            sw_rows["model_id"] = model_id
            sw_rows["source"] = "sliding_window"
            rows.append(sw_rows)

            ss_rows = ss_by_model[model_id][session_id]
            ss_rows = ss_rows.loc[ss_rows["trial_index"].isin(common_index)].copy()
            ss_rows["model_id"] = model_id
            ss_rows["source"] = "single_session"
            rows.append(ss_rows)

    if not rows:
        raise RuntimeError(
            f"No matched sliding-window/single-session trial population found for "
            f"mouse={mouse_id!r}: every model must share both a sliding-window test "
            "window and a numerically sound single-session fine-tune for at least one session."
        )
    table = pd.concat(rows, ignore_index=True, sort=False)
    return _add_trial_type_and_block_type(table)


_PSYCHOMETRIC_BLOCK_SUBSETS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all": lambda df: df["block_type"].isin(["short_block", "long_block"]),
    "short_block": lambda df: df["block_type"].eq("short_block"),
    "long_block": lambda df: df["block_type"].eq("long_block"),
    "neutral": lambda df: df["block_type"].eq("neutral"),
}
PSYCHOMETRIC_BLOCK_ORDER = list(_PSYCHOMETRIC_BLOCK_SUBSETS)
PSYCHOMETRIC_BLOCK_LABELS = {
    "all": "All short + long blocks", "short_block": "Short block",
    "long_block": "Long block", "neutral": "Neutral block",
}
PSYCHOMETRIC_SOURCE_LABELS = {
    "sliding_window": "Sliding-window (zero-shot)",
    "single_session": "Single-session (fine-tuned)",
}


def _four_bin_psychometric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Session-balanced psychometric summary in four duration quantile bins
    per trial type (short/long).

    Reimplemented (rather than imported) from
    ``src/m0_sensory_v1/analysis.py::four_bin_psychometric_summary`` so
    ``post_analysis`` -- which never imports from ``src/`` and is run with a
    different ``PYTHONPATH`` convention -- stays self-contained. Deterministic
    N(0, 1e-6) jitter (seed 42) breaks duration ties before quantile-cutting,
    matching the original exactly.
    """
    work = data.loc[data["trial_type"].isin(["short", "long"])].copy()
    parts = []
    for offset, trial_type in ((0, "short"), (4, "long")):
        subset = work.loc[work["trial_type"].eq(trial_type)].copy()
        if subset.empty:
            continue
        jitter = np.random.RandomState(42).normal(0.0, 1e-6, len(subset))
        labels = pd.qcut(
            subset["duration_s"].to_numpy(float) + jitter, q=4, labels=False, duplicates="drop",
        )
        subset["within_type_bin"] = np.asarray(labels, dtype=int)
        if subset["within_type_bin"].nunique() != 4:
            raise ValueError(f"Expected four {trial_type} bins")
        subset["bin_order"] = subset["within_type_bin"] + offset
        parts.append(subset)
    if not parts:
        return pd.DataFrame()
    binned = pd.concat(parts, ignore_index=True)
    metadata = binned.groupby("bin_order", observed=False).agg(
        bin_center_s=("duration_s", "mean"), n_trials=("duration_s", "size"),
    ).reset_index()
    per_session = binned.groupby(["bin_order", "session_id"], observed=False).agg(
        mouse_probability_long=("choice_long", "mean"),
        model_probability_long=("model_probability_long", "mean"),
    ).reset_index()
    summary = per_session.groupby("bin_order", observed=False).agg(
        mouse_probability_long=("mouse_probability_long", "mean"),
        mouse_sem=("mouse_probability_long", "sem"),
        model_probability_long=("model_probability_long", "mean"),
        model_sem=("model_probability_long", "sem"),
        n_sessions=("session_id", "nunique"),
    ).reset_index()
    summary = summary.merge(metadata, on="bin_order", validate="one_to_one")
    summary["trial_type"] = np.where(summary["bin_order"] < 4, "short", "long")
    return summary.sort_values("bin_order").reset_index(drop=True)


def summarize_psychometric_comparison(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (model, source, block_subset): the four-quantile-bin psychometric
    summary Figure 18 plots, precomputed here so ``figures.py`` only ever
    reads a tidy CSV (this module's usual contract)."""
    rows = []
    for model_id in MODEL_ORDER:
        for source in ("sliding_window", "single_session"):
            subset = trials.loc[(trials["model_id"] == model_id) & (trials["source"] == source)]
            if subset.empty:
                continue
            for block_key in PSYCHOMETRIC_BLOCK_ORDER:
                panel = subset.loc[_PSYCHOMETRIC_BLOCK_SUBSETS[block_key](subset)]
                summary = _four_bin_psychometric_summary(panel)
                if summary.empty:
                    continue
                summary.insert(0, "model_id", model_id)
                summary.insert(1, "source", source)
                summary.insert(2, "block_subset", block_key)
                summary["n_matched_sessions"] = subset["session_id"].nunique()
                rows.append(summary)
    columns = [
        "model_id", "source", "block_subset", "bin_order", "trial_type",
        "bin_center_s", "n_trials", "n_sessions", "n_matched_sessions",
        "mouse_probability_long", "mouse_sem", "model_probability_long", "model_sem",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True, sort=False)[columns]


def common_trial_subset_window_ids(
    sw_trial_subset: pd.DataFrame, mouse_id: str, split_role: str, trial_subset: str,
) -> set:
    """Windows with one finite subset NLL for every model, for one
    (split_role, trial_subset) slice -- generalizes ``common_metric_window_ids``
    to the trial-subset grid (Figure 01). A window only counts for a given
    subset if every model actually had at least one qualifying trial there
    (e.g. a short test session could have zero rare-long trials).
    """
    common_ids = common_window_ids(mouse_id)
    subset = sw_trial_subset[
        (sw_trial_subset["split_role"] == split_role)
        & (sw_trial_subset["trial_subset"] == trial_subset)
        & (sw_trial_subset["window_id"].isin(common_ids))
    ]
    matrix = mc_stats.complete_case_matrix(subset, "window_id", "mean_nll")
    return common_ids & set(matrix.index)


def common_trial_subset_session_ids(
    ss_trial_subset: pd.DataFrame, analysis_role: str, trial_subset: str,
) -> set:
    """Sessions with one finite subset NLL for every model, for one
    (analysis_role, trial_subset) slice -- generalizes ``common_session_ids``
    to the trial-subset grid (Figure 03)."""
    subset = ss_trial_subset[
        (ss_trial_subset["analysis_role"] == analysis_role)
        & (ss_trial_subset["trial_subset"] == trial_subset)
    ]
    matrix = mc_stats.complete_case_matrix(subset, "session_id", "mean_nll")
    return set(matrix.index)


def compute_sliding_window_trial_subset_paired_tests(
    sw_trial_subset: pd.DataFrame, mouse_id: str,
) -> pd.DataFrame:
    """Four adjacent-model paired tests per (split_role, trial_subset) --
    the Figure 01 trial-subset grid's per-subplot statistics. Mirrors
    ``compute_sliding_window_paired_tests`` with an extra trial_subset loop.
    """
    pairwise_frames = []
    for split_role in ("train", "validation", "test"):
        for trial_subset in TRIAL_SUBSET_ORDER:
            common_ids = common_trial_subset_window_ids(sw_trial_subset, mouse_id, split_role, trial_subset)
            subset = sw_trial_subset[
                (sw_trial_subset["split_role"] == split_role)
                & (sw_trial_subset["trial_subset"] == trial_subset)
                & (sw_trial_subset["window_id"].isin(common_ids))
            ]
            matrix = mc_stats.complete_case_matrix(subset, "window_id", "mean_nll")
            pairwise = mc_stats.adjacent_pairwise_tests(matrix)
            pairwise.insert(0, "trial_subset", trial_subset)
            pairwise.insert(0, "split_role", split_role)
            pairwise["units_independent"] = False
            pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


def compute_single_session_trial_subset_paired_tests(ss_trial_subset: pd.DataFrame) -> pd.DataFrame:
    """Mirrors ``compute_sliding_window_trial_subset_paired_tests`` for the
    single-session (fine_tune_prefix/heldout) trial-subset grid (Figure 03).
    """
    pairwise_frames = []
    for analysis_role in ("fine_tune_prefix", "heldout"):
        for trial_subset in TRIAL_SUBSET_ORDER:
            common_ids = common_trial_subset_session_ids(ss_trial_subset, analysis_role, trial_subset)
            subset = ss_trial_subset[
                (ss_trial_subset["analysis_role"] == analysis_role)
                & (ss_trial_subset["trial_subset"] == trial_subset)
                & (ss_trial_subset["session_id"].isin(common_ids))
            ]
            matrix = mc_stats.complete_case_matrix(subset, "session_id", "mean_nll")
            pairwise = mc_stats.adjacent_pairwise_tests(matrix)
            pairwise.insert(0, "trial_subset", trial_subset)
            pairwise.insert(0, "analysis_role", analysis_role)
            pairwise["units_independent"] = True
            pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


def collect_sliding_window_trial_subset_pooled_diamonds(
    sw_trial_subset: pd.DataFrame, mouse_id: str,
) -> pd.DataFrame:
    """Trial-weighted pooled NLL per (model, split_role, trial_subset) -- the
    Figure 01 diamond marker for every subset column, not just "all" (Figure
    03 stays diamond-free, matching its current design). Computed as
    sum(mean_nll * n_trials) / sum(n_trials) over the windows common to that
    (split, subset); algebraically identical to pooling the raw per-trial
    NLL directly (sum_j mean_j*n_j / sum_j n_j == sum_j sum_i nll_ij / sum_j n_j),
    since n_trials uses the exact same evaluated-trial filter as mean_nll
    (both built together in ``_trial_subset_rows``).
    """
    rows = []
    for split_role in ("train", "validation", "test"):
        for trial_subset in TRIAL_SUBSET_ORDER:
            common_ids = common_trial_subset_window_ids(sw_trial_subset, mouse_id, split_role, trial_subset)
            subset = sw_trial_subset[
                (sw_trial_subset["split_role"] == split_role)
                & (sw_trial_subset["trial_subset"] == trial_subset)
                & (sw_trial_subset["window_id"].isin(common_ids))
            ]
            for model_id in MODEL_ORDER:
                model_subset = subset[subset["model_id"] == model_id]
                total_trials = model_subset["n_trials"].sum()
                pooled_nll = (
                    (model_subset["mean_nll"] * model_subset["n_trials"]).sum() / total_trials
                    if total_trials else float("nan")
                )
                rows.append({
                    "model_id": model_id, "split_role": split_role, "trial_subset": trial_subset,
                    "mean_nll": pooled_nll, "n_trials": int(total_trials), "n_windows": len(model_subset),
                })
    return pd.DataFrame(rows)


def event_aligned_unit_scalar(
    event_aligned: pd.DataFrame, unit_col: str, event_category: str,
    relative_positions: Optional[tuple[int, ...]] = None,
    split_role: Optional[str] = None,
) -> pd.DataFrame:
    """Collapse one event trajectory to a single value per (model, unit).

    ``relative_positions=None`` means the category's complete registered
    display range; ``(0,)`` gives the event-anchor NLL. Exact position
    coverage and row uniqueness are required before averaging so a finite
    scalar can never silently represent only part of the requested range.

    ``split_role`` selects one of Figure 16's four pages
    (``EVENT_ALIGNED_SPLIT_ROLES``) when ``event_aligned`` carries a
    ``split_role`` column (sliding-window only); left ``None`` for
    single-session tables, which have no such column and only one implicit
    page.
    """
    subset = event_aligned[event_aligned["event_category"] == event_category]
    if split_role is not None:
        subset = subset[subset["split_role"] == split_role]
    columns = [unit_col, "model_id", "mean_nll"]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    positions = tuple(
        EVENT_RELATIVE_POSITIONS[event_category]
        if relative_positions is None else relative_positions
    )
    if not positions:
        raise ValueError("Event-aligned scalar requires at least one relative position")
    subset = subset[subset["relative_position"].isin(positions)].copy()

    finite_nll = np.isfinite(pd.to_numeric(subset["mean_nll"], errors="coerce"))
    if not finite_nll.all():
        bad = subset.loc[~finite_nll, [unit_col, "model_id", "relative_position"]]
        raise ValueError(
            "Non-finite event-aligned NLL in requested positions for "
            f"{event_category}: {bad.head().to_dict('records')}"
        )

    counts = subset.groupby([unit_col, "model_id", "relative_position"]).size()
    if (counts > 1).any():
        duplicates = counts[counts > 1].index.tolist()
        raise ValueError(
            f"Duplicate event-aligned (unit, model, position) rows: {duplicates[:5]}"
        )
    expected = frozenset(positions)
    observed = subset.groupby([unit_col, "model_id"])["relative_position"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    incomplete = observed[observed != expected]
    if not incomplete.empty:
        raise ValueError(
            "Incomplete event-aligned relative-position coverage for "
            f"{event_category}: {incomplete.index.tolist()[:5]}"
        )
    return subset.groupby([unit_col, "model_id"])["mean_nll"].mean().reset_index()


def _validate_matched_event_counts(
    event_aligned: pd.DataFrame, unit_col: str, event_category: str, unit_ids: set,
    split_role: Optional[str] = None,
) -> None:
    """Require the matched models to summarize the same event count per unit."""
    if not unit_ids:
        return
    subset = event_aligned[
        (event_aligned["event_category"] == event_category)
        & (event_aligned[unit_col].isin(unit_ids))
    ]
    if split_role is not None:
        subset = subset[subset["split_role"] == split_role]
    required_count_columns = {"n_events_in_unit", "n_events_at_position"}
    missing_columns = required_count_columns - set(subset.columns)
    if missing_columns:
        raise ValueError(
            "Event-aligned table is missing count provenance columns: "
            f"{sorted(missing_columns)}; rerun collection"
        )
    count_values = subset[list(required_count_columns)].apply(pd.to_numeric, errors="coerce")
    invalid_counts = (
        ~np.isfinite(count_values).all(axis=1)
        | count_values.le(0).any(axis=1)
        | count_values["n_events_at_position"].ne(count_values["n_events_in_unit"])
    )
    if invalid_counts.any():
        bad = subset.loc[
            invalid_counts,
            [unit_col, "model_id", "relative_position", *sorted(required_count_columns)],
        ]
        raise ValueError(
            "Not every event contributes at every requested relative position for "
            f"{event_category}: {bad.head().to_dict('records')}"
        )
    per_pair_nunique = subset.groupby([unit_col, "model_id"])["n_events_in_unit"].nunique()
    if (per_pair_nunique > 1).any():
        raise ValueError(
            f"Inconsistent event counts across positions for {event_category}"
        )
    per_pair = subset.groupby([unit_col, "model_id"])["n_events_in_unit"].first()
    wide = per_pair.unstack("model_id").reindex(columns=MODEL_ORDER)
    wide = wide.reindex(sorted(unit_ids))
    if wide.isna().any(axis=None) or not wide.nunique(axis=1, dropna=False).eq(1).all():
        raise ValueError(
            f"Event counts are not identical across models for {event_category}"
        )


def common_event_window_ids(
    event_aligned: pd.DataFrame, mouse_id: str, event_category: str,
    relative_positions: Optional[tuple[int, ...]] = None,
    split_role: Optional[str] = None,
) -> set:
    """Windows with one finite windowed-average NLL for every model, for one
    event_category -- generalizes ``common_metric_window_ids`` to the
    event-aligned tables (a window can have zero qualifying events of one
    category, e.g. zero rare-short events, even with plenty of rare-long).

    ``split_role`` picks one of Figure 16's four pages; see
    ``event_aligned_unit_scalar``.
    """
    common_ids = common_window_ids(mouse_id)
    scalars = event_aligned_unit_scalar(
        event_aligned, "window_id", event_category, relative_positions, split_role,
    )
    scalars = scalars[scalars["window_id"].isin(common_ids)]
    matrix = mc_stats.complete_case_matrix(scalars, "window_id", "mean_nll")
    matched_ids = common_ids & set(matrix.index)
    _validate_matched_event_counts(
        event_aligned, "window_id", event_category, matched_ids, split_role,
    )
    return matched_ids


def common_event_session_ids(
    event_aligned: pd.DataFrame, event_category: str,
    relative_positions: Optional[tuple[int, ...]] = None,
) -> set:
    """Sessions with one finite windowed-average NLL for every model, for one
    event_category -- generalizes ``common_session_ids`` to the event-aligned
    tables."""
    scalars = event_aligned_unit_scalar(
        event_aligned, "session_id", event_category, relative_positions,
    )
    matrix = mc_stats.complete_case_matrix(scalars, "session_id", "mean_nll")
    matched_ids = set(matrix.index)
    _validate_matched_event_counts(event_aligned, "session_id", event_category, matched_ids)
    return matched_ids


def compute_sliding_window_event_aligned_paired_tests(
    sw_event_aligned: pd.DataFrame, mouse_id: str, split_role: str = "all",
) -> pd.DataFrame:
    """Four adjacent-model paired tests per event_category for one page of
    Figure 16 (``split_role`` in ``EVENT_ALIGNED_SPLIT_ROLES``; ``"all"`` is
    the original page-1 pooled behavior). Each window's paired unit is its
    windowed-average NLL over the whole displayed relative-position range
    (see ``event_aligned_unit_scalar``).
    """
    pairwise_frames = []
    for event_category in EVENT_CATEGORIES:
        common_ids = common_event_window_ids(
            sw_event_aligned, mouse_id, event_category, split_role=split_role,
        )
        scalars = event_aligned_unit_scalar(
            sw_event_aligned, "window_id", event_category, split_role=split_role,
        )
        scalars = scalars[scalars["window_id"].isin(common_ids)]
        matrix = mc_stats.complete_case_matrix(scalars, "window_id", "mean_nll")
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        if pairwise.empty:
            continue
        pairwise.insert(0, "event_category", event_category)
        pairwise.insert(1, "split_role", split_role)
        pairwise["units_independent"] = False
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


def compute_sliding_window_event_aligned_trial0_paired_tests(
    sw_event_aligned: pd.DataFrame, mouse_id: str, split_role: str = "all",
) -> pd.DataFrame:
    """Adjacent paired tests at event trial 0 for one page of Figure 16: the
    first trial of the new block for the two transition categories, and the
    rare trial itself for the two rare categories (``split_role`` as above).

    Each rolling window contributes one already-within-window-aggregated NLL
    at that anchor trial. These p-values are exploratory because the rolling
    windows reuse sessions and fitted history.
    """
    pairwise_frames = []
    for event_category in EVENT_CATEGORIES:
        common_ids = common_event_window_ids(
            sw_event_aligned, mouse_id, event_category, (0,), split_role,
        )
        scalars = event_aligned_unit_scalar(
            sw_event_aligned, "window_id", event_category, (0,), split_role,
        )
        scalars = scalars[scalars["window_id"].isin(common_ids)]
        matrix = mc_stats.complete_case_matrix(scalars, "window_id", "mean_nll")
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        if pairwise.empty:
            continue
        pairwise.insert(0, "event_category", event_category)
        pairwise.insert(1, "split_role", split_role)
        pairwise["units_independent"] = False
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


def compute_single_session_event_aligned_paired_tests(ss_event_aligned: pd.DataFrame) -> pd.DataFrame:
    """Mirrors ``compute_sliding_window_event_aligned_paired_tests`` for
    Figure 17 (single-session fine-tuning)."""
    pairwise_frames = []
    for event_category in EVENT_CATEGORIES:
        common_ids = common_event_session_ids(ss_event_aligned, event_category)
        scalars = event_aligned_unit_scalar(ss_event_aligned, "session_id", event_category)
        scalars = scalars[scalars["session_id"].isin(common_ids)]
        matrix = mc_stats.complete_case_matrix(scalars, "session_id", "mean_nll")
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        if pairwise.empty:
            continue
        pairwise.insert(0, "event_category", event_category)
        pairwise["units_independent"] = True
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Information criteria (AIC/BIC): penalized in-sample fit comparison, a
# different lens from the held-out NLL comparisons above. See
# ``figures.fig_information_criteria`` and ``report.py`` for why BIC is
# highlighted as the primary panel.
# ---------------------------------------------------------------------------

def collect_information_criteria(mouse_id: str, sw_split_session: pd.DataFrame) -> pd.DataFrame:
    """Per (model, window) AIC/BIC from that window's own pooled TRAINING
    fit (``split_role="train"``, the actual fitted objective -- see
    ``docs/models/M0_sensory_v1.md`` Part II's ``J_pooled`` and Part IX's
    ``ELPD``/``MeanNLL`` -- so ``nll_total = -elpd`` is the exact quantity
    the optimizer maximized the negative of, not a reconstruction).

    Restricted to ``common_metric_window_ids`` (all 5 models, finite metrics
    in all three panels, and only converged
    winners by construction -- see that function's docstring) so AIC/BIC
    are always compared on genuinely fitted, equal-population likelihoods.

    ``k`` = number of free parameters for that model (nested family, 5-9).
    ``n`` = that window's pooled training trial count. Both AIC (2k penalty)
    and BIC (k*ln(n) penalty) are computed; BIC is reported as the primary
    criterion in the figure/report (see their docstrings) but AIC is kept
    for transparency. Per-window delta-AIC/BIC and Akaike/Schwarz weights
    are normalized against that window's own best (lowest-IC) model, the
    standard Burnham & Anderson convention, so magnitudes are comparable
    across windows with different trial counts.
    """
    common_ids = common_metric_window_ids(sw_split_session, mouse_id)
    subset = sw_split_session[
        (sw_split_session["split_role"] == "train")
        & (sw_split_session["session_id"] == "ALL_SESSIONS")
        & (sw_split_session["window_id"].isin(common_ids))
        & (sw_split_session["converged"] == True)  # noqa: E712 -- explicit bool column check
    ].copy()
    same_n_ids = _same_positive_trial_count_ids(subset, "window_id")
    subset = subset[subset["window_id"].isin(same_n_ids)]
    rows = []
    for _, row in subset.iterrows():
        model_id = row["model_id"]
        k = len(MODEL_SPECS[model_id].param_names)
        n = row["valid_trial_count"]
        nll_total = -row["elpd"]
        rows.append({
            "model_id": model_id, "window_id": row["window_id"], "test_rank": row["test_rank"],
            "n_trials": int(n), "n_params": k, "nll_total": float(nll_total),
            "aic": 2 * k + 2 * nll_total, "bic": k * np.log(n) + 2 * nll_total,
        })
    columns = [
        "model_id", "window_id", "test_rank", "n_trials", "n_params",
        "nll_total", "aic", "bic",
    ]
    return _finish_information_criteria_table(pd.DataFrame(rows, columns=columns), "window_id")


def _same_positive_trial_count_ids(table: pd.DataFrame, unit_col: str) -> set:
    """Complete units whose five models use one identical positive trial N."""
    matrix = mc_stats.complete_case_matrix(table, unit_col, "valid_trial_count")
    if matrix.empty:
        return set()
    same_n = matrix.nunique(axis=1).eq(1)
    positive = matrix.gt(0).all(axis=1)
    return set(matrix.index[same_n & positive])


def _finish_information_criteria_table(table: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """Add within-unit deltas and normalized IC weights."""
    if table.empty:
        for metric in ("aic", "bic"):
            table[f"delta_{metric}"] = pd.Series(dtype=float)
            table[f"{metric}_weight"] = pd.Series(dtype=float)
        return table
    matrix = mc_stats.complete_case_matrix(table, unit_col, "aic")
    table = table[table[unit_col].isin(matrix.index)].copy()
    for metric in ("aic", "bic"):
        best = table.groupby(unit_col)[metric].transform("min")
        table[f"delta_{metric}"] = table[metric] - best
        raw_weight = np.exp(-0.5 * table[f"delta_{metric}"])
        table[f"{metric}_weight"] = raw_weight / table.groupby(unit_col)[f"delta_{metric}"].transform(
            lambda values: np.exp(-0.5 * values).sum()
        )
    return table


def collect_single_session_information_criteria(ss_metrics: pd.DataFrame) -> pd.DataFrame:
    """AIC/BIC from numerically sound single-session fine-tune fits.

    Only the fine-tune prefix enters the likelihood.  A session is retained
    when all five models have exactly one finite training ELPD, are
    numerically sound (``numerically_sound``, see ``_numerically_sound_fit``),
    and use the same valid-trial count.

    This deliberately does NOT require each model's own binary ``converged``
    flag: on MC06 that flag's absolute gradient-norm/tail-occupancy checks
    (``_SCALE_SENSITIVE_CONVERGENCE_CHECKS``) were calibrated for the much
    larger pooled rolling-window training set and collapse the joint-success
    population from 27 sessions to 5, even though the great majority of the
    "non-converged" single-session fits are a few-fold over that absolute
    tolerance rather than genuinely diverged. Sessions that did hit a real
    numerical breakdown (a raised parameter-transform/probability-
    normalization/optimizer-runtime error) still fail ``numerically_sound``
    and are excluded. See the report's Figure 15 reliability note for the
    counts behind this choice.
    """
    if "numerically_sound" not in ss_metrics.columns:
        raise ValueError(
            "single-session information criteria require the numerically_sound column"
        )
    training = ss_metrics[
        (ss_metrics["analysis_role"] == "fine_tune_prefix")
        & (ss_metrics["numerically_sound"] == True)  # noqa: E712
    ].copy()
    matrix = mc_stats.complete_case_matrix(training, "session_id", "elpd")
    training = training[training["session_id"].isin(matrix.index)].copy()

    eligible_ids = _same_positive_trial_count_ids(training, "session_id")
    training = training[training["session_id"].isin(eligible_ids)]

    rows = []
    for _, row in training.iterrows():
        model_id = row["model_id"]
        k = len(MODEL_SPECS[model_id].param_names)
        n = int(row["valid_trial_count"])
        nll_total = -float(row["elpd"])
        rows.append({
            "model_id": model_id,
            "session_id": row["session_id"],
            "chronological_rank": row["chronological_rank"],
            "n_trials": n,
            "n_params": k,
            "nll_total": nll_total,
            "aic": 2 * k + 2 * nll_total,
            "bic": k * np.log(n) + 2 * nll_total,
        })
    columns = [
        "model_id", "session_id", "chronological_rank", "n_trials", "n_params",
        "nll_total", "aic", "bic",
    ]
    return _finish_information_criteria_table(pd.DataFrame(rows, columns=columns), "session_id")


def summarize_information_criteria(ic_table: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary of the AIC/BIC table: mean/median IC, mean delta-IC
    and mean IC weight relative to each window's own best model, and how
    many of the common windows that model was the single best BIC model on
    (ties broken by ``delta_bic == 0`` -- exact ties are effectively
    impossible with continuous NLL, so this is a plain win count in
    practice).
    """
    if ic_table.empty:
        return pd.DataFrame(columns=[
            "model_id", "n_windows", "n_windows_total", "n_params",
            "mean_aic", "median_aic", "mean_bic", "median_bic",
            "mean_delta_aic", "mean_delta_bic", "mean_aic_weight",
            "mean_bic_weight", "n_windows_best_by_bic", "n_windows_best_by_aic",
        ])
    n_windows = ic_table["window_id"].nunique()
    rows = []
    for model_id in MODEL_ORDER:
        subset = ic_table[ic_table["model_id"] == model_id]
        if subset.empty:
            continue
        rows.append({
            "model_id": model_id, "n_windows": len(subset), "n_windows_total": n_windows,
            "n_params": int(subset["n_params"].iloc[0]),
            "mean_aic": subset["aic"].mean(), "median_aic": subset["aic"].median(),
            "mean_bic": subset["bic"].mean(), "median_bic": subset["bic"].median(),
            "mean_delta_aic": subset["delta_aic"].mean(), "mean_delta_bic": subset["delta_bic"].mean(),
            "mean_aic_weight": subset["aic_weight"].mean(), "mean_bic_weight": subset["bic_weight"].mean(),
            "n_windows_best_by_bic": int((subset["delta_bic"] == 0).sum()),
            "n_windows_best_by_aic": int((subset["delta_aic"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def summarize_single_session_information_criteria(ic_table: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary of the converged common-session IC table."""
    if ic_table.empty:
        return pd.DataFrame(columns=[
            "model_id", "n_sessions", "n_sessions_total", "n_params",
            "mean_aic", "median_aic", "mean_bic", "median_bic",
            "mean_delta_aic", "mean_delta_bic", "mean_aic_weight",
            "mean_bic_weight", "n_sessions_best_by_bic", "n_sessions_best_by_aic",
        ])
    n_sessions = ic_table["session_id"].nunique()
    rows = []
    for model_id in MODEL_ORDER:
        subset = ic_table[ic_table["model_id"] == model_id]
        if subset.empty:
            continue
        rows.append({
            "model_id": model_id,
            "n_sessions": len(subset),
            "n_sessions_total": n_sessions,
            "n_params": int(subset["n_params"].iloc[0]),
            "mean_aic": subset["aic"].mean(),
            "median_aic": subset["aic"].median(),
            "mean_bic": subset["bic"].mean(),
            "median_bic": subset["bic"].median(),
            "mean_delta_aic": subset["delta_aic"].mean(),
            "mean_delta_bic": subset["delta_bic"].mean(),
            "mean_aic_weight": subset["aic_weight"].mean(),
            "mean_bic_weight": subset["bic_weight"].mean(),
            "n_sessions_best_by_bic": int((subset["delta_bic"] == 0).sum()),
            "n_sessions_best_by_aic": int((subset["delta_aic"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def compute_information_criteria_paired_tests(
    ic_table: pd.DataFrame, unit_col: str, analysis_level: str,
) -> pd.DataFrame:
    """Four adjacent paired contrasts for AIC and BIC at one analysis level."""
    if ic_table.empty:
        return pd.DataFrame(columns=[
            "analysis_level", "metric", *mc_stats.PAIRWISE_RESULT_COLUMNS,
            "units_independent",
        ])
    pairwise_frames = []
    for metric in ("aic", "bic"):
        matrix = mc_stats.complete_case_matrix(ic_table, unit_col, metric)
        pairwise = mc_stats.adjacent_pairwise_tests(matrix)
        pairwise.insert(0, "metric", metric)
        pairwise.insert(0, "analysis_level", analysis_level)
        pairwise["units_independent"] = analysis_level == "single_session"
        pairwise_frames.append(pairwise)
    return pd.concat(pairwise_frames, ignore_index=True, sort=False) if pairwise_frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Chronological pattern: which model wins over time
# ---------------------------------------------------------------------------

PHASE_BOUNDARIES = [
    (13, 23, "early (13-23)"),
    (24, 33, "mid (24-33)"),
    (34, 44, "late (34-44)"),
]


def collect_chronological_winner_timeline(
    mouse_id: str, sw_per_window: pd.DataFrame, ss_per_session: pd.DataFrame,
) -> pd.DataFrame:
    """Long-format table: one row per (analysis_level, chronological_rank),
    covering EVERY matched rank (13-44), not just the ranks where all five
    models happened to produce a comparable winner.

    ``comparable=False`` rows have no winner (fewer than 5 models had a
    usable fit at that rank) and must be rendered as "not comparable" rather
    than silently dropped: the missing ranks are disproportionately early
    in this dataset (M4's non-converged windows are 13, 14, 15, 17, 21), so
    dropping them would misrepresent the early period specifically.

    The two ``analysis_level`` values are NOT paired replicate estimates of
    the same thing: ``sliding_window`` evaluates a whole held-out session
    from a fit trained on 9 earlier sessions (true walk-forward), while
    ``single_session`` fine-tunes on that session's own prefix and evaluates
    only its final two blocks. Never present them as if one row = one trial.
    """
    all_ranks = _all_matched_ranks(mouse_id)
    backbone = pd.DataFrame({"chronological_rank": all_ranks})

    frames = []
    for level, per_unit, rank_col in [
        ("sliding_window", sw_per_window, "test_rank"),
        ("single_session", ss_per_session, "chronological_rank"),
    ]:
        detail = per_unit.rename(columns={rank_col: "chronological_rank"}).copy()
        duplicate_ranks = detail.loc[detail["chronological_rank"].duplicated(), "chronological_rank"]
        if not duplicate_ranks.empty:
            raise RuntimeError(
                f"Duplicate chronological_rank in {level} win-count table: "
                f"{sorted(duplicate_ranks.unique())}"
            )
        merged = backbone.merge(detail, on="chronological_rank", how="left")
        merged.insert(0, "analysis_level", level)
        merged["comparable"] = merged["winning_model"].notna()
        frames.append(merged)

    table = pd.concat(frames, ignore_index=True, sort=False)
    prefix_cols = [
        "analysis_level", "chronological_rank", "comparable", "winning_model",
        "winning_mean_nll", "runner_up_model", "runner_up_mean_nll", "winning_margin_nll",
    ]
    ordered_cols = prefix_cols + [c for c in MODEL_ORDER if c in table.columns]
    remaining = [c for c in table.columns if c not in ordered_cols]
    return table[ordered_cols + remaining]


def collect_chronological_winner_phase_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """Per (analysis_level, phase, model): win counts and share, restricted
    to comparable ranks within that fixed chronological phase.

    Denominators differ by phase and by analysis level (missing ranks are
    disproportionately early), so ``n_comparable_units_in_phase`` is reported
    alongside every count -- a win share is meaningless without it. Fixed
    chronological ranges are used rather than equal-count terciles, because
    the missing ranks would otherwise shift the phase boundaries and break
    the plain meaning of "early/mid/late".
    """
    rows = []
    for level, group in timeline.groupby("analysis_level"):
        comparable = group.loc[group["comparable"]]
        for lo, hi, label in PHASE_BOUNDARIES:
            phase = comparable.loc[comparable["chronological_rank"].between(lo, hi)]
            n_units = len(phase)
            counts = phase["winning_model"].value_counts().reindex(MODEL_ORDER, fill_value=0)
            for model_id in MODEL_ORDER:
                rows.append({
                    "analysis_level": level,
                    "phase": label,
                    "model_id": model_id,
                    "n_wins": int(counts[model_id]),
                    "n_comparable_units_in_phase": n_units,
                    "win_share": (counts[model_id] / n_units) if n_units else float("nan"),
                })
    return pd.DataFrame(rows)


def collect_train_vs_test_winner_agreement(
    test_timeline: pd.DataFrame, train_timeline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (analysis_level, chronological_rank): does the model with the
    lowest in-sample training NLL also have the lowest held-out NLL?

    Disagreement is the interesting case -- it flags a rank where the model
    that fit its own training data best did NOT generalize best, i.e. likely
    overfit relative to a simpler nested model. Restricted to ranks where
    both the train-based and test-based winner are defined (comparable in
    both timelines).
    """
    test_cols = test_timeline[["analysis_level", "chronological_rank", "comparable", "winning_model"]].rename(
        columns={"comparable": "test_comparable", "winning_model": "test_winning_model"}
    )
    train_cols = train_timeline[["analysis_level", "chronological_rank", "comparable", "winning_model"]].rename(
        columns={"comparable": "train_comparable", "winning_model": "train_winning_model"}
    )
    merged = test_cols.merge(train_cols, on=["analysis_level", "chronological_rank"], how="outer")
    merged["both_comparable"] = merged["test_comparable"].fillna(False) & merged["train_comparable"].fillna(False)
    both = merged.loc[merged["both_comparable"]].copy()
    both["agree"] = both["test_winning_model"] == both["train_winning_model"]
    both = both.sort_values(["analysis_level", "chronological_rank"]).reset_index(drop=True)

    summary_rows = []
    for level, group in both.groupby("analysis_level"):
        n = len(group)
        n_agree = int(group["agree"].sum())
        summary_rows.append({
            "analysis_level": level,
            "n_ranks_compared": n,
            "n_agree": n_agree,
            "n_disagree": n - n_agree,
            "agreement_rate": (n_agree / n) if n else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)

    keep_cols = [
        "analysis_level", "chronological_rank", "train_winning_model",
        "test_winning_model", "agree",
    ]
    return both[keep_cols], summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(mouse_id: str) -> dict[str, Path]:
    tables_dir, _ = common.ensure_output_dirs("model_comparison", mouse_id)
    outputs: dict[str, Path] = {}

    def save(name: str, df: pd.DataFrame) -> None:
        path = tables_dir / name
        # Never expose a partially written CSV to a later selective figure run.
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv.tmp", prefix=f".{name}.",
                dir=tables_dir, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                df.to_csv(temporary, index=False)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        outputs[name] = path

    sw_split_session = collect_sliding_window_split_session_metrics(mouse_id)
    save("sliding_window_split_session_metrics.csv", sw_split_session)

    sw_global = collect_sliding_window_global_pooled_metrics(mouse_id, sw_split_session)
    save("sliding_window_global_pooled_metrics.csv", sw_global)

    sw_params = collect_sliding_window_parameters(mouse_id)
    save("sliding_window_parameters.csv", sw_params)

    ss_metrics = collect_single_session_metrics(mouse_id)
    save("single_session_metrics.csv", ss_metrics)

    ss_availability = collect_single_session_availability(mouse_id)
    save("single_session_availability.csv", ss_availability)

    ss_params = collect_single_session_parameters(mouse_id)
    save("single_session_parameters.csv", ss_params)

    per_session_wins, ss_win_summary = compute_single_session_win_counts(ss_metrics)
    save("single_session_win_counts_per_session.csv", per_session_wins)
    save("single_session_win_counts_summary.csv", ss_win_summary)

    per_window_wins, sw_win_summary = compute_sliding_window_win_counts(sw_split_session)
    save("sliding_window_win_counts_per_window.csv", per_window_wins)
    save("sliding_window_win_counts_summary.csv", sw_win_summary)

    timeline = collect_chronological_winner_timeline(mouse_id, per_window_wins, per_session_wins)
    save("chronological_winner_timeline.csv", timeline)

    phase_summary = collect_chronological_winner_phase_summary(timeline)
    save("chronological_winner_phase_summary.csv", phase_summary)

    # Training-set counterparts of the above: in-sample fit quality, not a
    # generalization signal (see compute_*_win_counts docstrings).
    per_session_train_wins, ss_train_win_summary = compute_single_session_win_counts(
        ss_metrics, analysis_role="fine_tune_prefix",
    )
    save("single_session_train_win_counts_per_session.csv", per_session_train_wins)
    save("single_session_train_win_counts_summary.csv", ss_train_win_summary)

    per_window_train_wins, sw_train_win_summary = compute_sliding_window_win_counts(
        sw_split_session, split_role="train",
    )
    save("sliding_window_train_win_counts_per_window.csv", per_window_train_wins)
    save("sliding_window_train_win_counts_summary.csv", sw_train_win_summary)

    train_timeline = collect_chronological_winner_timeline(
        mouse_id, per_window_train_wins, per_session_train_wins,
    )
    save("chronological_winner_timeline_train.csv", train_timeline)

    train_phase_summary = collect_chronological_winner_phase_summary(train_timeline)
    save("chronological_winner_phase_summary_train.csv", train_phase_summary)

    agreement_per_rank, agreement_summary = collect_train_vs_test_winner_agreement(timeline, train_timeline)
    save("chronological_train_vs_test_winner_agreement_per_rank.csv", agreement_per_rank)
    save("chronological_train_vs_test_winner_agreement_summary.csv", agreement_summary)

    # Four pre-specified adjacent-model paired tests (fig01/fig03).
    ss_paired = compute_single_session_paired_tests(ss_metrics)
    save("single_session_paired_pairwise.csv", ss_paired)

    sw_paired = compute_sliding_window_paired_tests(sw_split_session, mouse_id)
    save("sliding_window_paired_pairwise.csv", sw_paired)

    # Trial-subset breakdown (fig01/fig03), event-aligned NLL (fig16/fig17),
    # and block-transition performance by trial type (fig19). Each
    # window/session's trial_predictions.csv is read exactly once here,
    # shared between the three tables.
    sw_trial_subset, sw_event_aligned, sw_block_transition_performance = (
        collect_sliding_window_trial_level_tables(mouse_id)
    )
    save("sliding_window_trial_subset_metrics.csv", sw_trial_subset)
    save("sliding_window_block_transition_performance.csv", sw_block_transition_performance)

    ss_trial_subset, ss_event_aligned = collect_single_session_trial_level_tables(mouse_id)
    save("single_session_trial_subset_metrics.csv", ss_trial_subset)

    sw_trial_subset_paired = compute_sliding_window_trial_subset_paired_tests(sw_trial_subset, mouse_id)
    save("sliding_window_trial_subset_paired_pairwise.csv", sw_trial_subset_paired)

    ss_trial_subset_paired = compute_single_session_trial_subset_paired_tests(ss_trial_subset)
    save("single_session_trial_subset_paired_pairwise.csv", ss_trial_subset_paired)

    sw_trial_subset_diamonds = collect_sliding_window_trial_subset_pooled_diamonds(sw_trial_subset, mouse_id)
    save("sliding_window_trial_subset_pooled_diamonds.csv", sw_trial_subset_diamonds)

    # One set of paired tests per Figure 16 page (all/train/validation/test).
    sw_event_paired = pd.concat(
        [
            compute_sliding_window_event_aligned_paired_tests(sw_event_aligned, mouse_id, split_role)
            for split_role in EVENT_ALIGNED_SPLIT_ROLES
        ],
        ignore_index=True, sort=False,
    )
    sw_event_trial0_paired = pd.concat(
        [
            compute_sliding_window_event_aligned_trial0_paired_tests(sw_event_aligned, mouse_id, split_role)
            for split_role in EVENT_ALIGNED_SPLIT_ROLES
        ],
        ignore_index=True, sort=False,
    )
    ss_event_paired = compute_single_session_event_aligned_paired_tests(ss_event_aligned)

    # Compute every dependent event result before replacing any event CSV, so
    # a validation failure cannot mix a fresh base table with stale tests.
    save("sliding_window_event_aligned_nll.csv", sw_event_aligned)
    save("sliding_window_event_aligned_paired_pairwise.csv", sw_event_paired)
    save("sliding_window_event_aligned_trial0_paired_pairwise.csv", sw_event_trial0_paired)
    save("single_session_event_aligned_nll.csv", ss_event_aligned)
    save("single_session_event_aligned_paired_pairwise.csv", ss_event_paired)

    # Sliding-window vs. single-session psychometrics on a matched trial
    # population, identical across all 5 models (fig18/report §10).
    psychometric_trials = collect_psychometric_comparison_trials(mouse_id)
    save("psychometric_comparison_trials.csv", psychometric_trials)
    psychometric_summary = summarize_psychometric_comparison(psychometric_trials)
    save("psychometric_comparison_summary.csv", psychometric_summary)

    # Information criteria at both analysis levels (fig15/report §7).
    ic_table = collect_information_criteria(mouse_id, sw_split_session)
    save("information_criteria_sliding_window.csv", ic_table)
    ic_summary = summarize_information_criteria(ic_table)
    save("information_criteria_summary.csv", ic_summary)

    ss_ic_table = collect_single_session_information_criteria(ss_metrics)
    save("information_criteria_single_session.csv", ss_ic_table)
    ss_ic_summary = summarize_single_session_information_criteria(ss_ic_table)
    save("information_criteria_single_session_summary.csv", ss_ic_summary)

    sw_ic_paired = compute_information_criteria_paired_tests(
        ic_table, "window_id", "sliding_window",
    )
    ss_ic_paired = compute_information_criteria_paired_tests(
        ss_ic_table, "session_id", "single_session",
    )
    ic_paired = pd.concat([sw_ic_paired, ss_ic_paired], ignore_index=True, sort=False)
    save("information_criteria_paired_pairwise.csv", ic_paired)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC06", help="Mouse/subject ID, e.g. MC06")
    args = parser.parse_args()
    outputs = run(args.mouse)
    for name, path in outputs.items():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
