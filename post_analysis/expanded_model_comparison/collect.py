"""Collect and strictly validate expanded MC08 sliding-window comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from post_analysis import common
from post_analysis.model_comparison.stats import friedman_test

from . import stats
from .config import (
    ANALYSIS_NAME,
    EXPECTED_TRAIN_SESSIONS,
    EXPECTED_VALIDATION_SESSIONS,
    MODEL_BY_ID,
    MODEL_ORDER,
    MODEL_SPECS,
    SPLIT_ROLES,
    TIE_TOLERANCE_NLL,
    WORKFLOW_DIRECTORY,
    WORKFLOW_MANIFEST,
    split_registry_path,
    validate_mouse_id,
)

_OBSERVED_COLUMNS = (
    "choice_long", "category_long", "duration_s", "choice_valid",
    "feedback_valid", "source_sha256",
)
_TRIAL_KEY = ("split_role", "session_id", "source_trial_index")


class WorkflowMismatchError(RuntimeError):
    """A result exists under the matched root but declares another workflow."""


def canonical_schedule(mouse_id: str) -> tuple[pd.DataFrame, dict[int, dict]]:
    mouse_id = validate_mouse_id(mouse_id)
    registry = pd.read_csv(split_registry_path())
    subject = registry[registry["mouse_id"].astype(str).eq(mouse_id)].copy()
    if subject.empty:
        raise ValueError(f"No split-registry sessions found for {mouse_id}")
    subject["chronological_rank"] = pd.to_numeric(subject["chronological_rank"], errors="raise").astype(int)
    subject = subject.sort_values("chronological_rank").reset_index(drop=True)
    expected_ranks = list(range(1, len(subject) + 1))
    if subject["chronological_rank"].tolist() != expected_ranks:
        raise ValueError(f"{mouse_id} split registry does not have consecutive chronological ranks")
    window_count = len(subject) - EXPECTED_TRAIN_SESSIONS - EXPECTED_VALIDATION_SESSIONS
    if window_count < 1:
        raise ValueError(f"{mouse_id} has too few sessions for a 9/3/1 rolling window")
    windows = {}
    for offset in range(window_count):
        train = subject.iloc[offset:offset + EXPECTED_TRAIN_SESSIONS]
        validation = subject.iloc[
            offset + EXPECTED_TRAIN_SESSIONS:
            offset + EXPECTED_TRAIN_SESSIONS + EXPECTED_VALIDATION_SESSIONS
        ]
        test = subject.iloc[offset + EXPECTED_TRAIN_SESSIONS + EXPECTED_VALIDATION_SESSIONS]
        windows[offset + 1] = {
            "training_session_ids": train["session_id"].astype(str).tolist(),
            "validation_session_ids": validation["session_id"].astype(str).tolist(),
            "test_session_id": str(test["session_id"]),
            "training_ranks": train["chronological_rank"].astype(int).tolist(),
            "validation_ranks": validation["chronological_rank"].astype(int).tolist(),
            "test_rank": int(test["chronological_rank"]),
        }
    return subject, windows


def _bool_mask(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0", "1.0", "0.0"})
    if unknown:
        raise ValueError(f"{label} contains non-boolean values: {unknown[:5]}")
    return normalized.isin({"true", "1", "1.0"})


def _file_snapshot(paths: list[Path]) -> tuple[tuple[str, int, int, str], ...]:
    rows = []
    for path in sorted(paths):
        if path.is_file():
            stat = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append((str(path), stat.st_size, stat.st_mtime_ns, digest.hexdigest()))
        else:
            rows.append((str(path), -1, -1, "missing"))
    return tuple(rows)


def _snapshot_digest(snapshot: tuple[tuple[str, int, int, str], ...]) -> str:
    return hashlib.sha256(json.dumps(snapshot, separators=(",", ":")).encode()).hexdigest()


def _tracked_source_paths(spec, mouse_id: str, expected_ids: set[int]) -> list[Path]:
    root = spec.result_root(mouse_id)
    directories = _window_directories(root)
    paths = [root / "experiment_manifest.json", root / "all_session_block_cut_audit.csv"]
    for window_id in sorted(expected_ids):
        directory = directories[window_id]
        paths.extend((
            directory / "window_manifest.json",
            directory / "split_and_session_metrics.csv",
            directory / "trial_predictions.csv",
            directory / "starts" / "multistart_summary.json",
            directory / "fit_result.json",
        ))
    return paths


def source_fingerprint(spec, mouse_id: str, expected_ids: set[int]) -> str:
    """Hash the same source set used by the strict audit."""
    return _snapshot_digest(_file_snapshot(_tracked_source_paths(spec, mouse_id, expected_ids)))


def _converged_winner(window_dir: Path) -> dict | None:
    """Read either result layout used by the configured model families."""
    winner = common.find_multistart_winner(window_dir)
    if winner is not None:
        return winner
    summary = common.read_json(window_dir / "fit_result.json") or {}
    winner_id = summary.get("winner_start_id")
    return next(
        (run for run in summary.get("runs", []) if run.get("start_id") == winner_id),
        None,
    )


def _window_directories(root: Path) -> dict[int, Path]:
    result = {}
    for directory in sorted(root.glob("window_*")):
        if not directory.is_dir():
            continue
        manifest = common.read_json(directory / "window_manifest.json")
        if manifest is None or not isinstance(manifest.get("window_id"), int):
            continue
        window_id = int(manifest["window_id"])
        if window_id in result:
            raise ValueError(f"Duplicate window_id={window_id} under {root}")
        result[window_id] = directory
    return result


def _validate_trial_table(
    path: Path,
    schedule: dict,
    reference: pd.DataFrame | None,
    expected_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    required = set(_TRIAL_KEY + _OBSERVED_COLUMNS) | {
        "model_evaluated", "model_probability_long", "per_trial_negative_log_likelihood",
    }
    trials = pd.read_csv(path, usecols=lambda column: column in required)
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"trial predictions missing columns {missing}")
    evaluated = trials.loc[_bool_mask(trials["model_evaluated"], "model_evaluated")].copy()
    if evaluated.empty:
        raise ValueError("trial predictions contain no evaluated trials")
    if evaluated.duplicated(list(_TRIAL_KEY)).any():
        raise ValueError("evaluated trial keys are not unique")
    allowed = {
        "train": set(schedule["training_session_ids"]),
        "validation": set(schedule["validation_session_ids"]),
        "test": {schedule["test_session_id"]},
    }
    for split_role, group in evaluated.groupby("split_role"):
        if split_role not in allowed or set(group["session_id"].astype(str)) != allowed[split_role]:
            raise ValueError(f"evaluated {split_role!r} sessions disagree with canonical schedule")
    if set(evaluated["split_role"].astype(str)) != set(SPLIT_ROLES):
        raise ValueError("evaluated predictions do not contain exactly train/validation/test")
    probabilities = pd.to_numeric(evaluated["model_probability_long"], errors="coerce").to_numpy(float)
    nll = pd.to_numeric(evaluated["per_trial_negative_log_likelihood"], errors="coerce").to_numpy(float)
    choices = pd.to_numeric(evaluated["choice_long"], errors="coerce").to_numpy(float)
    if not np.isin(choices, (0.0, 1.0)).all():
        raise ValueError("evaluated choice_long must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("evaluated probabilities are nonfinite or outside [0,1]")
    if not np.isfinite(nll).all():
        raise ValueError("evaluated NLL contains nonfinite values")
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    reconstructed = -(choices * np.log(clipped) + (1 - choices) * np.log1p(-clipped))
    if not np.allclose(nll, reconstructed, rtol=1e-8, atol=1e-10):
        raise ValueError("saved per-trial NLL does not match choice/probability")

    comparison = evaluated[list(_TRIAL_KEY + _OBSERVED_COLUMNS)].copy()
    comparison = comparison.sort_values(list(_TRIAL_KEY)).reset_index(drop=True)
    if reference is not None:
        if not comparison[list(_TRIAL_KEY)].equals(reference[list(_TRIAL_KEY)]):
            raise ValueError("evaluated trial identities differ across models")
        for column in _OBSERVED_COLUMNS:
            left, right = comparison[column], reference[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                equal = np.allclose(left.to_numpy(float), right.to_numpy(float), equal_nan=True)
            else:
                equal = left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))
            if not equal:
                raise ValueError(f"observed trial column {column!r} differs across models")

    observed_counts = trials.groupby("session_id").agg(
        n_timeline_trials=("source_trial_index", "size"),
        n_valid_choices=("choice_valid", lambda values: int(_bool_mask(values, "choice_valid").sum())),
    ).sort_index()
    scheduled_sessions = set().union(
        schedule["training_session_ids"], schedule["validation_session_ids"], {schedule["test_session_id"]},
    )
    expected_window_counts = expected_counts.loc[sorted(scheduled_sessions)].sort_index()
    if not observed_counts.equals(expected_window_counts):
        raise ValueError("per-session timeline/valid-choice counts disagree with block-cut audit")

    summaries = {}
    for split_role, group in evaluated.groupby("split_role"):
        values = pd.to_numeric(group["per_trial_negative_log_likelihood"], errors="raise")
        summaries[str(split_role)] = {
            "valid_trial_count": int(len(values)),
            "elpd": -float(values.sum()),
            "mean_nll": float(values.mean()),
        }
    return comparison, summaries


def _validate_metrics(path: Path, recomputed: dict[str, dict]) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    required = {"split_role", "session_id", "valid_trial_count", "elpd", "mean_nll"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"metrics table missing {sorted(required - set(metrics.columns))}")
    pooled = metrics[metrics["session_id"].astype(str).eq("ALL_SESSIONS")].copy()
    if pooled["split_role"].duplicated().any() or set(pooled["split_role"].astype(str)) != set(SPLIT_ROLES):
        raise ValueError("metrics must have one ALL_SESSIONS row per split")
    for split_role in SPLIT_ROLES:
        row = pooled[pooled["split_role"].eq(split_role)].iloc[0]
        expected = recomputed[split_role]
        saved_count = float(row["valid_trial_count"])
        if not np.isfinite(saved_count) or not saved_count.is_integer() or int(saved_count) != expected["valid_trial_count"]:
            raise ValueError(f"{split_role} valid-trial count disagrees with predictions")
        for field in ("elpd", "mean_nll"):
            if not np.isclose(float(row[field]), expected[field], rtol=1e-9, atol=1e-10):
                raise ValueError(f"{split_role} {field} disagrees with predictions")
    return pooled


def audit_and_collect(mouse_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    mouse_id = validate_mouse_id(mouse_id)
    _, schedule = canonical_schedule(mouse_id)
    expected_ids = set(schedule)
    reference_trials: dict[int, pd.DataFrame] = {}
    status_rows = []
    metric_frames = []

    for spec in MODEL_SPECS:
        root = spec.result_root(mouse_id)
        issues: list[str] = []
        complete_ids: set[int] = set()
        observed_ids: set[int] = set()
        fingerprint = ""
        local_metrics = []
        local_reference: dict[int, pd.DataFrame] = {}
        try:
            if root.name != WORKFLOW_DIRECTORY or not root.is_dir():
                raise FileNotFoundError(f"missing result root {root}")
            directories = _window_directories(root)
            observed_ids = set(directories)
            unexpected = sorted(observed_ids - expected_ids)
            if unexpected:
                issues.append(f"unexpected window IDs {unexpected}")
            for window_id in sorted(expected_ids & observed_ids):
                manifest = common.read_json(directories[window_id] / "window_manifest.json") or {}
                declared_workflow = manifest.get("workflow")
                if declared_workflow is not None and declared_workflow != WORKFLOW_MANIFEST:
                    raise WorkflowMismatchError(
                        f"Mixed workflows rejected: {spec.label} window {window_id} "
                        f"declares {declared_workflow!r}, expected {WORKFLOW_MANIFEST!r}"
                    )
                if manifest.get("status") == "complete" and manifest.get("workflow") == WORKFLOW_MANIFEST:
                    complete_ids.add(window_id)
            if complete_ids != expected_ids:
                missing = sorted(expected_ids - complete_ids)
                issues.append(f"complete windows {len(complete_ids)}/{len(expected_ids)}; missing {missing}")
            if issues:
                raise ValueError("; ".join(issues))

            tracked_paths: list[Path] = [root / "experiment_manifest.json"]
            audit_path = root / "all_session_block_cut_audit.csv"
            tracked_paths.append(audit_path)
            for window_id in sorted(expected_ids):
                directory = directories[window_id]
                tracked_paths.extend((
                    directory / "window_manifest.json",
                    directory / "split_and_session_metrics.csv",
                    directory / "trial_predictions.csv",
                ))
                tracked_paths.extend((
                    directory / "starts" / "multistart_summary.json",
                    directory / "fit_result.json",
                ))
            before = _file_snapshot(tracked_paths)
            audit = pd.read_csv(audit_path)
            audit_required = {"session_id", "n_timeline_trials", "n_valid_choices"}
            if not audit_required.issubset(audit.columns):
                raise ValueError("block-cut audit lacks session/count columns")
            expected_counts = audit.groupby("session_id").agg(
                n_timeline_trials=("n_timeline_trials", "sum"),
                n_valid_choices=("n_valid_choices", "sum"),
            ).astype(int).sort_index()
            canonical_sessions = set().union(*(set(item["training_session_ids"] + item["validation_session_ids"] + [item["test_session_id"]]) for item in schedule.values()))
            if set(expected_counts.index.astype(str)) != canonical_sessions:
                raise ValueError("block-cut audit sessions disagree with canonical registry")
            experiment = common.read_json(root / "experiment_manifest.json") or {}
            if experiment.get("workflow") != WORKFLOW_MANIFEST:
                raise WorkflowMismatchError(
                    f"Mixed workflows rejected: {spec.label} experiment declares "
                    f"{experiment.get('workflow')!r}, expected {WORKFLOW_MANIFEST!r}"
                )
            raw_window_count = experiment.get("window_count", -1)
            if not isinstance(raw_window_count, int) or isinstance(raw_window_count, bool) or raw_window_count != len(expected_ids):
                raise ValueError("experiment window count disagrees with canonical schedule")
            for window_id in sorted(expected_ids):
                directory = directories[window_id]
                manifest_path = directory / "window_manifest.json"
                metrics_path = directory / "split_and_session_metrics.csv"
                predictions_path = directory / "trial_predictions.csv"
                manifest = common.read_json(manifest_path) or {}
                expected = schedule[window_id]
                if manifest.get("workflow") != WORKFLOW_MANIFEST or manifest.get("status") != "complete":
                    raise ValueError(f"window {window_id} is not complete full_session_matched")
                if manifest.get("model_id") != spec.directory or manifest.get("mouse_id") != mouse_id:
                    raise ValueError(f"window {window_id} model/mouse identity mismatch")
                for field, expected_value in expected.items():
                    if manifest.get(field) != expected_value:
                        raise ValueError(f"window {window_id} {field} disagrees with canonical schedule")
                winner = _converged_winner(directory)
                if winner is None or winner.get("converged") is not True:
                    raise ValueError(f"window {window_id} has no converged multistart winner")
                reference = reference_trials.get(window_id) if reference_trials else None
                comparison, recomputed = _validate_trial_table(predictions_path, expected, reference, expected_counts)
                pooled = _validate_metrics(metrics_path, recomputed)
                pooled.insert(0, "test_rank", expected["test_rank"])
                pooled.insert(0, "window_id", window_id)
                pooled.insert(0, "model_label", spec.label)
                pooled.insert(0, "model_id", spec.model_id)
                local_metrics.append(pooled)
                local_reference[window_id] = comparison
            after = _file_snapshot(tracked_paths)
            if before != after:
                raise RuntimeError("source files changed during audit")
            fingerprint = _snapshot_digest(after)
            if not reference_trials:
                reference_trials = local_reference
            metric_frames.extend(local_metrics)
        except WorkflowMismatchError:
            raise
        except (FileNotFoundError, ValueError, RuntimeError, OSError, TypeError, AttributeError, pd.errors.ParserError) as exc:
            issues = [str(exc)]
            local_metrics = []

        eligible = not issues and len(local_metrics) == len(expected_ids)
        status_rows.append({
            "model_id": spec.model_id,
            "model_label": spec.label,
            "directory": spec.directory,
            "oracle_benchmark": spec.oracle,
            "expected_windows": len(expected_ids),
            "observed_window_ids": " ".join(map(str, sorted(observed_ids))),
            "complete_window_count": len(complete_ids),
            "eligible_for_comparison": eligible,
            "status": "complete" if eligible else "not completely run",
            "reason": "" if eligible else "; ".join(issues),
            "source_snapshot_sha256": fingerprint,
        })

    status = pd.DataFrame(status_rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return status, metrics, reference_trials


def _rankings(metrics: pd.DataFrame, eligible: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = {model_id: MODEL_BY_ID[model_id].label for model_id in eligible}
    oracle = {model_id: MODEL_BY_ID[model_id].oracle for model_id in eligible}
    ranking_frames, pairwise_frames, omnibus_rows = [], [], []
    final_all_models = len(eligible) == len(MODEL_ORDER)
    for split_role in SPLIT_ROLES:
        split = metrics[metrics["split_role"].eq(split_role)]
        matrix = stats.complete_matrix(split, eligible)
        for scope, scoped_models in (
            ("all_models", eligible),
            ("deployable_only", tuple(model for model in eligible if not oracle[model])),
        ):
            if not scoped_models:
                continue
            table = stats.ranking_table(matrix[list(scoped_models)], labels, oracle, scope)
            table.insert(0, "split_role", split_role)
            ranking_frames.append(table)
        pairs = stats.pairwise_table(matrix, split_role, final_all_models)
        pairwise_frames.append(pairs)
        friedman = friedman_test(matrix)
        omnibus_rows.append({
            "split_role": split_role,
            **friedman,
            "units_independent": False,
            "inferential_status": "final_exploratory" if final_all_models else "provisional_descriptive_do_not_interpret",
        })
    return (
        pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame(),
        pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame(),
        pd.DataFrame(omnibus_rows),
    )


def _pooled_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (model_id, split_role), group in metrics.groupby(["model_id", "split_role"], sort=False):
        n = pd.to_numeric(group["valid_trial_count"]).sum()
        total_nll = -pd.to_numeric(group["elpd"]).sum()
        rows.append({
            "model_id": model_id,
            "model_label": MODEL_BY_ID[model_id].label,
            "split_role": split_role,
            "window_instance_count": len(group),
            "evaluated_trial_instances": int(n),
            "pooled_trial_instance_mean_nll": total_nll / n,
            "mean_window_nll": pd.to_numeric(group["mean_nll"]).mean(),
        })
    return pd.DataFrame(rows)


def _win_tables(metrics: pd.DataFrame, eligible: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_window_rows, summary_rows = [], []
    for split_role in SPLIT_ROLES:
        matrix = stats.complete_matrix(metrics[metrics["split_role"].eq(split_role)], eligible)
        credits = {model: 0.0 for model in eligible}
        strict = {model: 0 for model in eligible}
        for window_id, row in matrix.iterrows():
            best = float(row.min())
            tied = [model for model in eligible if float(row[model]) - best <= TIE_TOLERANCE_NLL]
            for model in tied:
                credits[model] += 1 / len(tied)
            if len(tied) == 1:
                strict[tied[0]] += 1
            ordered = row.sort_values()
            per_window_rows.append({
                "split_role": split_role,
                "window_id": window_id,
                "winning_models": "|".join(tied),
                "tie_count": len(tied),
                "best_mean_nll": best,
                "runner_up_margin_nll": float(ordered.iloc[1] - ordered.iloc[0]) if len(ordered) > 1 else np.nan,
                **{model: row[model] for model in eligible},
            })
        for model in eligible:
            summary_rows.append({
                "split_role": split_role,
                "model_id": model,
                "model_label": MODEL_BY_ID[model].label,
                "n_windows": len(matrix),
                "strict_wins": strict[model],
                "tie_adjusted_win_credit": credits[model],
                "tie_tolerance_nll": TIE_TOLERANCE_NLL,
            })
    return pd.DataFrame(per_window_rows), pd.DataFrame(summary_rows)


def _information_criteria(metrics: pd.DataFrame, eligible: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = metrics[metrics["split_role"].eq("train")].copy()
    rows = []
    for _, row in train.iterrows():
        spec = MODEL_BY_ID[row["model_id"]]
        n = int(row["valid_trial_count"])
        total_nll = -float(row["elpd"])
        k = len(spec.parameter_names)
        rows.append({
            "model_id": spec.model_id, "model_label": spec.label,
            "oracle_benchmark": spec.oracle, "window_id": int(row["window_id"]),
            "test_rank": int(row["test_rank"]), "n_trials": n, "n_params": k,
            "nll_total": total_nll, "aic": 2 * k + 2 * total_nll,
            "bic": np.log(n) * k + 2 * total_nll,
        })
    table = pd.DataFrame(rows)
    scoped_frames = []
    for scope, models in (
        ("all_models", eligible),
        ("deployable_only", tuple(model for model in eligible if not MODEL_BY_ID[model].oracle)),
    ):
        scoped = table[table["model_id"].isin(models)].copy()
        for metric in ("aic", "bic"):
            scoped[f"delta_{metric}"] = scoped[metric] - scoped.groupby("window_id")[metric].transform("min")
            raw = np.exp(-0.5 * scoped[f"delta_{metric}"])
            scoped[f"{metric}_weight"] = raw / raw.groupby(scoped["window_id"]).transform("sum")
        scoped.insert(0, "ranking_scope", scope)
        scoped_frames.append(scoped)
    full = pd.concat(scoped_frames, ignore_index=True) if scoped_frames else pd.DataFrame()
    if full.empty:
        return full, pd.DataFrame()
    summary = full.groupby(["ranking_scope", "model_id", "model_label", "oracle_benchmark"], as_index=False).agg(
        n_windows=("window_id", "nunique"), mean_delta_aic=("delta_aic", "mean"),
        mean_delta_bic=("delta_bic", "mean"), mean_aic_weight=("aic_weight", "mean"),
        mean_bic_weight=("bic_weight", "mean"),
    )
    return full, summary


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            table.to_csv(handle, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(mouse_id: str) -> dict[str, Path]:
    mouse_id = validate_mouse_id(mouse_id)
    canonical_schedule(mouse_id)
    tables_dir, _ = common.ensure_output_dirs(ANALYSIS_NAME, mouse_id)
    status, metrics, _ = audit_and_collect(mouse_id)
    eligible = tuple(model for model in MODEL_ORDER if bool(status.set_index("model_id").loc[model, "eligible_for_comparison"]))
    for _, row in status.iterrows():
        if not bool(row["eligible_for_comparison"]):
            print(f"{row['model_label']}: not completely run")
    deployable = tuple(model for model in eligible if not MODEL_BY_ID[model].oracle)
    if len(eligible) < 2 or not deployable:
        raise RuntimeError("Comparison requires at least two complete models, including one non-oracle model")
    expected_counts = set(status.loc[status["model_id"].isin(deployable), "expected_windows"].astype(int))
    if len(expected_counts) != 1:
        raise RuntimeError("Eligible deployable models disagree on the expected window count")

    rankings, pairwise, omnibus = _rankings(metrics, eligible)
    equivalence = stats.equivalence_table(
        metrics, deployable, expected_window_count=expected_counts.pop(),
    )
    per_window_wins, win_summary = _win_tables(metrics, eligible)
    ic, ic_summary = _information_criteria(metrics, eligible)
    outputs = {
        "model_completion_status.csv": status,
        "sliding_window_metrics.csv": metrics,
        "figure00_rankings.csv": rankings,
        "figure00_pairwise_comparisons.csv": pairwise,
        "figure00_friedman_exploratory.csv": omnibus,
        "figure00_equivalence_tost.csv": equivalence,
        "figure00_family_summary.csv": stats.family_summary_table(equivalence),
        "pooled_metrics.csv": _pooled_metrics(metrics),
        "win_counts_per_window.csv": per_window_wins,
        "win_counts_summary.csv": win_summary,
        "information_criteria.csv": ic,
        "information_criteria_summary.csv": ic_summary,
    }
    from .extended_collect import collect as collect_extended
    outputs.update(collect_extended(mouse_id, status))
    paths = {}
    for filename, table in outputs.items():
        path = tables_dir / filename
        _atomic_csv(path, table)
        paths[filename] = path

    manifest = {
        "analysis": ANALYSIS_NAME,
        "mouse_id": mouse_id,
        "workflow_directory": WORKFLOW_DIRECTORY,
        "workflow_manifest_value": WORKFLOW_MANIFEST,
        "configured_models": [asdict(spec) | {"parameter_names": list(spec.parameter_names)} for spec in MODEL_SPECS],
        "eligible_models": list(eligible),
        "provisional": len(eligible) != len(MODEL_ORDER),
        "source_snapshots": dict(zip(status["model_id"], status["source_snapshot_sha256"])),
        "statistical_note": "Rolling windows overlap; all tests are exploratory within-mouse summaries.",
    }
    manifest_path = tables_dir.parent / "analysis_manifest.json"
    _atomic_json(manifest_path, manifest)
    paths["analysis_manifest.json"] = manifest_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", default="MC08")
    args = parser.parse_args()
    for path in run(args.mouse).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
