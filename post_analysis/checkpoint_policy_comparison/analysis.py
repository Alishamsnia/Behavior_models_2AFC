"""Replay saved L-BFGS checkpoints without changing any fitted artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from m0_sliding_window_v1.experiment import prediction_performance
from post_analysis.common import POST_ANALYSIS_ROOT, REPO_ROOT, RESULTS_SLIDING_ROOT

from .config import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MODEL_SPECS,
    NUMERICAL_TIE_TOLERANCE,
    POLICIES,
    PRACTICAL_TEST_TIE_NLL,
    VALIDATION_TOLERANCE_NLL,
    WORKFLOW,
)


def _resolve(path: str):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _winner(payload: dict) -> dict | None:
    winner_id = payload.get("winner_start_id")
    return next((run for run in payload.get("runs", []) if run.get("start_id") == winner_id), None)


def _window_id(path: Path) -> int:
    manifest = json.loads((path / "window_manifest.json").read_text())
    return int(manifest["window_id"])


def _observed_trials(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    generated = {
        "per_trial_negative_log_likelihood", "model_evaluated", "trial_type",
        "block_type", "mouse_correct",
    }
    keep = [
        column for column in frame.columns
        if column not in generated
        and not column.startswith("model_")
        and not column.startswith("fitted_")
    ]
    result = frame[keep].copy()
    result["session_id"] = result["session_id"].astype(str)
    return result.sort_values(["chronological_rank", "trial_index"]).reset_index(drop=True)


def _raw_parameters(record: dict, spec, registry) -> list[float]:
    for key in ("raw_parameters", "raw_parameters_after_step"):
        values = record.get(key)
        if values is not None:
            array = np.asarray(values, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite raw parameters")
            return array.tolist()
    for key in ("constrained_parameters", "constrained_parameters_after_step"):
        values = record.get(key)
        if values is not None:
            if spec.encoder is None:
                raise ValueError("constrained checkpoint has no registered inverse encoder")
            encoder = _resolve(spec.encoder)
            constrained = torch.as_tensor(values, dtype=torch.float64)
            raw = encoder(
                constrained,
                lapse_max=float(registry.lapse_max),
                epsilon=float(registry.parameter_epsilon),
            ).detach().cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError("inverse transform produced nonfinite raw parameters")
            return raw.tolist()
    raise ValueError("history record has no checkpoint parameters")


def select_checkpoints(history: list[dict], spec, registry) -> dict[str, dict]:
    """Select both prespecified policies from ordered, finite L-BFGS records."""
    candidates = []
    for ordinal, record in enumerate(history):
        if str(record.get("optimizer_phase", "")).lower() != "lbfgs":
            continue
        train = float(record.get("fresh_training_mean_nll", math.nan))
        validation = float(record.get("validation_mean_nll", math.nan))
        if not np.isfinite(train) or not np.isfinite(validation):
            continue
        try:
            raw = _raw_parameters(record, spec, registry)
        except (TypeError, ValueError, RuntimeError):
            continue
        candidates.append({
            "record": record, "ordinal": ordinal, "raw": raw,
            "train_nll_logged": train, "validation_nll_logged": validation,
        })
    if not candidates:
        raise ValueError("no finite L-BFGS history record with replayable parameters")
    minimum = min(item["validation_nll_logged"] for item in candidates)
    exact = next(
        item for item in candidates
        if item["validation_nll_logged"] <= minimum + NUMERICAL_TIE_TOLERANCE
    )
    conservative = next(
        item for item in candidates
        if item["validation_nll_logged"] <= minimum + VALIDATION_TOLERANCE_NLL + NUMERICAL_TIE_TOLERANCE
    )
    for item in candidates:
        item["candidate_count"] = len(candidates)
        item["checkpoint_fraction"] = 0.0 if len(candidates) == 1 else item["ordinal"] / (len(history) - 1)
    return {"validation_min": exact, "validation_conservative": conservative}


def _mean_nll(predictions: pd.DataFrame, split: str) -> tuple[float, int]:
    selected = predictions.loc[
        predictions["split_role"].eq(split) & predictions["model_evaluated"].astype(bool)
    ]
    values = pd.to_numeric(selected["per_trial_negative_log_likelihood"], errors="coerce")
    if selected.empty or not np.isfinite(values).all():
        raise ValueError(f"invalid {split} predictions")
    return float(values.mean()), int(len(values))


def _scientific_gate(predictions: pd.DataFrame, registry) -> tuple[bool, str]:
    messages = []
    for split in ("train", "validation"):
        part = predictions.loc[predictions["split_role"].eq(split)]
        if "model_projection_activated" in part:
            valid = part.loc[part["feedback_valid"].astype(bool), "model_projection_activated"]
            fraction = float(pd.to_numeric(valid, errors="coerce").mean()) if len(valid) else 0.0
            limit = getattr(registry, "maximum_accepted_projection_fraction", None)
            if limit is not None and (not np.isfinite(fraction) or fraction > float(limit)):
                messages.append(f"{split} projection_fraction={fraction:.6g}>{float(limit):.6g}")
        if "model_tail_probability_pre" in part:
            mean_tail = float(pd.to_numeric(part["model_tail_probability_pre"], errors="coerce").mean())
            limit = getattr(registry, "maximum_accepted_mean_tail_probability", None)
            if limit is not None and (not np.isfinite(mean_tail) or mean_tail > float(limit)):
                messages.append(f"{split} mean_tail={mean_tail:.6g}>{float(limit):.6g}")
    return not messages, "; ".join(messages)


def _parameter_hash(values: list[float]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()[:16]


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    block = min(BOOTSTRAP_BLOCK_LENGTH, n)
    starts = np.arange(n - block + 1)
    results = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(values[start:start + block])
        results[index] = np.mean(sample[:n])
    return tuple(np.quantile(results, [0.025, 0.975]))


def _saved_metrics(window: Path) -> pd.DataFrame:
    path = window / "split_and_session_metrics.csv"
    if not path.exists():
        path = window / "prefix_and_heldout_metrics.csv"
    return pd.read_csv(path)


def _replay_policy(spec, registry, predictor, observed, selected, *, include_test: bool):
    pretest = observed.loc[~observed["split_role"].eq("test")].copy()
    predicted_pretest = predictor(pretest, selected["raw"], registry, allow_test=False)
    replayed_validation, validation_count = _mean_nll(predicted_pretest, "validation")
    logged = selected["validation_nll_logged"]
    if not np.isclose(replayed_validation, logged, rtol=1e-8, atol=1e-9):
        raise ValueError(
            f"validation replay mismatch: logged={logged:.12g}, replayed={replayed_validation:.12g}"
        )
    gate, gate_reason = _scientific_gate(predicted_pretest, registry)
    if not gate:
        raise ValueError(f"scientific acceptance gate failed: {gate_reason}")
    if not include_test:
        return predicted_pretest, validation_count
    test = observed.loc[observed["split_role"].eq("test")].copy()
    predicted_test = predictor(test, selected["raw"], registry, allow_test=True)
    return pd.concat([predicted_pretest, predicted_test], ignore_index=True), validation_count


def _baseline_audit(spec, registry, predictor, window: Path, winner: dict) -> None:
    observed = _observed_trials(window / "trial_predictions.csv")
    predicted = predictor(observed, winner["best_raw_parameters"], registry, allow_test=True)
    replayed = prediction_performance(predicted)
    saved = _saved_metrics(window)
    for split in ("train", "validation", "test"):
        left = replayed.loc[(replayed["split_role"] == split) & (replayed["session_id"] == "ALL_SESSIONS"), "mean_nll"]
        right = saved.loc[(saved["split_role"] == split) & (saved["session_id"].astype(str) == "ALL_SESSIONS"), "mean_nll"]
        if len(left) != 1 or len(right) != 1 or not np.isclose(float(left.iloc[0]), float(right.iloc[0]), rtol=1e-9, atol=1e-10):
            raise ValueError(f"baseline replay failed for {split}")


def _derive_tables(metrics: pd.DataFrame, selections: pd.DataFrame, available_models: list[str]):
    pooled = metrics.loc[metrics["session_id"].astype(str).eq("ALL_SESSIONS")].copy()
    current = pooled.loc[pooled["policy"].eq("current_train_mle"), ["model_id", "window_id", "split_role", "mean_nll"]]
    current = current.rename(columns={"mean_nll": "current_mean_nll"})
    deltas = pooled.loc[~pooled["policy"].eq("current_train_mle")].merge(
        current, on=["model_id", "window_id", "split_role"], validate="many_to_one"
    )
    deltas["delta_mean_nll"] = deltas["mean_nll"] - deltas["current_mean_nll"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    for keys, group in deltas.sort_values("window_id").groupby(["model_id", "policy", "split_role"], sort=False):
        values = group["delta_mean_nll"].to_numpy(float)
        low, high = _bootstrap_mean(values, rng)
        summary_rows.append({
            "model_id": keys[0], "policy": keys[1], "split_role": keys[2],
            "n_windows": len(values), "mean_delta_nll": float(np.mean(values)),
            "median_delta_nll": float(np.median(values)), "bootstrap_ci_low": low,
            "bootstrap_ci_high": high, "bootstrap_block_length": min(BOOTSTRAP_BLOCK_LENGTH, len(values)),
        })
    summary = pd.DataFrame(summary_rows)

    test = pooled.loc[pooled["split_role"].eq("test") & pooled["model_id"].isin(available_models)].copy()
    support = set.intersection(*(
        set(test.loc[(test["model_id"] == model) & (test["policy"] == policy), "window_id"])
        for model in available_models for policy in POLICIES
    )) if available_models else set()
    ranking = test.loc[test["window_id"].isin(support)].groupby(["policy", "model_id"], as_index=False).agg(
        mean_test_nll=("mean_nll", "mean"), median_test_nll=("mean_nll", "median"), n_windows=("window_id", "nunique")
    )
    ranking["rank"] = ranking.groupby("policy")["mean_test_nll"].rank(method="min")

    test_delta = deltas.loc[deltas["split_role"].eq("test")].copy()
    test_delta["outcome"] = np.where(
        test_delta["delta_mean_nll"] < -PRACTICAL_TEST_TIE_NLL, "win",
        np.where(test_delta["delta_mean_nll"] > PRACTICAL_TEST_TIE_NLL, "loss", "tie"),
    )
    return deltas, summary, ranking, test_delta, sorted(support)


def run_analysis(mouse_id: str = "MC08", *, include_test: bool = True) -> Path:
    output = POST_ANALYSIS_ROOT / "checkpoint_policy_comparison" / mouse_id
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    status_rows, selection_rows, metric_frames = [], [], []
    replayable = []

    for spec in MODEL_SPECS:
        root = RESULTS_SLIDING_ROOT / spec.directory / mouse_id / WORKFLOW
        windows = sorted((path for path in root.glob("window_*") if path.is_dir()), key=_window_id)
        if spec.loader is None or spec.predictor is None:
            status_rows.append({"model_id": spec.model_id, "directory": spec.directory, "status": "not_replayable", "windows_found": len(windows), "windows_scored": 0, "reason": "saved histories contain no per-step parameter vectors"})
            continue
        if not windows:
            status_rows.append({"model_id": spec.model_id, "directory": spec.directory, "status": "missing", "windows_found": 0, "windows_scored": 0, "reason": "result directory absent"})
            continue
        registry, predictor = _resolve(spec.loader)(), _resolve(spec.predictor)
        first_payload = json.loads((windows[0] / "fit_result.json").read_text())
        first_winner = _winner(first_payload)
        try:
            if first_winner is None:
                raise ValueError("first window has no selected winner")
            _baseline_audit(spec, registry, predictor, windows[0], first_winner)
        except Exception as exc:
            status_rows.append({"model_id": spec.model_id, "directory": spec.directory, "status": "baseline_audit_failed", "windows_found": len(windows), "windows_scored": 0, "reason": str(exc)})
            continue

        scored = 0
        for window in windows:
            window_id = _window_id(window)
            payload = json.loads((window / "fit_result.json").read_text())
            winner = _winner(payload)
            if winner is None:
                selection_rows.append({"model_id": spec.model_id, "window_id": window_id, "policy": "all", "status": "no_accepted_winner", "reason": payload.get("selection_status", "unknown")})
                continue
            if str(winner.get("optimizer_strategy", "")).lower() != "lbfgs":
                selection_rows.append({"model_id": spec.model_id, "window_id": window_id, "policy": "all", "status": "unsupported_optimizer", "reason": winner.get("optimizer_strategy")})
                continue
            try:
                selected = select_checkpoints(winner.get("history", []), spec, registry)
            except Exception as exc:
                selection_rows.append({"model_id": spec.model_id, "window_id": window_id, "policy": "all", "status": "selection_failed", "reason": str(exc)})
                continue
            observed = _observed_trials(window / "trial_predictions.csv")
            baseline = _saved_metrics(window).copy()
            baseline["model_id"], baseline["window_id"], baseline["policy"] = spec.model_id, window_id, "current_train_mle"
            metric_frames.append(baseline)
            window_ok = True
            cache: dict[str, pd.DataFrame] = {}
            for policy in ("validation_min", "validation_conservative"):
                item = selected[policy]
                record = item["record"]
                row = {
                    "model_id": spec.model_id, "window_id": window_id, "policy": policy,
                    "status": "selected", "reason": "", "winner_start_id": winner.get("start_id"),
                    "history_ordinal": item["ordinal"], "optimizer_step": record.get("step", record.get("epoch")),
                    "history_length": len(winner.get("history", [])), "candidate_count": item["candidate_count"],
                    "checkpoint_fraction": item["checkpoint_fraction"],
                    "training_nll_logged": item["train_nll_logged"],
                    "validation_nll_logged": item["validation_nll_logged"],
                    "parameter_hash": _parameter_hash(item["raw"]),
                }
                try:
                    digest = row["parameter_hash"]
                    if digest not in cache:
                        predictions, validation_count = _replay_policy(
                            spec, registry, predictor, observed, item, include_test=include_test
                        )
                        cache[digest] = prediction_performance(predictions)
                    row["validation_valid_trial_count"] = validation_count if 'validation_count' in locals() else np.nan
                    metrics = cache[digest].copy()
                    metrics["model_id"], metrics["window_id"], metrics["policy"] = spec.model_id, window_id, policy
                    metric_frames.append(metrics)
                except Exception as exc:
                    row["status"], row["reason"] = "replay_failed", str(exc)
                    window_ok = False
                selection_rows.append(row)
            if window_ok:
                scored += 1
        status_rows.append({"model_id": spec.model_id, "directory": spec.directory, "status": "replayed", "windows_found": len(windows), "windows_scored": scored, "reason": ""})
        replayable.append(spec.model_id)

    status = pd.DataFrame(status_rows)
    selections = pd.DataFrame(selection_rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    deltas, summary, ranking, outcomes, common_support = _derive_tables(metrics, selections, replayable)
    status.to_csv(tables / "model_replay_status.csv", index=False)
    selections.to_csv(tables / "checkpoint_selection.csv", index=False)
    metrics.to_csv(tables / "split_metrics.csv", index=False)
    deltas.to_csv(tables / "policy_deltas.csv", index=False)
    summary.to_csv(tables / "bootstrap_summary.csv", index=False)
    ranking.to_csv(tables / "model_rankings.csv", index=False)
    outcomes.to_csv(tables / "test_win_tie_loss.csv", index=False)
    manifest = {
        "mouse_id": mouse_id, "policies": list(POLICIES),
        "validation_tolerance_mean_nll": VALIDATION_TOLERANCE_NLL,
        "test_practical_tie_mean_nll": PRACTICAL_TEST_TIE_NLL,
        "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cross_model_common_window_ids": common_support,
        "retrospective_exploratory": True,
        "source_results_unchanged": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
