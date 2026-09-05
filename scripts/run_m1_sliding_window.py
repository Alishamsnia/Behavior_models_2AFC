#!/usr/bin/env python3
"""Run the isolated M1 9/3/1 sliding and per-session experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _resume_staleness import (  # noqa: E402
    manifest_fingerprint,
    manifest_fingerprint_is_stale,
    result_registry_is_stale,
)

from m0_sensory_v1.m1_analysis import (  # noqa: E402
    build_m1_trial_predictions,
    save_m1_analysis_outputs,
)
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m0_sensory_v1.model import (  # noqa: E402
    LBFGSSettings,
    fit_all_initializations,
    fit_single_initialization_with_strategy,
    save_fit_run_result,
)
from m0_sensory_v1.model.m1_initialization import (  # noqa: E402
    M1InitializationSpec,
)
from m0_sensory_v1.model.m1_session import (  # noqa: E402
    m1_session_trials_from_frame,
)
from m1_sliding_window_v1 import (  # noqa: E402
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    WORKFLOW_CHOICES,
    aggregate_m1_session_initializations,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_m1_experiment_figures,
    match_window_winners_to_test_sessions,
    prediction_performance,
    prepare_sliding_data,
)


EXPERIMENT_ID = "M1_sliding_9train_3validation_1test_two_complete_blocks_v1"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit unchanged M1 in chronological 9/3/1 windows with eight starts, "
            "then fine-tune every session and score two held-out complete blocks."
        )
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--workflow", choices=WORKFLOW_CHOICES,
        default=LEGACY_PARTITIONED_WORKFLOW,
        help="Use the legacy block-partitioned workflow or the additive full-session/matched-session workflow.",
    )
    parser.add_argument(
        "--sliding-output-dir", type=Path, default=Path("results_sliding_window")
    )
    parser.add_argument(
        "--single-output-dir", type=Path, default=Path("results_single_session")
    )
    parser.add_argument(
        "--optimizer", choices=("adam", "lbfgs", "hybrid"), default="lbfgs"
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adam-warmup-epochs", type=int, default=1000)
    parser.add_argument("--lbfgs-max-steps", type=int, default=250)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--lbfgs-max-evaluations", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Concurrent optimizer-start workers inside each rolling window",
    )
    parser.add_argument(
        "--window-parallel-workers",
        type=int,
        default=1,
        help=(
            "Concurrent rolling windows fitted at once. Each window still uses "
            "--parallel-workers optimizer-start processes, so total CPU usage is "
            "roughly window-parallel-workers x parallel-workers x cpu-threads-per-worker."
        ),
    )
    parser.add_argument(
        "--single-session-workers",
        type=int,
        default=4,
        help="Concurrent session fine-tuning processes",
    )
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sliding-only", action="store_true", help="Stop after rolling windows"
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Reuse completed rolling outputs and run session fine-tuning only",
    )
    parser.add_argument(
        "--window-limit",
        type=int,
        default=None,
        help="Development/smoke limit; prevents the single-session stage",
    )
    args = parser.parse_args()
    if args.sliding_only and args.single_only:
        parser.error("--sliding-only and --single-only are mutually exclusive")
    positive = {
        "adam_warmup_epochs": args.adam_warmup_epochs,
        "lbfgs_max_steps": args.lbfgs_max_steps,
        "lbfgs_learning_rate": args.lbfgs_learning_rate,
        "lbfgs_history_size": args.lbfgs_history_size,
        "lbfgs_max_evaluations": args.lbfgs_max_evaluations,
        "lbfgs_tolerance_grad": args.lbfgs_tolerance_grad,
        "lbfgs_tolerance_change": args.lbfgs_tolerance_change,
        "parallel_workers": args.parallel_workers,
        "window_parallel_workers": args.window_parallel_workers,
        "single_session_workers": args.single_session_workers,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
    }
    if any(value <= 0 for value in positive.values()):
        parser.error("All optimizer and worker controls must be positive")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.window_limit is not None and args.window_limit <= 0:
        parser.error("--window-limit must be positive")
    if args.window_limit is not None and not args.sliding_only:
        parser.error("--window-limit requires --sliding-only")
    if args.optimizer != "adam" and args.max_epochs is not None:
        parser.error("--max-epochs applies only to Adam")
    return args


def _lbfgs(args: argparse.Namespace) -> LBFGSSettings:
    return LBFGSSettings(
        maximum_steps=args.lbfgs_max_steps,
        learning_rate=args.lbfgs_learning_rate,
        history_size=args.lbfgs_history_size,
        maximum_evaluations_per_step=args.lbfgs_max_evaluations,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
    )


def _sessions(prepared, session_ids: tuple[str, ...], role: str, scope: str = "prefix"):
    return tuple(
        m1_session_trials_from_frame(
            prepared.frame(session_id, scope, role), split_role=role
        )
        for session_id in session_ids
    )


def _window_canonical(prepared, window, scope: str = "prefix") -> pd.DataFrame:
    frames = [
        prepared.frame(session_id, scope, "train")
        for session_id in window.training_session_ids
    ]
    frames.extend(
        prepared.frame(session_id, scope, "validation")
        for session_id in window.validation_session_ids
    )
    frames.append(prepared.frame(window.test_session_id, scope, "test"))
    return pd.concat(frames, ignore_index=True)


def _winner(summary: dict[str, Any]) -> dict[str, Any]:
    winner_id = summary.get("winner_start_id")
    winner = next(
        (run for run in summary.get("runs", []) if run.get("start_id") == winner_id),
        None,
    )
    if winner is None:
        raise RuntimeError("M1 window has no accepted converged winner")
    return winner


def _continuous_single_session_predictions(
    prefix: pd.DataFrame,
    heldout: pd.DataFrame,
    raw_parameters,
    registry,
) -> pd.DataFrame:
    """Evaluate one causal session continuously, then restore train/test labels."""

    combined = pd.concat([prefix, heldout], ignore_index=True).sort_values(
        "trial_index"
    )
    if combined["session_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous M1 prediction requires exactly one session")
    timeline = combined.copy()
    timeline["split_role"] = "train"
    predictions = build_m1_trial_predictions(
        timeline, raw_parameters, registry, allow_test=False
    )
    partition_to_role = {"prefix": "train", "heldout": "test"}
    predictions["split_role"] = predictions["experiment_partition"].map(
        partition_to_role
    )
    if predictions["split_role"].isna().any():
        raise ValueError("Single-session predictions contain an unexpected partition")
    return predictions.sort_values("trial_index").reset_index(drop=True)


def _fit_window(prepared, window, registry, args, root: Path) -> dict[str, Any]:
    rolling_scope = "full" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "prefix"
    output = root / window.directory_name
    result_path = output / "fit_result.json"
    manifest_path = output / "window_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    figures_current = figure_outputs_are_current(output / "m1_m0_style_figures")
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    registry_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous_manifest, registry, SOURCE_ROOT)
    if registry_stale:
        print(
            f"window={window.window_id:03d} status=resume_stale_registry "
            "(config/initialization/split checksum or source code changed "
            "since this fit was saved -- refitting instead of resuming)",
            flush=True,
        )
    if (
        args.resume
        and previous_result is not None
        and previous_manifest.get("status") == "complete"
        and figures_current
        and not registry_stale
    ):
        print(f"window={window.window_id:03d} status=resume_skip", flush=True)
        return previous_result

    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    resumed_postfit = args.resume and previous_result is not None and not registry_stale
    if resumed_postfit:
        print(f"window={window.window_id:03d} status=resume_postfit", flush=True)
        summary = previous_result
    else:
        train = _sessions(prepared, window.training_session_ids, "train", rolling_scope)
        validation = _sessions(
            prepared, window.validation_session_ids, "validation", rolling_scope
        )
        print(
            f"window={window.window_id:03d} "
            f"train={window.training_ranks[0]:03d}-{window.training_ranks[-1]:03d} "
            f"validation={window.validation_ranks[0]:03d}-{window.validation_ranks[-1]:03d} "
            f"test={window.test_rank:03d} starts={len(registry.initializations)}",
            flush=True,
        )
        fit = fit_all_initializations(
            train,
            validation,
            registry,
            output_directory=output / "starts",
            max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            optimizer_strategy=args.optimizer,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs(args),
            parallel_workers=args.parallel_workers,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
        )
        summary = fit.as_dict()
    if len(summary.get("runs", [])) != 8:
        raise RuntimeError("Every rolling M1 window must preserve all eight starts")
    winner = _winner(summary)
    _atomic_json(summary, result_path)
    canonical = _window_canonical(prepared, window, rolling_scope)
    predictions = build_m1_trial_predictions(
        canonical, winner["best_raw_parameters"], registry, allow_test=True
    )
    save_m1_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "split_and_session_metrics.csv", index=False
    )
    generate_m1_experiment_figures(
        figure_ready(predictions),
        summary,
        output / "m1_m0_style_figures",
        output / "m1_figures",
    )
    relevant = set(
        (*window.training_session_ids, *window.validation_session_ids, window.test_session_id)
    )
    rolling_audit = prepared.full_session_block_audit() if rolling_scope == "full" else prepared.block_audit
    rolling_audit.loc[
        rolling_audit["session_id"].astype(str).isin(relevant)
    ].to_csv(output / "block_cut_audit.csv", index=False)
    latest_elapsed = time.monotonic() - started
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": args.workflow,
        "status": "complete",
        "resume_fingerprint": manifest_fingerprint(registry, SOURCE_ROOT),
        "model_id": registry.model_id,
        "mouse_id": prepared.mouse_id,
        **window.as_dict(),
        "registered_initialization_count": len(registry.initializations),
        "executed_start_ids": [run["start_id"] for run in summary["runs"]],
        "optimizer": args.optimizer,
        "winner_start_id": summary["winner_start_id"],
        "winner_best_constrained_parameters": winner[
            "best_constrained_parameters"
        ],
        "winner_biased_hazard": float(
            torch.sigmoid(torch.tensor(winner["best_constrained_parameters"][5]))
        ),
        "neutral_hazard_rule": "h_N=h_B/2; no additional parameter",
        "rolling_test_scope": (
            "full_session" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
            else "retained_prefix_only; final two complete blocks and terminal block excluded"
        ),
        "elapsed_seconds": previous_manifest.get("elapsed_seconds", latest_elapsed),
        "latest_postfit_elapsed_seconds": latest_elapsed,
        "resumed_from_saved_fit": resumed_postfit,
    }
    _atomic_json(manifest, manifest_path)
    print(
        f"window={window.window_id:03d} winner={summary['winner_start_id']} "
        f"status=complete elapsed_s={latest_elapsed:.1f}",
        flush=True,
    )
    return summary


def _initialize_window_worker(cpu_threads: int) -> None:
    """Pin one rolling-window process to its allotted Torch CPU threads."""

    torch.set_num_threads(cpu_threads)


def _run_windows(
    prepared,
    windows,
    registry,
    args: argparse.Namespace,
    root: Path,
) -> dict[int, list[float]]:
    """Fit every requested M1 window, running up to --window-parallel-workers at once.

    Windows are fully independent -- separate sessions, separate output
    directories, no warm-starting from each other -- so every window is
    submitted up front and the pool starts the next queued window the instant
    a worker frees up. A fast window is never held back waiting for slower
    siblings to finish before its replacement begins. Each window still runs
    its own self-contained resume check (unchanged from the sequential path),
    so ``--resume`` behaves identically regardless of how many workers are
    requested here.
    """

    winner_parameters: dict[int, list[float]] = {}
    workers = min(
        args.window_parallel_workers,
        len(windows),
        multiprocessing.cpu_count(),
    )
    print(
        f"rolling_window_jobs={len(windows)} window_parallel_workers={workers}",
        flush=True,
    )
    if workers == 1:
        for window in windows:
            summary = _fit_window(prepared, window, registry, args, root)
            winner_parameters[window.window_id] = list(
                _winner(summary)["best_constrained_parameters"]
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_window_worker,
            initargs=(args.cpu_threads_per_worker,),
        ) as executor:
            futures = {
                executor.submit(_fit_window, prepared, window, registry, args, root): window
                for window in windows
            }
            for future in as_completed(futures):
                window = futures[future]
                summary = future.result()
                winner_parameters[window.window_id] = list(
                    _winner(summary)["best_constrained_parameters"]
                )
    return winner_parameters


def _single_session_job(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(payload["cpu_threads"]))
    registry = payload["registry"]
    prefix = payload["prefix"]
    heldout = payload["heldout"]
    output = Path(payload["output"])
    args = payload["optimizer_args"]
    initialization_row = payload["initialization"]
    result_path = output / "fine_tune_fit_result.json"
    manifest_path = output / "single_session_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    figures_current = figure_outputs_are_current(output / "m1_m0_style_figures")
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    session_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous_manifest, registry, SOURCE_ROOT)
    if (
        payload["resume"]
        and previous_result is not None
        and previous_manifest.get("status") == "complete"
        and figures_current
        and not session_stale
    ):
        return {"session_id": payload["session_id"], "status": "resume_skip"}

    output.mkdir(parents=True, exist_ok=True)
    parameter_names = ("w", "s0", "ell_short", "ell_long", "kappa", "ah")
    source_values = [
        float(initialization_row[f"initial_{name}"])
        for name in parameter_names
    ]
    values, adjusted_names = clamp_fine_tune_initialization(
        source_values,
        parameter_names,
        parameter_epsilon=registry.parameter_epsilon,
        lapse_max=registry.lapse_max,
    )
    initialization = M1InitializationSpec(
        start_id=1,
        w=values[0],
        s0=values[1],
        ell_short=values[2],
        ell_long=values[3],
        kappa=values[4],
        ah=values[5],
        purpose="sliding-window initialization for M1 session fine-tuning",
    )
    experimental_registry = replace(registry, initializations=(initialization,))
    started = time.monotonic()
    resumed_postfit = (
        payload["resume"] and previous_result is not None and not session_stale
    )
    if resumed_postfit:
        run_record = previous_result
    else:
        train_session = m1_session_trials_from_frame(prefix, split_role="train")
        monitor_frame = prefix.copy()
        monitor_frame["split_role"] = "validation"
        monitor_session = m1_session_trials_from_frame(
            monitor_frame, split_role="validation"
        )
        result = fit_single_initialization_with_strategy(
            (train_session,),
            (monitor_session,),
            experimental_registry,
            initialization,
            optimizer_strategy=args["optimizer"],
            max_epochs_override=args["max_epochs"],
            initial_learning_rate_override=args["learning_rate"],
            adam_warmup_epochs=args["adam_warmup_epochs"],
            lbfgs_settings=LBFGSSettings(**args["lbfgs"]),
        )
        save_fit_run_result(result, result_path)
        run_record = result.as_dict()
    summary = {
        "result_schema_version": "m1_single_session_fine_tune_v1",
        "model_id": registry.model_id,
        "mouse_id": payload["mouse_id"],
        "winner_start_id": 1,
        "runs": [run_record],
    }
    predictions = _continuous_single_session_predictions(
        prefix,
        heldout,
        run_record["best_raw_parameters"],
        registry,
    )
    save_m1_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "prefix_and_heldout_metrics.csv", index=False
    )
    generate_m1_experiment_figures(
        figure_ready(predictions),
        summary,
        output / "m1_m0_style_figures",
        output / "m1_figures",
    )
    provenance = dict(initialization_row)
    for key, value in list(provenance.items()):
        if pd.isna(value):
            provenance[key] = None
    provenance["fine_tune_initial_constrained_parameters"] = dict(
        zip(parameter_names, values, strict=True)
    )
    provenance["open_bound_adjusted_parameters"] = list(adjusted_names)
    _atomic_json(provenance, output / "initialization_provenance.json")
    latest_elapsed = time.monotonic() - started
    best_ah = float(run_record["best_constrained_parameters"][5])
    best_hazard = float(torch.sigmoid(torch.tensor(best_ah)))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": payload["workflow"],
        "status": "complete",
        "resume_fingerprint": manifest_fingerprint(registry, SOURCE_ROOT),
        "model_id": registry.model_id,
        "mouse_id": payload["mouse_id"],
        "session_id": payload["session_id"],
        "chronological_rank": payload["rank"],
        "fine_tune_scope": (
            "all blocks before the literal final two blocks"
            if payload["workflow"] == FULL_SESSION_MATCHED_WORKFLOW
            else "all blocks before the two held-out complete blocks"
        ),
        "heldout_scope": (
            "literal final two chronological blocks, including the terminal block"
            if payload["workflow"] == FULL_SESSION_MATCHED_WORKFLOW
            else "two complete blocks preceding ignored terminal block"
        ),
        "heldout_state_policy": (
            "continuous causal belief from prefix; heldout category feedback updates "
            "only subsequent heldout trials"
        ),
        "heldout_choices_used_for_parameter_updates": False,
        "validation_monitor": (
            "duplicate of training prefix for optimizer API reporting only; "
            "not an independent validation set and never controls updates"
        ),
        "initial_constrained_parameters": values,
        "source_window_constrained_parameters": source_values,
        "open_bound_adjusted_parameters": list(adjusted_names),
        "best_constrained_parameters": list(
            run_record["best_constrained_parameters"]
        ),
        "best_biased_hazard": best_hazard,
        "derived_neutral_hazard": best_hazard / 2.0,
        "neutral_hazard_rule": "h_N=h_B/2; no additional parameter",
        "converged": bool(run_record["converged"]),
        "termination_reason": run_record["termination_reason"],
        "resumed_from_saved_fit": resumed_postfit,
        "elapsed_seconds": previous_manifest.get("elapsed_seconds", latest_elapsed),
        "latest_postfit_elapsed_seconds": latest_elapsed,
    }
    _atomic_json(manifest, manifest_path)
    return {
        "session_id": payload["session_id"],
        "status": "resume_postfit" if resumed_postfit else "complete",
        "converged": bool(run_record["converged"]),
        "elapsed_seconds": latest_elapsed,
    }


def _fine_tune_all(
    prepared,
    registry,
    args,
    single_root: Path,
    initializations: pd.DataFrame,
) -> None:
    jobs: list[dict[str, Any]] = []
    optimizer_args = {
        "optimizer": args.optimizer,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "adam_warmup_epochs": args.adam_warmup_epochs,
        "lbfgs": {
            "maximum_steps": args.lbfgs_max_steps,
            "learning_rate": args.lbfgs_learning_rate,
            "history_size": args.lbfgs_history_size,
            "maximum_evaluations_per_step": args.lbfgs_max_evaluations,
            "tolerance_grad": args.lbfgs_tolerance_grad,
            "tolerance_change": args.lbfgs_tolerance_change,
        },
    }
    by_session = initializations.set_index("session_id")
    prefix_scope = "final_two_prefix" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "prefix"
    heldout_scope = "final_two_heldout" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "heldout"
    for assignment in initializations.sort_values(
        "chronological_rank"
    ).itertuples(index=False):
        session_id = str(assignment.session_id)
        rank = int(assignment.chronological_rank)
        directory = single_root / f"session_{rank:03d}_{session_id}"
        jobs.append(
            {
                "registry": registry,
                "mouse_id": prepared.mouse_id,
                "session_id": session_id,
                "rank": rank,
                "prefix": prepared.frame(session_id, prefix_scope, "train"),
                "heldout": prepared.frame(session_id, heldout_scope, "test"),
                "initialization": by_session.loc[session_id].to_dict(),
                "output": str(directory),
                "optimizer_args": optimizer_args,
                "cpu_threads": args.cpu_threads_per_worker,
                "resume": args.resume,
                "workflow": args.workflow,
            }
        )
    workers = min(
        args.single_session_workers,
        len(jobs),
        multiprocessing.cpu_count(),
    )
    print(f"single_session_jobs={len(jobs)} parallel_workers={workers}", flush=True)
    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            result = _single_session_job(job)
            results.append(result)
            print(f"session={result['session_id']} status={result['status']}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(_single_session_job, job): job["session_id"]
                for job in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"session={result['session_id']} status={result['status']}",
                    flush=True,
                )
    results.sort(key=lambda item: str(item["session_id"]))
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id,
            "session_count": len(results),
            "results": results,
        },
        single_root / "single_session_run_summary.json",
    )


def _generate_sliding_global_figures(root: Path, windows) -> None:
    sources = [
        (
            window.directory_name,
            root / window.directory_name / "trial_predictions.csv",
        )
        for window in windows
    ]
    manifest = generate_global_prediction_figures(
        sources,
        root / "global_figures",
        source_kind="sliding_windows",
    )
    print(
        f"sliding_global_figures=complete sources={manifest['source_count']}",
        flush=True,
    )


def _generate_single_global_figures(initializations, root: Path) -> None:
    sources: list[tuple[str, Path]] = []
    for assignment in initializations.sort_values(
        "chronological_rank"
    ).itertuples(index=False):
        rank = int(assignment.chronological_rank)
        session_id = str(assignment.session_id)
        directory_name = f"session_{rank:03d}_{session_id}"
        sources.append(
            (directory_name, root / directory_name / "trial_predictions.csv")
        )
    manifest = generate_global_prediction_figures(
        sources,
        root / "global_figures",
        source_kind="single_session_fine_tuning",
    )
    print(
        f"single_session_global_figures=complete sources={manifest['source_count']}",
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    registry = load_m1_model_registry(args.registry)
    if len(registry.initializations) != 8:
        raise RuntimeError("Sliding M1 requires exactly eight registered initializations")
    prepared = prepare_sliding_data(args.raw_data_root, registry)
    windows = build_sliding_windows(prepared.assignments)
    workflow_suffix = "full_session_matched_v1" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else None
    sliding_root = (
        args.sliding_output_dir.expanduser().resolve()
        / registry.model_id
        / prepared.mouse_id
    )
    single_root = (
        args.single_output_dir.expanduser().resolve()
        / registry.model_id
        / prepared.mouse_id
    )
    if workflow_suffix is not None:
        sliding_root = sliding_root / workflow_suffix
        single_root = single_root / workflow_suffix
    sliding_root.mkdir(parents=True, exist_ok=True)
    prepared.assignments.to_csv(
        sliding_root / "chronological_sessions.csv", index=False
    )
    rolling_audit = prepared.full_session_block_audit() if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else prepared.block_audit
    rolling_audit.to_csv(
        sliding_root / "all_session_block_cut_audit.csv", index=False
    )
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "workflow": args.workflow,
            "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id,
            "session_count": len(prepared.assignments),
            "window_count": len(windows),
            "training_sessions_per_window": 9,
            "validation_sessions_per_window": 3,
            "rolling_test_sessions_per_window": 1,
            "step_size_sessions": 1,
            "registered_initialization_count_per_window": 8,
            "window_parallel_workers": args.window_parallel_workers,
            "parameter_order": ["w", "s0", "ell_short", "ell_long", "kappa", "ah"],
            "hazard_initialization_aggregation": (
                "direct matched rolling-test window winner; no aggregation"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "mean h_B across contributing winners, then ah=logit(mean h_B)"
            ),
            "neutral_hazard_rule": "h_N=h_B/2; no additional parameter",
            "session_state_reset": "one-hot neutral independently per session",
            "terminal_block_policy": (
                "included_in_rolling_and_final_two_fine_tune_test"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "right_censored_and_ignored"
            ),
            "heldout_policy": (
                "matched_test_sessions_only_literal_final_two_blocks"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "two_preceding_observed_complete_blocks"
            ),
            "source_files": list(prepared.source_files),
            "source_sha256": list(prepared.source_sha256),
        },
        sliding_root / "experiment_manifest.json",
    )
    selected_windows = windows[: args.window_limit] if args.window_limit else windows
    winner_parameters: dict[int, list[float]] = {}
    if not args.single_only:
        winner_parameters = _run_windows(
            prepared, selected_windows, registry, args, sliding_root
        )
        _generate_sliding_global_figures(sliding_root, selected_windows)
        if args.window_limit is not None:
            print(
                "Sliding M1 development subset complete; session stage skipped.",
                flush=True,
            )
            return 0
    if args.sliding_only:
        return 0
    if args.single_only:
        for window in windows:
            path = sliding_root / window.directory_name / "fit_result.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed M1 rolling result: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            winner_parameters[window.window_id] = list(
                _winner(summary)["best_constrained_parameters"]
            )
        _generate_sliding_global_figures(sliding_root, windows)

    if args.workflow == FULL_SESSION_MATCHED_WORKFLOW:
        initializations = match_window_winners_to_test_sessions(
            prepared.assignments, windows, winner_parameters,
            ("w", "s0", "ell_short", "ell_long", "kappa", "ah"),
        )
        initialization_filename = "matched_window_test_initializations.csv"
        block_audit = prepared.final_two_block_audit()
    else:
        initializations = aggregate_m1_session_initializations(
            prepared.assignments, windows, winner_parameters
        )
        initialization_filename = "session_window_mean_initializations.csv"
        block_audit = prepared.block_audit
    single_root.mkdir(parents=True, exist_ok=True)
    initializations.to_csv(
        single_root / initialization_filename, index=False
    )
    block_audit.to_csv(
        single_root / "all_session_block_cut_audit.csv", index=False
    )
    _fine_tune_all(prepared, registry, args, single_root, initializations)
    _generate_single_global_figures(initializations, single_root)
    print(
        f"experiment=complete sliding_results={sliding_root} "
        f"single_session_results={single_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
