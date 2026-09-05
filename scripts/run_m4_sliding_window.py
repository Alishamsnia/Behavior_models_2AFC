#!/usr/bin/env python3
"""Run the isolated M4 9/3/1 sliding and per-session experiment."""

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
import warnings

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
from m0_sensory_v1.model import LBFGSSettings  # noqa: E402
from m4_hsmm_delta_v1.analysis import (  # noqa: E402
    build_m4_hazard_curve,
    build_m4_trial_predictions,
    save_m4_trial_predictions,
)
from m4_hsmm_delta_v1.fit import (  # noqa: E402
    fit_m4_all_initializations,
    fit_m4_single_initialization,
)
from m4_hsmm_delta_v1.initialization import M4InitializationSpec  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402
from m4_hsmm_delta_v1.session import m4_session_trials_from_frame  # noqa: E402
from m4_sliding_window_v1 import (  # noqa: E402
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    M4_PARAMETER_NAMES,
    WORKFLOW_CHOICES,
    aggregate_m4_session_initializations,
    build_sliding_windows,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_identifiability_outputs,
    generate_m4_experiment_outputs,
    generate_truncation_outputs,
    identifiability_outputs_are_current,
    match_window_winners_to_test_sessions,
    prediction_performance,
    prepare_sliding_data,
    truncation_outputs_are_current,
)


EXPERIMENT_ID = "M4_sliding_9train_3validation_1test_two_complete_blocks_v1"
CPU_BUDGET = 32


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit unchanged M4 in chronological 9/3/1 windows with eight starts, "
            "then fine-tune every session and score two held-out complete blocks."
        )
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--workflow", choices=WORKFLOW_CHOICES, default=LEGACY_PARTITIONED_WORKFLOW)
    parser.add_argument("--sliding-output-dir", type=Path, default=Path("results_sliding_window"))
    parser.add_argument("--single-output-dir", type=Path, default=Path("results_single_session"))
    parser.add_argument("--optimizer", choices=("adam", "lbfgs", "hybrid"), default="lbfgs")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adam-warmup-epochs", type=int, default=1000)
    parser.add_argument("--lbfgs-max-steps", type=int, default=250)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--lbfgs-max-evaluations", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument("--rolling-workers", type=int, default=8)
    parser.add_argument("--rolling-threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--window-parallel-workers",
        type=int,
        default=1,
        help=(
            "Concurrent rolling windows fitted at once. Each window still uses "
            "--rolling-workers optimizer-start processes, so total CPU usage is "
            "roughly window-parallel-workers x rolling-workers x rolling-threads-per-worker "
            "and must stay within the 32-CPU budget."
        ),
    )
    parser.add_argument("--single-session-workers", type=int, default=32)
    parser.add_argument("--single-session-threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--identifiability-grid-points", type=int, default=0,
        help="Opt-in M4 ah/bh and alpha_theta/bh grids; 0 disables",
    )
    parser.add_argument(
        "--truncation-sensitivity", action="store_true",
        help="Opt in to fixed-parameter Rmax 65/75/100 sensitivity per fit",
    )
    parser.add_argument(
        "--include-full-run-length-distributions", action="store_true",
        help="Include 100 pre/post run-length marginal columns in predictions",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sliding-only", action="store_true")
    parser.add_argument("--single-only", action="store_true")
    parser.add_argument("--window-limit", type=int, default=None)
    args = parser.parse_args()
    if args.sliding_only and args.single_only:
        parser.error("--sliding-only and --single-only are mutually exclusive")
    positive = (
        args.adam_warmup_epochs, args.lbfgs_max_steps, args.lbfgs_learning_rate,
        args.lbfgs_history_size, args.lbfgs_max_evaluations,
        args.lbfgs_tolerance_grad, args.lbfgs_tolerance_change,
        args.rolling_workers, args.rolling_threads_per_worker,
        args.window_parallel_workers,
        args.single_session_workers, args.single_session_threads_per_worker,
    )
    if any(value <= 0 for value in positive):
        parser.error("All optimizer and worker controls must be positive")
    if (
        args.window_parallel_workers * min(args.rolling_workers, 8) * args.rolling_threads_per_worker
        > CPU_BUDGET
    ):
        parser.error(
            "window-parallel-workers x rolling worker allocation exceeds the "
            "32-CPU budget; reduce --window-parallel-workers, --rolling-workers, "
            "or --rolling-threads-per-worker"
        )
    if args.single_session_workers * args.single_session_threads_per_worker > CPU_BUDGET:
        parser.error("Single-session worker allocation exceeds the 32-CPU budget")
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
    if args.optimizer == "lbfgs" and args.learning_rate is not None:
        parser.error("--learning-rate configures Adam, not pure L-BFGS")
    if args.identifiability_grid_points != 0 and args.identifiability_grid_points < 3:
        parser.error("--identifiability-grid-points must be 0 or at least 3")
    return args


def _lbfgs(args: argparse.Namespace | dict[str, Any]) -> LBFGSSettings:
    if isinstance(args, dict):
        return LBFGSSettings(**args)
    return LBFGSSettings(
        maximum_steps=args.lbfgs_max_steps,
        learning_rate=args.lbfgs_learning_rate,
        history_size=args.lbfgs_history_size,
        maximum_evaluations_per_step=args.lbfgs_max_evaluations,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
    )


def _clamp_initialization_bounds(values: list[float], registry) -> list[float]:
    """Keep every sigmoid/softplus-decoded coordinate strictly inside its open bounds.

    A weakly identified coordinate can legitimately decode to exactly a
    float64 boundary through sigmoid/softplus saturation -- a near-zero
    lapse rate leaves ``ell_long`` unconstrained, a near-zero learning rate leaves
    ``alpha_theta`` unconstrained, and the same risk applies to every other
    free coordinate (``w``, ``s0``, ``ell_short``, ``delta0``, ``bh``). The raw
    parameterization needs each value strictly inside its open interval to
    encode a finite optimizer starting point (``logit(0)``/``logit(1)`` and
    inverse-softplus at the floor are all undefined), so this adjusts only
    the values used to start this fit; the honestly aggregated means are
    still recorded verbatim in ``initialization_provenance.json``.
    """

    margin = 1e-6
    floor = registry.parameter_epsilon + margin
    clamped = list(values)
    clamped[0] = max(clamped[0], floor)  # w
    clamped[1] = max(clamped[1], floor)  # s0
    clamped[2] = min(max(clamped[2], margin), registry.lapse_max - margin)  # ell_short
    clamped[3] = min(max(clamped[3], margin), registry.lapse_max - margin)  # ell_long
    clamped[4] = max(clamped[4], margin)  # kappa
    clamped[6] = max(clamped[6], floor)  # delta0
    clamped[7] = min(max(clamped[7], margin), 1.0 - margin)  # alpha_theta
    clamped[8] = max(clamped[8], floor)  # bh
    return clamped


def _sessions(prepared, session_ids: tuple[str, ...], role: str, scope: str = "prefix"):
    return tuple(
        m4_session_trials_from_frame(
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


def _find_winner_run(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the winning start's run record, or None if the window has no winner.

    Unlike a raising lookup, this never stops the caller -- it hands back
    ``None`` so a window with no eligible start can be handled gracefully
    instead of aborting every other rolling window still in flight.
    """

    winner_id = summary.get("winner_start_id")
    if winner_id is None:
        return None
    return next(
        (run for run in summary.get("runs", []) if run.get("start_id") == winner_id),
        None,
    )


def _warn_no_winner(window) -> None:
    """Report a window with no eligible winner without stopping the run."""

    message = (
        f"M4 window={window.window_id:03d} has no converged, projection-acceptable, "
        "tail-acceptable winner among its eight starts; excluding it from session "
        "initialization aggregation and continuing with the remaining rolling windows."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"window={window.window_id:03d} status=no_winner", flush=True)


def _record_winner(
    window, summary: dict[str, Any], winner_parameters: dict[int, list[float]]
) -> None:
    """Add a window's winner to winner_parameters, or warn and skip if there is none."""

    winner = _find_winner_run(summary)
    if winner is None:
        _warn_no_winner(window)
        return
    winner_parameters[window.window_id] = list(winner["best_constrained_parameters"])


def _print_start_completion(window_id: int, result) -> None:
    """Print one live line per completed optimizer start, as it lands.

    This is purely a visibility improvement -- it changes no fitting,
    convergence, or selection logic. Without it a window's ~8 starts run as
    an opaque block for up to an hour or more with no intermediate signal.
    """

    print(
        f"window={window_id:03d} start={result.start_id:02d} "
        f"converged={result.converged} "
        f"projection_ok={result.projection_acceptable} "
        f"tail_ok={result.tail_occupancy_acceptable} "
        f"tail_prob={result.training_mean_tail_probability_pre:.4f} "
        f"train_nll={result.recalculated_best_training_mean_nll:.6f} "
        f"termination={result.termination_reason}",
        flush=True,
    )


def _continuous_single_session_predictions(
    prefix: pd.DataFrame,
    heldout: pd.DataFrame,
    raw_parameters,
    registry,
    *,
    include_full_run_length_distributions: bool = False,
) -> pd.DataFrame:
    """Continue M4's joint HSMM belief and learned delta across held-out blocks."""

    combined = pd.concat([prefix, heldout], ignore_index=True).sort_values("trial_index")
    if combined["session_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous M4 prediction requires exactly one session")
    timeline = combined.copy()
    timeline["split_role"] = "train"
    predictions = build_m4_trial_predictions(
        timeline,
        raw_parameters,
        registry,
        allow_test=False,
        include_full_run_length_distributions=include_full_run_length_distributions,
    )
    predictions["split_role"] = predictions["experiment_partition"].map(
        {"prefix": "train", "heldout": "test"}
    )
    if predictions["split_role"].isna().any():
        raise ValueError("Single-session M4 predictions contain an unknown partition")
    return predictions.sort_values("trial_index").reset_index(drop=True)


def _optional_outputs_current(output: Path, args) -> bool:
    return (
        identifiability_outputs_are_current(
            output / "m4_identifiability",
            grid_points=int(args.identifiability_grid_points),
        )
        and truncation_outputs_are_current(
            output / "m4_truncation_sensitivity",
            enabled=bool(args.truncation_sensitivity),
        )
    )


def _save_predictions_and_outputs(
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    best_raw_parameters,
    registry,
    output: Path,
    *,
    include_full_run_length_distributions: bool,
) -> None:
    _save_core_single_session_outputs(
        predictions, best_raw_parameters, registry, output
    )
    generate_m4_experiment_outputs(
        figure_ready(predictions), summary, best_raw_parameters, registry,
        output / "m4_m0_style_figures", output / "m4_figures",
        include_full_run_length_distributions=include_full_run_length_distributions,
    )


def _save_core_single_session_outputs(
    predictions: pd.DataFrame,
    best_raw_parameters,
    registry,
    output: Path,
) -> None:
    """Save auditable predictions and hazards even when a fit is ineligible."""

    save_m4_trial_predictions(predictions, output / "trial_predictions.csv")
    prediction_performance(predictions).to_csv(
        output / "split_and_session_metrics.csv", index=False
    )
    diagnostic_output = output / "m4_figures"
    diagnostic_output.mkdir(parents=True, exist_ok=True)
    build_m4_hazard_curve(best_raw_parameters, registry).to_csv(
        diagnostic_output / "hazard_curve.csv", index=False
    )


def _fit_window(prepared, window, registry, args, root: Path) -> dict[str, Any]:
    rolling_scope = "full" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "prefix"
    output = root / window.directory_name
    result_path = output / "fit_result.json"
    manifest_path = output / "window_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    requested_full = bool(args.include_full_run_length_distributions)
    current = (
        figure_outputs_are_current(output / "m4_m0_style_figures")
        and _optional_outputs_current(output, args)
        and previous.get("full_run_length_distributions_included") == requested_full
    )
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    registry_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous, registry, SOURCE_ROOT)
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
        and previous.get("status") == "complete"
        and current
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
        validation = _sessions(prepared, window.validation_session_ids, "validation", rolling_scope)
        print(
            f"window={window.window_id:03d} "
            f"train={window.training_ranks[0]:03d}-{window.training_ranks[-1]:03d} "
            f"validation={window.validation_ranks[0]:03d}-{window.validation_ranks[-1]:03d} "
            f"test={window.test_rank:03d} starts={len(registry.initializations)} "
            f"states=3x{registry.maximum_represented_run_length} "
            f"cpu={min(args.rolling_workers, 8)}x{args.rolling_threads_per_worker}",
            flush=True,
        )
        fit = fit_m4_all_initializations(
            train, validation, registry,
            optimizer_strategy=args.optimizer,
            max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs(args),
            parallel_workers=args.rolling_workers,
            cpu_threads_per_worker=args.rolling_threads_per_worker,
            output_directory=output / "starts",
            completion_callback=lambda result: _print_start_completion(
                window.window_id, result
            ),
        )
        summary = fit.as_dict()
    if len(summary.get("runs", [])) != 8:
        raise RuntimeError("Every rolling M4 window must preserve all eight starts")
    _atomic_json(summary, result_path)
    winner = _find_winner_run(summary)
    elapsed = time.monotonic() - started
    if winner is None:
        _warn_no_winner(window)
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "status": "no_winner",
            "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id,
            **window.as_dict(),
            "registered_initialization_count": 8,
            "executed_start_ids": [run["start_id"] for run in summary["runs"]],
            "optimizer": args.optimizer,
            "winner_start_id": None,
            "selection_status": summary.get("selection_status"),
            "elapsed_seconds": previous.get("elapsed_seconds", elapsed),
            "latest_postfit_elapsed_seconds": elapsed,
            "resumed_from_saved_fit": resumed_postfit,
        }
        _atomic_json(manifest, manifest_path)
        return summary
    raw = winner["best_raw_parameters"]
    predictions = build_m4_trial_predictions(
        _window_canonical(prepared, window, rolling_scope), raw, registry, allow_test=True,
        include_full_run_length_distributions=requested_full,
    )
    _save_predictions_and_outputs(
        predictions, summary, raw, registry, output,
        include_full_run_length_distributions=requested_full,
    )
    train_sessions = _sessions(prepared, window.training_session_ids, "train", rolling_scope)
    if args.identifiability_grid_points:
        generate_identifiability_outputs(
            train_sessions, raw, registry, output / "m4_identifiability",
            grid_points=args.identifiability_grid_points,
        )
    if args.truncation_sensitivity:
        generate_truncation_outputs(
            train_sessions, raw, registry, output / "m4_truncation_sensitivity"
        )
    relevant = set((*window.training_session_ids, *window.validation_session_ids, window.test_session_id))
    rolling_audit = prepared.full_session_block_audit() if rolling_scope == "full" else prepared.block_audit
    rolling_audit.loc[
        rolling_audit["session_id"].astype(str).isin(relevant)
    ].to_csv(output / "block_cut_audit.csv", index=False)
    best = winner["best_constrained_parameters"]
    curve = build_m4_hazard_curve(raw, registry).set_index("represented_run_length")
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "workflow": args.workflow,
            "status": "complete",
            "resume_fingerprint": manifest_fingerprint(registry, SOURCE_ROOT),
            "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id,
            **window.as_dict(),
            "registered_initialization_count": 8,
            "executed_start_ids": [run["start_id"] for run in summary["runs"]],
            "optimizer": args.optimizer,
            "winner_start_id": summary["winner_start_id"],
            "winner_best_constrained_parameters": best,
            "winner_projection_acceptable": winner["projection_acceptable"],
            "winner_tail_occupancy_acceptable": winner["tail_occupancy_acceptable"],
            "winner_training_mean_tail_probability_pre": winner["training_mean_tail_probability_pre"],
            "winner_biased_hazards": {
                "r1": float(curve.loc[1, "biased_hazard"]),
                "r23": float(curve.loc[registry.run_length_center, "biased_hazard"]),
                "r65": float(curve.loc[registry.maximum_represented_run_length, "biased_hazard"]),
            },
            "neutral_hazard_rule": "h_N(r)=h_B(r)/2; no additional parameter",
            "rolling_test_scope": (
                "full_session" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "retained prefix only; final two complete blocks and terminal block excluded"
            ),
            "full_run_length_distributions_included": requested_full,
            "identifiability_grid_points": args.identifiability_grid_points,
            "truncation_sensitivity": bool(args.truncation_sensitivity),
            "cpu_allocation": {
                "workers": min(args.rolling_workers, 8),
                "threads_per_worker": args.rolling_threads_per_worker,
                "maximum_threads": min(args.rolling_workers, 8) * args.rolling_threads_per_worker,
            },
            "elapsed_seconds": previous.get("elapsed_seconds", elapsed),
            "latest_postfit_elapsed_seconds": elapsed,
            "resumed_from_saved_fit": resumed_postfit,
        },
        manifest_path,
    )
    print(
        f"window={window.window_id:03d} winner={summary['winner_start_id']} "
        f"status=complete elapsed_s={elapsed:.1f}", flush=True,
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
    """Fit every requested M4 window, running up to --window-parallel-workers at once.

    Windows are fully independent -- separate sessions, separate output
    directories, no warm-starting from each other -- so every window is
    submitted up front and the pool starts the next queued window the instant
    a worker frees up. A fast window is never held back waiting for slower
    siblings to finish before its replacement begins. Each window still runs
    its own self-contained resume check (unchanged from the sequential path),
    so ``--resume`` behaves identically regardless of how many workers are
    requested here. The CLI already enforces window-parallel-workers x
    rolling-workers x rolling-threads-per-worker <= the 32-CPU budget.
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
            _record_winner(window, summary, winner_parameters)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_window_worker,
            initargs=(args.rolling_threads_per_worker,),
        ) as executor:
            futures = {
                executor.submit(_fit_window, prepared, window, registry, args, root): window
                for window in windows
            }
            for future in as_completed(futures):
                window = futures[future]
                summary = future.result()
                _record_winner(window, summary, winner_parameters)
    completed = len(winner_parameters)
    print(
        f"rolling_windows_summary total={len(windows)} "
        f"winner_selected={completed} no_winner={len(windows) - completed}",
        flush=True,
    )
    return winner_parameters


def _single_session_ineligibility_reasons(run_record: dict[str, Any]) -> list[str]:
    """Return the registered M4 acceptance failures for one session fit."""

    reasons: list[str] = []
    if not bool(run_record.get("converged", False)):
        reasons.append("not_converged")
    if not bool(run_record.get("projection_acceptable", False)):
        reasons.append("projection_activation_unacceptable")
    if not bool(run_record.get("tail_occupancy_acceptable", False)):
        reasons.append("tail_occupancy_unacceptable")
    return reasons


def _is_duration_tail_tolerance_error(error: RuntimeError) -> bool:
    return str(error) == "Duration distribution did not reach the requested tail tolerance"


def _manifest_eligible_for_global_analysis(manifest: dict[str, Any]) -> bool:
    """Read new eligibility metadata or infer it from legacy complete manifests."""

    if "eligible_for_global_analysis" in manifest:
        return bool(manifest["eligible_for_global_analysis"])
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("converged", False)
        and manifest.get("projection_acceptable", False)
        and manifest.get("tail_occupancy_acceptable", False)
    )


def _single_session_result_from_manifest(
    manifest: dict[str, Any], *, status: str
) -> dict[str, Any]:
    inferred_fit_eligible = bool(
        manifest.get("converged", False)
        and manifest.get("projection_acceptable", False)
        and manifest.get("tail_occupancy_acceptable", False)
    )
    return {
        "session_id": str(manifest["session_id"]),
        "chronological_rank": int(manifest["chronological_rank"]),
        "status": status,
        "fit_eligible": bool(
            manifest.get("fit_eligible", inferred_fit_eligible)
        ),
        "eligible_for_global_analysis": _manifest_eligible_for_global_analysis(
            manifest
        ),
        "ineligibility_reasons": list(manifest.get("ineligibility_reasons", [])),
        "converged": bool(manifest.get("converged", False)),
        "projection_acceptable": bool(
            manifest.get("projection_acceptable", False)
        ),
        "tail_occupancy_acceptable": bool(
            manifest.get("tail_occupancy_acceptable", False)
        ),
        "elapsed_seconds": float(manifest.get("elapsed_seconds", 0.0)),
    }


def _single_session_job(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(payload["threads_per_worker"]))
    registry = payload["registry"]
    prefix, heldout = payload["prefix"], payload["heldout"]
    output = Path(payload["output"])
    args = payload["optimizer_args"]
    row = payload["initialization"]
    result_path = output / "fine_tune_fit_result.json"
    manifest_path = output / "single_session_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    requested_full = bool(args["include_full_run_length_distributions"])
    current = (
        figure_outputs_are_current(output / "m4_m0_style_figures")
        and _optional_outputs_current(output, argparse.Namespace(**args))
        and previous.get("full_run_length_distributions_included") == requested_full
    )
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    registry_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous, registry, SOURCE_ROOT)
    if payload["resume"] and previous_result is not None and not registry_stale:
        if previous.get("status") == "ineligible":
            return _single_session_result_from_manifest(
                previous, status="resume_skip_ineligible"
            )
        if previous.get("status") == "complete" and current:
            legacy_reasons = _single_session_ineligibility_reasons(previous)
            if legacy_reasons:
                previous.update(
                    {
                        "status": "ineligible",
                        "fit_eligible": False,
                        "eligible_for_global_analysis": False,
                        "ineligibility_reasons": legacy_reasons,
                        "diagnostic_status": (
                            "legacy_outputs_retained_but_fit_reclassified_ineligible"
                        ),
                        "ineligible_session_policy": (
                            "retain existing outputs and exclude from pooled analysis"
                        ),
                        "automatic_retry_policy": (
                            "none; preserve direct matched rolling-window initialization"
                        ),
                    }
                )
                _atomic_json(previous, manifest_path)
                return _single_session_result_from_manifest(
                    previous, status="resume_reclassified_ineligible"
                )
            return _single_session_result_from_manifest(
                previous, status="resume_skip"
            )

    output.mkdir(parents=True, exist_ok=True)
    values = [float(row[f"initial_{name}"]) for name in M4_PARAMETER_NAMES]
    values = _clamp_initialization_bounds(values, registry)
    initialization = M4InitializationSpec(
        start_id=1, source_m3_start_id=1,
        w=values[0], s0=values[1], ell_short=values[2], ell_long=values[3],
        kappa=values[4], ah=values[5], delta0=values[6], alpha_theta=values[7],
        bh=values[8],
        purpose="sliding-window initialization for M4 session fine-tuning",
    )
    experimental_registry = replace(registry, initializations=(initialization,))
    started = time.monotonic()
    resumed_postfit = payload["resume"] and previous_result is not None and not registry_stale
    if resumed_postfit:
        run_record = previous_result
    else:
        train_session = m4_session_trials_from_frame(prefix, split_role="train")
        monitor_frame = prefix.copy()
        monitor_frame["split_role"] = "validation"
        monitor_session = m4_session_trials_from_frame(monitor_frame, split_role="validation")
        result = fit_m4_single_initialization(
            (train_session,), (monitor_session,), experimental_registry, initialization,
            optimizer_strategy=args["optimizer"],
            max_epochs_override=args["max_epochs"],
            initial_learning_rate_override=args["learning_rate"],
            adam_warmup_epochs=args["adam_warmup_epochs"],
            lbfgs_settings=_lbfgs(args["lbfgs"]),
        )
        run_record = result.as_dict()
        _atomic_json(run_record, result_path)
    summary = {
        "result_schema_version": "m4_single_session_fine_tune_v1",
        "model_id": registry.model_id,
        "mouse_id": payload["mouse_id"],
        "winner_start_id": 1,
        "runs": [run_record],
    }
    raw = run_record["best_raw_parameters"]
    predictions = _continuous_single_session_predictions(
        prefix, heldout, raw, registry,
        include_full_run_length_distributions=requested_full,
    )
    provenance = dict(row)
    for key, value in list(provenance.items()):
        if pd.isna(value):
            provenance[key] = None
    _atomic_json(provenance, output / "initialization_provenance.json")

    ineligibility_reasons = _single_session_ineligibility_reasons(run_record)
    fit_eligible = not ineligibility_reasons
    diagnostic_status = "skipped_ineligible_fit"
    if fit_eligible:
        try:
            _save_predictions_and_outputs(
                predictions, summary, raw, registry, output,
                include_full_run_length_distributions=requested_full,
            )
            diagnostic_status = "complete"
        except RuntimeError as error:
            if not _is_duration_tail_tolerance_error(error):
                raise
            ineligibility_reasons.append(
                "duration_distribution_tail_tolerance_not_reached"
            )
            diagnostic_status = "failed_tail_tolerance"
    else:
        _save_core_single_session_outputs(predictions, raw, registry, output)

    eligible_for_global_analysis = not ineligibility_reasons
    if eligible_for_global_analysis:
        train_sessions = (m4_session_trials_from_frame(prefix, split_role="train"),)
        if args["identifiability_grid_points"]:
            generate_identifiability_outputs(
                train_sessions, raw, registry, output / "m4_identifiability",
                grid_points=int(args["identifiability_grid_points"]),
            )
        if args["truncation_sensitivity"]:
            generate_truncation_outputs(
                train_sessions, raw, registry, output / "m4_truncation_sensitivity"
            )

    curve = build_m4_hazard_curve(raw, registry).set_index("represented_run_length")
    elapsed = time.monotonic() - started
    manifest = {
            "experiment_id": EXPERIMENT_ID,
            "workflow": payload["workflow"],
            "status": "complete" if eligible_for_global_analysis else "ineligible",
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
                "continuous 3x65 HSMM belief and delta_t from prefix; heldout category "
                "feedback affects only subsequent heldout trials"
            ),
            "heldout_choices_used_for_parameter_updates": False,
            "validation_monitor": "duplicate training prefix for optimizer API reporting only",
            "initial_constrained_parameters": values,
            "best_constrained_parameters": list(run_record["best_constrained_parameters"]),
            "best_biased_hazards": {
                "r1": float(curve.loc[1, "biased_hazard"]),
                "r23": float(curve.loc[registry.run_length_center, "biased_hazard"]),
                "r65": float(curve.loc[registry.maximum_represented_run_length, "biased_hazard"]),
            },
            "neutral_hazard_rule": "h_N(r)=h_B(r)/2; no additional parameter",
            "converged": bool(run_record["converged"]),
            "projection_acceptable": bool(run_record["projection_acceptable"]),
            "tail_occupancy_acceptable": bool(run_record["tail_occupancy_acceptable"]),
            "training_mean_tail_probability_pre": run_record["training_mean_tail_probability_pre"],
            "termination_reason": run_record["termination_reason"],
            "fit_eligible": fit_eligible,
            "eligible_for_global_analysis": eligible_for_global_analysis,
            "ineligibility_reasons": ineligibility_reasons,
            "diagnostic_status": diagnostic_status,
            "ineligible_session_policy": (
                "retain fit, predictions, metrics, hazards, and provenance; "
                "skip duration-dependent/optional diagnostics and exclude from pooled analysis"
            ),
            "automatic_retry_policy": (
                "none; preserve direct matched rolling-window initialization"
            ),
            "full_run_length_distributions_included": requested_full,
            "identifiability_grid_points": args["identifiability_grid_points"],
            "truncation_sensitivity": bool(args["truncation_sensitivity"]),
            "resumed_from_saved_fit": resumed_postfit,
            "elapsed_seconds": previous.get("elapsed_seconds", elapsed),
            "latest_postfit_elapsed_seconds": elapsed,
    }
    _atomic_json(manifest, manifest_path)
    return _single_session_result_from_manifest(
        manifest,
        status=(
            "ineligible"
            if not eligible_for_global_analysis
            else "resume_postfit" if resumed_postfit else "complete"
        ),
    )


def _fine_tune_all(prepared, registry, args, root: Path, initializations) -> None:
    optimizer_args = {
        "optimizer": args.optimizer, "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "adam_warmup_epochs": args.adam_warmup_epochs,
        "identifiability_grid_points": args.identifiability_grid_points,
        "truncation_sensitivity": args.truncation_sensitivity,
        "include_full_run_length_distributions": args.include_full_run_length_distributions,
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
    jobs = []
    prefix_scope = "final_two_prefix" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "prefix"
    heldout_scope = "final_two_heldout" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "heldout"
    for assignment in initializations.sort_values("chronological_rank").itertuples(index=False):
        session_id, rank = str(assignment.session_id), int(assignment.chronological_rank)
        jobs.append({
            "registry": registry, "mouse_id": prepared.mouse_id,
            "session_id": session_id, "rank": rank,
            "prefix": prepared.frame(session_id, prefix_scope, "train"),
            "heldout": prepared.frame(session_id, heldout_scope, "test"),
            "initialization": by_session.loc[session_id].to_dict(),
            "output": str(root / f"session_{rank:03d}_{session_id}"),
            "optimizer_args": optimizer_args,
            "threads_per_worker": args.single_session_threads_per_worker,
            "resume": args.resume,
            "workflow": args.workflow,
        })
    workers = min(args.single_session_workers, len(jobs), multiprocessing.cpu_count())
    print(
        f"single_session_jobs={len(jobs)} parallel_workers={workers} "
        f"threads_per_worker={args.single_session_threads_per_worker} "
        f"maximum_cpu_threads={workers * args.single_session_threads_per_worker}", flush=True,
    )
    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            result = _single_session_job(job)
            results.append(result)
            print(f"session={result['session_id']} status={result['status']}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_single_session_job, job): job["session_id"] for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"session={result['session_id']} status={result['status']}", flush=True)
    results.sort(key=lambda item: str(item["session_id"]))
    eligible_count = sum(
        bool(item["eligible_for_global_analysis"]) for item in results
    )
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID, "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id, "session_count": len(results),
            "eligible_session_count": eligible_count,
            "ineligible_session_count": len(results) - eligible_count,
            "global_analysis_policy": (
                "include only sessions passing convergence, projection, tail-occupancy, "
                "and duration-diagnostic acceptance"
            ),
            "cpu_allocation": {
                "workers": workers,
                "threads_per_worker": args.single_session_threads_per_worker,
                "maximum_threads": workers * args.single_session_threads_per_worker,
            },
            "results": results,
        },
        root / "single_session_run_summary.json",
    )


def _generate_sliding_global_figures(root: Path, windows) -> None:
    sources = [(window.directory_name, root / window.directory_name / "trial_predictions.csv") for window in windows]
    manifest = generate_global_prediction_figures(
        sources, root / "global_figures", source_kind="sliding_windows"
    )
    print(f"sliding_global_figures=complete sources={manifest['source_count']}", flush=True)


def _generate_single_global_figures(initializations, root: Path) -> None:
    sources: list[tuple[str, Path]] = []
    exclusions: list[dict[str, Any]] = []
    for assignment in initializations.sort_values("chronological_rank").itertuples(index=False):
        rank, session_id = int(assignment.chronological_rank), str(assignment.session_id)
        name = f"session_{rank:03d}_{session_id}"
        directory = root / name
        manifest_path = directory / "single_session_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        predictions_path = directory / "trial_predictions.csv"
        if _manifest_eligible_for_global_analysis(manifest) and predictions_path.is_file():
            sources.append((name, predictions_path))
        else:
            exclusions.append(
                {
                    "session_id": session_id,
                    "chronological_rank": rank,
                    "source_run_id": name,
                    "manifest_status": manifest.get("status", "missing"),
                    "ineligibility_reasons": manifest.get(
                        "ineligibility_reasons", ["missing_or_incomplete_manifest"]
                    ),
                    "trial_predictions_available": predictions_path.is_file(),
                }
            )
    selection = {
        "experiment_id": EXPERIMENT_ID,
        "source_kind": "single_session_fine_tuning",
        "eligibility_policy": (
            "converged and projection-acceptable and tail-occupancy-acceptable "
            "and duration diagnostics complete"
        ),
        "eligible_source_count": len(sources),
        "excluded_source_count": len(exclusions),
        "eligible_source_run_ids": [source_id for source_id, _ in sources],
        "excluded_sources": exclusions,
    }
    _atomic_json(selection, root / "single_session_global_selection.json")
    if not sources:
        print(
            f"single_session_global_figures=skipped eligible_sources=0 "
            f"excluded_sources={len(exclusions)}",
            flush=True,
        )
        return
    manifest = generate_global_prediction_figures(
        sources,
        root / "global_figures",
        source_kind="single_session_fine_tuning_eligible_only",
    )
    manifest.update(
        {
            "eligibility_policy": selection["eligibility_policy"],
            "excluded_source_count": len(exclusions),
            "excluded_sources": exclusions,
        }
    )
    _atomic_json(manifest, root / "global_figures" / "global_figure_manifest.json")
    print(
        f"single_session_global_figures=complete eligible_sources={manifest['source_count']} "
        f"excluded_sources={len(exclusions)}",
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    registry = load_m4_model_registry(args.registry)
    if len(registry.initializations) != 8:
        raise RuntimeError("Sliding M4 requires exactly eight registered initializations")
    prepared = prepare_sliding_data(args.raw_data_root, registry)
    windows = build_sliding_windows(prepared.assignments)
    workflow_suffix = "full_session_matched_v1" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else None
    sliding_root = args.sliding_output_dir.expanduser().resolve() / registry.model_id / prepared.mouse_id
    single_root = args.single_output_dir.expanduser().resolve() / registry.model_id / prepared.mouse_id
    if workflow_suffix is not None:
        sliding_root = sliding_root / workflow_suffix
        single_root = single_root / workflow_suffix
    sliding_root.mkdir(parents=True, exist_ok=True)
    prepared.assignments.to_csv(sliding_root / "chronological_sessions.csv", index=False)
    rolling_audit = prepared.full_session_block_audit() if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else prepared.block_audit
    rolling_audit.to_csv(sliding_root / "all_session_block_cut_audit.csv", index=False)
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID, "workflow": args.workflow, "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id, "session_count": len(prepared.assignments),
            "window_count": len(windows), "training_sessions_per_window": 9,
            "validation_sessions_per_window": 3, "rolling_test_sessions_per_window": 1,
            "step_size_sessions": 1, "registered_initialization_count_per_window": 8,
            "parameter_order": list(M4_PARAMETER_NAMES),
            "fine_tune_initialization_policy": (
                "direct matched rolling-test window winner; no aggregation"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "legacy window aggregation by parameter transform"
            ),
            "center_hazard_initialization_aggregation": (
                "none; direct matched winner"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "mean h_B(23), then ah=logit(mean h_B(23))"
            ),
            "prototype_initialization_aggregation": (
                "none; direct matched winner"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "mean theta_L(0), then delta0=logit(mean theta_L(0))"
            ),
            "learning_rate_initialization_aggregation": (
                "none; direct matched winner"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "arithmetic mean constrained alpha_theta"
            ),
            "hazard_slope_initialization_aggregation": (
                "none; direct matched winner"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "arithmetic mean positive constrained bh"
            ),
            "neutral_hazard_rule": "h_N(r)=h_B(r)/2; no additional parameter",
            "session_state_reset": "neutral run-length-one joint state and delta=delta0 independently per session",
            "terminal_block_policy": (
                "included_in_rolling_and_final_two_fine_tune_test"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "right_censored_and_ignored"
            ),
            "heldout_policy": (
                "matched_test_sessions_only_literal_final_two_blocks"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "two_preceding_observed_complete_blocks"
            ),
            "run_length_center": registry.run_length_center,
            "maximum_represented_run_length": registry.maximum_represented_run_length,
            "tail_acceptance_threshold": registry.maximum_accepted_mean_tail_probability,
            "full_run_length_distributions_included": bool(args.include_full_run_length_distributions),
            "identifiability_grid_points": args.identifiability_grid_points,
            "truncation_sensitivity": bool(args.truncation_sensitivity),
            "cpu_budget": CPU_BUDGET,
            "rolling_cpu_allocation": {"workers": min(args.rolling_workers, 8), "threads_per_worker": args.rolling_threads_per_worker, "window_parallel_workers": args.window_parallel_workers},
            "single_session_cpu_allocation": {"workers": args.single_session_workers, "threads_per_worker": args.single_session_threads_per_worker},
            "source_files": list(prepared.source_files), "source_sha256": list(prepared.source_sha256),
        },
        sliding_root / "experiment_manifest.json",
    )
    selected_windows = windows[:args.window_limit] if args.window_limit else windows
    winner_parameters: dict[int, list[float]] = {}
    if not args.single_only:
        winner_parameters = _run_windows(
            prepared, selected_windows, registry, args, sliding_root
        )
        succeeded_windows = [
            window for window in selected_windows if window.window_id in winner_parameters
        ]
        if not succeeded_windows:
            raise RuntimeError(
                "No M4 rolling window produced an eligible winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)
        if args.window_limit is not None:
            print("Sliding M4 development subset complete; session stage skipped.", flush=True)
            return 0
    if args.sliding_only:
        return 0
    if args.single_only:
        for window in windows:
            path = sliding_root / window.directory_name / "fit_result.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed M4 rolling result: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            _record_winner(window, summary, winner_parameters)
        succeeded_windows = [
            window for window in windows if window.window_id in winner_parameters
        ]
        if not succeeded_windows:
            raise RuntimeError(
                "No M4 rolling window produced an eligible winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)

    if args.workflow == FULL_SESSION_MATCHED_WORKFLOW:
        initializations = match_window_winners_to_test_sessions(
            prepared.assignments, succeeded_windows, winner_parameters, M4_PARAMETER_NAMES
        )
        initialization_filename = "matched_window_test_initializations.csv"
        block_audit = prepared.final_two_block_audit()
    else:
        initializations = aggregate_m4_session_initializations(
            prepared.assignments, succeeded_windows, winner_parameters,
            run_length_center=registry.run_length_center,
            maximum_run_length=registry.maximum_represented_run_length,
        )
        initialization_filename = "session_window_mean_initializations.csv"
        block_audit = prepared.block_audit
    single_root.mkdir(parents=True, exist_ok=True)
    initializations.to_csv(single_root / initialization_filename, index=False)
    block_audit.to_csv(single_root / "all_session_block_cut_audit.csv", index=False)
    _fine_tune_all(prepared, registry, args, single_root, initializations)
    _generate_single_global_figures(initializations, single_root)
    print(
        f"experiment=complete sliding_results={sliding_root} single_session_results={single_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
