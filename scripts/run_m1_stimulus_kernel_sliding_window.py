#!/usr/bin/env python3
"""Run the isolated M1_stimulus_kernel 9/3/1 sliding and per-session experiment.

Adapted from ``run_m1_known_switch_sliding_window.py`` (same generic
orchestration: chronological window construction, resume/staleness
detection, CPU-budgeted process-pool parallelism for both stages), with the
model-specific pieces rewritten for a scalar delta-rule prior instead of an
oracle-switch biased-subspace log-odds belief: six pooled parameters
``(w,s0,ell_short,ell_long,kappa,a)`` instead of five, no
``true_block``-derived predictive signal, no fixed category-rate
prototypes, no biased-subspace log-odds/in-Neutral/N-S-L diagnostics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
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
    manifest_fingerprint_for_write,
    manifest_fingerprint_is_stale,
    result_registry_is_stale,
)
from m0_sensory_v1.model import LBFGSSettings  # noqa: E402
from m0_sensory_v1.model.m1_session import m1_session_trials_from_frame  # noqa: E402
from m1_stimulus_kernel_v1.analysis import (  # noqa: E402
    build_m1_stimulus_kernel_trial_predictions,
    save_m1_stimulus_kernel_analysis_outputs,
)
from m1_stimulus_kernel_v1.fit import (  # noqa: E402
    fit_m1_stimulus_kernel_all_initializations,
    fit_m1_stimulus_kernel_single_initialization,
)
from m1_stimulus_kernel_v1.initialization import M1StimulusKernelInitializationSpec  # noqa: E402
from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry  # noqa: E402
from m1_stimulus_kernel_sliding_window_v1 import (  # noqa: E402
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    WORKFLOW_CHOICES,
    aggregate_m1_stimulus_kernel_session_initializations,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_m1_stimulus_kernel_experiment_figures,
    match_window_winners_to_test_sessions,
    prediction_performance,
    prepare_sliding_data,
)


EXPERIMENT_ID = "M1_stimulus_kernel_sliding_9train_3validation_1test_two_complete_blocks_v1"
CPU_BUDGET = 32
PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "a")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_fit_run_result(result, path: Path) -> None:
    _atomic_json(result.as_dict(), path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit M1_stimulus_kernel in chronological 9/3/1 windows with eight starts, "
            "then fine-tune every session and score two held-out complete blocks."
        )
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--workflow", choices=WORKFLOW_CHOICES, default=LEGACY_PARTITIONED_WORKFLOW)
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
        "--lbfgs-escalated-additional-steps",
        type=int,
        default=250,
        help=(
            "If zero of eight starts converge at --lbfgs-max-steps, each start "
            "whose sole failing check is small_gradient_norm is warm-started "
            "from its own best checkpoint and re-optimized for up to this many "
            "further L-BFGS steps."
        ),
    )
    parser.add_argument(
        "--gradient-norm-tolerance-relaxation-factor",
        type=float,
        default=2.0,
        help=(
            "If extended steps still leave zero starts converged, the "
            "gradient-norm-tolerance check is re-evaluated (no further "
            "optimization) at registry_tolerance times this factor."
        ),
    )
    parser.add_argument(
        "--disable-convergence-escalation",
        action="store_true",
        help="Reproduce pre-escalation behavior: give up as soon as zero starts converge",
    )
    parser.add_argument(
        "--rolling-workers",
        type=int,
        default=8,
        help="Concurrent optimizer-start processes per rolling window",
    )
    parser.add_argument(
        "--rolling-threads-per-worker",
        type=int,
        default=4,
        help="Torch CPU threads per rolling optimizer process",
    )
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
    parser.add_argument(
        "--single-session-workers",
        type=int,
        default=32,
        help="Concurrent session fine-tuning processes",
    )
    parser.add_argument(
        "--single-session-threads-per-worker",
        type=int,
        default=1,
        help="Torch CPU threads per session fine-tuning process",
    )
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
        "lbfgs_escalated_additional_steps": args.lbfgs_escalated_additional_steps,
        "rolling_workers": args.rolling_workers,
        "rolling_threads_per_worker": args.rolling_threads_per_worker,
        "window_parallel_workers": args.window_parallel_workers,
        "single_session_workers": args.single_session_workers,
        "single_session_threads_per_worker": args.single_session_threads_per_worker,
    }
    if any(value <= 0 for value in positive.values()):
        parser.error("All optimizer and worker controls must be positive")
    if not math.isfinite(args.gradient_norm_tolerance_relaxation_factor) or (
        args.gradient_norm_tolerance_relaxation_factor < 1.0
    ):
        parser.error("--gradient-norm-tolerance-relaxation-factor must be a finite number >= 1.0")
    rolling_processes = min(args.rolling_workers, 8)
    if (
        args.window_parallel_workers * rolling_processes * args.rolling_threads_per_worker
        > CPU_BUDGET
    ):
        parser.error(
            "window-parallel-workers x rolling worker allocation exceeds the "
            "32-CPU budget; reduce --window-parallel-workers, --rolling-workers, "
            "or --rolling-threads-per-worker"
        )
    if (
        args.single_session_workers * args.single_session_threads_per_worker
        > CPU_BUDGET
    ):
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


def _optimizer_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Return every CLI setting that can change a fitted checkpoint,
    including the convergence-escalation controls -- a saved fit produced
    under one relaxation factor/step budget/enabled-state must not be
    silently resumed as if it reflected today's settings.
    """

    return {
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
        "convergence_escalation": {
            "enabled": not args.disable_convergence_escalation,
            "escalated_lbfgs_additional_steps": args.lbfgs_escalated_additional_steps,
            "gradient_norm_tolerance_relaxation_factor": (
                args.gradient_norm_tolerance_relaxation_factor
            ),
        },
    }


def _runner_sha256() -> str:
    """Hash the workflow orchestration code omitted from SOURCE_ROOT."""

    return sha256(Path(__file__).read_bytes()).hexdigest()


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


def _find_winner_run(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the winning start's run record, or None if the window has no winner.

    Unlike a raising lookup, this never stops the caller -- it hands back
    ``None`` so a window with no eligible start (all eight starts fail
    gradient-norm convergence, even after the escalation ladder) can be
    handled gracefully instead of aborting every other rolling window still
    in flight. A genuine ``"winner_start_id": null`` (no eligible start; the
    key is present) is the only case returning ``None`` -- an entirely
    absent key (a malformed or schema-incompatible summary) and a non-null
    ``winner_start_id`` that does not match any saved run are both treated
    as corrupted/incomplete output and still raise, so a real data problem
    is never silently folded into the same "no eligible start" bucket.
    """

    if "winner_start_id" not in summary:
        raise RuntimeError(
            "M1_stimulus_kernel summary is missing the required winner_start_id "
            "key entirely (malformed or schema-incompatible)"
        )
    winner_id = summary["winner_start_id"]
    if winner_id is None:
        return None
    winner = next(
        (run for run in summary.get("runs", []) if run.get("start_id") == winner_id),
        None,
    )
    if winner is None:
        raise RuntimeError(
            f"M1_stimulus_kernel summary names winner_start_id={winner_id!r} but "
            "no run with that start_id is present"
        )
    return winner


def _warn_no_winner(window) -> None:
    """Report a window with no eligible winner without stopping the run."""

    message = (
        f"M1_stimulus_kernel window={window.window_id:03d} has no converged "
        "winner among its eight starts (even after the convergence-escalation "
        "ladder); excluding it from session initialization and continuing "
        "with the remaining rolling windows."
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


def single_session_run_failed(run_record: dict[str, Any]) -> bool:
    """Return whether a single-session fine-tune run must not be treated as
    a usable checkpoint.

    Purpose
    -------
    Gate `_single_session_job` against generating predictions from either
    an outright numerical failure (best_raw_parameters all-NaN, e.g. the
    initial parameter encode itself raised) or a recorded exception during
    optimization/restore (e.g. the restored-checkpoint gradient
    recomputation failed) -- the latter can leave best_raw_parameters
    finite (the last good checkpoint before the failure) while still
    meaning something broke.

    Parameter interpretation
    ------------------------
    Merely not converging (no exception, finite parameters, insufficient
    epochs) is deliberately NOT gated here: a single-start fine-tune has no
    multistart competition to fall back to, so that case still produces
    predictions, matching every sibling model's behavior, with
    ``"converged": false`` left in the manifest for a downstream consumer
    to check.

    Related tests
    -------------
    ``test_single_session_run_failed_gates_on_exception_or_nonfinite``.
    """

    if run_record.get("exception_message") is not None:
        return True
    return not np.all(np.isfinite(run_record["best_raw_parameters"]))


def _continuous_single_session_predictions(
    prefix: pd.DataFrame,
    heldout: pd.DataFrame,
    raw_parameters,
    registry,
) -> pd.DataFrame:
    """Evaluate one M1_stimulus_kernel session continuously, then restore
    train/test labels."""

    combined = pd.concat([prefix, heldout], ignore_index=True).sort_values(
        "trial_index"
    )
    if combined["session_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous M1_stimulus_kernel prediction requires exactly one session")
    timeline = combined.copy()
    timeline["split_role"] = "train"
    predictions = build_m1_stimulus_kernel_trial_predictions(
        timeline, raw_parameters, registry, allow_test=False
    )
    predictions["split_role"] = predictions["experiment_partition"].map(
        {"prefix": "train", "heldout": "test"}
    )
    if predictions["split_role"].isna().any():
        raise ValueError("Single-session M1_stimulus_kernel predictions contain an unknown partition")
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
    figures_current = figure_outputs_are_current(output / "m1_stimulus_kernel_figures")
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    registry_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous_manifest, registry, SOURCE_ROOT)
    if previous_result is not None or previous_manifest:
        registry_stale = registry_stale or previous_manifest.get(
            "source_sha256"
        ) != list(prepared.source_sha256)
        registry_stale = registry_stale or previous_manifest.get(
            "optimizer_configuration"
        ) != _optimizer_provenance(args)
        registry_stale = registry_stale or previous_manifest.get(
            "runner_sha256"
        ) != _runner_sha256()
    if registry_stale:
        print(
            f"window={window.window_id:03d} status=resume_stale_registry "
            "(data, optimizer/escalation settings, registry, split, or "
            "source code changed since this fit was saved -- refitting "
            "instead of resuming)",
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
            f"test={window.test_rank:03d} starts={len(registry.initializations)} "
            f"cpu={min(args.rolling_workers, 8)}x{args.rolling_threads_per_worker}",
            flush=True,
        )
        fit = fit_m1_stimulus_kernel_all_initializations(
            train,
            validation,
            registry,
            output_directory=output / "starts",
            max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            optimizer_strategy=args.optimizer,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs(args),
            enable_convergence_escalation=not args.disable_convergence_escalation,
            escalated_lbfgs_additional_steps=args.lbfgs_escalated_additional_steps,
            gradient_norm_tolerance_relaxation_factor=args.gradient_norm_tolerance_relaxation_factor,
            parallel_workers=args.rolling_workers,
            cpu_threads_per_worker=args.rolling_threads_per_worker,
        )
        summary = fit.as_dict()
    if len(summary.get("runs", [])) != 8:
        raise RuntimeError("Every rolling M1_stimulus_kernel window must preserve all eight starts")
    _atomic_json(summary, result_path)
    winner = _find_winner_run(summary)
    latest_elapsed = time.monotonic() - started
    if winner is None:
        # Not warned here: _fit_window is only ever called from _run_windows,
        # whose _record_winner(window, summary, ...) call right after this
        # returns already looks up the winner and warns exactly once -- an
        # additional warning here would double-log every no-winner window
        # (and, in parallel mode, from two different processes).
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "workflow": args.workflow,
            "status": "no_winner",
            "resume_fingerprint": manifest_fingerprint_for_write(
                resumed_postfit, previous_manifest, registry, SOURCE_ROOT
            ),
            "model_id": registry.model_id,
            "mouse_id": prepared.mouse_id,
            **window.as_dict(),
            "registered_initialization_count": len(registry.initializations),
            "executed_start_ids": [run["start_id"] for run in summary["runs"]],
            "optimizer": args.optimizer,
            "optimizer_configuration": _optimizer_provenance(args),
            "source_sha256": list(prepared.source_sha256),
            "runner_sha256": _runner_sha256(),
            "winner_start_id": None,
            "selection_status": summary.get("selection_status"),
            "convergence_escalation_stage": summary.get("convergence_escalation_stage"),
            "convergence_escalation_detail": summary.get("convergence_escalation_detail"),
            "rolling_test_scope": (
                "full_session" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
                else "retained prefix only; final two complete blocks and terminal block excluded"
            ),
            "cpu_allocation": {
                "workers": min(args.rolling_workers, 8),
                "threads_per_worker": args.rolling_threads_per_worker,
                "maximum_threads": min(args.rolling_workers, 8)
                * args.rolling_threads_per_worker,
            },
            "elapsed_seconds": previous_manifest.get("elapsed_seconds", latest_elapsed),
            "latest_postfit_elapsed_seconds": latest_elapsed,
            "resumed_from_saved_fit": resumed_postfit,
        }
        _atomic_json(manifest, manifest_path)
        return summary
    canonical = _window_canonical(prepared, window, rolling_scope)
    predictions = build_m1_stimulus_kernel_trial_predictions(
        canonical, winner["best_raw_parameters"], registry, allow_test=True
    )
    save_m1_stimulus_kernel_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "split_and_session_metrics.csv", index=False
    )
    generate_m1_stimulus_kernel_experiment_figures(
        figure_ready(predictions),
        summary,
        output / "m1_stimulus_kernel_figures",
        output / "m1_stimulus_kernel_figures",
    )
    relevant = set(
        (*window.training_session_ids, *window.validation_session_ids, window.test_session_id)
    )
    rolling_audit = prepared.full_session_block_audit() if rolling_scope == "full" else prepared.block_audit
    rolling_audit.loc[
        rolling_audit["session_id"].astype(str).isin(relevant)
    ].to_csv(output / "block_cut_audit.csv", index=False)
    best = winner["best_constrained_parameters"]
    fitted_alpha = float(torch.sigmoid(torch.tensor(best[5], dtype=torch.float64)))
    latest_elapsed = time.monotonic() - started  # recomputed after post-processing IO, not reused from above
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": args.workflow,
        "status": "complete",
        "resume_fingerprint": manifest_fingerprint_for_write(
            resumed_postfit, previous_manifest, registry, SOURCE_ROOT
        ),
        "model_id": registry.model_id,
        "mouse_id": prepared.mouse_id,
        **window.as_dict(),
        "registered_initialization_count": len(registry.initializations),
        "executed_start_ids": [run["start_id"] for run in summary["runs"]],
        "optimizer": args.optimizer,
        "optimizer_configuration": _optimizer_provenance(args),
        "source_sha256": list(prepared.source_sha256),
        "runner_sha256": _runner_sha256(),
        "winner_start_id": summary["winner_start_id"],
        "winner_best_constrained_parameters": best,
        "winner_fitted_alpha": fitted_alpha,
        "convergence_escalation_stage": summary.get("convergence_escalation_stage"),
        "convergence_escalation_detail": summary.get("convergence_escalation_detail"),
        "prior_update_rule": "pi_next = pi + sigmoid(a) * (category - pi); pi_1 = 0.5",
        "rolling_test_scope": (
            "full_session" if args.workflow == FULL_SESSION_MATCHED_WORKFLOW
            else "retained prefix only; final two complete blocks and terminal block excluded"
        ),
        "cpu_allocation": {
            "workers": min(args.rolling_workers, 8),
            "threads_per_worker": args.rolling_threads_per_worker,
            "maximum_threads": min(args.rolling_workers, 8)
            * args.rolling_threads_per_worker,
        },
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
    """Fit every requested M1_stimulus_kernel window, running up to
    --window-parallel-workers at once.

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


def _single_session_job(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(payload["threads_per_worker"]))
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
    figures_current = figure_outputs_are_current(output / "m1_stimulus_kernel_figures")
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
    source_values = [
        float(initialization_row[f"initial_{name}"])
        for name in PARAMETER_NAMES
    ]
    values, adjusted_names = clamp_fine_tune_initialization(
        source_values,
        PARAMETER_NAMES,
        parameter_epsilon=registry.parameter_epsilon,
        lapse_max=registry.lapse_max,
    )
    initialization = M1StimulusKernelInitializationSpec(
        start_id=1,
        w=values[0],
        s0=values[1],
        ell_short=values[2],
        ell_long=values[3],
        kappa=values[4],
        a=values[5],
        purpose="sliding-window initialization for M1_stimulus_kernel session fine-tuning",
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
        result = fit_m1_stimulus_kernel_single_initialization(
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
    if single_session_run_failed(run_record):
        latest_elapsed = time.monotonic() - started
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "workflow": payload["workflow"],
            "status": "failed",
            "model_id": registry.model_id,
            "mouse_id": payload["mouse_id"],
            "session_id": payload["session_id"],
            "chronological_rank": payload["rank"],
            "termination_reason": run_record.get("termination_reason"),
            "exception_message": run_record.get("exception_message"),
            "resumed_from_saved_fit": resumed_postfit,
            "elapsed_seconds": latest_elapsed,
        }
        _atomic_json(manifest, manifest_path)
        return {
            "session_id": payload["session_id"],
            "status": "failed",
            "converged": False,
            "elapsed_seconds": latest_elapsed,
        }
    summary = {
        "result_schema_version": "m1_stimulus_kernel_single_session_fine_tune_v1",
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
    save_m1_stimulus_kernel_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "prefix_and_heldout_metrics.csv", index=False
    )
    generate_m1_stimulus_kernel_experiment_figures(
        figure_ready(predictions),
        summary,
        output / "m1_stimulus_kernel_figures",
        output / "m1_stimulus_kernel_figures",
    )
    provenance = dict(initialization_row)
    for key, value in list(provenance.items()):
        if pd.isna(value):
            provenance[key] = None
    provenance["fine_tune_initial_constrained_parameters"] = dict(
        zip(PARAMETER_NAMES, values, strict=True)
    )
    provenance["open_bound_adjusted_parameters"] = list(adjusted_names)
    _atomic_json(provenance, output / "initialization_provenance.json")
    best = run_record["best_constrained_parameters"]
    fitted_alpha = float(torch.sigmoid(torch.tensor(best[5], dtype=torch.float64)))
    latest_elapsed = time.monotonic() - started
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": payload["workflow"],
        "status": "complete",
        "resume_fingerprint": manifest_fingerprint_for_write(
            resumed_postfit, previous_manifest, registry, SOURCE_ROOT
        ),
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
            "continuous causal pi_t from prefix; heldout category feedback updates "
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
        "best_constrained_parameters": list(best),
        "fitted_alpha": fitted_alpha,
        "prior_update_rule": "pi_next = pi + sigmoid(a) * (category - pi); pi_1 = 0.5",
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
                "threads_per_worker": args.single_session_threads_per_worker,
                "resume": args.resume,
                "workflow": args.workflow,
            }
        )
    workers = min(
        args.single_session_workers,
        len(jobs),
        multiprocessing.cpu_count(),
    )
    print(
        f"single_session_jobs={len(jobs)} parallel_workers={workers} "
        f"threads_per_worker={args.single_session_threads_per_worker} "
        f"maximum_cpu_threads={workers * args.single_session_threads_per_worker}",
        flush=True,
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
            "cpu_allocation": {
                "workers": workers,
                "threads_per_worker": args.single_session_threads_per_worker,
                "maximum_threads": workers
                * args.single_session_threads_per_worker,
            },
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
    registry = load_m1_stimulus_kernel_model_registry(args.registry)
    if len(registry.initializations) != 8:
        raise RuntimeError("Sliding M1_stimulus_kernel requires exactly eight registered initializations")
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
            "parameter_order": list(PARAMETER_NAMES),
            "prior_update_rule": "pi_next = pi + sigmoid(a) * (category - pi); pi_1 = 0.5",
            "hidden_state": "none_pi_t_is_itself_the_fitted_scalar_belief",
            "session_state_reset": "pi resets to 0.5 independently per session",
            "terminal_block_policy": (
                "included_in_rolling_and_final_two_fine_tune_test"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "right_censored_and_ignored"
            ),
            "heldout_policy": (
                "matched_test_sessions_only_literal_final_two_blocks"
                if args.workflow == FULL_SESSION_MATCHED_WORKFLOW else "two_preceding_observed_complete_blocks"
            ),
            "cpu_budget": CPU_BUDGET,
            "rolling_cpu_allocation": {
                "workers": min(args.rolling_workers, 8),
                "threads_per_worker": args.rolling_threads_per_worker,
                "window_parallel_workers": args.window_parallel_workers,
            },
            "single_session_cpu_allocation": {
                "workers": args.single_session_workers,
                "threads_per_worker": args.single_session_threads_per_worker,
            },
            "source_files": list(prepared.source_files),
            "source_sha256": list(prepared.source_sha256),
        },
        sliding_root / "experiment_manifest.json",
    )
    selected_windows = windows[: args.window_limit] if args.window_limit else windows
    winner_parameters: dict[int, list[float]] = {}
    succeeded_windows: list = []
    if not args.single_only:
        winner_parameters = _run_windows(
            prepared, selected_windows, registry, args, sliding_root
        )
        succeeded_windows = [
            window for window in selected_windows if window.window_id in winner_parameters
        ]
        if not succeeded_windows:
            raise RuntimeError(
                "No M1_stimulus_kernel rolling window produced an eligible "
                "winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)
        if args.window_limit is not None:
            print(
                "Sliding M1_stimulus_kernel development subset complete; session stage skipped.",
                flush=True,
            )
            return 0
    if args.sliding_only:
        return 0
    if args.single_only:
        for window in windows:
            path = sliding_root / window.directory_name / "fit_result.json"
            manifest_path = sliding_root / window.directory_name / "window_manifest.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed M1_stimulus_kernel rolling result: {path}")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing M1_stimulus_kernel rolling manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("workflow") != args.workflow:
                raise RuntimeError(
                    f"Rolling result is not a {args.workflow} result: {manifest_path}"
                )
            if manifest.get("status") not in ("complete", "no_winner"):
                raise RuntimeError(
                    f"Rolling result is not a completed or no-winner {args.workflow} result: {manifest_path}"
                )
            # Every provenance/staleness check below applies identically to a
            # "complete" and a "no_winner" manifest -- a no-winner exclusion
            # recorded under stale data/optimizer/escalation/code settings
            # must force a refit exactly like a stale complete result would,
            # not be silently trusted and folded into a smaller dataset.
            fingerprint = manifest.get("resume_fingerprint")
            if not isinstance(fingerprint, dict) or not isinstance(
                fingerprint.get("source_code_sha256"), str
            ):
                raise RuntimeError(
                    f"Rolling result lacks a required source fingerprint: {manifest_path}"
                )
            if manifest.get("source_sha256") != list(prepared.source_sha256):
                raise RuntimeError(
                    f"Rolling result was fitted to different raw data: {manifest_path}"
                )
            if manifest.get("optimizer_configuration") != _optimizer_provenance(args):
                raise RuntimeError(
                    f"Rolling result used different optimizer/escalation settings: {manifest_path}"
                )
            if manifest.get("runner_sha256") != _runner_sha256():
                raise RuntimeError(
                    f"Rolling result used different runner code: {manifest_path}"
                )
            if manifest_fingerprint_is_stale(manifest, registry, SOURCE_ROOT):
                raise RuntimeError(
                    f"Rolling result is stale and must be refitted before --single-only: {manifest_path}"
                )
            summary = json.loads(path.read_text(encoding="utf-8"))
            if result_registry_is_stale(summary, registry):
                raise RuntimeError(
                    f"Rolling fit registry is stale and must be refitted: {path}"
                )
            winner = _find_winner_run(summary)
            if winner is None:
                print(
                    f"window={window.window_id:03d} status=no_winner (single-only recovery)",
                    flush=True,
                )
                continue
            winner_parameters[window.window_id] = list(winner["best_constrained_parameters"])
        succeeded_windows = [
            window for window in windows if window.window_id in winner_parameters
        ]
        if not succeeded_windows:
            raise RuntimeError(
                "No M1_stimulus_kernel rolling window produced an eligible "
                "winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)

    if args.workflow == FULL_SESSION_MATCHED_WORKFLOW:
        initializations = match_window_winners_to_test_sessions(
            prepared.assignments, succeeded_windows, winner_parameters,
            PARAMETER_NAMES,
        )
        initialization_filename = "matched_window_test_initializations.csv"
        block_audit = prepared.final_two_block_audit()
    else:
        initializations = aggregate_m1_stimulus_kernel_session_initializations(
            prepared.assignments, succeeded_windows, winner_parameters
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
