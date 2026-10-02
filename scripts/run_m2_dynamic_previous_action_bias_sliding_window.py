#!/usr/bin/env python3
"""Run the isolated M2_dynamic_previous_action_bias 9/3/1 sliding and per-session experiment.

Adapted from ``run_m1_known_switch_sliding_window.py`` (same generic
orchestration: chronological window construction, resume/staleness
detection, CPU-budgeted process-pool parallelism for both stages), with the
model-specific pieces rewritten for a scalar delta-rule prior driven by the
mouse's own actions (not category feedback, and not an oracle-switch
biased-subspace log-odds belief): six pooled parameters
``(w,s0,ell_short,ell_long,kappa,a)`` instead of five, no
``true_block``-derived predictive signal, no fixed category-rate
prototypes, no biased-subspace log-odds/in-Neutral/N-S-L diagnostics. This
is the sibling of ``run_m1_stimulus_kernel_sliding_window.py``, identical in
every respect except the history-state update source; see
``docs/models/M2_dynamic_previous_action_bias_v1.md``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
import json
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
from m2_dynamic_previous_action_bias_v1.session import m2_dynamic_previous_action_bias_session_trials_from_frame  # noqa: E402
from m2_dynamic_previous_action_bias_v1.analysis import (  # noqa: E402
    build_m2_dynamic_previous_action_bias_trial_predictions,
    save_m2_dynamic_previous_action_bias_analysis_outputs,
)
from m2_dynamic_previous_action_bias_v1.fit import (  # noqa: E402
    fit_m2_dynamic_previous_action_bias_all_initializations,
    fit_m2_dynamic_previous_action_bias_single_initialization,
)
from m2_dynamic_previous_action_bias_v1.initialization import M2DynamicPreviousActionBiasInitializationSpec  # noqa: E402
from m2_dynamic_previous_action_bias_v1.registry import load_m2_dynamic_previous_action_bias_model_registry  # noqa: E402
from m2_dynamic_previous_action_bias_sliding_window_v1 import (  # noqa: E402
    FULL_SESSION_MATCHED_WORKFLOW,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_identifiability_outputs,
    generate_m2_dynamic_previous_action_bias_experiment_outputs,
    generate_truncation_outputs,
    identifiability_outputs_are_current,
    match_window_winners_to_test_sessions,
    prediction_performance,
    prepare_sliding_data,
    truncation_outputs_are_current,
)


EXPERIMENT_ID = "M2_dynamic_previous_action_bias_full_session_matched_sliding_9train_3validation_1test_v1"
WORKFLOW = FULL_SESSION_MATCHED_WORKFLOW
WORKFLOW_OUTPUT_SUFFIX = "full_session_matched_v1"
CPU_BUDGET = 32
PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_fit_run_result(result, path: Path) -> None:
    _atomic_json(result.as_dict(), path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit M2_dynamic_previous_action_bias in chronological 9/3/1 windows with eight starts, "
            "then fine-tune each matched rolling-test session and score its literal "
            "final two blocks."
        )
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
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
    parser.add_argument(
        "--identifiability-grid-points",
        type=int,
        default=0,
        help="Fixed-other-eight (eta_logit, bh) grid points per axis; 0 disables, else must be >=3",
    )
    parser.add_argument(
        "--truncation-sensitivity",
        action="store_true",
        help="Also run the opt-in fixed-parameter Rmax 65/75/100 sensitivity check",
    )
    parser.add_argument(
        "--include-full-run-length-distributions",
        action="store_true",
        help="Include the 130 per-trial run-length marginal columns in saved predictions",
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
    args = parser.parse_args(argv)
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
        "rolling_workers": args.rolling_workers,
        "rolling_threads_per_worker": args.rolling_threads_per_worker,
        "window_parallel_workers": args.window_parallel_workers,
        "single_session_workers": args.single_session_workers,
        "single_session_threads_per_worker": args.single_session_threads_per_worker,
    }
    if any(value <= 0 for value in positive.values()):
        parser.error("All optimizer and worker controls must be positive")
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
    if args.identifiability_grid_points != 0 and args.identifiability_grid_points < 3:
        parser.error("--identifiability-grid-points must be 0 (disabled) or at least 3")
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
    """Return every CLI setting that can change a fitted checkpoint."""

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
    }


def _runner_sha256() -> str:
    """Hash the workflow orchestration code omitted from SOURCE_ROOT."""

    return sha256(Path(__file__).read_bytes()).hexdigest()


def _optional_outputs_current(output: Path, args: argparse.Namespace) -> bool:
    """Return whether the opt-in identifiability/truncation outputs already
    on disk match the currently requested flags.

    Folded into the resume freshness check so re-running with
    ``--identifiability-grid-points``/``--truncation-sensitivity`` newly
    enabled (or at a different grid size) triggers regeneration instead of
    a silent ``resume_skip`` that never produces the newly requested output.
    """

    return (
        identifiability_outputs_are_current(
            output / "m2_dynamic_previous_action_bias_identifiability",
            grid_points=int(args.identifiability_grid_points),
        )
        and truncation_outputs_are_current(
            output / "m2_dynamic_previous_action_bias_truncation_sensitivity",
            enabled=bool(args.truncation_sensitivity),
        )
    )


def _sessions(prepared, session_ids: tuple[str, ...], role: str):
    return tuple(
        m2_dynamic_previous_action_bias_session_trials_from_frame(
            prepared.frame(session_id, "full", role), split_role=role
        )
        for session_id in session_ids
    )


def _window_canonical(prepared, window) -> pd.DataFrame:
    frames = [
        prepared.frame(session_id, "full", "train")
        for session_id in window.training_session_ids
    ]
    frames.extend(
        prepared.frame(session_id, "full", "validation")
        for session_id in window.validation_session_ids
    )
    frames.append(prepared.frame(window.test_session_id, "full", "test"))
    return pd.concat(frames, ignore_index=True)


def _find_winner_run(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Return the winning start's run record, or None if the window has no winner.

    Unlike a raising lookup, this never stops the caller -- it hands back
    ``None`` so a window with no eligible start (e.g. every one of its eight
    starts converges but fails the separate tail-occupancy acceptance gate)
    can be handled gracefully instead of aborting every other rolling window
    still in flight. A genuine ``"winner_start_id": null`` (no eligible
    start; the key is present) is the only case returning ``None`` --
    an entirely absent key (a malformed or schema-incompatible summary) and
    a non-null ``winner_start_id`` that does not match any saved run are
    both treated as corrupted/incomplete output and still raise, so a real
    data problem is never silently folded into the same "no eligible start"
    bucket.
    """

    if "winner_start_id" not in summary:
        raise RuntimeError(
            "M2_dynamic_previous_action_bias summary is missing the required "
            "winner_start_id key entirely (malformed or schema-incompatible)"
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
            f"M2_dynamic_previous_action_bias summary names winner_start_id="
            f"{winner_id!r} but no run with that start_id is present"
        )
    return winner


def _warn_no_winner(window) -> None:
    """Report a window with no eligible winner without stopping the run."""

    message = (
        f"M2_dynamic_previous_action_bias window={window.window_id:03d} has no "
        "converged, tail-acceptable winner among its eight starts; excluding it "
        "from session initialization matching and continuing with the "
        "remaining rolling windows."
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
    """Evaluate one M2_dynamic_previous_action_bias session continuously, then restore
    train/test labels."""

    combined = pd.concat([prefix, heldout], ignore_index=True).sort_values(
        "trial_index"
    )
    if combined["session_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous M2_dynamic_previous_action_bias prediction requires exactly one session")
    timeline = combined.copy()
    timeline["split_role"] = "train"
    predictions = build_m2_dynamic_previous_action_bias_trial_predictions(
        timeline, raw_parameters, registry, allow_test=False
    )
    predictions["split_role"] = predictions["experiment_partition"].map(
        {"prefix": "train", "heldout": "test"}
    )
    if predictions["split_role"].isna().any():
        raise ValueError("Single-session M2_dynamic_previous_action_bias predictions contain an unknown partition")
    return predictions.sort_values("trial_index").reset_index(drop=True)


def _fit_window(prepared, window, registry, args, root: Path) -> dict[str, Any]:
    output = root / window.directory_name
    result_path = output / "fit_result.json"
    manifest_path = output / "window_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    figures_current = (
        figure_outputs_are_current(output / "m2_dynamic_previous_action_bias_figures")
        and _optional_outputs_current(output, args)
    )
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    input_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous_manifest, registry, SOURCE_ROOT)
    if previous_result is not None or previous_manifest:
        input_stale = input_stale or previous_manifest.get("source_sha256") != list(
            prepared.source_sha256
        )
        input_stale = input_stale or previous_manifest.get(
            "optimizer_configuration"
        ) != _optimizer_provenance(args)
        input_stale = input_stale or previous_manifest.get(
            "runner_sha256"
        ) != _runner_sha256()
    if input_stale:
        print(
            f"window={window.window_id:03d} status=resume_stale_inputs "
            "(data, optimizer, registry, split, or source code changed "
            "since this fit was saved -- refitting instead of resuming)",
            flush=True,
        )
    if (
        args.resume
        and previous_result is not None
        and previous_manifest.get("status") == "complete"
        and figures_current
        and not input_stale
    ):
        print(f"window={window.window_id:03d} status=resume_skip", flush=True)
        return previous_result

    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    resumed_postfit = args.resume and previous_result is not None and not input_stale
    if resumed_postfit:
        print(f"window={window.window_id:03d} status=resume_postfit", flush=True)
        summary = previous_result
    else:
        train = _sessions(prepared, window.training_session_ids, "train")
        validation = _sessions(
            prepared, window.validation_session_ids, "validation"
        )
        print(
            f"window={window.window_id:03d} "
            f"train={window.training_ranks[0]:03d}-{window.training_ranks[-1]:03d} "
            f"validation={window.validation_ranks[0]:03d}-{window.validation_ranks[-1]:03d} "
            f"test={window.test_rank:03d} starts={len(registry.initializations)} "
            f"cpu={min(args.rolling_workers, 8)}x{args.rolling_threads_per_worker}",
            flush=True,
        )
        fit = fit_m2_dynamic_previous_action_bias_all_initializations(
            train,
            validation,
            registry,
            output_directory=output / "starts",
            max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            optimizer_strategy=args.optimizer,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs(args),
            parallel_workers=args.rolling_workers,
            cpu_threads_per_worker=args.rolling_threads_per_worker,
        )
        summary = fit.as_dict()
    if len(summary.get("runs", [])) != 8:
        raise RuntimeError("Every rolling M2_dynamic_previous_action_bias window must preserve all eight starts")
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
            "workflow": WORKFLOW,
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
            "rolling_test_scope": "full_session",
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
    canonical = _window_canonical(prepared, window)
    predictions = build_m2_dynamic_previous_action_bias_trial_predictions(
        canonical, winner["best_raw_parameters"], registry, allow_test=True
    )
    save_m2_dynamic_previous_action_bias_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "split_and_session_metrics.csv", index=False
    )
    generate_m2_dynamic_previous_action_bias_experiment_outputs(
        figure_ready(predictions),
        summary,
        winner["best_raw_parameters"],
        registry,
        output / "m2_dynamic_previous_action_bias_figures",
        output / "m2_dynamic_previous_action_bias_figures",
        include_full_run_length_distributions=args.include_full_run_length_distributions,
    )
    if args.identifiability_grid_points > 0 or args.truncation_sensitivity:
        # Built fresh here (not reused from the resumed_postfit/else branch above,
        # where `train` only exists on the refit path) and restricted to the single
        # "train" split role, matching identifiability.py/diagnostics.py's own
        # single-non-test-split requirement.
        train_sessions = _sessions(prepared, window.training_session_ids, "train")
    if args.identifiability_grid_points > 0:
        generate_identifiability_outputs(
            train_sessions, winner["best_raw_parameters"], registry,
            output / "m2_dynamic_previous_action_bias_identifiability",
            grid_points=args.identifiability_grid_points,
        )
    if args.truncation_sensitivity:
        generate_truncation_outputs(
            train_sessions, winner["best_raw_parameters"], registry,
            output / "m2_dynamic_previous_action_bias_truncation_sensitivity",
        )
    relevant = set(
        (*window.training_session_ids, *window.validation_session_ids, window.test_session_id)
    )
    rolling_audit = prepared.full_session_block_audit()
    rolling_audit.loc[
        rolling_audit["session_id"].astype(str).isin(relevant)
    ].to_csv(output / "block_cut_audit.csv", index=False)
    best = winner["best_constrained_parameters"]
    fitted_alpha = 1.0  # fixed registry constant, not a fitted coordinate in this model
    fitted_eta = float(torch.sigmoid(torch.tensor(best[8], dtype=torch.float64)))
    latest_elapsed = time.monotonic() - started  # recomputed after post-processing IO, not reused from above
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": WORKFLOW,
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
        "winner_fitted_eta": fitted_eta,
        "prior_update_rule": "pi_next = pi + alpha * (choice_long - pi); alpha is the fixed registry constant 1.0, not a fitted coordinate; pi_1 = 0.5; update gated on choice validity, not feedback_valid",
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t) -> lapse_mixture(m_t)",
        "rolling_test_scope": "full_session",
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
    """Fit every requested M2_dynamic_previous_action_bias window, running up to
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
    source_values = [
        float(initialization_row[f"initial_{name}"])
        for name in PARAMETER_NAMES
    ]
    result_path = output / "fine_tune_fit_result.json"
    manifest_path = output / "single_session_manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    figures_current = figure_outputs_are_current(output / "m2_dynamic_previous_action_bias_figures")
    previous_result = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    )
    session_stale = result_registry_is_stale(
        previous_result, registry
    ) or manifest_fingerprint_is_stale(previous_manifest, registry, SOURCE_ROOT)
    if previous_result is not None or previous_manifest:
        session_stale = session_stale or previous_manifest.get(
            "source_sha256"
        ) != list(payload["source_sha256"])
        session_stale = session_stale or previous_manifest.get(
            "optimizer_configuration"
        ) != args
        session_stale = session_stale or previous_manifest.get(
            "source_window_constrained_parameters"
        ) != source_values
        session_stale = session_stale or previous_manifest.get(
            "runner_sha256"
        ) != _runner_sha256()
    if (
        payload["resume"]
        and previous_result is not None
        and previous_manifest.get("status") == "complete"
        and figures_current
        and not session_stale
    ):
        return {"session_id": payload["session_id"], "status": "resume_skip"}

    output.mkdir(parents=True, exist_ok=True)
    values, adjusted_names = clamp_fine_tune_initialization(
        source_values,
        PARAMETER_NAMES,
        parameter_epsilon=registry.parameter_epsilon,
        lapse_max=registry.lapse_max,
    )
    initialization = M2DynamicPreviousActionBiasInitializationSpec(
        start_id=1, source_m2_dynamic_start_id=1,
        w=values[0], s0=values[1], ell_short=values[2], ell_long=values[3],
        kappa=values[4], ah=values[5], delta0=values[6], bh=values[7],
        eta_logit=values[8],
        purpose="sliding-window initialization for M2_dynamic_previous_action_bias session fine-tuning",
    )
    experimental_registry = replace(registry, initializations=(initialization,))
    started = time.monotonic()
    resumed_postfit = (
        payload["resume"]
        and previous_result is not None
        and not session_stale
        and previous_manifest.get("status") != "failed"
    )
    if resumed_postfit:
        run_record = previous_result
    else:
        train_session = m2_dynamic_previous_action_bias_session_trials_from_frame(prefix, split_role="train")
        monitor_frame = prefix.copy()
        monitor_frame["split_role"] = "validation"
        monitor_session = m2_dynamic_previous_action_bias_session_trials_from_frame(
            monitor_frame, split_role="validation"
        )
        result = fit_m2_dynamic_previous_action_bias_single_initialization(
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
            "workflow": WORKFLOW,
            "status": "failed",
            "source_sha256": list(payload["source_sha256"]),
            "optimizer_configuration": args,
            "source_window_constrained_parameters": source_values,
            "runner_sha256": _runner_sha256(),
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
        "result_schema_version": "m2_dynamic_previous_action_bias_single_session_fine_tune_v1",
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
    save_m2_dynamic_previous_action_bias_analysis_outputs(predictions, summary, output, test_unlocked=True)
    prediction_performance(predictions).to_csv(
        output / "prefix_and_heldout_metrics.csv", index=False
    )
    generate_m2_dynamic_previous_action_bias_experiment_outputs(
        figure_ready(predictions),
        summary,
        run_record["best_raw_parameters"],
        registry,
        output / "m2_dynamic_previous_action_bias_figures",
        output / "m2_dynamic_previous_action_bias_figures",
        include_full_run_length_distributions=payload["include_full_run_length_distributions"],
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
    fitted_alpha = 1.0  # fixed registry constant, not a fitted coordinate in this model
    fitted_eta = float(torch.sigmoid(torch.tensor(best[8], dtype=torch.float64)))
    latest_elapsed = time.monotonic() - started
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "workflow": WORKFLOW,
        "status": "complete",
        "source_sha256": list(payload["source_sha256"]),
        "optimizer_configuration": args,
        "runner_sha256": _runner_sha256(),
        "resume_fingerprint": manifest_fingerprint_for_write(
            resumed_postfit, previous_manifest, registry, SOURCE_ROOT
        ),
        "model_id": registry.model_id,
        "mouse_id": payload["mouse_id"],
        "session_id": payload["session_id"],
        "chronological_rank": payload["rank"],
        "fine_tune_scope": "all blocks before the literal final two blocks",
        "heldout_scope": (
            "literal final two chronological blocks, including the terminal block"
        ),
        "heldout_state_policy": (
            "continuous causal pi_t from prefix; each heldout trial's own observed "
            "mouse action updates only subsequent heldout trials, never earlier ones "
            "and never the fitted parameters"
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
        "fitted_eta": fitted_eta,
        "prior_update_rule": "pi_next = pi + alpha * (choice_long - pi); alpha is the fixed registry constant 1.0, not a fitted coordinate; pi_1 = 0.5; update gated on choice validity, not feedback_valid",
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t) -> lapse_mixture(m_t)",
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


def _build_single_session_jobs(
    prepared,
    registry,
    args,
    single_root: Path,
    initializations: pd.DataFrame,
) -> list[dict[str, Any]]:
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
                "prefix": prepared.frame(session_id, "final_two_prefix", "train"),
                "heldout": prepared.frame(session_id, "final_two_heldout", "test"),
                "initialization": by_session.loc[session_id].to_dict(),
                "output": str(directory),
                "optimizer_args": optimizer_args,
                "threads_per_worker": args.single_session_threads_per_worker,
                "resume": args.resume,
                "source_sha256": list(prepared.source_sha256),
                "include_full_run_length_distributions": args.include_full_run_length_distributions,
            }
        )
    return jobs


def _fine_tune_all(
    prepared,
    registry,
    args,
    single_root: Path,
    initializations: pd.DataFrame,
) -> None:
    jobs = _build_single_session_jobs(
        prepared, registry, args, single_root, initializations
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
            "workflow": WORKFLOW,
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
    excluded: list[str] = []
    for assignment in initializations.sort_values(
        "chronological_rank"
    ).itertuples(index=False):
        rank = int(assignment.chronological_rank)
        session_id = str(assignment.session_id)
        directory_name = f"session_{rank:03d}_{session_id}"
        manifest_path = root / directory_name / "single_session_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        if manifest.get("status") != "complete":
            excluded.append(session_id)
            continue
        sources.append(
            (directory_name, root / directory_name / "trial_predictions.csv")
        )
    if excluded:
        print(
            "single_session_global_figures_excluded_failed_or_incomplete="
            + ",".join(excluded),
            flush=True,
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


def _result_roots(
    args: argparse.Namespace, model_id: str, mouse_id: str
) -> tuple[Path, Path]:
    """Return isolated roots for the sole supported matched workflow."""

    sliding_root = (
        args.sliding_output_dir.expanduser().resolve()
        / model_id
        / mouse_id
        / WORKFLOW_OUTPUT_SUFFIX
    )
    single_root = (
        args.single_output_dir.expanduser().resolve()
        / model_id
        / mouse_id
        / WORKFLOW_OUTPUT_SUFFIX
    )
    return sliding_root, single_root


def main() -> int:
    args = _parse_args()
    registry = load_m2_dynamic_previous_action_bias_model_registry(args.registry)
    if len(registry.initializations) != 8:
        raise RuntimeError("Sliding M2_dynamic_previous_action_bias requires exactly eight registered initializations")
    prepared = prepare_sliding_data(args.raw_data_root, registry)
    windows = build_sliding_windows(prepared.assignments)
    sliding_root, single_root = _result_roots(
        args, registry.model_id, prepared.mouse_id
    )
    sliding_root.mkdir(parents=True, exist_ok=True)
    prepared.assignments.to_csv(
        sliding_root / "chronological_sessions.csv", index=False
    )
    prepared.full_session_block_audit().to_csv(
        sliding_root / "all_session_block_cut_audit.csv", index=False
    )
    _atomic_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "workflow": WORKFLOW,
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
            "prior_update_rule": "pi_next = pi + alpha * (choice_long - pi); alpha is the fixed registry constant 1.0, not a fitted coordinate; pi_1 = 0.5; update gated on choice validity, not feedback_valid",
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t) -> lapse_mixture(m_t)",
            "hidden_state": "joint_3x65_block_run_length_belief_unchanged_from_M2_dynamic_block_length_v1_plus_an_independent_scalar_action_history_state",
            "session_state_reset": "pi resets to 0.5 independently per session",
            "terminal_block_policy": "included_in_rolling_and_final_two_fine_tune_test",
            "heldout_policy": "matched_test_sessions_only_literal_final_two_blocks",
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
                "No M2_dynamic_previous_action_bias rolling window produced an "
                "eligible winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)
        if args.window_limit is not None:
            print(
                "Sliding M2_dynamic_previous_action_bias development subset complete; session stage skipped.",
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
                raise FileNotFoundError(f"Missing completed M2_dynamic_previous_action_bias rolling result: {path}")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing M2_dynamic_previous_action_bias rolling manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("workflow") != WORKFLOW:
                raise RuntimeError(
                    f"Rolling result is not a {WORKFLOW} result: {manifest_path}"
                )
            if manifest.get("status") not in ("complete", "no_winner"):
                raise RuntimeError(
                    f"Rolling result is not a completed or no-winner {WORKFLOW} result: {manifest_path}"
                )
            # Every provenance/staleness check below applies identically to a
            # "complete" and a "no_winner" manifest -- a no-winner exclusion
            # recorded under stale data/optimizer/code settings must force a
            # refit exactly like a complete result would, not be trusted as-is.
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
                    f"Rolling result used different optimizer settings: {manifest_path}"
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
            # The skip/use decision is derived from the freshly loaded summary
            # itself, not from manifest.get("status"): the manifest and
            # fit_result.json are written as two separate files, so an
            # interrupted refit could in principle leave a stale "no_winner"
            # manifest beside a summary that now has a real winner (or vice
            # versa). Re-deriving from summary means that possible desync
            # can never cause a real winner to be silently dropped or a
            # missing one to be silently accepted.
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
                "No M2_dynamic_previous_action_bias rolling window produced an "
                "eligible winner; cannot proceed"
            )
        _generate_sliding_global_figures(sliding_root, succeeded_windows)

    initializations = match_window_winners_to_test_sessions(
        prepared.assignments, succeeded_windows, winner_parameters, PARAMETER_NAMES
    )
    single_root.mkdir(parents=True, exist_ok=True)
    initializations.to_csv(
        single_root / "matched_window_test_initializations.csv", index=False
    )
    prepared.final_two_block_audit().to_csv(
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
