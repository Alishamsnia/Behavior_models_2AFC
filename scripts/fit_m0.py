#!/usr/bin/env python3
"""Run leak-safe M0 fitting with live train/validation NLL reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    EpochRecord,
    FitRunResult,
    LBFGSSettings,
    evaluate_split_metrics,
    fit_all_initializations,
    fit_single_initialization_with_strategy,
    save_evaluation_metrics,
    save_fit_run_result,
)
from m0_sensory_v1.analysis import (  # noqa: E402
    build_trial_predictions,
    generate_preliminary_figures,
    save_model_analysis_outputs,
)
from m0_sensory_v1.pipeline import (  # noqa: E402
    prepare_registered_fit_data,
    save_fit_input_audit,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def parse_args(*, description: str | None = None) -> argparse.Namespace:
    """Parse full-run or bounded-smoke command-line arguments."""

    parser = argparse.ArgumentParser(
        description=description or (
            "Fit M0 to frozen train/validation sessions. Locked-test MATLAB "
            "contents are never opened."
        )
    )
    parser.add_argument(
        "--raw-data-root",
        required=True,
        type=Path,
        help="Directory containing the registered raw session .mat files",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "lbfgs", "hybrid"),
        default="adam",
        help="Optimizer strategy; default 'adam' preserves the registered procedure",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print fresh train/validation NLL every N recorded outer steps",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only start 1 with bounded phase lengths",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=2,
        help="Epoch count used only with --smoke-test",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "Override the registry Adam maximum for a full Adam run; the "
            "override is recorded in every result"
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Override Adam's initial learning rate; omit to use the registered "
            "value (0.01)"
        ),
    )
    parser.add_argument(
        "--adam-warmup-epochs",
        type=int,
        default=1000,
        help="Adam epochs before L-BFGS; used only with --optimizer hybrid",
    )
    parser.add_argument(
        "--lbfgs-max-steps",
        type=int,
        default=250,
        help="Maximum recorded L-BFGS outer steps",
    )
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument(
        "--lbfgs-max-evaluations",
        type=int,
        default=20,
        help="Maximum strong-Wolfe closure evaluations per outer step",
    )
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Maximum concurrent start processes for a full run (default: 8)",
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=1,
        help="Torch intra-op CPU threads assigned to each start process (default: 1)",
    )
    args = parser.parse_args()
    if args.print_every <= 0:
        parser.error("--print-every must be positive")
    if args.smoke_epochs <= 0:
        parser.error("--smoke-epochs must be positive")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if args.smoke_test and args.max_epochs is not None:
        parser.error(
            "--max-epochs is for full runs; use --smoke-epochs for smoke tests"
        )
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.adam_warmup_epochs <= 0:
        parser.error("--adam-warmup-epochs must be positive")
    if args.lbfgs_max_steps <= 0:
        parser.error("--lbfgs-max-steps must be positive")
    if args.lbfgs_learning_rate <= 0.0:
        parser.error("--lbfgs-learning-rate must be positive")
    if args.lbfgs_history_size <= 0:
        parser.error("--lbfgs-history-size must be positive")
    if args.lbfgs_max_evaluations <= 0:
        parser.error("--lbfgs-max-evaluations must be positive")
    if args.lbfgs_tolerance_grad <= 0.0 or args.lbfgs_tolerance_change <= 0.0:
        parser.error("L-BFGS tolerances must be positive")
    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be positive")
    if args.cpu_threads_per_worker <= 0:
        parser.error("--cpu-threads-per-worker must be positive")
    if args.optimizer != "adam" and args.max_epochs is not None:
        parser.error("--max-epochs applies only to --optimizer adam")
    if args.optimizer == "lbfgs" and args.learning_rate is not None:
        parser.error(
            "--learning-rate configures Adam; use --lbfgs-learning-rate for L-BFGS"
        )
    return args


def _lbfgs_settings(args: argparse.Namespace, *, smoke_test: bool) -> LBFGSSettings:
    """Build validated L-BFGS settings from explicit command-line controls."""

    return LBFGSSettings(
        maximum_steps=(
            min(args.smoke_epochs, args.lbfgs_max_steps)
            if smoke_test
            else args.lbfgs_max_steps
        ),
        learning_rate=args.lbfgs_learning_rate,
        history_size=args.lbfgs_history_size,
        maximum_evaluations_per_step=args.lbfgs_max_evaluations,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
    )


def _validate_convergence_budget(args: argparse.Namespace, registry) -> None:
    """Reject full alternative runs that cannot fill the stability window."""

    if args.smoke_test or args.optimizer == "adam":
        return
    planned_steps = args.lbfgs_max_steps + (
        args.adam_warmup_epochs if args.optimizer == "hybrid" else 0
    )
    required_steps = registry.convergence.stable_training_window_epochs
    if planned_steps < required_steps:
        raise ValueError(
            "Optimizer step budget cannot fill the registered stability window: "
            f"planned={planned_steps}, required={required_steps}. Increase "
            "--lbfgs-max-steps or, for hybrid, --adam-warmup-epochs."
        )


def _print_initial(start_id: int, train_nll: float, validation_nll: float) -> None:
    """Print the epoch-zero fresh objectives with immediate flushing."""

    print(
        f"start={start_id:02d} epoch=0000 "
        f"train_mean_nll={train_nll:.12f} "
        f"validation_mean_nll={validation_nll:.12f} "
        "stage=initial",
        flush=True,
    )


def _print_runtime_settings(args: argparse.Namespace, registry) -> None:
    """Print the effective optimizer controls once before fitting begins."""

    optimizer_strategy = getattr(args, "optimizer", "adam")
    if optimizer_strategy != "adam":
        adam_warmup = (
            min(args.smoke_epochs, args.adam_warmup_epochs)
            if args.smoke_test
            else args.adam_warmup_epochs
        )
        lbfgs_steps = (
            min(args.smoke_epochs, args.lbfgs_max_steps)
            if args.smoke_test
            else args.lbfgs_max_steps
        )
        print(
            f"optimizer_strategy={optimizer_strategy} "
            f"adam_warmup_epochs={adam_warmup if optimizer_strategy == 'hybrid' else 0} "
            f"adam_initial_learning_rate={registry.optimizer.initial_learning_rate if args.learning_rate is None else args.learning_rate:.12g} "
            f"lbfgs_max_steps={lbfgs_steps} "
            f"lbfgs_learning_rate={args.lbfgs_learning_rate:.12g} "
            f"lbfgs_history_size={args.lbfgs_history_size} "
            f"lbfgs_max_evaluations={args.lbfgs_max_evaluations} "
            f"lbfgs_tolerance_grad={args.lbfgs_tolerance_grad:.6e} "
            f"lbfgs_tolerance_change={args.lbfgs_tolerance_change:.6e}",
            flush=True,
        )
        return
    effective_epochs = (
        args.smoke_epochs
        if args.smoke_test
        else (
            registry.optimizer.maximum_epochs
            if args.max_epochs is None
            else args.max_epochs
        )
    )
    effective_learning_rate = (
        registry.optimizer.initial_learning_rate
        if args.learning_rate is None
        else args.learning_rate
    )
    print(
        f"effective_max_epochs={effective_epochs} "
        f"effective_initial_learning_rate={effective_learning_rate:.12g} "
        f"epoch_override={int(args.max_epochs is not None)} "
        f"learning_rate_override={int(args.learning_rate is not None)}",
        flush=True,
    )


def _print_epoch(start_id: int, record: EpochRecord, print_every: int) -> None:
    """Print one live post-step fresh train/validation objective record."""

    if record.epoch % print_every != 0:
        return
    phase = getattr(record, "optimizer_phase", None)
    phase_text = (
        ""
        if phase is None
        else f"phase={phase} phase_step={getattr(record, 'phase_step')} "
    )
    print(
        f"start={start_id:02d} epoch={record.epoch:04d} {phase_text}"
        f"train_mean_nll={record.fresh_training_mean_nll:.12f} "
        f"validation_mean_nll={record.validation_mean_nll:.12f} "
        f"grad_norm={record.gradient_norm_before_step:.6e} "
        f"lr={record.learning_rate_after_scheduler:.6e} "
        f"checkpoint={int(record.training_checkpoint_improved)}",
        flush=True,
    )


def _print_completed_start(result: FitRunResult) -> None:
    """Report one safely serialized start after a worker returns it."""

    print(
        f"completed_start={result.start_id:02d} "
        f"training_mean_nll={result.recalculated_best_training_mean_nll:.12f} "
        f"validation_mean_nll={result.final_validation_mean_nll:.12f} "
        f"converged={result.converged} "
        f"termination={result.termination_reason}",
        flush=True,
    )


def main() -> int:
    """Load only non-test data, fit, serialize results, and score validation."""

    args = parse_args()
    registry = load_model_registry(args.registry)
    try:
        _validate_convergence_budget(args, registry)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr, flush=True)
        return 2
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(
        "Preparing frozen train/validation sessions; locked test remains unopened.",
        flush=True,
    )
    data = prepare_registered_fit_data(args.raw_data_root, registry)
    save_fit_input_audit(data.audit, output / "fit_input_audit.json")
    print(
        f"loaded_train_sessions={len(data.training_sessions)} "
        f"loaded_validation_sessions={len(data.validation_sessions)} "
        f"train_valid_decisions={data.audit.training_valid_decisions} "
        f"validation_valid_decisions={data.audit.validation_valid_decisions} "
        f"locked_test_files_opened={data.audit.locked_test_files_opened}",
        flush=True,
    )
    _print_runtime_settings(args, registry)

    if args.smoke_test:
        initialization = registry.initializations[0]
        result = fit_single_initialization_with_strategy(
            data.training_sessions,
            data.validation_sessions,
            registry,
            initialization,
            optimizer_strategy=args.optimizer,
            max_epochs_override=(
                args.smoke_epochs if args.optimizer == "adam" else None
            ),
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=min(args.smoke_epochs, args.adam_warmup_epochs),
            lbfgs_settings=_lbfgs_settings(args, smoke_test=True),
            allow_short_convergence_budget=True,
            initial_callback=lambda train, validation: _print_initial(
                initialization.start_id,
                train,
                validation,
            ),
            epoch_callback=lambda record: _print_epoch(
                initialization.start_id,
                record,
                args.print_every,
            ),
        )
        save_fit_run_result(result, output / "smoke_start_01.json")
        metrics = evaluate_split_metrics(
            data.validation_sessions,
            torch.tensor(result.best_raw_parameters, dtype=torch.float64),
            registry,
            required_split_role="validation",
            block_labels_by_session=data.validation_block_labels,
        )
        save_evaluation_metrics(metrics, output / "smoke_validation_metrics.json")
        smoke_summary = {
            "status": "passed",
            "purpose": "pipeline_smoke_only_not_a_registered_fit_or_winner",
            "epochs": args.smoke_epochs,
            "optimizer_strategy": args.optimizer,
            "start_id": initialization.start_id,
            "training_mean_nll": result.recalculated_best_training_mean_nll,
            "validation_mean_nll": metrics.mean_nll_per_valid_trial,
            "validation_elpd": metrics.elpd,
            "locked_test_files_opened": data.audit.locked_test_files_opened,
        }
        (output / "smoke_summary.json").write_text(
            json.dumps(smoke_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(smoke_summary, indent=2), flush=True)
        return 0

    print(
        f"Multistart execution: {len(registry.initializations)} independent starts, "
        f"up to {args.parallel_workers} CPU worker processes, "
        f"{args.cpu_threads_per_worker} Torch thread(s) per worker.",
        flush=True,
    )
    result = fit_all_initializations(
        data.training_sessions,
        data.validation_sessions,
        registry,
        output_directory=output / "starts",
        max_epochs_override=args.max_epochs,
        initial_learning_rate_override=args.learning_rate,
        optimizer_strategy=args.optimizer,
        adam_warmup_epochs=args.adam_warmup_epochs,
        lbfgs_settings=_lbfgs_settings(args, smoke_test=False),
        parallel_workers=args.parallel_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        initial_callback=_print_initial,
        epoch_callback=lambda start_id, record: _print_epoch(
            start_id,
            record,
            args.print_every,
        ),
        completion_callback=_print_completed_start,
    )
    if result.winner_start_id is None:
        print(
            "No converged run passed every frozen check; validation metrics were not "
            "used to choose a fallback.",
            flush=True,
        )
        return 2

    winner = next(run for run in result.runs if run.start_id == result.winner_start_id)
    validation_metrics = evaluate_split_metrics(
        data.validation_sessions,
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        registry,
        required_split_role="validation",
        block_labels_by_session=data.validation_block_labels,
    )
    save_evaluation_metrics(validation_metrics, output / "validation_metrics.json")
    predictions = build_trial_predictions(
        data.canonical_trials,
        winner.best_raw_parameters,
        registry,
    )
    save_model_analysis_outputs(
        predictions,
        result.as_dict(),
        output,
        test_unlocked=False,
    )
    generate_preliminary_figures(
        predictions,
        result.as_dict(),
        output / "preliminary_figures",
        test_unlocked=False,
    )
    print(
        f"winner_start={winner.start_id:02d} "
        f"training_mean_nll={winner.recalculated_best_training_mean_nll:.12f} "
        f"validation_mean_nll={validation_metrics.mean_nll_per_valid_trial:.12f} "
        f"validation_elpd={validation_metrics.elpd:.12f}",
        flush=True,
    )
    print(
        f"trial_predictions={output / 'trial_predictions.csv'} "
        f"figures={output / 'preliminary_figures'} test_status=LOCKED",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
