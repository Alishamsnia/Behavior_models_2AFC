#!/usr/bin/env python3
"""Fit the additive constant-step M3 model without opening locked-test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m0_sensory_v1.model.metrics import calculate_evaluation_metrics  # noqa: E402
from m0_sensory_v1.pipeline import save_fit_input_audit  # noqa: E402
from scripts.fit_m0 import (  # noqa: E402
    _lbfgs_settings,
    _print_initial,
    _print_runtime_settings,
    _validate_convergence_budget,
)
from m3_hmm_delta_v1.analysis import (  # noqa: E402
    build_m3_trial_predictions,
    save_m3_trial_predictions,
)
from m3_hmm_delta_v1.fit import (  # noqa: E402
    fit_m3_all_initializations,
    fit_m3_single_initialization,
)
from m3_hmm_delta_v1.figures import (  # noqa: E402
    generate_m3_diagnostic_figures,
    generate_m3_identifiability_figure,
    generate_m3_learning_figures,
)
from m3_hmm_delta_v1.identifiability import (  # noqa: E402
    evaluate_m3_delta_alpha_surface,
    evaluate_m3_dynamic_volume,
    evaluate_m3_hazard_alpha_surface,
    evaluate_m3_hazard_delta_surface,
)
from m3_hmm_delta_v1.m0_style_figures import generate_m3_m0_style_figures  # noqa: E402
from m3_hmm_delta_v1.pipeline import prepare_registered_m3_fit_data  # noqa: E402
from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402
from m3_hmm_delta_v1.session import evaluate_m3_mouse_objective  # noqa: E402


def _parse_m3_args() -> argparse.Namespace:
    """Parse M2-consistent fit controls plus M3 process-pool controls."""

    parser = argparse.ArgumentParser(description=(
        "Fit M3 with the same Adam, pure PyTorch L-BFGS, or hybrid optimizer "
        "strategies used by M2. Locked-test files remain unopened."
    ))
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--optimizer", choices=("adam", "lbfgs", "hybrid"), default="adam"
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
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
        help="Maximum concurrent M3 start processes (default: 8)",
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
        parser.error("--max-epochs is for full runs; use --smoke-epochs for smoke tests")
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
    if args.optimizer != "adam" and args.max_epochs is not None:
        parser.error("--max-epochs applies only to --optimizer adam")
    if args.optimizer == "lbfgs" and args.learning_rate is not None:
        parser.error("--learning-rate configures Adam; use --lbfgs-learning-rate for L-BFGS")
    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be positive")
    if args.cpu_threads_per_worker <= 0:
        parser.error("--cpu-threads-per-worker must be positive")
    return args


def _print_step(start_id, record, every):
    if record.step % every == 0:
        print(
            f"start={start_id:02d} step={record.step:04d} "
            f"phase={record.optimizer_phase} "
            f"train_mean_nll={record.fresh_training_mean_nll:.12f} "
            f"validation_mean_nll={record.validation_mean_nll:.12f} "
            f"grad_norm={record.gradient_norm:.6e} lr={record.learning_rate:.6e}",
            flush=True,
        )


def _print_completed_start(run):
    print(
        f"completed_start={run.start_id:02d} "
        f"training_mean_nll={run.recalculated_best_training_mean_nll:.12f} "
        f"validation_mean_nll={run.final_validation_mean_nll:.12f} "
        f"converged={run.converged} termination={run.termination_reason}",
        flush=True,
    )


def _metrics(sessions, raw, registry, role):
    objective = evaluate_m3_mouse_objective(
        sessions,
        torch.tensor(raw, dtype=torch.float64),
        registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role,
        lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        delta_min=registry.delta_min,
        first_long_probability=registry.first_long_probability,
    )
    return calculate_evaluation_metrics(
        objective.session_evaluations,
        model_id=registry.model_id,
        calibration_bins=registry.metrics.calibration_bins,
        psychometric_duration_decimals=registry.metrics.psychometric_duration_decimals,
        classification_threshold=registry.metrics.classification_threshold,
    )


def main() -> int:
    args = _parse_m3_args()
    registry = load_m3_model_registry(args.registry)
    try:
        _validate_convergence_budget(args, registry)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr, flush=True)
        return 2
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Preparing M3 train/validation timelines; locked test remains unopened.", flush=True)
    data = prepare_registered_m3_fit_data(args.raw_data_root, registry)
    save_fit_input_audit(data.audit, output / "fit_input_audit.json")
    print(
        f"model={registry.model_id} train_sessions={len(data.training_sessions)} "
        f"validation_sessions={len(data.validation_sessions)} "
        f"locked_test_files_opened={data.audit.locked_test_files_opened}",
        flush=True,
    )
    _print_runtime_settings(args, registry)
    if args.smoke_test:
        run = fit_m3_single_initialization(
            data.training_sessions,
            data.validation_sessions,
            registry,
            registry.initializations[0],
            optimizer_strategy=args.optimizer,
            max_epochs_override=args.smoke_epochs if args.optimizer == "adam" else None,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=min(args.smoke_epochs, args.adam_warmup_epochs),
            lbfgs_settings=_lbfgs_settings(args, smoke_test=True),
            allow_short_convergence_budget=True,
            initial_callback=lambda train, validation: _print_initial(1, train, validation),
            epoch_callback=lambda record: _print_step(1, record, args.print_every),
        )
        (output / "smoke_start_01.json").write_text(
            json.dumps(run.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        raw = run.best_raw_parameters
        fit_result = run.as_dict()
        prefix = "smoke_"
        winner_id = run.start_id
    else:
        print(
            f"M3 multistart execution: 8 independent starts, up to "
            f"{args.parallel_workers} CPU worker processes, "
            f"{args.cpu_threads_per_worker} Torch thread(s) per worker.",
            flush=True,
        )
        result = fit_m3_all_initializations(
            data.training_sessions,
            data.validation_sessions,
            registry,
            optimizer_strategy=args.optimizer,
            max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs_settings(args, smoke_test=False),
            parallel_workers=args.parallel_workers,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            initial_callback=lambda start_id, train, validation: _print_initial(
                start_id, train, validation
            ),
            epoch_callback=lambda start_id, record: _print_step(
                start_id, record, args.print_every
            ),
            completion_callback=_print_completed_start,
            output_directory=output / "starts",
        )
        if result.winner_start_id is None:
            print("No M3 run passed convergence and projection acceptance.", flush=True)
            return 2
        winner = next(item for item in result.runs if item.start_id == result.winner_start_id)
        raw = winner.best_raw_parameters
        fit_result = result.as_dict()
        prefix = ""
        winner_id = winner.start_id
    metrics = _metrics(data.validation_sessions, raw, registry, "validation")
    (output / f"{prefix}validation_metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    predictions = build_m3_trial_predictions(data.canonical_trials, raw, registry)
    save_m3_trial_predictions(predictions, output / f"{prefix}trial_predictions.csv")
    generate_m3_learning_figures(predictions, output / f"{prefix}figures")
    generate_m3_diagnostic_figures(predictions, output / f"{prefix}diagnostic_figures")
    (output / f"{prefix}fit_result.json").write_text(
        json.dumps(fit_result, indent=2) + "\n", encoding="utf-8"
    )
    if not args.smoke_test:
        generate_m3_m0_style_figures(
            predictions, fit_result, output / "m0_style_figures", test_unlocked=False
        )
        surface_functions = {
            "delta_alpha": evaluate_m3_delta_alpha_surface,
            "hazard_alpha": evaluate_m3_hazard_alpha_surface,
            "hazard_delta": evaluate_m3_hazard_delta_surface,
        }
        for stem, function in surface_functions.items():
            surface = function(
                data.training_sessions, torch.tensor(raw, dtype=torch.float64), registry
            )
            surface.to_csv(output / f"{stem}_identifiability_surface.csv", index=False)
            generate_m3_identifiability_figure(surface, output / "figures")
        volume = evaluate_m3_dynamic_volume(
            data.training_sessions, torch.tensor(raw, dtype=torch.float64), registry
        )
        volume.to_csv(output / "joint_dynamic_identifiability_volume.csv", index=False)
    summary = {
        "model_id": registry.model_id,
        "winner_start_id": winner_id,
        "validation_mean_nll": metrics.mean_nll_per_valid_trial,
        "validation_elpd": metrics.elpd,
        "locked_test_files_opened": data.audit.locked_test_files_opened,
        "smoke_test": args.smoke_test,
    }
    (output / f"{prefix}summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
