#!/usr/bin/env python3
"""Fit M2_dynamic_previous_action_bias_v1 (M2_dynamic_block_length_v1's static
prototypes + duration-dependent HSMM hazard, unchanged, plus a fitted
action-history bias mixed between the sensory readout and lapse) without
opening locked-test files."""

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
from scripts.fit_m0 import _lbfgs_settings, _print_initial, _print_runtime_settings, _validate_convergence_budget  # noqa: E402
from m2_dynamic_previous_action_bias_v1.analysis import (  # noqa: E402
    build_m2_dynamic_previous_action_bias_hazard_curve,
    build_m2_dynamic_previous_action_bias_trial_predictions,
    save_m2_dynamic_previous_action_bias_analysis_outputs,
)
from m2_dynamic_previous_action_bias_v1.diagnostics import (  # noqa: E402
    build_m2_dynamic_previous_action_bias_implied_duration_distribution,
    summarize_m2_dynamic_previous_action_bias_hazard_duration,
    summarize_m2_dynamic_previous_action_bias_prediction_diagnostics,
)
from m2_dynamic_previous_action_bias_v1.fit import (  # noqa: E402
    fit_m2_dynamic_previous_action_bias_all_initializations,
    fit_m2_dynamic_previous_action_bias_single_initialization,
)
from m2_dynamic_previous_action_bias_v1.figures import generate_m2_dynamic_previous_action_bias_duration_figures  # noqa: E402
from m2_dynamic_previous_action_bias_v1.m0_style_figures import generate_m2_dynamic_previous_action_bias_m0_style_figures  # noqa: E402
from m2_dynamic_previous_action_bias_v1.pipeline import prepare_registered_m2_dynamic_previous_action_bias_fit_data  # noqa: E402
from m2_dynamic_previous_action_bias_v1.registry import load_m2_dynamic_previous_action_bias_model_registry  # noqa: E402
from m2_dynamic_previous_action_bias_v1.session import evaluate_m2_dynamic_previous_action_bias_mouse_objective  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Fit M2_dynamic_previous_action_bias_v1 with eight independent starts, a "
        "3x65 capped-tail HSMM (unchanged from M2_dynamic_block_length_v1), and a "
        "fitted action-history bias mixed between the sensory readout and lapse. "
        "Locked-test files remain unopened."
    ))
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs", "hybrid"), default="adam")
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
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--include-full-run-length-distributions", action="store_true")
    args = parser.parse_args()
    positive = (
        "print_every", "smoke_epochs", "adam_warmup_epochs", "lbfgs_max_steps",
        "lbfgs_learning_rate", "lbfgs_history_size", "lbfgs_max_evaluations",
        "lbfgs_tolerance_grad", "lbfgs_tolerance_change", "parallel_workers",
        "cpu_threads_per_worker",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("Optimizer counts, rates, tolerances, and worker counts must be positive")
    if args.max_epochs is not None and args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.smoke_test and args.max_epochs is not None:
        parser.error("--max-epochs is for full runs; use --smoke-epochs for smoke tests")
    if args.optimizer != "adam" and args.max_epochs is not None:
        parser.error("--max-epochs applies only to --optimizer adam")
    if args.optimizer == "lbfgs" and args.learning_rate is not None:
        parser.error("--learning-rate configures Adam, not pure L-BFGS")
    return args


def _print_step(start_id, record, every):
    if record.step % every == 0:
        print(
            f"start={start_id:02d} step={record.step:04d} phase={record.optimizer_phase} "
            f"train_mean_nll={record.fresh_training_mean_nll:.12f} "
            f"validation_mean_nll={record.validation_mean_nll:.12f} "
            f"grad_norm={record.gradient_norm:.6e} lr={record.learning_rate:.6e}",
            flush=True,
        )


def _metrics(sessions, raw, registry, role):
    objective = evaluate_m2_dynamic_previous_action_bias_mouse_objective(
        sessions, torch.tensor(raw, dtype=torch.float64), registry.task_moments,
        objective_weighting=registry.objective_weighting, required_split_role=role,
        lapse_max=registry.lapse_max, parameter_epsilon=registry.parameter_epsilon,
        run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        log_probability_epsilon=registry.log_probability_epsilon,
        prior_probability_epsilon=registry.prior_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
        first_long_probability=registry.first_long_probability,
        initial_pi=registry.initial_pi,
    )
    return calculate_evaluation_metrics(
        objective.session_evaluations, model_id=registry.model_id,
        calibration_bins=registry.metrics.calibration_bins,
        psychometric_duration_decimals=registry.metrics.psychometric_duration_decimals,
        classification_threshold=registry.metrics.classification_threshold,
    )


def main() -> int:
    args = _parse_args()
    registry = load_m2_dynamic_previous_action_bias_model_registry(args.registry)
    try:
        _validate_convergence_budget(args, registry)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr, flush=True)
        return 2
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Preparing M2_dynamic_previous_action_bias train/validation timelines; locked test remains unopened.", flush=True)
    data = prepare_registered_m2_dynamic_previous_action_bias_fit_data(args.raw_data_root, registry)
    save_fit_input_audit(data.audit, output / "fit_input_audit.json")
    print(
        f"model={registry.model_id} states=3x{registry.maximum_represented_run_length} "
        f"center={registry.run_length_center} train_sessions={len(data.training_sessions)} "
        f"validation_sessions={len(data.validation_sessions)} "
        f"locked_test_files_opened={data.audit.locked_test_files_opened}", flush=True,
    )
    _print_runtime_settings(args, registry)
    if args.smoke_test:
        run = fit_m2_dynamic_previous_action_bias_single_initialization(
            data.training_sessions, data.validation_sessions, registry,
            registry.initializations[0], optimizer_strategy=args.optimizer,
            max_epochs_override=args.smoke_epochs if args.optimizer == "adam" else None,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=min(args.smoke_epochs, args.adam_warmup_epochs),
            lbfgs_settings=_lbfgs_settings(args, smoke_test=True),
            allow_short_convergence_budget=True,
            initial_callback=lambda train, validation: _print_initial(1, train, validation),
            epoch_callback=lambda record: _print_step(1, record, args.print_every),
        )
        fit_result = run.as_dict()
        raw = run.best_raw_parameters
        prefix = "smoke_"
        winner_id = run.start_id
    else:
        result = fit_m2_dynamic_previous_action_bias_all_initializations(
            data.training_sessions, data.validation_sessions, registry,
            optimizer_strategy=args.optimizer, max_epochs_override=args.max_epochs,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=args.adam_warmup_epochs,
            lbfgs_settings=_lbfgs_settings(args, smoke_test=False),
            parallel_workers=args.parallel_workers,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
            initial_callback=lambda start, train, validation: _print_initial(start, train, validation),
            epoch_callback=lambda start, record: _print_step(start, record, args.print_every),
            completion_callback=lambda run: print(
                f"completed_start={run.start_id:02d} training_mean_nll="
                f"{run.recalculated_best_training_mean_nll:.12f} converged={run.converged}",
                flush=True,
            ),
            output_directory=output / "starts",
        )
        if result.winner_start_id is None:
            print("No M2_dynamic_previous_action_bias run passed convergence and tail-occupancy acceptance.", flush=True)
            return 2
        winner = next(run for run in result.runs if run.start_id == result.winner_start_id)
        fit_result = result.as_dict()
        raw = winner.best_raw_parameters
        prefix = ""
        winner_id = winner.start_id
    metrics = _metrics(data.validation_sessions, raw, registry, "validation")
    predictions = build_m2_dynamic_previous_action_bias_trial_predictions(
        data.canonical_trials, raw, registry,
        include_full_run_length_distributions=args.include_full_run_length_distributions,
    )
    save_m2_dynamic_previous_action_bias_analysis_outputs(
        predictions, fit_result, output if not prefix else output / "smoke", test_unlocked=False,
    )
    summarize_m2_dynamic_previous_action_bias_prediction_diagnostics(predictions).to_csv(
        output / f"{prefix}prediction_diagnostics_by_session.csv", index=False
    )
    hazard_curve = build_m2_dynamic_previous_action_bias_hazard_curve(raw, registry)
    hazard_curve.to_csv(output / f"{prefix}hazard_curve.csv", index=False)
    build_m2_dynamic_previous_action_bias_implied_duration_distribution(raw, registry).to_csv(
        output / f"{prefix}implied_duration_distribution.csv", index=False
    )
    duration_summary = summarize_m2_dynamic_previous_action_bias_hazard_duration(raw, registry).as_dict()
    (output / f"{prefix}hazard_duration_summary.json").write_text(
        json.dumps(duration_summary, indent=2) + "\n", encoding="utf-8"
    )
    generate_m2_dynamic_previous_action_bias_duration_figures(
        predictions, hazard_curve, output / f"{prefix}duration_figures"
    )
    if not args.smoke_test:
        generate_m2_dynamic_previous_action_bias_m0_style_figures(
            predictions, fit_result, output / "m0_style_figures", test_unlocked=False,
        )
    (output / f"{prefix}validation_metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2) + "\n")
    (output / f"{prefix}fit_result.json").write_text(json.dumps(fit_result, indent=2) + "\n")
    summary = {
        "model_id": registry.model_id, "winner_start_id": winner_id,
        "validation_mean_nll": metrics.mean_nll_per_valid_trial,
        "validation_elpd": metrics.elpd,
        "run_length_center": registry.run_length_center,
        "maximum_represented_run_length": registry.maximum_represented_run_length,
        "tail_state_semantics": registry.tail_state_semantics,
        "tail_acceptance_threshold": registry.maximum_accepted_mean_tail_probability,
        "training_mean_tail_probability_pre": (
            run.training_mean_tail_probability_pre if args.smoke_test
            else winner.training_mean_tail_probability_pre
        ),
        "tail_occupancy_acceptable": (
            run.tail_occupancy_acceptable if args.smoke_test else winner.tail_occupancy_acceptable
        ),
        "hazard_duration_summary": duration_summary,
        "locked_test_files_opened": data.audit.locked_test_files_opened,
        "map_optimization_used": False,
        "full_run_length_distributions_included": args.include_full_run_length_distributions,
        "smoke_test": args.smoke_test,
        "combination_order": "sensory_readout(r_t) -> bias_mix(m_t=(1-eta)r_t+eta*pi_t, alpha fixed at 1.0) -> lapse_mixture(m_t)",
        "independent_of_m4_hsmm_delta_v1": True,
    }
    (output / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
