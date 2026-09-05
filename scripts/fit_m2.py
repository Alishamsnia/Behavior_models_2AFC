#!/usr/bin/env python3
"""Fit the causal M2 static-prototype HMM without opening locked-test files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fit_m0 import (  # noqa: E402
    _lbfgs_settings,
    _print_epoch,
    _print_completed_start,
    _print_initial,
    _print_runtime_settings,
    _validate_convergence_budget,
    parse_args,
)
from m0_sensory_v1.m2_analysis import (  # noqa: E402
    build_m2_trial_predictions,
    save_m2_analysis_outputs,
)
from m0_sensory_v1.m2_figures import (  # noqa: E402
    generate_m2_diagnostic_figures,
    generate_m2_identifiability_figure,
)
from m0_sensory_v1.m2_m0_style_figures import generate_m2_m0_style_figures  # noqa: E402
from m0_sensory_v1.m2_registry import load_m2_model_registry  # noqa: E402
from m0_sensory_v1.model import (  # noqa: E402
    evaluate_m2_hazard_contrast_surface,
    evaluate_split_metrics,
    fit_all_initializations,
    fit_single_initialization_with_strategy,
    save_evaluation_metrics,
    save_fit_run_result,
)
from m0_sensory_v1.pipeline import (  # noqa: E402
    prepare_registered_m2_fit_data,
    save_fit_input_audit,
)


def main() -> int:
    """Fit one M2 registry with Adam, L-BFGS, or hybrid optimization."""

    args = parse_args(
        description=(
            "Fit M2 static-prototype HMM to frozen train/validation sessions. "
            "Locked-test MATLAB contents are never opened."
        )
    )
    registry = load_m2_model_registry(args.registry)
    try:
        _validate_convergence_budget(args, registry)
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr, flush=True)
        return 2
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Preparing M2 train/validation timelines; locked test remains unopened.", flush=True)
    data = prepare_registered_m2_fit_data(args.raw_data_root, registry)
    save_fit_input_audit(data.audit, output / "fit_input_audit.json")
    print(
        f"model={registry.model_id} loaded_train_sessions={len(data.training_sessions)} "
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
            max_epochs_override=args.smoke_epochs if args.optimizer == "adam" else None,
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=min(args.smoke_epochs, args.adam_warmup_epochs),
            lbfgs_settings=_lbfgs_settings(args, smoke_test=True),
            allow_short_convergence_budget=True,
            initial_callback=lambda train, validation: _print_initial(initialization.start_id, train, validation),
            epoch_callback=lambda record: _print_epoch(initialization.start_id, record, args.print_every),
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
        predictions = build_m2_trial_predictions(
            data.canonical_trials, result.best_raw_parameters, registry
        )
        predictions.to_csv(output / "smoke_trial_predictions.csv", index=False)
        generate_m2_diagnostic_figures(predictions, output / "smoke_m2_figures")
        summary = {
            "status": "passed",
            "purpose": "pipeline_smoke_only_not_a_registered_fit_or_winner",
            "model_id": registry.model_id,
            "epochs": args.smoke_epochs,
            "optimizer_strategy": args.optimizer,
            "start_id": initialization.start_id,
            "training_mean_nll": result.recalculated_best_training_mean_nll,
            "validation_mean_nll": metrics.mean_nll_per_valid_trial,
            "validation_elpd": metrics.elpd,
            "locked_test_files_opened": data.audit.locked_test_files_opened,
        }
        (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    print(
        f"M2 multistart execution: {len(registry.initializations)} independent starts, "
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
        epoch_callback=lambda start_id, record: _print_epoch(start_id, record, args.print_every),
        completion_callback=_print_completed_start,
    )
    if result.winner_start_id is None:
        print("No converged M2 run passed every frozen check.", flush=True)
        return 2
    winner = next(run for run in result.runs if run.start_id == result.winner_start_id)
    metrics = evaluate_split_metrics(
        data.validation_sessions,
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        registry,
        required_split_role="validation",
        block_labels_by_session=data.validation_block_labels,
    )
    save_evaluation_metrics(metrics, output / "validation_metrics.json")
    predictions = build_m2_trial_predictions(
        data.canonical_trials, winner.best_raw_parameters, registry
    )
    save_m2_analysis_outputs(predictions, result.as_dict(), output, test_unlocked=False)
    generate_m2_diagnostic_figures(predictions, output / "m2_figures")
    generate_m2_m0_style_figures(
        predictions, result.as_dict(), output / "m2_m0_style_figures", test_unlocked=False
    )
    identifiability_surface = evaluate_m2_hazard_contrast_surface(
        data.training_sessions,
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        registry,
        grid_points=15,
    )
    generate_m2_identifiability_figure(
        identifiability_surface, output / "m2_identifiability"
    )
    print(
        f"winner_start={winner.start_id:02d} "
        f"training_mean_nll={winner.recalculated_best_training_mean_nll:.12f} "
        f"validation_mean_nll={metrics.mean_nll_per_valid_trial:.12f} "
        f"validation_elpd={metrics.elpd:.12f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
