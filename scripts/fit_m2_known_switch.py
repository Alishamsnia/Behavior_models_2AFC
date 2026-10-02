#!/usr/bin/env python3
"""Fit M2_known_switch_v1 to frozen train/validation sessions.

Mirrors ``scripts/fit_m1_known_switch.py`` exactly (see its module
docstring for the scope note and independence rationale).
"""

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

from m0_sensory_v1.model.fit import LBFGSSettings  # noqa: E402
from m2_known_switch_v1.analysis import evaluate_m2_known_switch_split_metrics  # noqa: E402
from m2_known_switch_v1.fit import fit_m2_known_switch_all_initializations  # noqa: E402
from m2_known_switch_v1.pipeline import prepare_registered_m2_known_switch_fit_data  # noqa: E402
from m2_known_switch_v1.registry import load_m2_known_switch_model_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit M2_known_switch_v1 (M2's fitted static subjective prototypes "
            "under the oracle block-switch schedule; no fitted hazard) to frozen "
            "train/validation sessions. Locked-test MATLAB contents are never opened."
        )
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs", "hybrid"), default="adam")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="Run only start 1, bounded")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adam-warmup-epochs", type=int, default=None)
    parser.add_argument("--lbfgs-max-steps", type=int, default=None)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--lbfgs-max-evaluations", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help=(
            "torch.set_num_threads() budget for this single-process sequential "
            "fitter (no parallel worker pool). Keep this modest on a machine "
            "running other concurrent fits."
        ),
    )
    return parser.parse_args()


def _lbfgs_settings(args: argparse.Namespace, registry, *, smoke: bool) -> LBFGSSettings:
    return LBFGSSettings(
        maximum_steps=(2 if smoke else (args.lbfgs_max_steps or registry.fitting.lbfgs_maximum_steps)),
        learning_rate=args.lbfgs_learning_rate,
        history_size=args.lbfgs_history_size,
        maximum_evaluations_per_step=args.lbfgs_max_evaluations,
        tolerance_grad=args.lbfgs_tolerance_grad,
        tolerance_change=args.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    registry = load_m2_known_switch_model_registry(args.registry)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"cpu_threads={args.cpu_threads} "
        "Preparing M2_known_switch train/validation timelines; locked test remains unopened.",
        flush=True,
    )
    data = prepare_registered_m2_known_switch_fit_data(args.raw_data_root, registry)
    print(
        f"model={registry.model_id} loaded_train_sessions={len(data.training_sessions)} "
        f"loaded_validation_sessions={len(data.validation_sessions)} "
        f"locked_test_files_opened={data.audit.locked_test_files_opened}",
        flush=True,
    )

    def epoch_printer(start_id: int, record) -> None:
        if record.step % max(1, args.print_every) == 0:
            print(
                f"start={start_id:02d} phase={record.optimizer_phase} step={record.step:04d} "
                f"train_mean_nll={record.fresh_training_mean_nll:.10f} "
                f"validation_mean_nll={record.validation_mean_nll:.10f} "
                f"grad_norm={record.gradient_norm:.6e} lr={record.learning_rate:.6g}",
                flush=True,
            )

    if args.smoke_test:
        initialization = registry.initializations[0]
        from m2_known_switch_v1.fit import fit_m2_known_switch_single_initialization

        result = fit_m2_known_switch_single_initialization(
            data.training_sessions, data.validation_sessions, registry, initialization,
            optimizer_strategy=args.optimizer,
            max_epochs_override=(args.smoke_epochs if args.optimizer == "adam" else None),
            initial_learning_rate_override=args.learning_rate,
            adam_warmup_epochs=(min(args.smoke_epochs, args.adam_warmup_epochs or args.smoke_epochs)),
            lbfgs_settings=_lbfgs_settings(args, registry, smoke=True),
            epoch_callback=lambda record: epoch_printer(initialization.start_id, record),
        )
        (output / "smoke_start_01.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
        summary = {
            "status": "passed",
            "purpose": "pipeline_smoke_only_not_a_registered_fit_or_winner",
            "model_id": registry.model_id,
            "optimizer_strategy": args.optimizer,
            "start_id": initialization.start_id,
            "training_mean_nll": result.recalculated_best_training_mean_nll,
            "converged": result.converged,
            "locked_test_files_opened": data.audit.locked_test_files_opened,
        }
        (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    print(f"M2_known_switch multistart execution: {len(registry.initializations)} sequential starts.", flush=True)
    result = fit_m2_known_switch_all_initializations(
        data.training_sessions, data.validation_sessions, registry,
        optimizer_strategy=args.optimizer,
        max_epochs_override=args.max_epochs,
        initial_learning_rate_override=args.learning_rate,
        adam_warmup_epochs=args.adam_warmup_epochs,
        lbfgs_settings=_lbfgs_settings(args, registry, smoke=False),
        epoch_callback=epoch_printer,
        completion_callback=lambda run: print(
            f"completed start={run.start_id:02d} converged={run.converged} "
            f"training_mean_nll={run.recalculated_best_training_mean_nll:.12f} "
            f"reason={run.termination_reason}",
            flush=True,
        ),
    )
    (output / "multistart_summary.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")

    if result.winner_start_id is None:
        print("No converged M2_known_switch run passed every registered check.", flush=True)
        return 2
    winner = next(run for run in result.runs if run.start_id == result.winner_start_id)
    metrics = evaluate_m2_known_switch_split_metrics(
        data.validation_sessions,
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        registry, required_split_role="validation",
    )
    (output / "validation_metrics.json").write_text(
        json.dumps(
            {
                "elpd": metrics.elpd,
                "mean_nll_per_valid_trial": metrics.mean_nll_per_valid_trial,
                "brier_score": metrics.brier_score,
                "classification_accuracy": metrics.classification_accuracy,
            },
            indent=2,
        ) + "\n"
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
