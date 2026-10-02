#!/usr/bin/env python3
"""Run ten-parameter M2_dynamic_previous_action_bias recovery on preserved
observed timelines (sequential, action-history-consistent simulation)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m2_dynamic_previous_action_bias_v1.pipeline import prepare_registered_m2_dynamic_previous_action_bias_fit_data  # noqa: E402
from m2_dynamic_previous_action_bias_v1.recovery import (  # noqa: E402
    iter_registered_m2_dynamic_previous_action_bias_recovery_grid,
    run_m2_dynamic_previous_action_bias_design_preserving_recovery,
)
from m2_dynamic_previous_action_bias_v1.registry import load_m2_dynamic_previous_action_bias_model_registry  # noqa: E402


def _nine_values(text: str) -> torch.Tensor:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Parameters must be comma-separated floats") from error
    if len(values) != 9:
        raise argparse.ArgumentTypeError("Exactly nine constrained M2_dynamic_previous_action_bias values are required")
    return torch.tensor(values, dtype=torch.float64)


def _five_values(text: str) -> tuple[float, float, float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Parameters must be comma-separated floats") from error
    if len(values) != 5:
        raise argparse.ArgumentTypeError("Exactly w,s0,ell_short,ell_long,kappa are required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    generating = parser.add_mutually_exclusive_group(required=True)
    generating.add_argument(
        "--generating-parameters", type=_nine_values,
        help="w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit in constrained coordinates",
    )
    generating.add_argument(
        "--registered-grid-base", type=_five_values,
        help="Run all registered cases using base w,s0,ell_short,ell_long,kappa",
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    registry = load_m2_dynamic_previous_action_bias_model_registry(args.registry)
    prepared = prepare_registered_m2_dynamic_previous_action_bias_fit_data(args.raw_data_root, registry)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = (
        [args.generating_parameters] if args.generating_parameters is not None
        else list(iter_registered_m2_dynamic_previous_action_bias_recovery_grid(args.registered_grid_base))
    )
    summaries = []
    for case_index, parameters in enumerate(cases, start=1):
        case_output = output / f"case_{case_index:03d}"
        case_output.mkdir(parents=True, exist_ok=True)
        result = run_m2_dynamic_previous_action_bias_design_preserving_recovery(
            parameters, prepared.training_sessions, prepared.validation_sessions, registry,
            random_seed=args.seed + 2 * (case_index - 1), maximum_steps=args.max_steps,
            output_directory=case_output / "starts",
        )
        (case_output / "recovery_summary.json").write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )
        summaries.append({
            "case_index": case_index, "generating_parameters": result.generating_parameters,
            "recovered_parameters": result.recovered_parameters, "parameter_errors": result.parameter_errors,
            "winner_start_id": result.winner_start_id, "trajectory_metrics": asdict(result.trajectory_metrics),
        })
        print(
            f"case={case_index:03d}/{len(cases):03d} winner_start={result.winner_start_id:02d} "
            f"parameter_errors={result.parameter_errors}", flush=True,
        )
    (output / "recovery_grid_summary.json").write_text(
        json.dumps({"case_count": len(summaries), "cases": summaries}, indent=2) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
