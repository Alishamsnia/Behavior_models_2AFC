#!/usr/bin/env python3
"""Run nine-parameter M4 recovery on preserved observed timelines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.pipeline import prepare_registered_m4_fit_data  # noqa: E402
from m4_hsmm_delta_v1.recovery import (  # noqa: E402
    iter_registered_m4_recovery_grid,
    run_m4_design_preserving_recovery,
)
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402


def _values(text: str, count: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated floats") from error
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"Exactly {count} {label} values are required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    generating = parser.add_mutually_exclusive_group(required=True)
    generating.add_argument(
        "--generating-parameters", type=lambda value: _values(value, 9, "constrained M4"),
        help="w,s0,ell_short,ell_long,kappa,ah,delta0,alpha_theta,bh",
    )
    generating.add_argument(
        "--registered-grid-base", type=lambda value: _values(value, 5, "common"),
        help="Run the registered dynamic grid using base w,s0,ell_short,ell_long,kappa",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    if min(args.repetitions, args.max_steps, args.parallel_workers, args.cpu_threads_per_worker) <= 0:
        parser.error("counts and optimizer steps must be positive")
    registry = load_m4_model_registry(args.registry)
    prepared = prepare_registered_m4_fit_data(args.raw_data_root, registry)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = (
        [torch.tensor(args.generating_parameters, dtype=torch.float64)]
        if args.generating_parameters is not None
        else list(iter_registered_m4_recovery_grid(args.registered_grid_base))
    )
    rows: list[dict[str, object]] = []
    run_index = 0
    for case_index, parameters in enumerate(cases, start=1):
        for repetition in range(1, args.repetitions + 1):
            run_index += 1
            case_output = output / f"case_{case_index:03d}_rep_{repetition:03d}"
            result = run_m4_design_preserving_recovery(
                parameters, prepared.training_sessions, prepared.validation_sessions, registry,
                random_seed=args.seed + 2 * (run_index - 1), maximum_steps=args.max_steps,
                output_directory=case_output / "starts",
                parallel_workers=args.parallel_workers,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
            )
            case_output.mkdir(parents=True, exist_ok=True)
            (case_output / "recovery_summary.json").write_text(
                json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
            )
            row: dict[str, object] = {
                "case_index": case_index, "repetition": repetition,
                "winner_start_id": result.winner_start_id,
            }
            for name, generated, recovered, error in zip(
                registry.parameter_order, result.generating_parameters,
                result.recovered_parameters, result.parameter_errors, strict=True,
            ):
                row[f"generating_{name}"] = generated
                row[f"recovered_{name}"] = recovered
                row[f"error_{name}"] = error
            row.update({f"trajectory_{key}": value for key, value in asdict(result.trajectory_metrics).items()})
            rows.append(row)
            print(f"case={case_index}/{len(cases)} repetition={repetition}/{args.repetitions} winner={result.winner_start_id}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "recovery_results.csv", index=False)
    interval_rows: list[dict[str, object]] = []
    group_columns = [f"generating_{name}" for name in registry.parameter_order]
    for generating_values, group in table.groupby(group_columns, observed=False):
        values = generating_values if isinstance(generating_values, tuple) else (generating_values,)
        generating_map = dict(zip(registry.parameter_order, values, strict=True))
        for name in registry.parameter_order:
            recovered = group[f"recovered_{name}"].to_numpy(float)
            lower = float(np.quantile(recovered, .025)) if len(recovered) >= 2 else float("nan")
            upper = float(np.quantile(recovered, .975)) if len(recovered) >= 2 else float("nan")
            truth = float(generating_map[name])
            interval_rows.append({
                **{f"generating_{key}": value for key, value in generating_map.items()},
                "parameter": name, "repetitions": len(recovered),
                "empirical_lower_2_5_percent": lower,
                "empirical_upper_97_5_percent": upper,
                "covers_generating_value": (
                    None if not np.isfinite(lower) else bool(lower <= truth <= upper)
                ),
            })
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(output / "case_recovery_intervals.csv", index=False)
    aggregate = []
    for name in registry.parameter_order:
        error = table[f"error_{name}"].to_numpy(float)
        generated = table[f"generating_{name}"].to_numpy(float)
        recovered = table[f"recovered_{name}"].to_numpy(float)
        rank = pd.Series(generated).corr(pd.Series(recovered), method="spearman")
        coverage_values = intervals.loc[
            intervals.parameter.eq(name), "covers_generating_value"
        ].dropna()
        aggregate.append({
            "parameter": name, "n": len(error), "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "spearman_rank_correlation": None if pd.isna(rank) else float(rank),
            "empirical_95_interval_coverage": (
                None if coverage_values.empty else float(coverage_values.astype(bool).mean())
            ),
        })
    pd.DataFrame(aggregate).to_csv(output / "parameter_recovery_metrics.csv", index=False)
    (output / "recovery_manifest.json").write_text(json.dumps({
        "model_id": registry.model_id, "case_count": len(cases),
        "repetitions_per_case": args.repetitions, "run_count": len(rows),
        "optimizer": "full_batch_lbfgs", "map_optimization_used": False,
        "starts_per_fit": len(registry.initializations),
        "parallel_workers": args.parallel_workers,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
