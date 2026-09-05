#!/usr/bin/env python3
"""Run reproducible multi-case synthetic M0 recovery and save a report."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import run_parameter_recovery_case  # noqa: E402
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse recovery sample-size, epoch, and output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--trials-per-duration", type=int, default=300)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.trials_per_duration <= 0 or args.sessions <= 0:
        parser.error("epochs, trials-per-duration, and sessions must be positive")
    return args


def _duration_sessions(count: int, repetitions: int) -> tuple[torch.Tensor, ...]:
    """Build balanced sessions spanning the real truncated task supports."""

    levels = torch.tensor(
        [0.2, 0.35, 0.5, 0.65, 0.8, 1.7, 1.85, 2.0, 2.15, 2.3],
        dtype=torch.float64,
    )
    return tuple(levels.repeat_interleave(repetitions) for _ in range(count))


def main() -> int:
    """Execute low/high noise/lapse cases and atomically save JSON results."""

    args = parse_args()
    registry = load_model_registry(args.registry)
    training_durations = _duration_sessions(args.sessions, args.trials_per_duration)
    validation_durations = _duration_sessions(
        max(1, args.sessions // 2),
        max(1, args.trials_per_duration // 2),
    )
    case_indices = (0, 3, 4, 6)
    labels = (
        "reference_symmetric_lapse",
        "high_short_category_lapse",
        "near_zero_category_lapses",
        "high_sensory_and_lapse_noise",
    )
    records: list[dict] = []
    for offset, (index, label) in enumerate(zip(case_indices, labels, strict=True)):
        generating = registry.initializations[index].constrained_tensor()
        result = run_parameter_recovery_case(
            generating,
            training_durations,
            validation_durations,
            registry,
            registry.initializations[0],
            random_seed=registry.seeds.simulation_seed + offset * 100,
            maximum_epochs=args.epochs,
        )
        record = asdict(result)
        record.pop("fit_result")
        record["label"] = label
        record["accepted_predictive_recovery"] = bool(
            result.probability_rmse_on_duration_grid <= 0.05
            and result.training_nll_improvement >= -1e-12
        )
        records.append(record)
        print(
            f"case={label} probability_rmse={result.probability_rmse_on_duration_grid:.6f} "
            f"nll_improvement={result.training_nll_improvement:.6f} "
            f"generated={result.generating_parameters} recovered={result.recovered_parameters}",
            flush=True,
        )

    payload = {
        "model_id": registry.model_id,
        "purpose": "Part X synthetic parameter and predictive recovery",
        "precision": "torch.float64",
        "epochs_per_case": args.epochs,
        "training_sessions_per_case": args.sessions,
        "trials_per_duration_per_session": args.trials_per_duration,
        "all_predictive_recovery_cases_accepted": all(
            record["accepted_predictive_recovery"] for record in records
        ),
        "identification_note": (
            "Both short- and long-category trials are included so ell_short and "
            "ell_long each influence their corresponding psychometric branch."
        ),
        "cases": records,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return 0 if payload["all_predictive_recovery_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
