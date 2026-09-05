#!/usr/bin/env python3
"""Run validation-only M3-versus-M4 model recovery on observed designs."""

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

from m4_hsmm_delta_v1.pipeline import prepare_registered_m4_fit_data  # noqa: E402
from m4_hsmm_delta_v1.recovery import run_m3_m4_model_recovery  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402


def _parameters(value: str) -> torch.Tensor:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Parameters must be comma-separated floats") from error
    if len(values) != 9:
        raise argparse.ArgumentTypeError(
            "Exactly w,s0,ell_short,ell_long,kappa,ah,delta0,alpha_theta,bh are required"
        )
    return torch.tensor(values, dtype=torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--generating-parameters", required=True, type=_parameters)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    registry = load_m4_model_registry(args.registry)
    prepared = prepare_registered_m4_fit_data(args.raw_data_root, registry)
    result = run_m3_m4_model_recovery(
        args.generating_parameters, prepared.training_sessions, prepared.validation_sessions,
        registry, random_seed=args.seed, maximum_steps=args.max_steps,
        parallel_workers=args.parallel_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
