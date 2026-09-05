#!/usr/bin/env python3
"""Run real-design M2 recovery at hazard 1/23 across three contrasts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m0_sensory_v1.m2_registry import load_m2_model_registry  # noqa: E402
from m0_sensory_v1.model import run_m2_real_design_recovery_case  # noqa: E402
from m0_sensory_v1.pipeline import prepare_registered_m2_fit_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    registry = load_m2_model_registry(args.registry)
    design = prepare_registered_m2_fit_data(args.raw_data_root, registry)
    block_maps = {
        str(session_id): tuple(
            frame.sort_values("trial_index")["true_block"].astype(str)
        )
        for session_id, frame in design.canonical_trials.groupby(
            "session_id", sort=False, observed=False
        )
    }
    training_blocks = {
        session.session_id: block_maps[session.session_id]
        for session in design.training_sessions
    }
    validation_blocks = {
        session.session_id: block_maps[session.session_id]
        for session in design.validation_sessions
    }
    common = registry.initializations[0].constrained_tensor()
    hazard = 1.0 / 23.0
    contrasts = (
        0.1,
        float(torch.log(torch.tensor(4.0, dtype=torch.float64))),
        2.2,
    )
    records = []
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, delta0 in enumerate(contrasts):
        generating = common.clone()
        generating[4] = torch.logit(torch.tensor(hazard, dtype=torch.float64))
        generating[5] = delta0
        condition_output = output / f"delta0_{delta0:.6f}" / "starts"
        result = run_m2_real_design_recovery_case(
            generating,
            design.training_sessions,
            design.validation_sessions,
            training_blocks,
            validation_blocks,
            registry,
            random_seed=registry.seeds.simulation_seed + index * 100,
            maximum_epochs=args.epochs,
            output_directory=condition_output,
        )
        record = asdict(result)
        record.pop("multistart_result")
        records.append(record)
        print(f"generated_h={hazard:.4f} recovered_h={result.recovered_hazard:.4f} generated_delta0={delta0:.4f} recovered_delta0={result.recovered_delta0:.4f}", flush=True)
    (output / "m2_recovery.json").write_text(json.dumps({
        "design": "real session lengths, durations, omissions, feedback masks, and true-block schedules",
        "generating_hazard": hazard,
        "hazard_rationale": "inverse of the observed median block duration, 23 trials",
        "cases": records,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, generated_key, recovered_key, label in (
        (axes[0], "generating_hazard", "recovered_hazard", "Hazard"),
        (axes[1], "generating_delta0", "recovered_delta0", "δ₀"),
    ):
        x = [record[generated_key] for record in records]
        y = [record[recovered_key] for record in records]
        axis.scatter(x, y, color="#7b1fa2", s=55)
        lo, hi = min(x + y), max(x + y)
        axis.plot([lo, hi], [lo, hi], "k--")
        axis.set(xlabel=f"Generating {label}", ylabel=f"Recovered {label}", title=f"M2 {label} recovery")
        axis.grid(alpha=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"m2_recovery.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
