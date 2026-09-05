#!/usr/bin/env python3
"""Run the registered four-condition M1 hazard-recovery grid."""

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
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m0_sensory_v1.model import run_m1_hazard_recovery_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--trials-per-session", type=int, default=300)
    args = parser.parse_args()
    if min(args.epochs, args.sessions, args.trials_per_session) <= 0:
        parser.error("numeric controls must be positive")
    registry = load_m1_model_registry(args.registry)
    levels = torch.tensor([.2,.35,.5,.65,.8,1.7,1.85,2.,2.15,2.3], dtype=torch.float64)
    repetitions = max(1, args.trials_per_session // len(levels))
    durations = tuple(levels.repeat(repetitions) for _ in range(args.sessions))
    validation = durations[:max(1, args.sessions // 3)]
    common = registry.initializations[0].constrained_tensor()
    hazards = (0.02, 1/30, 0.08, 0.20)
    records = []
    for index, hazard in enumerate(hazards):
        generating = common.clone()
        generating[4] = torch.logit(torch.tensor(hazard, dtype=torch.float64))
        start = min(registry.initializations, key=lambda item: abs(item.hazard-hazard))
        result = run_m1_hazard_recovery_case(
            generating, durations, validation, registry, start,
            random_seed=registry.seeds.simulation_seed + index*100,
            maximum_epochs=args.epochs,
        )
        record = asdict(result); record.pop("fit_result"); records.append(record)
        print(f"generated_h={hazard:.6f} recovered_h={result.recovered_hazard:.6f}", flush=True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output/"hazard_recovery.json").write_text(json.dumps({"cases":records},indent=2)+"\n")
    x=[r["generating_hazard"] for r in records]; y=[r["recovered_hazard"] for r in records]
    fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(x,y,color="#c62828",s=55); lo=min(x+y); hi=max(x+y)
    ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="Generating hazard",ylabel="Recovered hazard",title="M1 hazard recovery"); ax.grid(alpha=.25)
    for suffix in ("png","pdf"): fig.savefig(output/f"hazard_recovery.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__": raise SystemExit(main())
