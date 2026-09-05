#!/usr/bin/env python3
"""Compare paired M2 and M3 trial-prediction exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.analysis import (  # noqa: E402
    build_m3_m2_validation_comparison,
    compare_m3_to_m2_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-predictions", required=True, type=Path)
    parser.add_argument("--m3-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    m3 = pd.read_csv(args.m3_predictions)
    m2 = pd.read_csv(args.m2_predictions)
    sessions = compare_m3_to_m2_predictions(m3, m2)
    sessions.to_csv(output / "paired_session_comparison.csv", index=False)
    aggregate = sessions.groupby("split_role", observed=False).agg(
        session_count=("session_id", "size"),
        valid_trial_count=("n_valid_trials", "sum"),
        total_delta_bits_m3_vs_m2=("total_delta_bits_m3_vs_m2", "sum"),
    ).reset_index()
    aggregate["mean_delta_bits_per_trial_m3_vs_m2"] = (
        aggregate["total_delta_bits_m3_vs_m2"] / aggregate["valid_trial_count"]
    )
    summary = aggregate.to_dict(orient="records")
    (output / "paired_model_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if "validation" in set(m3.split_role.astype(str)):
        detailed = build_m3_m2_validation_comparison(m3, m2)
        for name in (
            "overall", "per_session", "context", "transition_aligned",
            "surprising_category_aligned", "psychometric_by_block", "calibration",
        ):
            getattr(detailed, name).to_csv(
                output / f"validation_{name}.csv", index=False
            )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
