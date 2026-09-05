#!/usr/bin/env python3
"""Compare paired M2 and M2_dynamic_block_length trial-prediction exports.

This is the scientific test of whether duration-dependent hazard estimation
improves over M2's constant hazard: paired per-session and aggregate
held-out log-score (bits/trial) comparisons on matching trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _paired(m2: pd.DataFrame, m2d: pd.DataFrame, *, split_role: str) -> pd.DataFrame:
    required = {"session_id", "trial_index", "split_role", "model_evaluated", "per_trial_negative_log_likelihood"}
    for name, frame in (("M2", m2), ("M2_dynamic_block_length", m2d)):
        if missing := sorted(required.difference(frame.columns)):
            raise KeyError(f"{name} predictions missing columns: {missing}")
    left = m2.loc[m2.split_role.eq(split_role) & m2.model_evaluated.astype(bool)]
    right = m2d.loc[m2d.split_role.eq(split_role) & m2d.model_evaluated.astype(bool)]
    merged = left[["session_id", "trial_index", "per_trial_negative_log_likelihood"]].merge(
        right[["session_id", "trial_index", "per_trial_negative_log_likelihood"]],
        on=["session_id", "trial_index"], suffixes=("_m2", "_m2d"), validate="one_to_one",
    )
    merged["delta_bits_m2d_vs_m2"] = (
        merged.per_trial_negative_log_likelihood_m2 - merged.per_trial_negative_log_likelihood_m2d
    ) / np.log(2.0)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-predictions", required=True, type=Path)
    parser.add_argument("--m2-dynamic-block-length-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-role", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    m2 = pd.read_csv(args.m2_predictions)
    m2d = pd.read_csv(args.m2_dynamic_block_length_predictions)
    merged = _paired(m2, m2d, split_role=args.split_role)
    merged.to_csv(output / "paired_trial_comparison.csv", index=False)
    per_session = merged.groupby("session_id", observed=False).agg(
        n_valid_trials=("delta_bits_m2d_vs_m2", "size"),
        total_delta_bits_m2d_vs_m2=("delta_bits_m2d_vs_m2", "sum"),
        mean_delta_bits_per_trial_m2d_vs_m2=("delta_bits_m2d_vs_m2", "mean"),
    ).reset_index()
    per_session.to_csv(output / "paired_session_comparison.csv", index=False)
    session_gain = per_session.mean_delta_bits_per_trial_m2d_vs_m2.to_numpy(float)
    summary = {
        "split_role": args.split_role,
        "session_count": int(len(per_session)),
        "valid_trial_count": int(merged.shape[0]),
        "total_delta_bits_m2_dynamic_block_length_vs_m2": float(merged.delta_bits_m2d_vs_m2.sum()),
        "mean_delta_bits_per_trial_m2_dynamic_block_length_vs_m2": float(merged.delta_bits_m2d_vs_m2.mean()),
        "session_level_mean_delta_bits_per_trial": float(np.mean(session_gain)) if len(session_gain) else float("nan"),
        "session_level_standard_error_per_trial": (
            float(np.std(session_gain, ddof=1) / np.sqrt(len(session_gain))) if len(session_gain) > 1 else float("nan")
        ),
    }
    (output / "paired_model_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
