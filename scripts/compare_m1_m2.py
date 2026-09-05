#!/usr/bin/env python3
"""Compare aligned M2 and M1 trial predictions on a frozen split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-predictions", required=True, type=Path)
    parser.add_argument("--m2-predictions", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    keys = ["mouse_id", "session_id", "trial_index", "split_role"]
    m1 = pd.read_csv(args.m1_predictions)
    m2 = pd.read_csv(args.m2_predictions)
    columns = keys + ["choice_long", "model_evaluated", "per_trial_negative_log_likelihood"]
    paired = m2[columns].merge(
        m1[columns], on=keys, suffixes=("_m2", "_m1"), validate="one_to_one"
    )
    paired = paired[paired["split_role"].eq(args.split)]
    if not paired["choice_long_m2"].equals(paired["choice_long_m1"]):
        raise ValueError("M1 and M2 observed choices disagree")
    if not paired["model_evaluated_m2"].equals(paired["model_evaluated_m1"]):
        raise ValueError("M1 and M2 valid-choice masks disagree")
    paired = paired[paired["model_evaluated_m2"].astype(bool)].copy()
    paired["delta_elpd_m2_minus_m1"] = paired["per_trial_negative_log_likelihood_m1"] - paired["per_trial_negative_log_likelihood_m2"]
    session = paired.groupby("session_id").delta_elpd_m2_minus_m1.sum()
    result = {
        "split": args.split,
        "n_valid_trials": int(len(paired)),
        "delta_elpd_m2_minus_m1": float(paired.delta_elpd_m2_minus_m1.sum()),
        "delta_bits_per_trial_m2_minus_m1": float(paired.delta_elpd_m2_minus_m1.sum() / (len(paired) * math.log(2))),
        "per_session_delta_elpd": {str(key): float(value) for key, value in session.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(max(7, len(session) * .3), 4))
    ax.bar(range(len(session)), session.values, color=["#2e7d32" if value >= 0 else "#c62828" for value in session.values])
    ax.axhline(0, color="black")
    ax.set(xlabel="Session", ylabel="ΔELPD M2 − M1", title=f"Paired {args.split} comparison")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
