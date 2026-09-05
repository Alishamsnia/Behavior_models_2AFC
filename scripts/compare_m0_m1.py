#!/usr/bin/env python3
"""Compare aligned M0 and M1 trial predictions on a frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model_comparison import (  # noqa: E402
    compare_m1_to_m0_predictions,
    save_model_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-predictions", required=True, type=Path)
    parser.add_argument("--m1-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()
    result = compare_m1_to_m0_predictions(
        pd.read_csv(args.m0_predictions),
        pd.read_csv(args.m1_predictions),
        split_role=args.split,
        allow_test=args.unlock_test,
    )
    save_model_comparison(result, args.output)
    labels = [item[0] for item in result.per_session_delta_elpd]
    values = [item[1] for item in result.per_session_delta_elpd]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.35), 4))
    ax.bar(range(len(values)), values, color=["#2e7d32" if x >= 0 else "#c62828" for x in values])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="Session", ylabel="ΔELPD M1 − M0", title=f"Paired {args.split} comparison")
    ax.grid(axis="y", alpha=0.25)
    figure_path = args.output.with_suffix(".png")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
