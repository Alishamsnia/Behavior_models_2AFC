#!/usr/bin/env python3
"""Compare aligned held-out M4 and M3 trial predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.analysis import build_m4_m3_validation_comparison  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired held-out M4-versus-M3 comparison")
    parser.add_argument("--m4-predictions", required=True, type=Path)
    parser.add_argument("--m3-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-role", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    result = build_m4_m3_validation_comparison(
        pd.read_csv(args.m4_predictions), pd.read_csv(args.m3_predictions),
        split_role=args.split_role,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.per_session.to_csv(args.output, index=False)
    stem = args.output.stem
    for name in ("overall", "calibration", "context", "transition_aligned", "rare_common"):
        getattr(result, name).to_csv(args.output.parent / f"{stem}_{name}.csv", index=False)
    print(result.overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
