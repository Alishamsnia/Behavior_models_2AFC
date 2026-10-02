#!/usr/bin/env python3
"""Generate M2_dynamic_previous_action_bias diagnostic figures from an
existing trial-predictions export (no refitting)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m2_dynamic_previous_action_bias_v1.analysis import build_m2_dynamic_previous_action_bias_hazard_curve  # noqa: E402
from m2_dynamic_previous_action_bias_v1.figures import generate_m2_dynamic_previous_action_bias_diagnostic_figures  # noqa: E402
from m2_dynamic_previous_action_bias_v1.registry import load_m2_dynamic_previous_action_bias_model_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-dynamic-action-kernel-bias-predictions", required=True, type=Path)
    parser.add_argument("--raw-parameters", required=True, help="comma-separated nine fitted values")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    registry = load_m2_dynamic_previous_action_bias_model_registry(args.registry)
    predictions = pd.read_csv(args.m2_dynamic_previous_action_bias_predictions)
    raw = [float(value) for value in args.raw_parameters.split(",")]
    if len(raw) != 9:
        parser.error("--raw-parameters must contain exactly nine comma-separated values")
    hazard_curve = build_m2_dynamic_previous_action_bias_hazard_curve(raw, registry)
    manifest = generate_m2_dynamic_previous_action_bias_diagnostic_figures(predictions, hazard_curve, args.output_dir)
    print(json.dumps({"status": "complete", **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
