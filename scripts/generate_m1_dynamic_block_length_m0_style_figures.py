#!/usr/bin/env python3
"""Generate the five-family M0-style figure suite for
M1_dynamic_block_length from an existing predictions CSV + fit summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m1_dynamic_block_length_v1.m0_style_figures import generate_m1_dynamic_block_length_m0_style_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dynamic-block-length-predictions", required=True, type=Path)
    parser.add_argument("--fit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.m1_dynamic_block_length_predictions)
    fit_summary = json.loads(args.fit_summary.read_text())
    test_unlocked = "test" in set(predictions["split_role"].astype(str))
    manifest = generate_m1_dynamic_block_length_m0_style_figures(
        predictions, fit_summary, args.output_dir, test_unlocked=test_unlocked,
    )
    print(json.dumps({"status": "complete", **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
