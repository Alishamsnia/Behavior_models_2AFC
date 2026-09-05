#!/usr/bin/env python3
"""Generate all M0-style figures from completed M3 prediction and fit files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.m0_style_figures import generate_m3_m0_style_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-predictions", required=True, type=Path)
    parser.add_argument("--fit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.m3_predictions.expanduser().resolve())
    summary = json.loads(args.fit_summary.expanduser().resolve().read_text(encoding="utf-8"))
    manifest = generate_m3_m0_style_figures(
        predictions, summary, args.output_dir,
        test_unlocked="test" in set(predictions.split_role.astype(str)),
    )
    print(json.dumps({
        "status": "complete", "model_id": manifest["model_id"],
        "figure_families": manifest["m0_figure_families_reproduced"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
