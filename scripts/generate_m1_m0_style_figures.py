#!/usr/bin/env python3
"""Generate all M0-style figures from an existing completed M1 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.m1_m0_style_figures import (  # noqa: E402
    generate_m1_m0_style_figures,
)


def parse_args() -> argparse.Namespace:
    """Parse paths to an existing M1 prediction table and fit summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-predictions", required=True, type=Path)
    parser.add_argument("--fit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Create M0-equivalent M1 figures without loading raw data or refitting."""

    args = parse_args()
    predictions_path = args.m1_predictions.expanduser().resolve()
    summary_path = args.fit_summary.expanduser().resolve()
    predictions = pd.read_csv(predictions_path)
    fit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_unlocked = "test" in set(predictions["split_role"].astype(str))
    manifest = generate_m1_m0_style_figures(
        predictions,
        fit_summary,
        args.output_dir,
        test_unlocked=test_unlocked,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "model_id": manifest["model_id"],
                "mouse_id": manifest["mouse_id"],
                "test_unlocked": manifest["test_unlocked"],
                "output_directory": str(args.output_dir.expanduser().resolve()),
                "figure_families": manifest["m0_figure_families_reproduced"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
