#!/usr/bin/env python3
"""Generate all M0-style figures from completed M4 prediction and fit files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.m0_style_figures import (  # noqa: E402
    generate_m4_m0_style_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-predictions", required=True, type=Path)
    parser.add_argument("--fit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.m4_predictions.expanduser().resolve())
    summary = json.loads(
        args.fit_summary.expanduser().resolve().read_text(encoding="utf-8")
    )
    manifest = generate_m4_m0_style_figures(
        predictions,
        summary,
        args.output_dir,
        test_unlocked="test" in set(predictions["split_role"].astype(str)),
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
