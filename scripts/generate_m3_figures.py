#!/usr/bin/env python3
"""Generate complete M3 diagnostics, with optional M0/M1/M2 comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.figures import generate_m3_diagnostic_figures  # noqa: E402


def _read(path: Path | None):
    return None if path is None else pd.read_csv(path.expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-predictions", required=True, type=Path)
    parser.add_argument("--m2-predictions", type=Path)
    parser.add_argument("--m1-predictions", type=Path)
    parser.add_argument("--m0-predictions", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = generate_m3_diagnostic_figures(
        _read(args.m3_predictions), args.output_dir,
        m2_predictions=_read(args.m2_predictions),
        m1_predictions=_read(args.m1_predictions),
        m0_predictions=_read(args.m0_predictions),
    )
    print(json.dumps({"status": "complete", **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
