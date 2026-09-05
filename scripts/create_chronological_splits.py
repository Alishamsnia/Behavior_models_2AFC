#!/usr/bin/env python3
"""Create frozen per-mouse chronological session assignments from metadata."""

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

from m0_sensory_v1.data import (  # noqa: E402
    DEFAULT_SPLIT_VERSION,
    create_chronological_splits,
)


def parse_args() -> argparse.Namespace:
    """Parse metadata-only split inputs and output paths."""

    parser = argparse.ArgumentParser(
        description="Freeze per-mouse chronological 70/15/15 session splits."
    )
    parser.add_argument("sessions_audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-version", default=DEFAULT_SPLIT_VERSION)
    return parser.parse_args()


def main() -> None:
    """Create, save, and summarize deterministic split assignments."""

    args = parse_args()
    sessions = pd.read_csv(args.sessions_audit)
    assignments = create_chronological_splits(
        sessions,
        split_version=args.split_version,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    counts = (
        assignments.groupby(["mouse_id", "split_role"], observed=False)
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    print(
        json.dumps(
            {
                "split_version": args.split_version,
                "assignment_file": str(output),
                "counts_by_mouse": counts,
                "locked_test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
