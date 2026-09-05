#!/usr/bin/env python3
"""Prepare canonical and legacy-compatible behavioral CSV files for M0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data import (  # noqa: E402
    estimate_task_moments,
    prepare_dataset,
    validate_prepared_data,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit raw inputs and output-policy arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert behavioral MATLAB SessionData files into the canonical "
            "M0 trial table and the legacy-compatible valid-choice view."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Behavioral .mat files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-isi-bins", type=int, default=5)
    parser.add_argument("--isi-decimals", type=int, default=3)
    parser.add_argument(
        "--include-nonstandard-isi-sessions",
        action="store_true",
        help="Keep sessions that do not have exactly the legacy 5 short/5 long levels",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare, validate, and save data tables plus a compact manifest."""

    args = parse_args()
    result = prepare_dataset(
        args.inputs,
        require_legacy_isi_bins=not args.include_nonstandard_isi_sessions,
        expected_isi_bins=args.expected_isi_bins,
        isi_decimals=args.isi_decimals,
    )
    validate_prepared_data(result)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_trials.csv"
    model_path = output_dir / "m0_model_trials.csv"
    sessions_path = output_dir / "sessions_audit.csv"
    manifest_path = output_dir / "preparation_manifest.json"
    moments_path = output_dir / "task_moments.json"

    result.canonical_trials.to_csv(canonical_path, index=False)
    result.model_trials.to_csv(model_path, index=False)
    result.sessions.to_csv(sessions_path, index=False)
    task_moments = estimate_task_moments(
        result.canonical_trials,
        source="all presented category/duration pairs in canonical_trials.csv",
    )
    moments_path.write_text(
        json.dumps(task_moments.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "canonical_trial_rows": len(result.canonical_trials),
        "model_trial_rows": len(result.model_trials),
        "sessions_seen": len(result.sessions),
        "sessions_included": int(result.sessions["included_in_model_trials"].sum()),
        "legacy_isi_filter_required": not args.include_nonstandard_isi_sessions,
        "expected_isi_bins_per_category": args.expected_isi_bins,
        "isi_rounding_decimals": args.isi_decimals,
        "outputs": {
            "canonical_trials": str(canonical_path),
            "m0_model_trials": str(model_path),
            "sessions_audit": str(sessions_path),
            "task_moments": str(moments_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
