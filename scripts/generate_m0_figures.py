#!/usr/bin/env python3
"""Generate trial exports and preliminary M0 figures from a completed fit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.analysis import (  # noqa: E402
    build_trial_predictions,
    generate_preliminary_figures,
    save_model_analysis_outputs,
)
from m0_sensory_v1.pipeline import (  # noqa: E402
    prepare_locked_test_analysis_trials,
    prepare_registered_fit_data,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M0 figures without refitting the model."
    )
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--fit-summary", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help=(
            "Explicitly open and evaluate the frozen test sessions. Omit this "
            "until the model and analysis plan are frozen."
        ),
    )
    return parser.parse_args()


def _resolve_registry_path(
    requested: Path | None,
    fit_summary: dict,
) -> Path:
    """Resolve an explicit registry or infer it from the fitted mouse ID."""

    if requested is not None:
        return requested.expanduser().resolve()
    mouse_id = str(fit_summary.get("mouse_id", "")).strip()
    if not mouse_id:
        raise ValueError("Fit summary does not contain a mouse_id")
    subject_registry = REPOSITORY_ROOT / "config" / f"m0_sensory_v1_{mouse_id.lower()}.json"
    if subject_registry.is_file():
        return subject_registry.resolve()
    if mouse_id == "MC08":
        return (REPOSITORY_ROOT / "config" / "m0_sensory_v1.json").resolve()
    raise FileNotFoundError(
        f"Could not infer a registry for {mouse_id}; pass --registry explicitly"
    )


def main() -> int:
    args = parse_args()
    results = args.results_dir.expanduser().resolve()
    summary_path = (
        args.fit_summary.expanduser().resolve()
        if args.fit_summary is not None
        else results / "starts" / "multistart_summary.json"
    )
    fit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winner_id = fit_summary.get("winner_start_id")
    if winner_id is None:
        raise RuntimeError("The fit summary has no converged winning start")
    winner = next(
        run for run in fit_summary["runs"] if run["start_id"] == winner_id
    )
    registry_path = _resolve_registry_path(args.registry, fit_summary)
    observed_registry_sha = sha256(registry_path.read_bytes()).hexdigest()
    expected_registry_sha = winner.get("registry_config_sha256")
    if observed_registry_sha != expected_registry_sha:
        raise ValueError(
            "Analysis registry does not match the winning fit: "
            f"expected {expected_registry_sha}, observed {observed_registry_sha} "
            f"at {registry_path}"
        )
    registry = load_model_registry(registry_path)

    print(
        f"Loading {fit_summary['mouse_id']} timelines with registry={registry_path}.",
        flush=True,
    )
    prepared = prepare_registered_fit_data(args.raw_data_root, registry)
    canonical = prepared.canonical_trials
    if args.unlock_test:
        print("Explicit test unlock requested: opening frozen test timelines.", flush=True)
        test = prepare_locked_test_analysis_trials(
            args.raw_data_root, registry, unlock_test=True
        )
        import pandas as pd

        canonical = pd.concat([canonical, test], ignore_index=True)
    else:
        print("Test remains locked; test panels will be marked LOCKED.", flush=True)

    predictions = build_trial_predictions(
        canonical,
        winner["best_raw_parameters"],
        registry,
        allow_test=args.unlock_test,
    )
    output_summary = save_model_analysis_outputs(
        predictions, fit_summary, results, test_unlocked=args.unlock_test
    )
    manifest = generate_preliminary_figures(
        predictions,
        fit_summary,
        results / "preliminary_figures",
        test_unlocked=args.unlock_test,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "trial_table": str(results / "trial_predictions.csv"),
                "valid_decisions": output_summary["saved_valid_decision_count"],
                "figure_directory": str(results / "preliminary_figures"),
                "test_unlocked": manifest["test_unlocked"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
