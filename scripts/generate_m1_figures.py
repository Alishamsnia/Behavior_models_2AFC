#!/usr/bin/env python3
"""Generate M1 analysis outputs from predictions or a completed fit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.m1_analysis import (  # noqa: E402
    build_m1_trial_predictions,
    save_m1_analysis_outputs,
)
from m0_sensory_v1.m1_figures import generate_m1_diagnostic_figures  # noqa: E402
from m0_sensory_v1.m1_m0_style_figures import (  # noqa: E402
    generate_m1_m0_style_figures,
)
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m0_sensory_v1.pipeline import (  # noqa: E402
    prepare_locked_test_analysis_trials,
    prepare_registered_m1_fit_data,
)


def parse_args() -> argparse.Namespace:
    """Parse either standalone-prediction or completed-fit analysis options."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--m1-predictions",
        type=Path,
        help="Existing M1 trial_predictions.csv (figure-only mode).",
    )
    source.add_argument(
        "--results-dir",
        type=Path,
        help="Completed M1 fit directory (rebuild-analysis mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Figure destination. Required in figure-only mode; defaults to "
            "RESULTS_DIR/m1_figures in rebuild-analysis mode."
        ),
    )
    parser.add_argument("--m0-predictions", type=Path)
    parser.add_argument("--raw-data-root", type=Path)
    parser.add_argument("--fit-summary", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help=(
            "Explicitly open and evaluate frozen test sessions. This is only "
            "valid with --results-dir after the analysis plan is frozen."
        ),
    )
    args = parser.parse_args()
    if args.m1_predictions is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --m1-predictions")
        if args.unlock_test or args.raw_data_root is not None:
            parser.error(
                "--unlock-test and --raw-data-root require completed-fit mode"
            )
    elif args.raw_data_root is None:
        parser.error("--raw-data-root is required with --results-dir")
    return args


def _resolve_m1_registry_path(
    requested: Path | None,
    fit_summary: dict,
) -> Path:
    """Resolve an explicit M1 registry or infer it from the fitted mouse ID."""

    if requested is not None:
        return requested.expanduser().resolve()
    mouse_id = str(fit_summary.get("mouse_id", "")).strip()
    if not mouse_id:
        raise ValueError("Fit summary does not contain a mouse_id")
    subject_registry = (
        REPOSITORY_ROOT / "config" / f"m1_hmm_true_v1_{mouse_id.lower()}.json"
    )
    if subject_registry.is_file():
        return subject_registry.resolve()
    if mouse_id == "MC08":
        return (REPOSITORY_ROOT / "config" / "m1_hmm_true_v1.json").resolve()
    raise FileNotFoundError(
        f"Could not infer an M1 registry for {mouse_id}; pass --registry explicitly"
    )


def _load_optional_m0_predictions(path: Path | None) -> pd.DataFrame | None:
    """Load optional paired M0 predictions for the registered M1 comparisons."""

    if path is None:
        return None
    return pd.read_csv(path.expanduser().resolve())


def _generate_from_completed_fit(args: argparse.Namespace) -> dict:
    """Rebuild M1 predictions without refitting and optionally unlock test data."""

    results = args.results_dir.expanduser().resolve()
    summary_path = (
        args.fit_summary.expanduser().resolve()
        if args.fit_summary is not None
        else results / "starts" / "multistart_summary.json"
    )
    fit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winner_id = fit_summary.get("winner_start_id")
    if winner_id is None:
        raise RuntimeError("The M1 fit summary has no converged winning start")
    winner = next(
        run for run in fit_summary["runs"] if run["start_id"] == winner_id
    )
    registry_path = _resolve_m1_registry_path(args.registry, fit_summary)
    observed_registry_sha = sha256(registry_path.read_bytes()).hexdigest()
    expected_registry_sha = winner.get("registry_config_sha256")
    if observed_registry_sha != expected_registry_sha:
        raise ValueError(
            "Analysis registry does not match the winning M1 fit: "
            f"expected {expected_registry_sha}, observed {observed_registry_sha} "
            f"at {registry_path}"
        )
    registry = load_m1_model_registry(registry_path)

    print(
        f"Loading {fit_summary['mouse_id']} M1 timelines with registry={registry_path}.",
        flush=True,
    )
    prepared = prepare_registered_m1_fit_data(args.raw_data_root, registry)
    canonical = prepared.canonical_trials
    if args.unlock_test:
        print("Explicit test unlock requested: opening frozen test timelines.", flush=True)
        test = prepare_locked_test_analysis_trials(
            args.raw_data_root, registry, unlock_test=True
        )
        canonical = pd.concat([canonical, test], ignore_index=True)
    else:
        print("Test remains locked; rebuilding train/validation outputs only.", flush=True)

    predictions = build_m1_trial_predictions(
        canonical,
        winner["best_raw_parameters"],
        registry,
        allow_test=args.unlock_test,
    )
    output_summary = save_m1_analysis_outputs(
        predictions,
        fit_summary,
        results,
        test_unlocked=args.unlock_test,
    )
    figure_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results / "m1_figures"
    )
    manifest = generate_m1_diagnostic_figures(
        predictions,
        figure_directory,
        m0_predictions=_load_optional_m0_predictions(args.m0_predictions),
    )
    m0_style_directory = results / "m1_m0_style_figures"
    generate_m1_m0_style_figures(
        predictions,
        fit_summary,
        m0_style_directory,
        test_unlocked=args.unlock_test,
    )
    return {
        "status": "complete",
        "trial_table": str(results / "trial_predictions.csv"),
        "valid_decisions": output_summary["saved_valid_decision_count"],
        "figure_directory": str(figure_directory),
        "m0_style_figure_directory": str(m0_style_directory),
        "test_unlocked": args.unlock_test,
        "m0_comparison_included": manifest["m0_comparison_included"],
    }


def main() -> int:
    """Run the selected M1 analysis mode without changing fitted parameters."""

    args = parse_args()
    if args.results_dir is not None:
        print(json.dumps(_generate_from_completed_fit(args), indent=2), flush=True)
        return 0

    manifest = generate_m1_diagnostic_figures(
        pd.read_csv(args.m1_predictions.expanduser().resolve()),
        args.output_dir,
        m0_predictions=_load_optional_m0_predictions(args.m0_predictions),
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
