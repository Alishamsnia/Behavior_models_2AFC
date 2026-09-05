#!/usr/bin/env python3
"""Generate M2 analysis outputs from predictions or a completed fit."""

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

from m0_sensory_v1.m2_analysis import build_m2_trial_predictions, save_m2_analysis_outputs  # noqa: E402
from m0_sensory_v1.m2_figures import (  # noqa: E402
    generate_m2_diagnostic_figures,
    generate_m2_identifiability_figure,
)
from m0_sensory_v1.m2_m0_style_figures import generate_m2_m0_style_figures  # noqa: E402
from m0_sensory_v1.m2_registry import load_m2_model_registry  # noqa: E402
from m0_sensory_v1.pipeline import prepare_locked_test_analysis_trials, prepare_registered_m2_fit_data  # noqa: E402
from m0_sensory_v1.model import evaluate_m2_hazard_contrast_surface  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--m2-predictions", type=Path, help="Existing M2 trial_predictions.csv")
    source.add_argument("--results-dir", type=Path, help="Completed M2 fit directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--m1-predictions", type=Path)
    parser.add_argument("--m0-predictions", type=Path)
    parser.add_argument("--raw-data-root", type=Path)
    parser.add_argument("--fit-summary", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument(
        "--identifiability-grid-points",
        type=int,
        default=0,
        help=(
            "Opt-in post-fit ah/delta0 objective slice; 0 skips this potentially "
            "expensive analysis and values >=3 set the grid size"
        ),
    )
    parser.add_argument("--identifiability-ah-half-width", type=float, default=2.0)
    parser.add_argument("--identifiability-delta0-half-width", type=float, default=1.25)
    args = parser.parse_args()
    if args.m2_predictions is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --m2-predictions")
        if args.unlock_test or args.raw_data_root is not None:
            parser.error("--unlock-test and --raw-data-root require completed-fit mode")
        if args.identifiability_grid_points:
            parser.error("identifiability analysis requires completed-fit mode")
    elif args.raw_data_root is None:
        parser.error("--raw-data-root is required with --results-dir")
    if args.identifiability_grid_points not in {0} and args.identifiability_grid_points < 3:
        parser.error("--identifiability-grid-points must be 0 or at least 3")
    if args.identifiability_ah_half_width <= 0 or args.identifiability_delta0_half_width <= 0:
        parser.error("identifiability half widths must be positive")
    return args


def _resolve_m2_registry_path(requested: Path | None, fit_summary: dict) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    mouse_id = str(fit_summary.get("mouse_id", "")).strip()
    if not mouse_id:
        raise ValueError("Fit summary does not contain a mouse_id")
    subject = REPOSITORY_ROOT / "config" / f"m2_hmm_static_v1_{mouse_id.lower()}.json"
    if subject.is_file():
        return subject.resolve()
    if mouse_id == "MC08":
        return (REPOSITORY_ROOT / "config" / "m2_hmm_static_v1.json").resolve()
    raise FileNotFoundError(f"Could not infer an M2 registry for {mouse_id}; pass --registry explicitly")


def _optional(path: Path | None) -> pd.DataFrame | None:
    return None if path is None else pd.read_csv(path.expanduser().resolve())


def _generate_from_completed_fit(args: argparse.Namespace) -> dict:
    results = args.results_dir.expanduser().resolve()
    summary_path = args.fit_summary.expanduser().resolve() if args.fit_summary is not None else results / "starts" / "multistart_summary.json"
    fit_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winner_id = fit_summary.get("winner_start_id")
    if winner_id is None:
        raise RuntimeError("The M2 fit summary has no converged winning start")
    winner = next(run for run in fit_summary["runs"] if run["start_id"] == winner_id)
    registry_path = _resolve_m2_registry_path(args.registry, fit_summary)
    observed_sha = sha256(registry_path.read_bytes()).hexdigest()
    expected_sha = winner.get("registry_config_sha256")
    if observed_sha != expected_sha:
        raise ValueError(f"Analysis registry does not match the winning M2 fit: expected {expected_sha}, observed {observed_sha} at {registry_path}")
    registry = load_m2_model_registry(registry_path)
    prepared = prepare_registered_m2_fit_data(args.raw_data_root, registry)
    canonical = prepared.canonical_trials
    if args.unlock_test:
        test = prepare_locked_test_analysis_trials(args.raw_data_root, registry, unlock_test=True)
        canonical = pd.concat([canonical, test], ignore_index=True)
    predictions = build_m2_trial_predictions(
        canonical, winner["best_raw_parameters"], registry, allow_test=args.unlock_test
    )
    output_summary = save_m2_analysis_outputs(
        predictions, fit_summary, results, test_unlocked=args.unlock_test
    )
    figure_directory = args.output_dir.expanduser().resolve() if args.output_dir is not None else results / "m2_figures"
    manifest = generate_m2_diagnostic_figures(
        predictions,
        figure_directory,
        m1_predictions=_optional(args.m1_predictions),
        m0_predictions=_optional(args.m0_predictions),
    )
    m0_style_directory = results / "m2_m0_style_figures"
    generate_m2_m0_style_figures(
        predictions, fit_summary, m0_style_directory, test_unlocked=args.unlock_test
    )
    identifiability_directory = None
    if args.identifiability_grid_points:
        surface = evaluate_m2_hazard_contrast_surface(
            prepared.training_sessions,
            torch.tensor(winner["best_raw_parameters"], dtype=torch.float64),
            registry,
            grid_points=args.identifiability_grid_points,
            ah_half_width=args.identifiability_ah_half_width,
            delta0_half_width=args.identifiability_delta0_half_width,
        )
        identifiability_directory = results / "m2_identifiability"
        generate_m2_identifiability_figure(surface, identifiability_directory)
    return {
        "status": "complete",
        "trial_table": str(results / "trial_predictions.csv"),
        "valid_decisions": output_summary["saved_valid_decision_count"],
        "figure_directory": str(figure_directory),
        "m0_style_figure_directory": str(m0_style_directory),
        "test_unlocked": args.unlock_test,
        "m1_comparison_included": manifest["m1_comparison_included"],
        "m0_comparison_included": manifest["m0_comparison_included"],
        "identifiability_directory": (
            None if identifiability_directory is None else str(identifiability_directory)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.results_dir is not None:
        print(json.dumps(_generate_from_completed_fit(args), indent=2), flush=True)
        return 0
    manifest = generate_m2_diagnostic_figures(
        pd.read_csv(args.m2_predictions.expanduser().resolve()),
        args.output_dir,
        m1_predictions=_optional(args.m1_predictions),
        m0_predictions=_optional(args.m0_predictions),
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
