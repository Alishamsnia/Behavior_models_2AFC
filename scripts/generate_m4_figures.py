#!/usr/bin/env python3
"""Generate complete M4 diagnostics from a completed fit, optionally unlocking test."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.pipeline import prepare_locked_test_analysis_trials  # noqa: E402
from m4_hsmm_delta_v1.analysis import build_m4_hazard_curve, build_m4_trial_predictions  # noqa: E402
from m4_hsmm_delta_v1.diagnostics import (  # noqa: E402
    build_m4_implied_duration_distribution,
    evaluate_m4_truncation_sensitivity,
    summarize_m4_hazard_duration,
    summarize_m4_prediction_diagnostics,
)
from m4_hsmm_delta_v1.figures import generate_m4_diagnostic_figures  # noqa: E402
from m4_hsmm_delta_v1.identifiability import evaluate_m4_dynamic_surface, profile_m4_dynamic_surface  # noqa: E402
from m4_hsmm_delta_v1.m0_style_figures import generate_m4_m0_style_figures  # noqa: E402
from m4_hsmm_delta_v1.pipeline import prepare_registered_m4_fit_data  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402


def _winner(summary: dict) -> dict:
    if "runs" not in summary:
        if "best_raw_parameters" not in summary:
            raise ValueError("Fit result contains neither runs nor best_raw_parameters")
        return summary
    winner_id = summary.get("winner_start_id")
    if winner_id is None:
        raise RuntimeError("Fit summary has no accepted converged winner")
    return next(run for run in summary["runs"] if run["start_id"] == winner_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", required=True, type=Path)
    parser.add_argument("--fit-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=ROOT / "config" / "m4_hsmm_delta_v1.json")
    parser.add_argument("--m3-predictions", type=Path)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--include-full-run-length-distributions", action="store_true")
    parser.add_argument("--identifiability-grid-points", type=int, default=15)
    parser.add_argument("--profile-identifiability", action="store_true")
    parser.add_argument("--profile-lbfgs-steps", type=int, default=40)
    args = parser.parse_args()
    summary = json.loads(args.fit_result.expanduser().resolve().read_text(encoding="utf-8"))
    winner = _winner(summary)
    registry_path = args.registry.expanduser().resolve()
    expected = winner.get("registry_config_sha256")
    observed = sha256(registry_path.read_bytes()).hexdigest()
    if expected is not None and expected != observed:
        raise ValueError(f"Analysis registry hash mismatch: expected {expected}, observed {observed}")
    registry = load_m4_model_registry(registry_path)
    prepared = prepare_registered_m4_fit_data(args.raw_data_root, registry)
    canonical = prepared.canonical_trials
    if args.unlock_test:
        test = prepare_locked_test_analysis_trials(args.raw_data_root, registry, unlock_test=True)
        canonical = pd.concat((canonical, test), ignore_index=True)
    raw = winner["best_raw_parameters"]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = build_m4_trial_predictions(
        canonical, raw, registry, allow_test=args.unlock_test,
        include_full_run_length_distributions=args.include_full_run_length_distributions,
    )
    predictions.to_csv(output / "trial_predictions.csv", index=False)
    hazard = build_m4_hazard_curve(raw, registry)
    hazard.to_csv(output / "hazard_curve.csv", index=False)
    build_m4_implied_duration_distribution(raw, registry).to_csv(
        output / "implied_duration_distribution.csv", index=False
    )
    summarize_m4_prediction_diagnostics(predictions).to_csv(
        output / "prediction_diagnostics_by_session.csv", index=False
    )
    duration_summary = summarize_m4_hazard_duration(raw, registry).as_dict()
    (output / "hazard_duration_summary.json").write_text(
        json.dumps(duration_summary, indent=2) + "\n", encoding="utf-8"
    )
    raw_tensor = torch.tensor(raw, dtype=torch.float64)
    for role, sessions in (
        ("train", prepared.training_sessions),
        ("validation", prepared.validation_sessions),
    ):
        evaluate_m4_truncation_sensitivity(sessions, raw, registry).to_csv(
            output / f"truncation_sensitivity_{role}.csv", index=False
        )
    surfaces = []
    for pair in (("ah", "bh"), ("alpha_theta", "bh")):
        function = profile_m4_dynamic_surface if args.profile_identifiability else evaluate_m4_dynamic_surface
        options = {"maximum_lbfgs_steps": args.profile_lbfgs_steps} if args.profile_identifiability else {}
        surface = function(
            prepared.training_sessions, raw_tensor, registry, pair=pair,
            grid_points=args.identifiability_grid_points, **options,
        )
        surface.to_csv(output / f"identifiability_{pair[0]}_{pair[1]}.csv", index=False)
        surfaces.append(surface)
    m3 = None if args.m3_predictions is None else pd.read_csv(args.m3_predictions)
    manifest = generate_m4_diagnostic_figures(
        predictions, hazard, output / "figures", m3_predictions=m3,
        identifiability_surfaces=tuple(surfaces),
    )
    m0_style_manifest = generate_m4_m0_style_figures(
        predictions,
        summary,
        output / "m0_style_figures",
        test_unlocked=args.unlock_test,
    )
    final = {
        "status": "complete", "model_id": registry.model_id,
        "test_unlocked": args.unlock_test,
        "locked_test_files_opened": bool(args.unlock_test),
        "model_selection_uses_test": False,
        "map_optimization_used": False,
        "full_run_length_distributions_included": args.include_full_run_length_distributions,
        "duration_summary": duration_summary,
        "figure_manifest": manifest,
        "m0_style_figure_manifest": m0_style_manifest,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
