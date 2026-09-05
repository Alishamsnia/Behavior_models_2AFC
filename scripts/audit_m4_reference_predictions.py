#!/usr/bin/env python3
"""Recompute and audit the deterministic M4 release prediction fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.parameters import decode_m4_raw_parameters  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402
from m4_hsmm_delta_v1.session import M4SessionTrials, evaluate_m4_session  # noqa: E402


def main() -> int:
    expected = json.loads((ROOT / "config/releases/m4_reference_predictions_v2.json").read_text())
    registry = load_m4_model_registry()
    initialization = registry.initializations[expected["initialization_start_id"] - 1]
    parameters = decode_m4_raw_parameters(initialization.raw_tensor())
    duration = torch.tensor(expected["duration_s"], dtype=torch.float64)
    category = torch.tensor(expected["category_long"], dtype=torch.float64)
    session = M4SessionTrials(
        mouse_id="reference", session_id="toy", duration_s=duration,
        choice_long=category.clone(), category_long=category,
        feedback_valid=torch.ones(len(duration), dtype=torch.bool), split_role="train",
    )
    result = evaluate_m4_session(session, parameters, registry.task_moments)
    observed = {
        "negative_log_likelihood": result.negative_log_likelihood,
        "probability_long": result.predictions.probability_long,
        "prior_long": result.predictions.prior_long,
        "delta_pre": result.predictions.delta_pre,
        "expected_run_length_pre": result.predictions.expected_run_length_pre,
        "expected_hazard_pre": result.predictions.expected_hazard_pre,
        "tail_probability_pre": result.predictions.tail_probability_pre,
    }
    for name, value in observed.items():
        reference = torch.as_tensor(expected[name], dtype=torch.float64)
        torch.testing.assert_close(value.detach(), reference, rtol=0, atol=2e-15)
    print("M4 deterministic reference-prediction audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
