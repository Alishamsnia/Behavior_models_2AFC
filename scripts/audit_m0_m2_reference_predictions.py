#!/usr/bin/env python3
"""Recompute and verify the deterministic M0/M1/M2 pre-M3 reference fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model.m0 import predict_m0_trials_with_parameters  # noqa: E402
from m0_sensory_v1.model.m1_parameters import M1DecodedParameters  # noqa: E402
from m0_sensory_v1.model.m1_session import M1SessionTrials, evaluate_m1_session  # noqa: E402
from m0_sensory_v1.model.m2_parameters import M2DecodedParameters  # noqa: E402
from m0_sensory_v1.model.m2_session import evaluate_m2_session  # noqa: E402
from m0_sensory_v1.model.parameters import DecodedParameters  # noqa: E402
from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402


def main() -> int:
    path = ROOT / "config/releases/m0_m1_m2_tied_neutral_hazard_reference_predictions_v2.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    fixture = reference["fixture"]
    values = [
        torch.tensor(value, dtype=torch.float64)
        for value in fixture["parameters_w_s0_ell_short_ell_long_kappa_ah_delta0"]
    ]
    w, s0, ell_short, ell_long, kappa, ah, delta0 = values
    hazard = torch.sigmoid(ah)
    duration = torch.tensor(fixture["duration_s"], dtype=torch.float64)
    choice = torch.tensor(fixture["choice_long"], dtype=torch.float64)
    session = M1SessionTrials(
        mouse_id="reference", session_id="reference_session", duration_s=duration,
        choice_long=choice, category_long=torch.tensor(fixture["category_long"], dtype=torch.float64),
        feedback_valid=torch.tensor(fixture["feedback_valid"], dtype=torch.bool), split_role="train",
    )
    moments = load_m3_model_registry().task_moments
    m0 = predict_m0_trials_with_parameters(
        duration, session.category_long, choice,
        DecodedParameters(
            w=w, s0=s0, ell_short=ell_short, ell_long=ell_long, kappa=kappa
        ),
        moments,
    )
    m1 = evaluate_m1_session(
        session, M1DecodedParameters(
            w=w, s0=s0, ell_short=ell_short, ell_long=ell_long, kappa=kappa,
            ah=ah, hazard=hazard,
        ), moments
    )
    m2 = evaluate_m2_session(
        session, M2DecodedParameters(
            w=w, s0=s0, ell_short=ell_short, ell_long=ell_long, kappa=kappa,
            ah=ah, delta0=delta0, hazard=hazard,
        ), moments
    )
    observed = {
        "M0_sensory_v1": {"probability_long": m0.probability_long},
        "M1_hmm_true_v1": {
            "probability_long": m1.predictions.probability_long,
            "prior_long": m1.predictions.prior_long,
            "pre_choice_block_probability": m1.predictions.pre_choice_block_probability,
            "negative_log_likelihood": m1.negative_log_likelihood,
        },
        "M2_hmm_static_v1": {
            "probability_long": m2.predictions.probability_long,
            "prior_long": m2.predictions.prior_long,
            "pre_choice_block_probability": m2.predictions.pre_choice_block_probability,
            "negative_log_likelihood": m2.negative_log_likelihood,
        },
    }
    tolerance = float(reference["absolute_tolerance"])
    for model, fields in observed.items():
        for field, value in fields.items():
            actual = value.detach().cpu().numpy().reshape(-1)
            expected = np.asarray(reference[model][field], dtype=float).reshape(-1)
            if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
                raise AssertionError(f"Frozen reference mismatch: {model}.{field}")
    print(json.dumps({"status": "passed", "artifact": reference["artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
