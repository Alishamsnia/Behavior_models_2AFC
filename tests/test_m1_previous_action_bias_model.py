"""Integration tests for M1_previous_action_bias_v1's registry, parameters,
session wrapper, and pooled objective around the shared
``sensory_previous_bias_core_v1`` evaluator.

The readout/mixture/mask correctness itself is covered in
``test_sensory_previous_bias_core.py``; these tests verify that this
package decodes ``eta_logit`` correctly, wires ``history_driver="action"``
into the core, fixes alpha at exactly 1, and preserves autograd and
session-order invariance for the pooled objective.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m1_previous_action_bias_v1 import (
    M1SessionTrials,
    M1_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER,
    decode_m1_previous_action_bias_raw_parameters,
    encode_m1_previous_action_bias_constrained_parameters,
    evaluate_m1_previous_action_bias_mouse_objective,
    evaluate_m1_previous_action_bias_session,
    load_m1_previous_action_bias_model_registry,
)
from sensory_previous_bias_core_v1 import evaluate_sensory_previous_bias_session


DTYPE = torch.float64
NAN = float("nan")


def _session(durations, choices, categories, feedback, *, session_id="s1", role="train"):
    return M1SessionTrials(
        mouse_id="mouse", session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


class RegistryTests(unittest.TestCase):
    def test_registry_loads_for_every_subject(self) -> None:
        for suffix in ("", "_mc06", "_mc07"):
            registry = load_m1_previous_action_bias_model_registry(
                ROOT / "config" / f"m1_previous_action_bias_v1{suffix}.json"
            )
            self.assertEqual(registry.model_id, "M1_previous_action_bias_v1")
            self.assertEqual(registry.parameter_order, M1_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER)
            self.assertEqual(len(registry.initializations), 8)
            self.assertEqual(
                [round(start.eta, 12) for start in registry.initializations],
                [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
            )
            self.assertNotIn("no_probability_normalization_errors", registry.convergence.required_checks)


class ParameterTests(unittest.TestCase):
    def test_parameter_round_trip_and_eta_constraints(self) -> None:
        registry = load_m1_previous_action_bias_model_registry()
        for start in registry.initializations:
            raw = start.raw_tensor(lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon)
            decoded = decode_m1_previous_action_bias_raw_parameters(
                raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            )
            self.assertAlmostEqual(float(decoded.eta), start.eta, places=12)
            self.assertEqual(decoded.alpha, 1.0)
            re_encoded = encode_m1_previous_action_bias_constrained_parameters(
                decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            )
            torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)

    def test_rejects_infinite_bias_logit_and_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            encode_m1_previous_action_bias_constrained_parameters(
                torch.tensor([0.04, 0.5, 0.1, 0.1, 1.0, float("inf")], dtype=DTYPE)
            )
        with self.assertRaises(ValueError):
            decode_m1_previous_action_bias_raw_parameters(torch.zeros(5, dtype=DTYPE))


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_previous_action_bias_model_registry()
        cls.parameters = decode_m1_previous_action_bias_raw_parameters(
            cls.registry.initializations[3].raw_tensor(lapse_max=cls.registry.lapse_max)
        )

    def test_matches_last_valid_action_and_mixture(self) -> None:
        session = _session(
            durations=[0.5, 2.0, 0.6, 2.1, 0.4],
            choices=[1.0, 0.0, NAN, 1.0, 0.0],
            categories=[0.0, 1.0, 0.0, 0.0, 1.0],  # disagree with choices
            feedback=[False] * 5,                   # ignored by the action driver
        )
        evaluation = evaluate_m1_previous_action_bias_session(session, self.parameters, self.registry.task_moments)
        self.assertEqual(evaluation.predictions.history_state.tolist(), [0.5, 1.0, 0.0, 0.0, 1.0])
        core = evaluate_sensory_previous_bias_session(
            session, self.parameters.common(), self.parameters.eta, self.registry.task_moments, "action"
        )
        torch.testing.assert_close(
            evaluation.predictions.probability_long, core.predictions.probability_long, equal_nan=True
        )
        self.assertIs(evaluation.parameters, self.parameters)

    def test_pooled_objective_is_differentiable_and_session_order_invariant(self) -> None:
        sessions = (
            _session([0.5, 2.0, 0.6], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [True] * 3, session_id="a"),
            _session([2.2, 0.4], [1.0, 0.0], [1.0, 0.0], [True] * 2, session_id="b"),
        )
        raw = self.registry.initializations[0].raw_tensor(lapse_max=self.registry.lapse_max).requires_grad_(True)
        forward = evaluate_m1_previous_action_bias_mouse_objective(sessions, raw, self.registry.task_moments)
        backward = evaluate_m1_previous_action_bias_mouse_objective(sessions[::-1], raw, self.registry.task_moments)
        torch.testing.assert_close(forward.loss, backward.loss)
        forward.loss.backward()
        self.assertTrue(torch.all(torch.isfinite(raw.grad)))
        self.assertNotEqual(float(raw.grad[5]), 0.0)

    def test_rejects_test_split_sessions(self) -> None:
        raw = self.registry.initializations[0].raw_tensor(lapse_max=self.registry.lapse_max)
        with self.assertRaises(ValueError):
            evaluate_m1_previous_action_bias_mouse_objective(
                (_session([0.5], [1.0], [0.0], [True], role="test"),), raw, self.registry.task_moments
            )


if __name__ == "__main__":
    unittest.main()
