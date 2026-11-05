"""Integration tests for M1_previous_wsls_bias_v1's registry, parameters,
session wrapper, and pooled objective around the shared
``sensory_previous_bias_core_v1`` evaluator.

The readout/mixture/mask correctness itself is covered in
``test_sensory_previous_bias_core.py``; these tests verify that this
package wires ``history_driver="category"`` (gate: ``feedback_valid``) into
the core, and that it agrees with / diverges from the action-driven sibling
``m1_previous_action_bias_v1`` exactly when it should.
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
    decode_m1_previous_action_bias_raw_parameters,
    evaluate_m1_previous_action_bias_session,
    load_m1_previous_action_bias_model_registry,
)
from m1_previous_wsls_bias_v1 import (
    M1SessionTrials,
    M1_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER,
    decode_m1_previous_wsls_bias_raw_parameters,
    encode_m1_previous_wsls_bias_constrained_parameters,
    evaluate_m1_previous_wsls_bias_mouse_objective,
    evaluate_m1_previous_wsls_bias_session,
    load_m1_previous_wsls_bias_model_registry,
)


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
            registry = load_m1_previous_wsls_bias_model_registry(
                ROOT / "config" / f"m1_previous_wsls_bias_v1{suffix}.json"
            )
            self.assertEqual(registry.model_id, "M1_previous_wsls_bias_v1")
            self.assertEqual(registry.parameter_order, M1_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER)
            self.assertEqual(
                [round(start.eta, 12) for start in registry.initializations],
                [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
            )

    def test_shares_every_start_with_the_action_sibling(self) -> None:
        wsls = load_m1_previous_wsls_bias_model_registry()
        action = load_m1_previous_action_bias_model_registry()
        for a, b in zip(wsls.initializations, action.initializations, strict=True):
            torch.testing.assert_close(a.constrained_tensor(), b.constrained_tensor(), rtol=0, atol=0)


class ParameterTests(unittest.TestCase):
    def test_parameter_round_trip(self) -> None:
        registry = load_m1_previous_wsls_bias_model_registry()
        raw = registry.initializations[5].raw_tensor(lapse_max=registry.lapse_max)
        decoded = decode_m1_previous_wsls_bias_raw_parameters(raw, lapse_max=registry.lapse_max)
        self.assertEqual(decoded.alpha, 1.0)
        torch.testing.assert_close(
            encode_m1_previous_wsls_bias_constrained_parameters(decoded.as_tensor(), lapse_max=registry.lapse_max),
            raw, rtol=1e-9, atol=1e-10,
        )


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_previous_wsls_bias_model_registry()
        cls.raw = cls.registry.initializations[3].raw_tensor(lapse_max=cls.registry.lapse_max)
        cls.parameters = decode_m1_previous_wsls_bias_raw_parameters(cls.raw)
        cls.action_registry = load_m1_previous_action_bias_model_registry()
        cls.action_parameters = decode_m1_previous_action_bias_raw_parameters(cls.raw)

    def test_matches_last_feedback_valid_category_and_mixture(self) -> None:
        session = _session(
            durations=[0.5, 2.0, 0.6, 2.1, 0.4],
            choices=[1.0, NAN, 1.0, 0.0, 1.0],     # trial 1 omitted; choices never drive D
            categories=[0.0, 1.0, NAN, 1.0, 0.0],
            feedback=[True, True, False, True, True],
        )
        evaluation = evaluate_m1_previous_wsls_bias_session(session, self.parameters, self.registry.task_moments)
        self.assertEqual(evaluation.predictions.history_state.tolist(), [0.5, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(evaluation.predictions.history_update_mask.tolist(), [True, True, False, True, True])

    def test_equals_action_sibling_when_choices_equal_categories(self) -> None:
        """When every choice equals its category and feedback mirrors choice
        validity (as in every registered real dataset), both gates and both
        drivers coincide, so the two models must be identical."""

        session = _session(
            durations=[0.5, 2.0, 0.6, 2.1, 1.2, 0.4],
            choices=[0.0, 1.0, NAN, 1.0, 1.0, 0.0],
            categories=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            feedback=[True, True, False, True, True, True],
        )
        wsls = evaluate_m1_previous_wsls_bias_session(session, self.parameters, self.registry.task_moments)
        action = evaluate_m1_previous_action_bias_session(
            session, self.action_parameters, self.action_registry.task_moments
        )
        self.assertTrue(torch.equal(wsls.predictions.history_state, action.predictions.history_state))
        self.assertTrue(torch.equal(wsls.negative_log_likelihood, action.negative_log_likelihood))

    def test_diverges_from_action_sibling_when_choices_differ(self) -> None:
        session = _session([0.5, 2.0, 0.6], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [True] * 3)
        wsls = evaluate_m1_previous_wsls_bias_session(session, self.parameters, self.registry.task_moments)
        action = evaluate_m1_previous_action_bias_session(
            session, self.action_parameters, self.action_registry.task_moments
        )
        self.assertEqual(wsls.predictions.history_state.tolist(), [0.5, 0.0, 1.0])
        self.assertEqual(action.predictions.history_state.tolist(), [0.5, 1.0, 0.0])

    def test_pooled_objective_is_differentiable_and_session_order_invariant(self) -> None:
        sessions = (
            _session([0.5, 2.0, 0.6], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [True] * 3, session_id="a"),
            _session([2.2, 0.4], [1.0, 0.0], [1.0, 0.0], [True] * 2, session_id="b"),
        )
        raw = self.raw.clone().requires_grad_(True)
        forward = evaluate_m1_previous_wsls_bias_mouse_objective(sessions, raw, self.registry.task_moments)
        backward = evaluate_m1_previous_wsls_bias_mouse_objective(sessions[::-1], raw, self.registry.task_moments)
        torch.testing.assert_close(forward.loss, backward.loss)
        forward.loss.backward()
        self.assertTrue(torch.all(torch.isfinite(raw.grad)))

    def test_rejects_test_split_sessions(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_m1_previous_wsls_bias_mouse_objective(
                (_session([0.5], [1.0], [0.0], [True], role="test"),), self.raw, self.registry.task_moments
            )


if __name__ == "__main__":
    unittest.main()
