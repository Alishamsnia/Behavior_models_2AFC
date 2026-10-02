"""Integration tests for Extended_WSLS_v1's registry, parameters, session,
and fit wrappers around the shared ``wsls_core_v1`` evaluator.

The recurrence/masking/lapse correctness itself is covered exhaustively in
``test_wsls_core.py``; these tests instead verify that this package's thin
wrappers decode parameters correctly, wire the fitted alpha=sigmoid(a)
into the shared core, and preserve autograd/session-order invariance for
the pooled objective -- exactly mirroring what ``test_pure_wsls_model.py``
verifies for the sibling fixed-alpha model.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from extended_wsls_v1 import (
    M1SessionTrials,
    decode_extended_wsls_raw_parameters,
    encode_extended_wsls_constrained_parameters,
    evaluate_extended_wsls_mouse_objective,
    evaluate_extended_wsls_session,
    load_extended_wsls_model_registry,
)


DTYPE = torch.float64


def _session(categories, choices, feedback, *, session_id="s1", role="train") -> M1SessionTrials:
    n = len(categories)
    return M1SessionTrials(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.ones(n, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


class ExtendedWslsRegistryTests(unittest.TestCase):
    def test_registry_loads_and_validates(self) -> None:
        registry = load_extended_wsls_model_registry()
        self.assertEqual(registry.model_id, "Extended_WSLS_v1")
        self.assertEqual(registry.parameter_order, ("ell_short", "ell_long", "a"))
        self.assertEqual(len(registry.initializations), 8)
        self.assertEqual(registry.initial_pi, 0.5)


class ExtendedWslsParameterTests(unittest.TestCase):
    def test_parameter_round_trip_and_alpha_constraints(self) -> None:
        registry = load_extended_wsls_model_registry()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        decoded = decode_extended_wsls_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        self.assertGreater(float(decoded.alpha), 0.0)
        self.assertLess(float(decoded.alpha), 1.0)
        re_encoded = encode_extended_wsls_constrained_parameters(
            decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)

    def test_rejects_infinite_learning_rate_logit(self) -> None:
        with self.assertRaises(ValueError):
            encode_extended_wsls_constrained_parameters(
                torch.tensor([0.1, 0.1, float("inf")], dtype=DTYPE)
            )

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            decode_extended_wsls_raw_parameters(torch.zeros(2, dtype=DTYPE))


class ExtendedWslsSessionEvaluationTests(unittest.TestCase):
    def _parameters(self, registry, a_value: float):
        raw = torch.tensor([0.0, 0.0, a_value], dtype=DTYPE)
        return decode_extended_wsls_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )

    def test_matches_shared_core_closed_form(self) -> None:
        registry = load_extended_wsls_model_registry()
        categories = [1.0, 0.0, 1.0, 1.0, 0.0]
        session = _session(
            categories=categories,
            choices=[1.0, 0.0, 1.0, 0.0, 1.0],
            feedback=[True, True, True, True, True],
        )
        parameters = self._parameters(registry, a_value=-0.5)
        evaluation = evaluate_extended_wsls_session(session, parameters, registry.task_moments)
        alpha = float(parameters.alpha)
        pi = 0.5
        expected = [pi]
        for category in categories[:-1]:
            pi = pi + alpha * (category - pi)
            expected.append(pi)
        torch.testing.assert_close(
            evaluation.predictions.prior_long, torch.tensor(expected, dtype=DTYPE)
        )

    def test_alpha_near_zero_flattens_prior_to_one_half(self) -> None:
        """Identifiability sanity check: as alpha->0, pi_t stays near 0.5
        for every trial (the flat-likelihood region flagged in the plan)."""

        registry = load_extended_wsls_model_registry()
        session = _session(
            categories=[1.0, 1.0, 1.0, 1.0],
            choices=[1.0, 0.0, 1.0, 0.0],
            feedback=[True, True, True, True],
        )
        parameters = self._parameters(registry, a_value=-15.0)  # alpha ~ 3e-7
        evaluation = evaluate_extended_wsls_session(session, parameters, registry.task_moments)
        for value in evaluation.predictions.prior_long.tolist():
            self.assertAlmostEqual(value, 0.5, places=4)


class ExtendedWslsMouseObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_extended_wsls_model_registry()
        self.raw = self.registry.initializations[0].raw_tensor(
            lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        ).clone().requires_grad_(True)

    def test_pooled_objective_is_differentiable_and_session_order_invariant(self) -> None:
        session_a = _session([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [True, True, True], session_id="a")
        session_b = _session([0.0, 1.0], [1.0, 1.0], [True, True], session_id="b")

        forward = evaluate_extended_wsls_mouse_objective(
            (session_a, session_b), self.raw, self.registry.task_moments,
            lapse_max=self.registry.lapse_max, parameter_epsilon=self.registry.parameter_epsilon,
        )
        backward = evaluate_extended_wsls_mouse_objective(
            (session_b, session_a), self.raw, self.registry.task_moments,
            lapse_max=self.registry.lapse_max, parameter_epsilon=self.registry.parameter_epsilon,
        )
        self.assertAlmostEqual(
            float(forward.total_negative_log_likelihood.detach()),
            float(backward.total_negative_log_likelihood.detach()),
            places=12,
        )
        forward.loss.backward()
        self.assertTrue(torch.all(torch.isfinite(self.raw.grad)))

    def test_rejects_test_split_role(self) -> None:
        session = _session([1.0], [1.0], [True], role="test")
        with self.assertRaises(ValueError):
            evaluate_extended_wsls_mouse_objective(
                (session,), self.raw, self.registry.task_moments,
            )


if __name__ == "__main__":
    unittest.main()
