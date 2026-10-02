"""Integration tests for Pure_WSLS_v1's registry, parameters, session, and
fit wrappers around the shared ``wsls_core_v1`` evaluator.

The recurrence/masking/lapse correctness itself is covered exhaustively in
``test_wsls_core.py``; these tests instead verify that this package's thin
wrappers decode parameters correctly, wire alpha=1.0 fixed into the shared
core, and preserve autograd/session-order invariance for the pooled
objective -- exactly mirroring what
``test_m1_stimulus_kernel_model.py`` verifies for its own family.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pure_wsls_v1 import (
    M1SessionTrials,
    PURE_WSLS_FIXED_ALPHA,
    decode_pure_wsls_raw_parameters,
    encode_pure_wsls_constrained_parameters,
    evaluate_pure_wsls_mouse_objective,
    evaluate_pure_wsls_session,
    load_pure_wsls_model_registry,
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


class PureWslsRegistryTests(unittest.TestCase):
    def test_registry_loads_and_validates(self) -> None:
        registry = load_pure_wsls_model_registry()
        self.assertEqual(registry.model_id, "Pure_WSLS_v1")
        self.assertEqual(registry.parameter_order, ("ell_short", "ell_long"))
        self.assertEqual(len(registry.initializations), 8)
        self.assertEqual(registry.initial_pi, 0.5)


class PureWslsParameterTests(unittest.TestCase):
    def test_parameter_round_trip(self) -> None:
        registry = load_pure_wsls_model_registry()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        decoded = decode_pure_wsls_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        self.assertEqual(decoded.alpha, PURE_WSLS_FIXED_ALPHA)
        self.assertEqual(decoded.alpha, 1.0)
        re_encoded = encode_pure_wsls_constrained_parameters(
            decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            decode_pure_wsls_raw_parameters(torch.zeros(3, dtype=DTYPE))
        with self.assertRaises(ValueError):
            encode_pure_wsls_constrained_parameters(torch.zeros(3, dtype=DTYPE))

    def test_rejects_out_of_bounds_lapse(self) -> None:
        with self.assertRaises(ValueError):
            encode_pure_wsls_constrained_parameters(torch.tensor([1.5, 0.1], dtype=DTYPE))


class PureWslsSessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_pure_wsls_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )
        cls.parameters = decode_pure_wsls_raw_parameters(
            cls.raw, lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )

    def test_evaluate_pure_wsls_session_matches_previous_category_exactly(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 1.0, 1.0],
            choices=[1.0, 0.0, 1.0, 0.0],
            feedback=[True, True, True, True],
        )
        evaluation = evaluate_pure_wsls_session(session, self.parameters, self.registry.task_moments)
        expected = torch.tensor([0.5, 1.0, 0.0, 1.0], dtype=DTYPE)
        self.assertTrue(torch.equal(evaluation.predictions.prior_long, expected))
        self.assertIs(evaluation.parameters, self.parameters)

    def test_moments_argument_is_unused(self) -> None:
        """This model has no sensory readout: passing None as moments must
        not change anything (call-site parity only)."""

        session = _session([1.0, 0.0], [1.0, 0.0], [True, True])
        with_moments = evaluate_pure_wsls_session(session, self.parameters, self.registry.task_moments)
        without_moments = evaluate_pure_wsls_session(session, self.parameters, None)
        torch.testing.assert_close(
            with_moments.predictions.prior_long, without_moments.predictions.prior_long
        )


class PureWslsMouseObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_pure_wsls_model_registry()
        self.raw = self.registry.initializations[0].raw_tensor(
            lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        ).clone().requires_grad_(True)

    def test_pooled_objective_is_differentiable_and_session_order_invariant(self) -> None:
        session_a = _session([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [True, True, True], session_id="a")
        session_b = _session([0.0, 1.0], [1.0, 1.0], [True, True], session_id="b")

        forward = evaluate_pure_wsls_mouse_objective(
            (session_a, session_b), self.raw, self.registry.task_moments,
            lapse_max=self.registry.lapse_max, parameter_epsilon=self.registry.parameter_epsilon,
        )
        backward = evaluate_pure_wsls_mouse_objective(
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
            evaluate_pure_wsls_mouse_objective(
                (session,), self.raw, self.registry.task_moments,
            )

    def test_rejects_multiple_mice(self) -> None:
        session_a = _session([1.0], [1.0], [True])
        session_b = M1SessionTrials(
            mouse_id="other_mouse",
            session_id="s2",
            duration_s=torch.ones(1, dtype=DTYPE),
            choice_long=torch.tensor([1.0], dtype=DTYPE),
            category_long=torch.tensor([1.0], dtype=DTYPE),
            feedback_valid=torch.tensor([True]),
            split_role="train",
        )
        with self.assertRaises(ValueError):
            evaluate_pure_wsls_mouse_objective(
                (session_a, session_b), self.raw, self.registry.task_moments,
            )


if __name__ == "__main__":
    unittest.main()
