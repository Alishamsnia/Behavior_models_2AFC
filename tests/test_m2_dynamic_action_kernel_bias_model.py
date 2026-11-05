"""Scientific tests for M2_dynamic_action_kernel_bias_v1
(sensory_learnRatio_learnDynamicBlockSwitch_actionHistory_lapse).

Correctness is anchored to the two exact nesting boundaries, both
constructed explicitly because the free sigmoid decoders cannot reach them:

- alpha = 1 reproduces ``M2_dynamic_previous_action_bias_v1`` bit-for-bit;
- eta = 0 reproduces ``M2_dynamic_block_length_v1``;

plus an independent reference for the fitted-alpha action-history
recursion, causal/masking hand checks, finite-difference gradients over all
ten coordinates, and the frozen registry/start-provenance contract.
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model.m1_session import M1SessionTrials
from m2_dynamic_block_length_v1.parameters import M2DynamicBlockLengthParameters
from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_session
from m2_dynamic_action_kernel_bias_v1.parameters import (
    M2_DYNAMIC_ACTION_KERNEL_BIAS_PARAMETER_ORDER,
    decode_m2_dynamic_action_kernel_bias_raw_parameters,
    encode_m2_dynamic_action_kernel_bias_constrained_parameters,
)
from m2_dynamic_action_kernel_bias_v1.registry import load_m2_dynamic_action_kernel_bias_model_registry
from m2_dynamic_action_kernel_bias_v1.session import (
    evaluate_m2_dynamic_action_kernel_bias_mouse_objective,
    evaluate_m2_dynamic_action_kernel_bias_session,
)
from m2_dynamic_previous_action_bias_v1.parameters import decode_m2_dynamic_previous_action_bias_raw_parameters
from m2_dynamic_previous_action_bias_v1.registry import load_m2_dynamic_previous_action_bias_model_registry
from m2_dynamic_previous_action_bias_v1.session import evaluate_m2_dynamic_previous_action_bias_session


DTYPE = torch.float64
NAN = float("nan")
REGISTRY = load_m2_dynamic_action_kernel_bias_model_registry()
MOMENTS = REGISTRY.task_moments


def _session(duration, choice, category, feedback, *, session_id="s", role="train") -> M1SessionTrials:
    return M1SessionTrials(
        mouse_id="m", session_id=session_id,
        duration_s=torch.tensor(duration, dtype=DTYPE),
        choice_long=torch.tensor(choice, dtype=DTYPE),
        category_long=torch.tensor(category, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


def _random_session(seed: int, n: int = 30, *, role: str = "train", session_id: str = "s") -> M1SessionTrials:
    rng = random.Random(seed)
    category = [float(rng.randint(0, 1)) for _ in range(n)]
    duration = [rng.uniform(0.2, 0.8) if c == 0 else rng.uniform(1.7, 2.3) for c in category]
    choice = [float(rng.randint(0, 1)) if rng.random() > 0.1 else NAN for _ in range(n)]
    feedback = [c == c for c in choice]
    return _session(duration, choice, category, feedback, role=role, session_id=session_id)


def _raw(eta: float = 0.3, alpha: float = 0.4) -> torch.Tensor:
    start = REGISTRY.initializations[2].constrained_tensor()
    start[8] = torch.logit(torch.tensor(eta, dtype=DTYPE))
    start[9] = torch.logit(torch.tensor(alpha, dtype=DTYPE))
    return encode_m2_dynamic_action_kernel_bias_constrained_parameters(start)


def _evaluate(session, parameters):
    return evaluate_m2_dynamic_action_kernel_bias_session(session, parameters, MOMENTS)


class RegistryTests(unittest.TestCase):
    def test_registry_loads_for_every_subject_with_ten_coordinates(self) -> None:
        for suffix in ("", "_mc06", "_mc07"):
            registry = load_m2_dynamic_action_kernel_bias_model_registry(
                ROOT / "config" / f"m2_dynamic_action_kernel_bias_v1{suffix}.json"
            )
            self.assertEqual(registry.parameter_order, M2_DYNAMIC_ACTION_KERNEL_BIAS_PARAMETER_ORDER)
            self.assertEqual(len(registry.parameter_order), 10)
            self.assertEqual(len(registry.initializations), 8)
            self.assertIn("tail_occupancy_acceptable", registry.convergence.required_checks)
            self.assertIn("no_probability_normalization_errors", registry.convergence.required_checks)
            self.assertEqual(len(registry.multi_start_selection.parameter_distance_scale), 10)

    def test_starts_preserve_parent_rows_and_reverse_learning_rates(self) -> None:
        parent = load_m2_dynamic_previous_action_bias_model_registry()
        alphas = [round(start.alpha, 12) for start in REGISTRY.initializations]
        self.assertEqual(alphas, [0.9, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02])
        for mine, theirs in zip(REGISTRY.initializations, parent.initializations, strict=True):
            torch.testing.assert_close(
                mine.constrained_tensor()[:9], theirs.constrained_tensor(), rtol=0, atol=0
            )


class ParameterTests(unittest.TestCase):
    def test_parameter_round_trip_and_bounds(self) -> None:
        for start in REGISTRY.initializations:
            raw = start.raw_tensor(lapse_max=REGISTRY.lapse_max)
            decoded = decode_m2_dynamic_action_kernel_bias_raw_parameters(raw)
            self.assertTrue(0.0 < float(decoded.alpha) < 1.0 and 0.0 < float(decoded.eta) < 1.0)
            self.assertAlmostEqual(float(decoded.alpha), start.alpha, places=12)
            torch.testing.assert_close(
                encode_m2_dynamic_action_kernel_bias_constrained_parameters(decoded.as_tensor()),
                raw, rtol=1e-9, atol=1e-10,
            )

    def test_rejects_infinite_logits_and_wrong_shape(self) -> None:
        bad = REGISTRY.initializations[0].constrained_tensor()
        bad[9] = float("inf")
        with self.assertRaises(ValueError):
            encode_m2_dynamic_action_kernel_bias_constrained_parameters(bad)
        with self.assertRaises(ValueError):
            decode_m2_dynamic_action_kernel_bias_raw_parameters(torch.zeros(9, dtype=DTYPE))
        with self.assertRaises(ValueError):
            decode_m2_dynamic_action_kernel_bias_raw_parameters(_raw()).with_explicit_alpha(1.5)


class NestingTests(unittest.TestCase):
    def test_m2_dynamic_action_kernel_bias_at_alpha_one_equals_previous_action_bias(self) -> None:
        for seed in (1, 2, 3):
            session = _random_session(seed)
            raw = _raw(eta=0.35, alpha=0.6)
            mine = decode_m2_dynamic_action_kernel_bias_raw_parameters(raw).with_explicit_alpha(1.0)
            theirs = decode_m2_dynamic_previous_action_bias_raw_parameters(raw[:9])
            a = _evaluate(session, mine)
            b = evaluate_m2_dynamic_previous_action_bias_session(session, theirs, MOMENTS)
            self.assertTrue(torch.equal(a.negative_log_likelihood, b.negative_log_likelihood))
            self.assertTrue(torch.equal(a.predictions.action_history_state, b.predictions.action_history_state))
            torch.testing.assert_close(
                a.predictions.probability_long, b.predictions.probability_long, rtol=0, atol=0, equal_nan=True
            )

    def test_eta_zero_reproduces_m2_dynamic_block_length(self) -> None:
        session = _random_session(11)
        parameters = decode_m2_dynamic_action_kernel_bias_raw_parameters(_raw(alpha=0.3)).with_explicit_eta(0.0)
        block_length = M2DynamicBlockLengthParameters(
            w=parameters.w, s0=parameters.s0, ell_short=parameters.ell_short,
            ell_long=parameters.ell_long, kappa=parameters.kappa, ah=parameters.ah,
            delta0=parameters.delta0, bh=parameters.bh,
        )
        a = _evaluate(session, parameters)
        b = evaluate_m2_dynamic_block_length_session(session, block_length, MOMENTS)
        torch.testing.assert_close(a.predictions.probability_long, b.predictions.probability_long, equal_nan=True)
        torch.testing.assert_close(a.negative_log_likelihood, b.negative_log_likelihood)


class ActionHistoryTests(unittest.TestCase):
    def test_matches_reference_recurrence_on_valid_actions_only(self) -> None:
        rng = random.Random(4)
        for _ in range(5):
            session = _random_session(rng.randint(0, 10_000), n=25)
            alpha = rng.uniform(0.05, 0.95)
            parameters = decode_m2_dynamic_action_kernel_bias_raw_parameters(_raw(alpha=alpha))
            alpha = float(parameters.alpha)
            expected, pi = [], 0.5
            for choice in session.choice_long.tolist():
                expected.append(pi)
                if choice in (0.0, 1.0):
                    pi = pi + alpha * (choice - pi)
            actual = _evaluate(session, parameters).predictions.action_history_state.tolist()
            for e, a in zip(expected, actual, strict=True):
                self.assertAlmostEqual(e, a, places=12)

    def test_feedback_and_duration_independence_and_invalid_choice(self) -> None:
        session = _session(
            duration=[0.5, NAN, 2.0, 0.6], choice=[1.0, 1.0, NAN, 0.0],
            category=[0.0, 0.0, 1.0, 1.0], feedback=[False, True, True, False],
        )
        parameters = decode_m2_dynamic_action_kernel_bias_raw_parameters(_raw(alpha=0.5))
        state = _evaluate(session, parameters).predictions.action_history_state.tolist()
        # Valid choice with invalid feedback (trial 0) and invalid duration
        # (trial 1) still update; the omission (trial 2) leaves pi unchanged.
        self.assertEqual(state, [0.5, 0.75, 0.875, 0.875])

    def test_no_current_trial_leakage(self) -> None:
        base = _session([0.5, 2.0, 0.6, 2.1], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [True] * 4)
        flip = _session([0.5, 2.0, 0.6, 2.1], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [True] * 4)
        parameters = decode_m2_dynamic_action_kernel_bias_raw_parameters(_raw())
        a, b = _evaluate(base, parameters), _evaluate(flip, parameters)
        torch.testing.assert_close(a.predictions.probability_long[:2], b.predictions.probability_long[:2])
        self.assertNotAlmostEqual(
            float(a.predictions.probability_long[2]), float(b.predictions.probability_long[2])
        )


class ObjectiveTests(unittest.TestCase):
    def test_finite_difference_gradient_all_ten_coordinates(self) -> None:
        sessions = (_random_session(21, n=18),)
        raw0 = _raw()

        def loss(raw: torch.Tensor) -> torch.Tensor:
            return evaluate_m2_dynamic_action_kernel_bias_mouse_objective(sessions, raw, MOMENTS).loss

        raw = raw0.clone().requires_grad_(True)
        loss(raw).backward()
        h = 1e-6
        for index in range(10):
            step = torch.zeros(10, dtype=DTYPE)
            step[index] = h
            with torch.no_grad():
                fd = (loss(raw0 + step) - loss(raw0 - step)) / (2 * h)
            self.assertAlmostEqual(float(raw.grad[index]), float(fd), places=5,
                                   msg=M2_DYNAMIC_ACTION_KERNEL_BIAS_PARAMETER_ORDER[index])
        self.assertNotEqual(float(raw.grad[9]), 0.0)

    def test_session_order_invariance_and_shape_and_test_split_guards(self) -> None:
        sessions = (_random_session(1, 12, session_id="a"), _random_session(2, 15, session_id="b"))
        raw = _raw()
        forward = evaluate_m2_dynamic_action_kernel_bias_mouse_objective(sessions, raw, MOMENTS)
        backward = evaluate_m2_dynamic_action_kernel_bias_mouse_objective(sessions[::-1], raw, MOMENTS)
        torch.testing.assert_close(forward.loss, backward.loss)
        with self.assertRaises(ValueError):
            evaluate_m2_dynamic_action_kernel_bias_mouse_objective(sessions, raw[:9], MOMENTS)
        with self.assertRaises(ValueError):
            evaluate_m2_dynamic_action_kernel_bias_mouse_objective(
                (_random_session(3, 5, role="test"),), raw, MOMENTS
            )


if __name__ == "__main__":
    unittest.main()
