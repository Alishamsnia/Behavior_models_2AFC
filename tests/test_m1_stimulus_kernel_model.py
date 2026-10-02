"""Scientific and integration tests for M1_stimulus_kernel_v1.

These tests are the primary correctness anchor for this model: like
M1_known_switch_v1, M1_stimulus_kernel_v1 has no exact-nesting boundary to
an already-verified parent (M0/M1), because its recursion mechanism (a
scalar delta rule with no discrete hidden state) is structurally different
from M1's hazard-driven HMM. Correctness is instead established here by
(a) an independently-derived closed-form exponential-moving-average
reference, cross-checked against the recursive implementation across many
random category/feedback sequences and learning rates, and (b) direct
hand-checkable cases.
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

from m0_sensory_v1.registry import load_model_registry
from m1_stimulus_kernel_v1 import (
    M1SessionTrials,
    decode_m1_stimulus_kernel_raw_parameters,
    encode_m1_stimulus_kernel_constrained_parameters,
    evaluate_m1_stimulus_kernel_mouse_objective,
    evaluate_m1_stimulus_kernel_session,
    load_m1_stimulus_kernel_model_registry,
)


DTYPE = torch.float64


def _session(
    categories,
    choices,
    feedback,
    durations,
    *,
    session_id="s1",
    role="train",
) -> M1SessionTrials:
    return M1SessionTrials(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


def _raw_with_a(base_raw: torch.Tensor, a_value: float) -> torch.Tensor:
    raw = base_raw.clone()
    raw[5] = a_value
    return raw


def _reference_exponential_kernel(categories, feedback, alpha: float, *, initial_pi: float = 0.5):
    """Independent, non-vectorized reference: plain Python delta-rule loop."""

    trajectory = []
    pi = initial_pi
    for category, valid in zip(categories, feedback):
        trajectory.append(pi)
        if valid:
            pi = pi + alpha * (category - pi)
    return trajectory


class M1StimulusKernelParameterTests(unittest.TestCase):
    def test_parameter_round_trip_and_alpha_constraints(self) -> None:
        registry = load_model_registry()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        raw = torch.cat([raw, torch.tensor([1.2345], dtype=DTYPE)])
        decoded = decode_m1_stimulus_kernel_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        self.assertGreater(float(decoded.alpha), 0.0)
        self.assertLess(float(decoded.alpha), 1.0)
        re_encoded = encode_m1_stimulus_kernel_constrained_parameters(
            decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)

    def test_rejects_infinite_learning_rate_logit(self) -> None:
        registry = load_model_registry()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        constrained = torch.cat([raw, torch.tensor([float("inf")], dtype=DTYPE)])
        with self.assertRaises(ValueError):
            encode_m1_stimulus_kernel_constrained_parameters(constrained)


class M1StimulusKernelSessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )
        cls.raw6 = torch.cat([cls.raw, torch.tensor([0.3], dtype=DTYPE)])

    def _parameters(self, a_value: float):
        raw = _raw_with_a(self.raw6, a_value)
        return decode_m1_stimulus_kernel_raw_parameters(
            raw, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        )

    def _evaluate(self, session, a_value: float = 0.3):
        return evaluate_m1_stimulus_kernel_session(
            session, self._parameters(a_value), self.registry.task_moments
        )

    def test_first_trial_prior_is_one_half(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 1.0],
            choices=[1.0, 0.0, 1.0],
            feedback=[True, True, True],
            durations=[0.5, 2.0, 0.5],
        )
        evaluation = self._evaluate(session)
        self.assertAlmostEqual(float(evaluation.predictions.prior_long[0]), 0.5, places=12)

    def test_matches_closed_form_exponential_kernel_across_random_sequences(self) -> None:
        rng = random.Random(20260909)
        for trial in range(30):
            length = rng.randint(1, 15)
            categories = [float(rng.randint(0, 1)) for _ in range(length)]
            feedback = [rng.random() < 0.8 for _ in range(length)]
            choices = [float(rng.randint(0, 1)) for _ in range(length)]
            durations = [rng.choice([0.5, 2.0]) for _ in range(length)]
            a_value = rng.uniform(-4.0, 4.0)
            alpha = float(torch.sigmoid(torch.tensor(a_value, dtype=DTYPE)))

            session = _session(categories, choices, feedback, durations)
            evaluation = self._evaluate(session, a_value=a_value)
            actual = evaluation.predictions.prior_long.detach().tolist()
            expected = _reference_exponential_kernel(categories, feedback, alpha)
            for got, want in zip(actual, expected):
                self.assertAlmostEqual(got, want, places=9, msg=(trial, a_value))

    def test_alpha_near_zero_freezes_prior_near_one_half(self) -> None:
        session = _session(
            categories=[1.0, 1.0, 1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0, 1.0, 1.0],
            feedback=[True, True, True, True, True],
            durations=[0.5] * 5,
        )
        evaluation = self._evaluate(session, a_value=-15.0)
        for value in evaluation.predictions.prior_long.detach().tolist():
            self.assertAlmostEqual(value, 0.5, places=4)

    def test_alpha_near_one_closely_tracks_previous_category(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 1.0, 1.0],
            choices=[1.0, 0.0, 1.0, 1.0],
            feedback=[True, True, True, True],
            durations=[0.5] * 4,
        )
        evaluation = self._evaluate(session, a_value=15.0)
        prior = evaluation.predictions.prior_long.detach().tolist()
        self.assertAlmostEqual(prior[0], 0.5, places=6)
        self.assertAlmostEqual(prior[1], 1.0, places=4)
        self.assertAlmostEqual(prior[2], 0.0, places=4)
        self.assertAlmostEqual(prior[3], 1.0, places=4)

    def test_missing_feedback_retains_prior_exactly(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 0.0, 1.0],
            choices=[1.0, 0.0, 0.0, 1.0],
            feedback=[True, True, False, True],
            durations=[0.5] * 4,
        )
        evaluation = self._evaluate(session)
        prior = evaluation.predictions.prior_long
        post = evaluation.predictions.post_update_prior_long
        # Trial 2 has no feedback: pi_3 (prior at index 2's successor) must
        # equal the post-update value from trial 1 unchanged.
        torch.testing.assert_close(post[1], post[2])
        torch.testing.assert_close(prior[3], post[2])

    def test_no_current_trial_leakage(self) -> None:
        base_categories = [0.0, 1.0, 0.0]
        durations = [0.5, 0.6, 0.55]
        choices = [0.0, 1.0, 0.0]
        feedback = [True, True, True]
        session_a = _session(base_categories, choices, feedback, durations)
        flipped_categories = [0.0, 1.0, 1.0]  # trial-2 category flipped
        session_b = _session(flipped_categories, choices, feedback, durations)
        eval_a = self._evaluate(session_a)
        eval_b = self._evaluate(session_b)
        # prior_long at trial 2 (index 2) depends only on trials 0-1, which
        # are identical between the two sessions, so it must match exactly
        # even though trial 2's own category differs.
        torch.testing.assert_close(
            eval_a.predictions.prior_long[2], eval_b.predictions.prior_long[2]
        )
        # But the post-update value at trial 2 (which uses trial 2's own
        # category) must differ.
        self.assertNotAlmostEqual(
            float(eval_a.predictions.post_update_prior_long[2]),
            float(eval_b.predictions.post_update_prior_long[2]),
        )

    def test_gradient_finite_difference(self) -> None:
        session = _session(
            categories=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            choices=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            feedback=[True, True, True, False, True, True],
            durations=[0.5, 0.6, 0.55, 2.1, 1.9, 0.45],
        )
        raw = self.raw6.clone().requires_grad_(True)

        def loss_fn(vector: torch.Tensor) -> torch.Tensor:
            parameters = decode_m1_stimulus_kernel_raw_parameters(
                vector, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
            )
            return evaluate_m1_stimulus_kernel_session(
                session, parameters, self.registry.task_moments
            ).negative_log_likelihood

        loss = loss_fn(raw)
        loss.backward()
        analytic = raw.grad.detach().clone()

        step = 1e-6
        numeric = torch.zeros_like(analytic)
        for index in range(raw.numel()):
            bump = torch.zeros_like(raw)
            bump[index] = step
            with torch.no_grad():
                plus = loss_fn(self.raw6 + bump)
                minus = loss_fn(self.raw6 - bump)
            numeric[index] = (plus - minus) / (2 * step)
        torch.testing.assert_close(analytic, numeric, rtol=1e-3, atol=1e-4)

    def test_pooled_objective_session_order_invariance(self) -> None:
        first = _session(
            categories=[0.0, 0.0, 1.0],
            choices=[0.0, 0.0, 1.0],
            feedback=[True, True, True],
            durations=[0.5, 0.5, 2.0],
            session_id="a",
        )
        second = _session(
            categories=[1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0],
            feedback=[True, True, True],
            durations=[2.0, 2.1, 1.9],
            session_id="b",
        )
        forward = evaluate_m1_stimulus_kernel_mouse_objective(
            [first, second], self.raw6, self.registry.task_moments
        )
        backward = evaluate_m1_stimulus_kernel_mouse_objective(
            [second, first], self.raw6, self.registry.task_moments
        )
        torch.testing.assert_close(forward.loss, backward.loss)

    def test_session_boundary_resets_prior_to_one_half(self) -> None:
        drifting = _session(
            categories=[1.0, 1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0, 1.0],
            feedback=[True, True, True, True],
            durations=[0.5] * 4,
            session_id="drift",
        )
        fresh = _session(
            categories=[0.0],
            choices=[0.0],
            feedback=[True],
            durations=[0.5],
            session_id="fresh",
        )
        result = evaluate_m1_stimulus_kernel_mouse_objective(
            [drifting, fresh], self.raw6, self.registry.task_moments
        )
        fresh_evaluation = next(
            record for record in result.session_evaluations if record.session_id == "fresh"
        )
        self.assertAlmostEqual(
            float(fresh_evaluation.predictions.prior_long[0]), 0.5, places=12
        )

    def test_pooled_trials_vs_equal_sessions_weighting_differ_with_unequal_counts(self) -> None:
        short = _session(
            categories=[0.0, 0.0],
            choices=[0.0, 0.0],
            feedback=[True, True],
            durations=[0.5, 0.5],
            session_id="short",
        )
        long_session = _session(
            categories=[1.0, 1.0, 1.0, 0.0, 0.0],
            choices=[1.0, 1.0, 1.0, 0.0, 0.0],
            feedback=[True, True, True, True, True],
            durations=[2.0, 2.1, 1.9, 0.5, 0.5],
            session_id="long",
        )
        pooled = evaluate_m1_stimulus_kernel_mouse_objective(
            [short, long_session], self.raw6, self.registry.task_moments,
            objective_weighting="pooled_trials",
        )
        equal = evaluate_m1_stimulus_kernel_mouse_objective(
            [short, long_session], self.raw6, self.registry.task_moments,
            objective_weighting="equal_sessions",
        )
        session_nll = torch.stack(
            [record.negative_log_likelihood for record in pooled.session_evaluations]
        )
        counts = torch.tensor(
            [record.valid_trial_count for record in pooled.session_evaluations], dtype=DTYPE
        )
        expected_pooled = session_nll.sum() / counts.sum()
        expected_equal = torch.mean(session_nll / counts)
        torch.testing.assert_close(pooled.loss, expected_pooled)
        torch.testing.assert_close(equal.loss, expected_equal)
        self.assertNotAlmostEqual(float(pooled.loss), float(equal.loss), places=6)

    def test_invalid_choice_contributes_no_likelihood_but_updates_state(self) -> None:
        with_omission = _session(
            categories=[0.0, 1.0, 0.0, 0.0],
            choices=[0.0, float("nan"), 0.0, 0.0],
            feedback=[True, True, True, True],
            durations=[0.5, 0.5, 0.5, 0.5],
        )
        evaluation = self._evaluate(with_omission, a_value=0.3)
        self.assertEqual(evaluation.valid_trial_count, 3)
        self.assertTrue(bool(torch.isnan(evaluation.per_trial_negative_log_likelihood[1])))
        # Feedback on the omitted trial (index 1, category=1) must still
        # apply the full update, not be silently skipped.
        alpha = float(self._parameters(0.3).alpha)
        expected = _reference_exponential_kernel(
            [0.0, 1.0, 0.0, 0.0], [True, True, True, True], alpha
        )
        actual = evaluation.predictions.prior_long.detach().tolist()
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=10)

    def test_all_invalid_choices_session(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 1.0],
            choices=[float("nan"), float("nan"), float("nan")],
            feedback=[True, True, True],
            durations=[0.5, 0.6, 0.55],
        )
        evaluation = self._evaluate(session)
        self.assertEqual(evaluation.valid_trial_count, 0)
        self.assertTrue(bool(torch.all(torch.isnan(evaluation.per_trial_negative_log_likelihood))))
        self.assertAlmostEqual(float(evaluation.negative_log_likelihood), 0.0, places=12)
        # Prior trajectory still evolves from feedback even with no choices.
        alpha = float(self._parameters(0.3).alpha)
        expected = _reference_exponential_kernel([1.0, 0.0, 1.0], [True, True, True], alpha)
        actual = evaluation.predictions.prior_long.detach().tolist()
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=10)

    def test_single_trial_session(self) -> None:
        session = _session(
            categories=[1.0], choices=[1.0], feedback=[True], durations=[0.5],
        )
        evaluation = self._evaluate(session)
        self.assertEqual(evaluation.valid_trial_count, 1)
        self.assertAlmostEqual(float(evaluation.predictions.prior_long[0]), 0.5, places=12)

    def test_no_feedback_anywhere_keeps_constant_one_half_prior(self) -> None:
        session = _session(
            categories=[1.0, 0.0, 1.0, 0.0],
            choices=[1.0, 0.0, 1.0, 0.0],
            feedback=[False, False, False, False],
            durations=[0.5] * 4,
        )
        evaluation = self._evaluate(session)
        torch.testing.assert_close(
            evaluation.predictions.prior_long, torch.full((4,), 0.5, dtype=DTYPE)
        )

    def test_rejects_test_split_sessions(self) -> None:
        test_session = _session(
            categories=[0.0, 0.0], choices=[0.0, 0.0], feedback=[True, True],
            durations=[0.5, 0.5], role="test",
        )
        with self.assertRaises(ValueError):
            evaluate_m1_stimulus_kernel_mouse_objective(
                [test_session], self.raw6, self.registry.task_moments
            )


class M1StimulusKernelRegistryTests(unittest.TestCase):
    def test_registry_loads_and_preserves_m0_starts(self) -> None:
        registry = load_m1_stimulus_kernel_model_registry()
        self.assertEqual(registry.model_id, "M1_stimulus_kernel_v1")
        self.assertEqual(
            registry.parameter_order, ("w", "s0", "ell_short", "ell_long", "kappa", "a")
        )
        self.assertEqual(registry.initial_pi, 0.5)
        self.assertEqual(len(registry.initializations), 8)
        alphas = [start.alpha for start in registry.initializations]
        self.assertEqual(len(set(round(value, 6) for value in alphas)), 8)
        for start in registry.initializations:
            self.assertGreater(start.alpha, 0.0)
            self.assertLess(start.alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
