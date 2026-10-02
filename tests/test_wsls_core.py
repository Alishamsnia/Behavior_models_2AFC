"""Scientific tests for the shared ``wsls_core_v1`` evaluator.

This is the primary correctness anchor for BOTH ``pure_wsls_v1`` and
``extended_wsls_v1``, since both models delegate their entire predict/
update/mask/likelihood computation to this one module. Correctness is
established by (a) an independently-derived closed-form exponential-moving-
average reference for the learned-alpha case, (b) an exact-equality check
for the fixed alpha=1.0 case, and (c) direct causal/masking hand checks.
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

from wsls_core_v1 import evaluate_wsls_session


DTYPE = torch.float64


def _tensors(categories, choices, feedback, durations=None):
    n = len(categories)
    if durations is None:
        durations = [1.0] * n
    return dict(
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
    )


def _reference_exponential_kernel(categories, feedback, alpha: float, *, initial_pi: float = 0.5):
    """Independent, non-vectorized reference: plain Python delta-rule loop."""

    trajectory = []
    pi = initial_pi
    for category, valid in zip(categories, feedback):
        trajectory.append(pi)
        if valid:
            pi = pi + alpha * (category - pi)
    return trajectory


class WslsCoreRecurrenceTests(unittest.TestCase):
    def test_first_trial_prior_is_one_half(self) -> None:
        tensors = _tensors([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [True, True, True])
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.3, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **tensors,
        )
        self.assertAlmostEqual(float(result.predictions.prior_long[0]), 0.5, places=12)

    def test_alpha_one_gives_previous_category_exactly(self) -> None:
        """The fixed pure_wsls case: pi_t must equal C_{t-1} bit-for-bit."""

        rng = random.Random(7)
        categories = [float(rng.randint(0, 1)) for _ in range(25)]
        tensors = _tensors(categories, categories, [True] * 25)
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=1.0, ell_short=torch.tensor(0.05, dtype=DTYPE), ell_long=torch.tensor(0.05, dtype=DTYPE), **tensors,
        )
        prior = result.predictions.prior_long
        expected = torch.tensor([0.5] + categories[:-1], dtype=DTYPE)
        self.assertTrue(torch.equal(prior, expected), "alpha=1.0 must be bit-exact, not merely close")

    def test_matches_closed_form_exponential_kernel_across_random_sequences(self) -> None:
        rng = random.Random(20260909)
        for _ in range(30):
            length = rng.randint(1, 15)
            categories = [float(rng.randint(0, 1)) for _ in range(length)]
            feedback = [rng.random() > 0.2 for _ in range(length)]
            choices = [float(rng.randint(0, 1)) for _ in range(length)]
            alpha = rng.uniform(0.01, 0.99)
            tensors = _tensors(categories, choices, feedback)
            result = evaluate_wsls_session(
                mouse_id="m", session_id="s", split_role="train",
                alpha=torch.tensor(alpha, dtype=DTYPE),
                ell_short=torch.tensor(0.05, dtype=DTYPE), ell_long=torch.tensor(0.05, dtype=DTYPE), **tensors,
            )
            expected = _reference_exponential_kernel(categories, feedback, alpha)
            actual = result.predictions.prior_long.tolist()
            for e, a in zip(expected, actual, strict=True):
                self.assertAlmostEqual(e, a, places=10)

    def test_feedback_invalid_trials_do_not_update_state(self) -> None:
        tensors = _tensors(
            categories=[1.0, 0.0, 1.0, 0.0],
            choices=[1.0, 0.0, 1.0, 0.0],
            feedback=[True, False, False, True],
        )
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=1.0, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **tensors,
        )
        prior = result.predictions.prior_long.tolist()
        # pi_1=0.5; feedback valid at t=1 with C_1=1 -> pi_2=1.0
        # feedback invalid at t=2 -> pi_3=pi_2=1.0 (carried forward)
        # feedback invalid at t=3 -> pi_4=pi_3=1.0 (carried forward)
        self.assertEqual(prior, [0.5, 1.0, 1.0, 1.0])

    def test_omission_with_valid_feedback_still_updates_state(self) -> None:
        """An invalid CHOICE (omission) with a valid category must still
        update pi for the next trial -- masking gates the likelihood, not
        the update."""

        tensors = _tensors(
            categories=[1.0, 0.0],
            choices=[float("nan"), 0.0],
            feedback=[True, True],
        )
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=1.0, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **tensors,
        )
        self.assertTrue(torch.isnan(result.predictions.probability_long[0]))
        self.assertFalse(bool(result.predictions.valid_choice_mask[0]))
        # pi_2 must still be C_1=1.0 despite trial 1's choice being invalid.
        self.assertEqual(float(result.predictions.prior_long[1]), 1.0)

    def test_current_trial_category_and_choice_cannot_affect_own_prediction(self) -> None:
        """Changing trial t's own category/choice must never change trial
        t's own prior_long (only future trials, via the update)."""

        base = _tensors(
            categories=[1.0, 0.0, 1.0],
            choices=[1.0, 0.0, 1.0],
            feedback=[True, True, True],
        )
        altered = _tensors(
            categories=[1.0, 1.0, 1.0],  # trial index 1's category flipped
            choices=[1.0, 1.0, 1.0],
            feedback=[True, True, True],
        )
        result_base = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.4, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **base,
        )
        result_altered = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.4, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **altered,
        )
        # prior_long[1] (predicted BEFORE trial 1's own category is seen)
        # must be identical in both cases.
        self.assertAlmostEqual(
            float(result_base.predictions.prior_long[1]),
            float(result_altered.predictions.prior_long[1]),
            places=12,
        )

    def test_duration_never_enters_prediction_or_likelihood(self) -> None:
        tensors = _tensors(
            categories=[1.0, 0.0, 1.0, 0.0, 1.0],
            choices=[1.0, 0.0, 0.0, 1.0, 1.0],
            feedback=[True, True, True, True, True],
        )
        baseline = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.4, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.2, dtype=DTYPE), **tensors,
        )
        perturbed = dict(tensors)
        perturbed["duration_s"] = torch.tensor(
            [999.0, -5.0, float("nan"), 0.0, 1e9], dtype=DTYPE
        )
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.4, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.2, dtype=DTYPE), **perturbed,
        )
        torch.testing.assert_close(baseline.predictions.prior_long, result.predictions.prior_long)
        torch.testing.assert_close(
            baseline.predictions.probability_long, result.predictions.probability_long, equal_nan=True
        )

    def test_session_reset_to_one_half(self) -> None:
        """Two independent sessions must each start from pi_1=0.5, regardless
        of how the first session ended."""

        tensors_1 = _tensors([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [True, True, True])
        tensors_2 = _tensors([0.0], [0.0], [True])
        result_1 = evaluate_wsls_session(
            mouse_id="m", session_id="s1", split_role="train",
            alpha=1.0, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **tensors_1,
        )
        result_2 = evaluate_wsls_session(
            mouse_id="m", session_id="s2", split_role="train",
            alpha=1.0, ell_short=torch.tensor(0.1, dtype=DTYPE), ell_long=torch.tensor(0.1, dtype=DTYPE), **tensors_2,
        )
        self.assertEqual(float(result_1.predictions.prior_long[-1]), 1.0)
        self.assertEqual(float(result_2.predictions.prior_long[0]), 0.5)

    def test_zero_eligible_trials_returns_zero_nll_and_nan_row(self) -> None:
        tensors = _tensors(
            categories=[1.0, 0.0],
            choices=[float("nan"), float("nan")],
            feedback=[False, False],
        )
        ell_short = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
        ell_long = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=1.0, ell_short=ell_short, ell_long=ell_long, **tensors,
        )
        self.assertEqual(result.valid_trial_count, 0)
        self.assertEqual(float(result.negative_log_likelihood.detach()), 0.0)
        self.assertTrue(torch.all(torch.isnan(result.per_trial_negative_log_likelihood)))
        # Must remain differentiable (used inside a pooled multi-session sum).
        result.negative_log_likelihood.backward()

    def test_lapse_mixture_matches_hand_computed_formula(self) -> None:
        """Trial 0's q_1=pi_1=0.5 exactly, at which point (1-eps)*0.5+0.5*eps
        equals 0.5 for ANY eps (0.5 is a fixed point of the lapse mixture) --
        so this check must use a LATER trial where q!=0.5, or it could never
        detect a wrong lapse rate, a short/long swap, or a broken formula."""

        tensors = _tensors([1.0, 1.0], [1.0, 0.0], [True, True])
        ell_short = 0.2
        ell_long = 0.3
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=0.5, ell_short=torch.tensor(ell_short, dtype=DTYPE),
            ell_long=torch.tensor(ell_long, dtype=DTYPE), **tensors,
        )
        # pi_1=0.5; category_1=1 (long), feedback valid -> pi_2 = 0.5+0.5*(1-0.5) = 0.75
        pi_2 = 0.75
        self.assertAlmostEqual(float(result.predictions.prior_long[1]), pi_2, places=12)
        # category_2=1 (long) -> epsilon = ell_long; q_2=pi_2=0.75 != 0.5, so
        # this genuinely depends on ell_long (a bug swapping ell_short in
        # would give a different, detectably wrong number).
        expected = (1.0 - ell_long) * pi_2 + 0.5 * ell_long
        self.assertAlmostEqual(float(result.predictions.probability_long[1]), expected, places=12)
        self.assertNotAlmostEqual(expected, (1.0 - ell_short) * pi_2 + 0.5 * ell_short, places=6)

    def test_finite_difference_gradient_matches_autograd(self) -> None:
        rng = random.Random(11)
        categories = [float(rng.randint(0, 1)) for _ in range(12)]
        feedback = [True] * 12
        choices = [float(rng.randint(0, 1)) for _ in range(12)]
        tensors = _tensors(categories, choices, feedback)

        def nll(a_value: float, ell_s: float, ell_l: float) -> float:
            with torch.no_grad():
                result = evaluate_wsls_session(
                    mouse_id="m", session_id="s", split_role="train",
                    alpha=torch.sigmoid(torch.tensor(a_value, dtype=DTYPE)),
                    ell_short=torch.tensor(ell_s, dtype=DTYPE),
                    ell_long=torch.tensor(ell_l, dtype=DTYPE), **tensors,
                )
            return float(result.negative_log_likelihood)

        a0, ell_s0, ell_l0 = 0.2, 0.1, 0.15
        h = 1e-6
        finite_diff_a = (nll(a0 + h, ell_s0, ell_l0) - nll(a0 - h, ell_s0, ell_l0)) / (2 * h)
        finite_diff_ell_s = (nll(a0, ell_s0 + h, ell_l0) - nll(a0, ell_s0 - h, ell_l0)) / (2 * h)
        finite_diff_ell_l = (nll(a0, ell_s0, ell_l0 + h) - nll(a0, ell_s0, ell_l0 - h)) / (2 * h)

        a = torch.tensor(a0, dtype=DTYPE, requires_grad=True)
        ell_s = torch.tensor(ell_s0, dtype=DTYPE, requires_grad=True)
        ell_l = torch.tensor(ell_l0, dtype=DTYPE, requires_grad=True)
        result = evaluate_wsls_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=torch.sigmoid(a), ell_short=ell_s, ell_long=ell_l, **tensors,
        )
        result.negative_log_likelihood.backward()
        self.assertAlmostEqual(float(a.grad), finite_diff_a, places=4)
        self.assertAlmostEqual(float(ell_s.grad), finite_diff_ell_s, places=4)
        self.assertAlmostEqual(float(ell_l.grad), finite_diff_ell_l, places=4)


if __name__ == "__main__":
    unittest.main()
