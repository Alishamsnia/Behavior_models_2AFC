"""Scientific tests for the shared ``action_core_v1`` evaluator.

This is the primary correctness anchor for BOTH ``pure_action_v1`` and
``extended_action_v1``, since both models delegate their entire predict/
update/mask/likelihood computation to this one module. It mirrors
``test_wsls_core.py`` check-for-check, but every sequence is chosen so that
the mouse's choices DISAGREE with the true categories -- otherwise a core
that silently tracked the category (the WSLS sibling's driver) would pass.
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

from action_core_v1 import evaluate_action_session
from wsls_core_v1 import evaluate_wsls_session


DTYPE = torch.float64


def _tensors(categories, choices, durations=None):
    n = len(categories)
    if durations is None:
        durations = [1.0] * n
    return dict(
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
    )


def _lapse(short=0.1, long=0.1):
    return dict(ell_short=torch.tensor(short, dtype=DTYPE), ell_long=torch.tensor(long, dtype=DTYPE))


def _evaluate(tensors, alpha, **lapse):
    return evaluate_action_session(
        mouse_id="m", session_id="s", split_role="train",
        alpha=alpha, **(lapse or _lapse()), **tensors,
    )


def _reference_action_kernel(choices, alpha: float, *, initial_pi: float = 0.5):
    """Independent, non-vectorized reference: plain Python delta-rule loop on
    valid (finite, binary) choices only."""

    trajectory = []
    pi = initial_pi
    for choice in choices:
        trajectory.append(pi)
        if choice == choice and choice in (0.0, 1.0):  # finite and binary
            pi = pi + alpha * (choice - pi)
    return trajectory


class ActionCoreRecurrenceTests(unittest.TestCase):
    def test_first_trial_prior_is_one_half(self) -> None:
        result = _evaluate(_tensors([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]), 0.3)
        self.assertAlmostEqual(float(result.predictions.prior_long[0]), 0.5, places=12)

    def test_alpha_one_gives_last_valid_action_exactly(self) -> None:
        """The fixed pure_action case: pi_t must equal the last valid choice
        bit-for-bit, including across omitted choices."""

        rng = random.Random(7)
        choices = [float(rng.randint(0, 1)) for _ in range(25)]
        for index in (3, 4, 11):
            choices[index] = float("nan")
        categories = [1.0 - c if c == c else 1.0 for c in choices]  # always disagree
        result = _evaluate(_tensors(categories, choices), 1.0)
        expected, last = [], 0.5
        for choice in choices:
            expected.append(last)
            if choice == choice:
                last = choice
        self.assertTrue(
            torch.equal(result.predictions.prior_long, torch.tensor(expected, dtype=DTYPE)),
            "alpha=1.0 must be bit-exact, not merely close",
        )

    def test_matches_reference_kernel_across_random_sequences(self) -> None:
        rng = random.Random(20261028)
        for _ in range(30):
            length = rng.randint(1, 15)
            choices = [
                float(rng.randint(0, 1)) if rng.random() > 0.2 else float("nan")
                for _ in range(length)
            ]
            categories = [float(rng.randint(0, 1)) for _ in range(length)]
            alpha = rng.uniform(0.01, 0.99)
            result = _evaluate(_tensors(categories, choices), torch.tensor(alpha, dtype=DTYPE))
            expected = _reference_action_kernel(choices, alpha)
            for e, a in zip(expected, result.predictions.prior_long.tolist(), strict=True):
                self.assertAlmostEqual(e, a, places=10)

    def test_driver_is_choice_not_category(self) -> None:
        """Same categories, different choices -> different pi trajectories;
        same choices, different categories -> identical pi trajectories."""

        base = _evaluate(_tensors([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]), 0.4)
        other_choices = _evaluate(_tensors([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]), 0.4)
        other_categories = _evaluate(_tensors([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]), 0.4)
        self.assertFalse(torch.allclose(base.predictions.prior_long, other_choices.predictions.prior_long))
        self.assertTrue(torch.equal(base.predictions.prior_long, other_categories.predictions.prior_long))

    def test_invalid_choice_does_not_update_state(self) -> None:
        result = _evaluate(_tensors([0.0, 1.0, 0.0, 1.0], [1.0, float("nan"), 2.0, 0.0]), 1.0)
        # pi_1=0.5; A_1=1 -> pi_2=1; A_2 NaN -> pi_3=1; A_3=2 (non-binary) -> pi_4=1
        self.assertEqual(result.predictions.prior_long.tolist(), [0.5, 1.0, 1.0, 1.0])
        self.assertEqual(
            result.predictions.action_update_mask.tolist(), [True, False, False, True]
        )

    def test_valid_choice_with_missing_category_updates_but_adds_no_likelihood(self) -> None:
        """The two masks are independent: a valid action on a trial whose
        category is missing still updates pi, but cannot be scored (the
        category-dependent lapse needs a binary category)."""

        result = _evaluate(_tensors([1.0, float("nan"), 1.0], [0.0, 1.0, 1.0]), 1.0)
        self.assertEqual(result.predictions.valid_choice_mask.tolist(), [True, False, True])
        self.assertEqual(result.predictions.action_update_mask.tolist(), [True, True, True])
        self.assertTrue(torch.isnan(result.predictions.probability_long[1]))
        self.assertEqual(float(result.predictions.prior_long[2]), 1.0)
        self.assertEqual(result.valid_trial_count, 2)

    def test_current_trial_choice_cannot_affect_own_prediction(self) -> None:
        """Flipping trial t's own choice changes trial t+1 onward only."""

        base = _evaluate(_tensors([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]), 0.4)
        altered = _evaluate(_tensors([1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]), 0.4)
        for name in ("prior_long", "probability_long"):
            torch.testing.assert_close(
                getattr(base.predictions, name)[:2], getattr(altered.predictions, name)[:2]
            )
        self.assertNotAlmostEqual(
            float(base.predictions.prior_long[2]), float(altered.predictions.prior_long[2])
        )

    def test_duration_never_enters_prediction_or_likelihood(self) -> None:
        tensors = _tensors([1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0])
        baseline = _evaluate(tensors, 0.4, **_lapse(0.1, 0.2))
        perturbed = dict(tensors)
        perturbed["duration_s"] = torch.tensor([999.0, -5.0, float("nan"), 0.0, 1e9], dtype=DTYPE)
        result = _evaluate(perturbed, 0.4, **_lapse(0.1, 0.2))
        torch.testing.assert_close(baseline.predictions.prior_long, result.predictions.prior_long)
        torch.testing.assert_close(
            baseline.predictions.probability_long, result.predictions.probability_long, equal_nan=True
        )
        torch.testing.assert_close(baseline.negative_log_likelihood, result.negative_log_likelihood)

    def test_session_reset_to_one_half(self) -> None:
        first = _evaluate(_tensors([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1.0)
        second = _evaluate(_tensors([1.0], [0.0]), 1.0)
        self.assertEqual(float(first.predictions.prior_long[-1]), 1.0)
        self.assertEqual(float(second.predictions.prior_long[0]), 0.5)

    def test_zero_eligible_trials_returns_zero_nll_and_nan_row(self) -> None:
        ell_short = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
        ell_long = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
        a = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
        result = evaluate_action_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=torch.sigmoid(a), ell_short=ell_short, ell_long=ell_long,
            **_tensors([1.0, 0.0], [float("nan"), float("nan")]),
        )
        self.assertEqual(result.valid_trial_count, 0)
        self.assertEqual(float(result.negative_log_likelihood.detach()), 0.0)
        self.assertTrue(torch.all(torch.isnan(result.per_trial_negative_log_likelihood)))
        result.negative_log_likelihood.backward()

    def test_lapse_mixture_matches_hand_computed_formula(self) -> None:
        """Uses a later trial where q != 0.5 (0.5 is a fixed point of the lapse
        mixture), on a short-category trial, so a short/long swap is caught."""

        ell_short, ell_long = 0.2, 0.3
        result = _evaluate(_tensors([1.0, 0.0], [1.0, 1.0]), 0.5, **_lapse(ell_short, ell_long))
        pi_2 = 0.75  # 0.5 + 0.5*(A_1=1 - 0.5)
        self.assertAlmostEqual(float(result.predictions.prior_long[1]), pi_2, places=12)
        expected = (1.0 - ell_short) * pi_2 + 0.5 * ell_short  # C_2=0 -> ell_short
        self.assertAlmostEqual(float(result.predictions.probability_long[1]), expected, places=12)
        self.assertNotAlmostEqual(expected, (1.0 - ell_long) * pi_2 + 0.5 * ell_long, places=6)

    def test_finite_difference_gradient_matches_autograd(self) -> None:
        rng = random.Random(11)
        choices = [float(rng.randint(0, 1)) for _ in range(12)]
        categories = [float(rng.randint(0, 1)) for _ in range(12)]
        tensors = _tensors(categories, choices)

        def nll(a_value: float, ell_s: float, ell_l: float) -> float:
            with torch.no_grad():
                result = _evaluate(
                    tensors, torch.sigmoid(torch.tensor(a_value, dtype=DTYPE)), **_lapse(ell_s, ell_l)
                )
            return float(result.negative_log_likelihood)

        a0, s0, l0, h = 0.2, 0.1, 0.15, 1e-6
        fd = (
            (nll(a0 + h, s0, l0) - nll(a0 - h, s0, l0)) / (2 * h),
            (nll(a0, s0 + h, l0) - nll(a0, s0 - h, l0)) / (2 * h),
            (nll(a0, s0, l0 + h) - nll(a0, s0, l0 - h)) / (2 * h),
        )
        a = torch.tensor(a0, dtype=DTYPE, requires_grad=True)
        ell_s = torch.tensor(s0, dtype=DTYPE, requires_grad=True)
        ell_l = torch.tensor(l0, dtype=DTYPE, requires_grad=True)
        result = evaluate_action_session(
            mouse_id="m", session_id="s", split_role="train",
            alpha=torch.sigmoid(a), ell_short=ell_s, ell_long=ell_l, **tensors,
        )
        result.negative_log_likelihood.backward()
        for grad, expected in zip((a.grad, ell_s.grad, ell_l.grad), fd, strict=True):
            self.assertAlmostEqual(float(grad), expected, places=4)

    def test_equals_wsls_core_when_choices_equal_categories(self) -> None:
        """Cross-model consistency: when every choice equals its category and
        every trial is valid (both gates coincide), the action and WSLS cores
        must produce identical trajectories and likelihoods."""

        rng = random.Random(3)
        for alpha in (1.0, torch.tensor(0.37, dtype=DTYPE)):
            sequence = [float(rng.randint(0, 1)) for _ in range(20)]
            tensors = _tensors(sequence, sequence)
            action = _evaluate(tensors, alpha, **_lapse(0.07, 0.21))
            wsls = evaluate_wsls_session(
                mouse_id="m", session_id="s", split_role="train", alpha=alpha,
                feedback_valid=torch.ones(20, dtype=torch.bool), **_lapse(0.07, 0.21), **tensors,
            )
            self.assertTrue(torch.equal(action.predictions.prior_long, wsls.predictions.prior_long))
            self.assertTrue(torch.equal(action.negative_log_likelihood, wsls.negative_log_likelihood))


if __name__ == "__main__":
    unittest.main()
