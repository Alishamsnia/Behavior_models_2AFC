"""Scientific and integration tests for M1_known_switch_v1.

These tests are the primary correctness anchor for this model: unlike every
other model in this family, M1_known_switch_v1 has no exact-nesting boundary
to an already-verified parent (M0/M1), because its recursion mechanism
(deterministic switch-conditioned belief) is structurally different from
M1's hazard-driven HMM. Correctness is instead established here by (a) an
independently-authored brute-force reference filter operating in raw
probability space, cross-checked against the closed-form log-odds recursion
across many random valid schedules, and (b) direct hand-checkable cases.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.registry import load_model_registry
from m1_known_switch_v1 import (
    M1_KNOWN_SWITCH_FIXED_PROTOTYPES,
    M1KnownSwitchSessionTrials,
    decode_m1_known_switch_raw_parameters,
    encode_m1_known_switch_constrained_parameters,
    evaluate_m1_known_switch_mouse_objective,
    evaluate_m1_known_switch_session,
    m1_known_switch_session_trials_from_frame,
)


DTYPE = torch.float64
_, THETA_S, THETA_L = M1_KNOWN_SWITCH_FIXED_PROTOTYPES


def _session(
    true_block,
    categories,
    choices,
    feedback,
    durations,
    *,
    session_id="s1",
    role="train",
) -> M1KnownSwitchSessionTrials:
    is_switch = [False] + [
        true_block[index] != true_block[index - 1] for index in range(1, len(true_block))
    ]
    return M1KnownSwitchSessionTrials(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        is_switch=torch.tensor(is_switch, dtype=torch.bool),
        split_role=role,
    )


def _reference_probability_long_trajectory(true_block, categories, feedback) -> list[float]:
    """Independent brute-force forward filter in raw probability space.

    Maintains ``b=(P(S),P(L))`` once biased. On a later switch, swaps the two
    components (equivalent to conditioning on the realized transition
    branch); on no switch, the belief passes through unchanged. On valid
    feedback, applies an elementwise Bayes update with the fixed prototypes.
    This is deliberately written without reference to the log-odds formula
    under test, so it independently checks the sign-flip and update
    magnitude.
    """

    trajectory: list[float] = []
    belief = None  # (p_short, p_long), None while still Neutral
    previous_label = None
    for index, label in enumerate(true_block):
        if label == "N":
            trajectory.append(0.5)
            previous_label = label
            continue
        if belief is None:
            # First switch: leaving Neutral, unresolved 50/50.
            belief = (0.5, 0.5)
        elif label != previous_label:
            belief = (belief[1], belief[0])
        trajectory.append(belief[1])
        if feedback[index]:
            category = categories[index]
            likelihood_short = THETA_S if category == 1 else (1.0 - THETA_S)
            likelihood_long = THETA_L if category == 1 else (1.0 - THETA_L)
            unnormalized = (belief[0] * likelihood_short, belief[1] * likelihood_long)
            total = sum(unnormalized)
            belief = (unnormalized[0] / total, unnormalized[1] / total)
        previous_label = label
    return trajectory


class M1KnownSwitchParameterTests(unittest.TestCase):
    def test_parameter_round_trip_matches_m0(self) -> None:
        registry = load_model_registry()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        decoded = decode_m1_known_switch_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        re_encoded = encode_m1_known_switch_constrained_parameters(
            decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)


class M1KnownSwitchSessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )
        cls.parameters = decode_m1_known_switch_raw_parameters(
            cls.raw, lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )

    def _evaluate(self, session):
        return evaluate_m1_known_switch_session(session, self.parameters, self.registry.task_moments)

    def test_zero_switches_gives_constant_neutral_prior(self) -> None:
        session = _session(
            true_block=["N", "N", "N", "N"],
            categories=[0.0, 1.0, 0.0, 1.0],
            choices=[0.0, 1.0, 0.0, 1.0],
            feedback=[True, True, True, True],
            durations=[0.5, 2.0, 0.5, 2.0],
        )
        evaluation = self._evaluate(session)
        torch.testing.assert_close(
            evaluation.predictions.prior_long,
            torch.full((4,), 0.5, dtype=DTYPE),
        )

    def test_feedback_during_neutral_has_no_effect(self) -> None:
        with_feedback = _session(
            true_block=["N", "N", "N"],
            categories=[1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0],
            feedback=[True, True, True],
            durations=[0.5, 2.0, 0.5],
        )
        without_feedback = _session(
            true_block=["N", "N", "N"],
            categories=[1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0],
            feedback=[False, False, False],
            durations=[0.5, 2.0, 0.5],
        )
        first = self._evaluate(with_feedback)
        second = self._evaluate(without_feedback)
        torch.testing.assert_close(first.predictions.prior_long, second.predictions.prior_long)
        torch.testing.assert_close(
            first.negative_log_likelihood, second.negative_log_likelihood
        )

    def test_missing_feedback_retains_belief(self) -> None:
        session = _session(
            true_block=["N", "S", "S", "S"],
            categories=[1.0, 0.0, 0.0, 0.0],
            choices=[1.0, 0.0, 0.0, 0.0],
            feedback=[True, True, False, True],
            durations=[0.5, 0.5, 0.5, 0.5],
        )
        evaluation = self._evaluate(session)
        # Trial index 2 has no feedback: post belief at 2 must equal pre belief at 2,
        # which must equal post belief at 1 (nothing changed across the gap).
        log_odds = evaluation.predictions.biased_log_odds_post
        pre = evaluation.predictions.biased_log_odds_pre
        torch.testing.assert_close(pre[2], log_odds[1])
        torch.testing.assert_close(log_odds[2], pre[2])

    def test_matches_reference_switch_filter_across_random_schedules(self) -> None:
        rng = random.Random(20260906)
        for trial_index in range(30):
            true_block = ["N"] * rng.randint(1, 4)
            current = None
            for _ in range(rng.randint(3, 12)):
                if current is None:
                    current = rng.choice(["S", "L"])
                else:
                    current = "L" if current == "S" else "S"
                true_block.append(current)
            categories = [float(rng.randint(0, 1)) for _ in true_block]
            feedback = [rng.random() < 0.8 for _ in true_block]
            choices = [float(rng.randint(0, 1)) for _ in true_block]
            durations = [rng.choice([0.5, 2.0]) for _ in true_block]

            session = _session(true_block, categories, choices, feedback, durations)
            evaluation = self._evaluate(session)
            actual = evaluation.predictions.prior_long.detach().tolist()
            expected = _reference_probability_long_trajectory(true_block, categories, feedback)
            expected_prior = [
                0.5 if label == "N" else THETA_S + (THETA_L - THETA_S) * p_long
                for label, p_long in zip(true_block, expected)
            ]
            for a, e in zip(actual, expected_prior):
                self.assertAlmostEqual(a, e, places=9, msg=(trial_index, true_block))

    def test_gradient_finite_difference(self) -> None:
        session = _session(
            true_block=["N", "N", "S", "L", "L", "S"],
            categories=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            choices=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            feedback=[True, True, True, False, True, True],
            durations=[0.5, 0.6, 0.55, 2.1, 1.9, 0.45],
        )
        raw = self.raw.clone().requires_grad_(True)

        def loss_fn(vector: torch.Tensor) -> torch.Tensor:
            parameters = decode_m1_known_switch_raw_parameters(
                vector, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
            )
            return evaluate_m1_known_switch_session(
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
                plus = loss_fn(self.raw + bump)
                minus = loss_fn(self.raw - bump)
            numeric[index] = (plus - minus) / (2 * step)
        torch.testing.assert_close(analytic, numeric, rtol=1e-3, atol=1e-4)

    def test_pooled_objective_session_order_invariance(self) -> None:
        first = _session(
            true_block=["N", "S", "L"],
            categories=[0.0, 0.0, 1.0],
            choices=[0.0, 0.0, 1.0],
            feedback=[True, True, True],
            durations=[0.5, 0.5, 2.0],
            session_id="a",
        )
        second = _session(
            true_block=["N", "N", "L"],
            categories=[1.0, 1.0, 1.0],
            choices=[1.0, 1.0, 1.0],
            feedback=[True, True, True],
            durations=[2.0, 2.1, 1.9],
            session_id="b",
        )
        forward = evaluate_m1_known_switch_mouse_objective(
            [first, second], self.raw, self.registry.task_moments
        )
        backward = evaluate_m1_known_switch_mouse_objective(
            [second, first], self.raw, self.registry.task_moments
        )
        torch.testing.assert_close(forward.loss, backward.loss)

    def test_pooled_trials_vs_equal_sessions_weighting_differ_with_unequal_counts(self) -> None:
        short = _session(
            true_block=["N", "S"],
            categories=[0.0, 0.0],
            choices=[0.0, 0.0],
            feedback=[True, True],
            durations=[0.5, 0.5],
            session_id="short",
        )
        long_session = _session(
            true_block=["N", "N", "L", "L", "S"],
            categories=[1.0, 1.0, 1.0, 0.0, 0.0],
            choices=[1.0, 1.0, 1.0, 0.0, 0.0],
            feedback=[True, True, True, True, True],
            durations=[2.0, 2.1, 1.9, 0.5, 0.5],
            session_id="long",
        )
        pooled = evaluate_m1_known_switch_mouse_objective(
            [short, long_session], self.raw, self.registry.task_moments,
            objective_weighting="pooled_trials",
        )
        equal = evaluate_m1_known_switch_mouse_objective(
            [short, long_session], self.raw, self.registry.task_moments,
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
            true_block=["N", "S", "S", "S"],
            categories=[0.0, 0.0, 0.0, 0.0],
            choices=[0.0, float("nan"), 0.0, 0.0],
            feedback=[True, True, True, True],
            durations=[0.5, 0.5, 0.5, 0.5],
        )
        evaluation = self._evaluate(with_omission)
        self.assertEqual(evaluation.valid_trial_count, 3)
        self.assertTrue(bool(torch.isnan(evaluation.per_trial_negative_log_likelihood[1])))
        # Trial 1 is the first switch (r_pre=0); category=0 feedback on the
        # OMITTED choice must still apply the full -log(4) update, not be
        # silently skipped -- checked against the exact expected magnitude,
        # not merely "not NaN" (a suppressed update would also be non-NaN,
        # just wrongly left at 0).
        log_odds = evaluation.predictions.biased_log_odds_post
        expected_after_trial1 = -math.log(4.0)
        self.assertAlmostEqual(float(log_odds[1]), expected_after_trial1, places=10)
        # Trial 2 (no switch, category=0 again) compounds the same update.
        self.assertAlmostEqual(float(log_odds[2]), 2 * expected_after_trial1, places=10)

    def test_rejects_test_split_sessions(self) -> None:
        test_session = _session(
            true_block=["N", "S"],
            categories=[0.0, 0.0],
            choices=[0.0, 0.0],
            feedback=[True, True],
            durations=[0.5, 0.5],
            role="test",
        )
        with self.assertRaises(ValueError):
            evaluate_m1_known_switch_mouse_objective(
                [test_session], self.raw, self.registry.task_moments
            )


class M1KnownSwitchFrameAdapterTests(unittest.TestCase):
    def _frame(self, true_block, **overrides) -> pd.DataFrame:
        n = len(true_block)
        base = {
            "mouse_id": ["mouse"] * n,
            "session_id": ["s1"] * n,
            "trial_index": list(range(1, n + 1)),
            "duration_s": [0.5] * n,
            "choice_long": [0.0] * n,
            "category_long": [0.0] * n,
            "feedback_valid": [1.0] * n,
            "true_block": true_block,
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_derives_switch_flags(self) -> None:
        frame = self._frame(["N", "N", "S", "S", "L", "S"])
        session = m1_known_switch_session_trials_from_frame(frame, split_role="train")
        self.assertEqual(
            session.is_switch.tolist(), [False, False, True, False, True, True]
        )

    def test_rejects_non_neutral_start(self) -> None:
        frame = self._frame(["S", "L", "S"])
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_adjacent_identical_labels_merge_into_one_block(self) -> None:
        # Repeated consecutive labels are one run, not a validation failure:
        # with only {S,L} available once biased and no re-entry into N,
        # strict alternation is automatic once "starts N" and "never
        # re-enters N" both hold -- there is no independent way to encode
        # two separate same-category blocks in a row via this per-trial column.
        frame = self._frame(["N", "N", "S", "S", "S", "L", "L"])
        session = m1_known_switch_session_trials_from_frame(frame, split_role="train")
        self.assertEqual(
            session.is_switch.tolist(),
            [False, False, True, False, False, True, False],
        )

    def test_rejects_reentry_into_neutral(self) -> None:
        frame = self._frame(["N", "S", "N", "L"])
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_missing_true_block(self) -> None:
        frame = self._frame(["N", "S", "L"])
        frame.loc[1, "true_block"] = np.nan
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_invalid_true_block_label(self) -> None:
        frame = self._frame(["N", "S", "X"])
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_gap_in_trial_index(self) -> None:
        # A dropped row (e.g. trial_index 1,2,4 for a true N,S,S,S session)
        # must not silently change the derived switch timeline.
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [1, 2, 4]
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_duplicate_trial_index(self) -> None:
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [1, 2, 2]
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_dropped_leading_trial(self) -> None:
        # Indices 2,3,4 are gapless by max-min+1==count but silently discard
        # the session's true first (Neutral) trial -- must be rejected, not
        # accepted with the wrong row treated as trial 0.
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [2, 3, 4]
        with self.assertRaises(ValueError):
            m1_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_string_trial_index_does_not_get_lexicographically_reordered(self) -> None:
        # A naive sort-then-convert would put "10","11","12" before "2".."9".
        true_block = ["N", "N"] + ["S", "L"] * 5
        frame = self._frame(true_block)
        frame["trial_index"] = [str(i) for i in range(1, len(true_block) + 1)]
        session = m1_known_switch_session_trials_from_frame(frame, split_role="train")
        expected = [False] + [
            true_block[index] != true_block[index - 1] for index in range(1, len(true_block))
        ]
        self.assertEqual(session.is_switch.tolist(), expected)


class M1KnownSwitchSessionTrialsValidationTests(unittest.TestCase):
    def test_rejects_is_switch_true_at_index_zero(self) -> None:
        with self.assertRaises(ValueError):
            M1KnownSwitchSessionTrials(
                mouse_id="mouse",
                session_id="s1",
                duration_s=torch.tensor([0.5, 0.6], dtype=DTYPE),
                choice_long=torch.tensor([0.0, 1.0], dtype=DTYPE),
                category_long=torch.tensor([0.0, 1.0], dtype=DTYPE),
                feedback_valid=torch.tensor([True, True], dtype=torch.bool),
                is_switch=torch.tensor([True, False], dtype=torch.bool),
                split_role="train",
            )


if __name__ == "__main__":
    unittest.main()
