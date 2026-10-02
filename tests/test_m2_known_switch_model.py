"""Scientific and integration tests for M2_known_switch_v1.

Two correctness anchors are available here that M1_known_switch_v1 lacks:
an exact ``delta0=0`` boundary (constant 0.5 prior, matching M0's own fixed
neutral prior) and an exact ``delta0=log(4)`` nesting to
``m1_known_switch_v1`` (both models' fixed rates coincide there). The
cross-import of ``m1_known_switch_v1`` in this file is test-only, mirroring
how ``m1_dynamic_block_length_v1`` is cross-verified against
``m4_hsmm_delta_v1`` only in the test suite, never at runtime.
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
    M1KnownSwitchSessionTrials,
    decode_m1_known_switch_raw_parameters,
    evaluate_m1_known_switch_session,
)
from m2_known_switch_v1 import (
    M2KnownSwitchParameters,
    M2KnownSwitchSessionTrials,
    decode_m2_known_switch_raw_parameters,
    encode_m2_known_switch_constrained_parameters,
    evaluate_m2_known_switch_mouse_objective,
    evaluate_m2_known_switch_session,
    m2_known_switch_session_trials_from_frame,
)


DTYPE = torch.float64
THETA_S_M1, THETA_L_M1 = 0.2, 0.8


def _session(
    true_block,
    categories,
    choices,
    feedback,
    durations,
    *,
    session_id="s1",
    role="train",
    cls=M2KnownSwitchSessionTrials,
):
    is_switch = [False] + [
        true_block[index] != true_block[index - 1] for index in range(1, len(true_block))
    ]
    return cls(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        is_switch=torch.tensor(is_switch, dtype=torch.bool),
        split_role=role,
    )


def _reference_probability_long_trajectory(true_block, categories, feedback, theta_s, theta_l):
    trajectory: list[float] = []
    belief = None
    previous_label = None
    for index, label in enumerate(true_block):
        if label == "N":
            trajectory.append(0.5)
            previous_label = label
            continue
        if belief is None:
            belief = (0.5, 0.5)
        elif label != previous_label:
            belief = (belief[1], belief[0])
        trajectory.append(belief[1])
        if feedback[index]:
            category = categories[index]
            likelihood_short = theta_s if category == 1 else (1.0 - theta_s)
            likelihood_long = theta_l if category == 1 else (1.0 - theta_l)
            unnormalized = (belief[0] * likelihood_short, belief[1] * likelihood_long)
            total = sum(unnormalized)
            belief = (unnormalized[0] / total, unnormalized[1] / total)
        previous_label = label
    return trajectory


class M2KnownSwitchParameterTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        registry = load_model_registry()
        common_raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        raw = torch.cat((common_raw, torch.tensor([0.3], dtype=DTYPE)))
        decoded = decode_m2_known_switch_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        re_encoded = encode_m2_known_switch_constrained_parameters(
            decoded.as_tensor(), lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        torch.testing.assert_close(re_encoded, raw, rtol=1e-9, atol=1e-10)

    def test_prototypes_symmetry(self) -> None:
        registry = load_model_registry()
        common_raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        raw = torch.cat((common_raw, torch.tensor([1.5], dtype=DTYPE)))
        decoded = decode_m2_known_switch_raw_parameters(raw)
        theta_n, theta_s, theta_l = decoded.prototypes().unbind(-1)
        self.assertAlmostEqual(float(theta_n), 0.5)
        self.assertAlmostEqual(float(theta_s + theta_l), 1.0, places=10)

    def test_prototypes_supports_batched_delta0(self) -> None:
        registry = load_model_registry()
        common_raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        batched_common = common_raw.unsqueeze(0).expand(2, -1)
        raw = torch.cat((batched_common, torch.tensor([[0.3], [1.5]], dtype=DTYPE)), dim=-1)
        decoded = decode_m2_known_switch_raw_parameters(raw)
        prototypes = decoded.prototypes()
        self.assertEqual(tuple(prototypes.shape), (2, 3))
        torch.testing.assert_close(prototypes[:, 0], torch.full((2,), 0.5, dtype=DTYPE))


class M2KnownSwitchSessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.common_raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )

    def _parameters(self, delta0_raw: float) -> M2KnownSwitchParameters:
        raw = torch.cat((self.common_raw, torch.tensor([delta0_raw], dtype=DTYPE)))
        return decode_m2_known_switch_raw_parameters(
            raw, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        )

    def test_zero_contrast_gives_constant_half_prior(self) -> None:
        parameters = self._parameters(1.0).with_explicit_delta0(0.0)
        session = _session(
            true_block=["N", "S", "L", "S", "L"],
            categories=[0.0, 1.0, 0.0, 1.0, 0.0],
            choices=[0.0, 1.0, 0.0, 1.0, 0.0],
            feedback=[True, True, True, True, True],
            durations=[0.5, 0.5, 2.0, 0.5, 2.0],
        )
        evaluation = evaluate_m2_known_switch_session(session, parameters, self.registry.task_moments)
        torch.testing.assert_close(
            evaluation.predictions.prior_long, torch.full((5,), 0.5, dtype=DTYPE)
        )

    def test_nests_m1_known_switch_at_log4(self) -> None:
        log4 = math.log(4.0)
        m2_parameters = self._parameters(1.0).with_explicit_delta0(log4)
        m1_parameters = decode_m1_known_switch_raw_parameters(
            self.common_raw, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        )
        true_block = ["N", "N", "S", "S", "L", "S", "L", "L"]
        categories = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        choices = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        feedback = [True, True, True, False, True, True, True, False]
        durations = [0.5, 0.55, 0.45, 0.6, 2.0, 0.5, 1.9, 2.1]

        m2_session = _session(true_block, categories, choices, feedback, durations, cls=M2KnownSwitchSessionTrials)
        m1_session = _session(true_block, categories, choices, feedback, durations, cls=M1KnownSwitchSessionTrials)

        m2_eval = evaluate_m2_known_switch_session(m2_session, m2_parameters, self.registry.task_moments)
        m1_eval = evaluate_m1_known_switch_session(m1_session, m1_parameters, self.registry.task_moments)

        torch.testing.assert_close(
            m2_eval.predictions.prior_long, m1_eval.predictions.prior_long, rtol=1e-11, atol=1e-12
        )
        torch.testing.assert_close(
            m2_eval.predictions.probability_long,
            m1_eval.predictions.probability_long,
            rtol=1e-11,
            atol=1e-12,
        )
        torch.testing.assert_close(
            m2_eval.negative_log_likelihood, m1_eval.negative_log_likelihood, rtol=1e-11, atol=1e-12
        )

    def test_matches_reference_switch_filter_across_random_schedules(self) -> None:
        rng = random.Random(20260906)
        delta0 = 0.9
        theta_s = 1.0 / (1.0 + math.exp(delta0))
        theta_l = 1.0 - theta_s
        parameters = self._parameters(2.0).with_explicit_delta0(delta0)
        for trial_index in range(20):
            true_block = ["N"] * rng.randint(1, 4)
            current = None
            for _ in range(rng.randint(3, 12)):
                current = rng.choice(["S", "L"]) if current is None else ("L" if current == "S" else "S")
                true_block.append(current)
            categories = [float(rng.randint(0, 1)) for _ in true_block]
            feedback = [rng.random() < 0.8 for _ in true_block]
            choices = [float(rng.randint(0, 1)) for _ in true_block]
            durations = [rng.choice([0.5, 2.0]) for _ in true_block]

            session = _session(true_block, categories, choices, feedback, durations)
            evaluation = evaluate_m2_known_switch_session(session, parameters, self.registry.task_moments)
            actual = evaluation.predictions.prior_long.detach().tolist()
            expected_p_long = _reference_probability_long_trajectory(
                true_block, categories, feedback, theta_s, theta_l
            )
            expected_prior = [
                0.5 if label == "N" else theta_s + (theta_l - theta_s) * p_long
                for label, p_long in zip(true_block, expected_p_long)
            ]
            for a, e in zip(actual, expected_prior):
                self.assertAlmostEqual(a, e, places=8, msg=(trial_index, true_block))

    def test_gradient_finite_difference(self) -> None:
        raw = torch.cat((self.common_raw, torch.tensor([0.4], dtype=DTYPE))).requires_grad_(True)
        session = _session(
            true_block=["N", "N", "S", "L", "L", "S"],
            categories=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            choices=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            feedback=[True, True, True, False, True, True],
            durations=[0.5, 0.6, 0.55, 2.1, 1.9, 0.45],
        )

        def loss_fn(vector: torch.Tensor) -> torch.Tensor:
            parameters = decode_m2_known_switch_raw_parameters(
                vector, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
            )
            return evaluate_m2_known_switch_session(
                session, parameters, self.registry.task_moments
            ).negative_log_likelihood

        loss = loss_fn(raw)
        loss.backward()
        analytic = raw.grad.detach().clone()

        base = raw.detach().clone()
        step = 1e-6
        numeric = torch.zeros_like(analytic)
        for index in range(base.numel()):
            bump = torch.zeros_like(base)
            bump[index] = step
            with torch.no_grad():
                plus = loss_fn(base + bump)
                minus = loss_fn(base - bump)
            numeric[index] = (plus - minus) / (2 * step)
        torch.testing.assert_close(analytic, numeric, rtol=1e-3, atol=1e-4)

    def test_pooled_objective_session_order_invariance(self) -> None:
        raw = torch.cat((self.common_raw, torch.tensor([0.7], dtype=DTYPE)))
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
        forward = evaluate_m2_known_switch_mouse_objective(
            [first, second], raw, self.registry.task_moments
        )
        backward = evaluate_m2_known_switch_mouse_objective(
            [second, first], raw, self.registry.task_moments
        )
        torch.testing.assert_close(forward.loss, backward.loss)

    def test_pooled_trials_vs_equal_sessions_weighting_differ_with_unequal_counts(self) -> None:
        raw = torch.cat((self.common_raw, torch.tensor([0.7], dtype=DTYPE)))
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
        pooled = evaluate_m2_known_switch_mouse_objective(
            [short, long_session], raw, self.registry.task_moments,
            objective_weighting="pooled_trials",
        )
        equal = evaluate_m2_known_switch_mouse_objective(
            [short, long_session], raw, self.registry.task_moments,
            objective_weighting="equal_sessions",
        )
        session_nll = torch.stack(
            [record.negative_log_likelihood for record in pooled.session_evaluations]
        )
        counts = torch.tensor(
            [record.valid_trial_count for record in pooled.session_evaluations], dtype=DTYPE
        )
        torch.testing.assert_close(pooled.loss, session_nll.sum() / counts.sum())
        torch.testing.assert_close(equal.loss, torch.mean(session_nll / counts))
        self.assertNotAlmostEqual(float(pooled.loss), float(equal.loss), places=6)

    def test_invalid_choice_contributes_no_likelihood_but_updates_state(self) -> None:
        raw = torch.cat((self.common_raw, torch.tensor([0.7], dtype=DTYPE)))
        parameters = decode_m2_known_switch_raw_parameters(
            raw, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        )
        with_omission = _session(
            true_block=["N", "S", "S", "S"],
            categories=[0.0, 0.0, 0.0, 0.0],
            choices=[0.0, float("nan"), 0.0, 0.0],
            feedback=[True, True, True, True],
            durations=[0.5, 0.5, 0.5, 0.5],
        )
        evaluation = evaluate_m2_known_switch_session(
            with_omission, parameters, self.registry.task_moments
        )
        self.assertEqual(evaluation.valid_trial_count, 3)
        self.assertTrue(bool(torch.isnan(evaluation.per_trial_negative_log_likelihood[1])))
        # Category=0 feedback on the OMITTED trial 1 (first switch, r_pre=0)
        # must still apply the full -delta0 update -- checked against the
        # exact expected magnitude, not merely "not NaN" (a suppressed
        # update would also be non-NaN, just wrongly left at 0).
        log_odds = evaluation.predictions.biased_log_odds_post
        expected_after_trial1 = -float(parameters.delta0)
        self.assertAlmostEqual(float(log_odds[1]), expected_after_trial1, places=10)
        self.assertAlmostEqual(float(log_odds[2]), 2 * expected_after_trial1, places=10)

    def test_rejects_test_split_sessions(self) -> None:
        raw = torch.cat((self.common_raw, torch.tensor([0.7], dtype=DTYPE)))
        test_session = _session(
            true_block=["N", "S"],
            categories=[0.0, 0.0],
            choices=[0.0, 0.0],
            feedback=[True, True],
            durations=[0.5, 0.5],
            role="test",
        )
        with self.assertRaises(ValueError):
            evaluate_m2_known_switch_mouse_objective(
                [test_session], raw, self.registry.task_moments
            )


class M2KnownSwitchFrameAdapterTests(unittest.TestCase):
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
        session = m2_known_switch_session_trials_from_frame(frame, split_role="train")
        self.assertEqual(session.is_switch.tolist(), [False, False, True, False, True, True])

    def test_rejects_non_neutral_start(self) -> None:
        frame = self._frame(["S", "L", "S"])
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_reentry_into_neutral(self) -> None:
        frame = self._frame(["N", "S", "N", "L"])
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_missing_true_block(self) -> None:
        frame = self._frame(["N", "S", "L"])
        frame.loc[1, "true_block"] = np.nan
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_gap_in_trial_index(self) -> None:
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [1, 2, 4]
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_duplicate_trial_index(self) -> None:
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [1, 2, 2]
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_rejects_dropped_leading_trial(self) -> None:
        frame = self._frame(["N", "S", "S"])
        frame["trial_index"] = [2, 3, 4]
        with self.assertRaises(ValueError):
            m2_known_switch_session_trials_from_frame(frame, split_role="train")

    def test_string_trial_index_does_not_get_lexicographically_reordered(self) -> None:
        true_block = ["N", "N"] + ["S", "L"] * 5
        frame = self._frame(true_block)
        frame["trial_index"] = [str(i) for i in range(1, len(true_block) + 1)]
        session = m2_known_switch_session_trials_from_frame(frame, split_role="train")
        expected = [False] + [
            true_block[index] != true_block[index - 1] for index in range(1, len(true_block))
        ]
        self.assertEqual(session.is_switch.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
