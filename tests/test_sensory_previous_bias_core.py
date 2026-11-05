"""Scientific tests for the shared ``sensory_previous_bias_core_v1`` evaluator.

The primary correctness anchor for BOTH ``m1_previous_action_bias_v1``
(history_driver="action") and ``m1_previous_wsls_bias_v1``
(history_driver="category"). Correctness is established by (a) exact
nesting to M0 at eta=0, (b) agreement with an independent, non-vectorized
per-trial reference that calls the readout one trial at a time (the form
``M2_dynamic_previous_*_bias_v1`` use), and (c) direct causal/masking hand
checks for each driver's gate.
"""

from __future__ import annotations

from pathlib import Path
import inspect
import math
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability
from m0_sensory_v1.model.m1_session import M1SessionTrials
from m0_sensory_v1.model.parameters import decode_raw_parameters, encode_constrained_parameters
from m0_sensory_v1.model.session import SessionTrials, evaluate_session
from m0_sensory_v1.registry import load_model_registry
from sensory_previous_bias_core_v1 import evaluate_sensory_previous_bias_session


DTYPE = torch.float64
NAN = float("nan")
MOMENTS = load_model_registry(ROOT / "config" / "m0_sensory_v1.json").task_moments


def _session(durations, choices, categories, feedback, *, role="train") -> M1SessionTrials:
    return M1SessionTrials(
        mouse_id="m", session_id="s",
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


def _common(values=(0.04, 0.5, 0.1, 0.2, 1.1)):
    return decode_raw_parameters(encode_constrained_parameters(torch.tensor(values, dtype=DTYPE)))


def _random_session(rng: random.Random, length: int) -> M1SessionTrials:
    categories, durations, choices, feedback = [], [], [], []
    for _ in range(length):
        category = float(rng.randint(0, 1))
        duration = rng.uniform(0.2, 0.8) if category == 0.0 else rng.uniform(1.7, 2.3)
        if rng.random() < 0.15:
            duration = rng.uniform(0.9, 1.6)
        choice = float(rng.randint(0, 1)) if rng.random() > 0.15 else NAN
        if rng.random() < 0.05:
            duration = NAN  # valid choice, invalid duration
        valid_feedback = choice == choice if rng.random() > 0.1 else not (choice == choice)
        if not valid_feedback and rng.random() < 0.3:
            category = NAN  # missing category is allowed only where feedback is invalid
        categories.append(category); durations.append(duration)
        choices.append(choice); feedback.append(valid_feedback)
    return _session(durations, choices, categories, feedback)


def _reference(session, common, eta: float, driver: str):
    """Independent per-trial loop (no shared code with the core's vectorized path)."""

    pi, probabilities, states, nll = 0.5, [], [], 0.0
    for t in range(session.duration_s.numel()):
        d, a, c = (float(session.duration_s[t]), float(session.choice_long[t]),
                   float(session.category_long[t]))
        states.append(pi)
        eligible = (math.isfinite(d) and d > 0 and a in (0.0, 1.0) and c in (0.0, 1.0))
        if eligible:
            r = float(m1_bayesian_choice_probability(
                torch.tensor(d, dtype=DTYPE), common, MOMENTS, torch.tensor(0.5, dtype=DTYPE)
            ).probability_long)
            m = (1.0 - eta) * r + eta * pi
            epsilon = float(common.ell_long) if c == 1.0 else float(common.ell_short)
            p = (1.0 - epsilon) * m + epsilon / 2.0
            probabilities.append(p)
            nll -= a * math.log(p) + (1.0 - a) * math.log1p(-p)
        else:
            probabilities.append(NAN)
        if driver == "action" and a in (0.0, 1.0):
            pi = a
        elif driver == "category" and bool(session.feedback_valid[t]):
            pi = c
    return probabilities, states, nll


class EtaZeroNestingTests(unittest.TestCase):
    def test_eta_zero_reproduces_m0_exactly(self) -> None:
        rng = random.Random(5)
        common = _common()
        for driver in ("action", "category"):
            for _ in range(5):
                session = _random_session(rng, 40)
                m0_session = SessionTrials(**{
                    name: getattr(session, name)
                    for name in inspect.signature(SessionTrials).parameters
                })
                m0 = evaluate_session(m0_session, common, MOMENTS)
                core = evaluate_sensory_previous_bias_session(
                    session, common, torch.tensor(0.0, dtype=DTYPE), MOMENTS, driver
                )
                self.assertEqual(core.valid_trial_count, m0.valid_trial_count)
                self.assertEqual(float(core.negative_log_likelihood), float(m0.negative_log_likelihood))
                torch.testing.assert_close(
                    core.predictions.probability_long, m0.predictions.probability_long,
                    equal_nan=True, rtol=0, atol=0,
                )


class ReferenceAgreementTests(unittest.TestCase):
    def test_matches_per_trial_reference_loop(self) -> None:
        rng = random.Random(20261028)
        for driver in ("action", "category"):
            for _ in range(12):
                session = _random_session(rng, rng.randint(1, 30))
                eta = rng.uniform(0.01, 0.95)
                common = _common((rng.uniform(0.005, 0.2), rng.uniform(0.15, 1.2),
                                  rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3),
                                  rng.uniform(0.5, 2.0)))
                core = evaluate_sensory_previous_bias_session(
                    session, common, torch.tensor(eta, dtype=DTYPE), MOMENTS, driver
                )
                probabilities, states, nll = _reference(session, common, eta, driver)
                self.assertEqual(core.predictions.history_state.tolist(), states)
                for expected, actual in zip(probabilities, core.predictions.probability_long.tolist()):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertAlmostEqual(expected, actual, places=12)
                self.assertAlmostEqual(nll, float(core.negative_log_likelihood), places=9)


class GateTests(unittest.TestCase):
    def test_action_driver_tracks_last_valid_choice_and_ignores_feedback_and_category(self) -> None:
        session = _session(
            durations=[0.5, NAN, 2.0, 0.6, 2.1],
            choices=[1.0, 0.0, NAN, 1.0, 0.0],
            categories=[0.0, 1.0, NAN, 0.0, 1.0],   # disagree with choices
            feedback=[False, False, False, False, False],
        )
        core = evaluate_sensory_previous_bias_session(
            session, _common(), torch.tensor(0.3, dtype=DTYPE), MOMENTS, "action"
        )
        # trial 1 has an invalid duration but a valid choice: it still updates.
        self.assertEqual(core.predictions.history_state.tolist(), [0.5, 1.0, 0.0, 0.0, 1.0])
        self.assertEqual(core.predictions.history_update_mask.tolist(), [True, True, False, True, True])
        self.assertEqual(core.predictions.valid_choice_mask.tolist(), [True, False, False, True, True])

    def test_category_driver_tracks_last_feedback_valid_category_and_ignores_choice(self) -> None:
        session = _session(
            durations=[0.5, 2.0, 0.6, 2.1, 0.4],
            choices=[1.0, NAN, 1.0, 0.0, 1.0],      # trial 1 omitted
            categories=[0.0, 1.0, NAN, 1.0, 0.0],
            feedback=[True, True, False, True, True],
        )
        core = evaluate_sensory_previous_bias_session(
            session, _common(), torch.tensor(0.3, dtype=DTYPE), MOMENTS, "category"
        )
        # A feedback-valid omission (trial 1) still updates; trial 2's
        # invalid feedback carries the state forward.
        self.assertEqual(core.predictions.history_state.tolist(), [0.5, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(core.predictions.history_update_mask.tolist(), [True, True, False, True, True])
        # Missing category on a valid choice/duration: no likelihood term, no crash.
        self.assertEqual(core.predictions.valid_choice_mask.tolist(), [True, False, False, True, True])

    def test_valid_choice_with_missing_category_updates_action_state_without_likelihood(self) -> None:
        session = _session([0.5, 0.6, 2.0], [1.0, 0.0, 1.0], [1.0, NAN, 1.0], [True, False, True])
        core = evaluate_sensory_previous_bias_session(
            session, _common(), torch.tensor(0.4, dtype=DTYPE), MOMENTS, "action"
        )
        self.assertEqual(core.valid_trial_count, 2)
        self.assertTrue(math.isnan(float(core.predictions.probability_long[1])))
        self.assertEqual(float(core.predictions.history_state[2]), 0.0)

    def test_session_reset_and_first_trial_state(self) -> None:
        for driver in ("action", "category"):
            core = evaluate_sensory_previous_bias_session(
                _session([0.5], [1.0], [1.0], [True]), _common(),
                torch.tensor(0.5, dtype=DTYPE), MOMENTS, driver,
            )
            self.assertEqual(core.predictions.history_state.tolist(), [0.5])
            self.assertEqual(core.predictions.prior_long.tolist(), [0.5])


class CausalityTests(unittest.TestCase):
    def test_no_current_trial_leakage(self) -> None:
        base = _session([0.5, 2.0, 0.6, 2.1], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [True] * 4)
        flipped = _session([0.5, 2.0, 0.6, 2.1], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [True] * 4)
        for driver in ("action", "category"):
            a = evaluate_sensory_previous_bias_session(base, _common(), torch.tensor(0.4, dtype=DTYPE), MOMENTS, driver)
            b = evaluate_sensory_previous_bias_session(flipped, _common(), torch.tensor(0.4, dtype=DTYPE), MOMENTS, driver)
            # Trial 1's own choice/category never enter its own state.
            self.assertEqual(a.predictions.history_state[:2].tolist(), b.predictions.history_state[:2].tolist())
            self.assertNotEqual(float(a.predictions.history_state[2]), float(b.predictions.history_state[2]))
        # With the action driver the current category enters only the lapse:
        # trial 1's pre-lapse mixed probability is unchanged by its own flip.
        a = evaluate_sensory_previous_bias_session(base, _common(), torch.tensor(0.4, dtype=DTYPE), MOMENTS, "action")
        b = evaluate_sensory_previous_bias_session(flipped, _common(), torch.tensor(0.4, dtype=DTYPE), MOMENTS, "action")
        self.assertEqual(float(a.predictions.bias_mixed_probability_long[1]),
                         float(b.predictions.bias_mixed_probability_long[1]))

    def test_mixture_and_lapse_order(self) -> None:
        session = _session([0.5, 2.0], [1.0, 1.0], [0.0, 1.0], [True, True])
        common = _common()
        core = evaluate_sensory_previous_bias_session(
            session, common, torch.tensor(0.25, dtype=DTYPE), MOMENTS, "action"
        )
        r = core.predictions.non_lapse_probability_long[1]
        expected_m = 0.75 * r + 0.25 * 1.0
        torch.testing.assert_close(core.predictions.bias_mixed_probability_long[1], expected_m)
        torch.testing.assert_close(
            core.predictions.probability_long[1],
            lapse_mixture(expected_m, torch.tensor(1.0, dtype=DTYPE), common),
        )
        torch.testing.assert_close(
            core.predictions.non_bias_probability_long[1],
            lapse_mixture(r, torch.tensor(1.0, dtype=DTYPE), common),
        )


class NumericalTests(unittest.TestCase):
    def test_finite_difference_gradient_all_six_coordinates(self) -> None:
        session = _random_session(random.Random(9), 30)
        raw0 = torch.cat((encode_constrained_parameters(torch.tensor((0.04, 0.5, 0.1, 0.2, 1.1), dtype=DTYPE)),
                          torch.tensor([-0.7], dtype=DTYPE)))
        for driver in ("action", "category"):
            def nll(raw: torch.Tensor) -> torch.Tensor:
                return evaluate_sensory_previous_bias_session(
                    session, decode_raw_parameters(raw[:5]), torch.sigmoid(raw[5]), MOMENTS, driver
                ).negative_log_likelihood
            raw = raw0.clone().requires_grad_(True)
            nll(raw).backward()
            h = 1e-6
            for index in range(6):
                step = torch.zeros(6, dtype=DTYPE); step[index] = h
                with torch.no_grad():
                    fd = (nll(raw0 + step) - nll(raw0 - step)) / (2 * h)
                self.assertAlmostEqual(float(raw.grad[index]), float(fd), places=4)

    def test_zero_eligible_trials_returns_zero_nll_and_nan_row(self) -> None:
        session = _session([0.5, 2.0], [NAN, NAN], [1.0, 0.0], [False, False])
        eta_logit = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)
        core = evaluate_sensory_previous_bias_session(
            session, _common(), torch.sigmoid(eta_logit), MOMENTS, "action"
        )
        self.assertEqual(core.valid_trial_count, 0)
        self.assertEqual(float(core.negative_log_likelihood.detach()), 0.0)
        self.assertTrue(torch.all(torch.isnan(core.per_trial_negative_log_likelihood)))
        core.negative_log_likelihood.backward()

    def test_rejects_unknown_driver(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_sensory_previous_bias_session(
                _session([0.5], [1.0], [1.0], [True]), _common(),
                torch.tensor(0.1, dtype=DTYPE), MOMENTS, "reward",
            )


if __name__ == "__main__":
    unittest.main()
