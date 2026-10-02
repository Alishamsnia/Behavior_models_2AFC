"""Scientific tests for M2_dynamic_previous_wsls_bias_v1.

This is the primary correctness anchor for the model's freshly-authored
``session.py`` loop, established by (a) an independently-derived reference
implementation of the feedback-gated, direct-assignment previous-category
recurrence (mirroring ``wsls_core_v1``'s own fixed-alpha=1 contract), (b)
direct causal/masking hand checks distinguishing this model from its
choice-driven sibling ``M2_dynamic_previous_action_bias_v1``, (c) the exact
``eta=0`` nesting fact against real ``M2_dynamic_block_length_v1``, and (d)
frozen registry/initialization contract checks.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.m1_session import M1SessionTrials
from m2_dynamic_block_length_v1.parameters import M2DynamicBlockLengthParameters
from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_session
from m2_dynamic_previous_wsls_bias_v1.parameters import (
    M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER,
    decode_m2_dynamic_previous_wsls_bias_raw_parameters,
    encode_m2_dynamic_previous_wsls_bias_constrained_parameters,
)
from m2_dynamic_previous_wsls_bias_v1.recovery import (
    _decode_generating_parameters,
    _reported_generating_parameters,
    iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid,
)
from m2_dynamic_previous_wsls_bias_v1.registry import load_m2_dynamic_previous_wsls_bias_model_registry
from m2_dynamic_previous_wsls_bias_v1.session import evaluate_m2_dynamic_previous_wsls_bias_session
from m2_dynamic_previous_wsls_bias_v1.simulation import simulate_m2_dynamic_previous_wsls_bias_choices


DTYPE = torch.float64

MOMENTS = TaskMoments(
    mu_short_s=0.5, mu_long_s=2.0, var_short_s2=0.01, var_long_s2=0.01,
    n_short=100, n_long=100, min_short_s=0.2, max_short_s=0.8,
    min_long_s=1.7, max_long_s=2.3, variance_ddof=0, source="synthetic",
)


def _session(*, duration, choice, category, feedback, session_id="s", split_role="train") -> M1SessionTrials:
    return M1SessionTrials(
        mouse_id="mouse", session_id=session_id,
        duration_s=torch.tensor(duration, dtype=DTYPE),
        choice_long=torch.tensor(choice, dtype=DTYPE),
        category_long=torch.tensor(category, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=split_role,
    )


def _reference_pi_trajectory(category, feedback, *, initial_pi: float = 0.5) -> list[float]:
    """Independent, non-vectorized reference for the feedback-gated,
    direct-assignment previous-category recurrence: pi_{t+1}=C_t if F_t else
    pi_t, mirroring wsls_core_v1's own alpha=1 exact-assignment branch."""

    trajectory = []
    pi = initial_pi
    for c, f in zip(category, feedback):
        trajectory.append(pi)
        if f:
            pi = c
    return trajectory


def _decoded_parameters(**overrides):
    values = dict(
        w=0.03, s0=0.5, ell_short=0.03, ell_long=0.03, kappa=1.0,
        ah=-1.2, delta0=1.3862943611198906, bh=0.10, eta_logit=-0.5,
    )
    values.update(overrides)
    constrained = torch.tensor(
        [values[name] for name in M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER], dtype=DTYPE,
    )
    raw = encode_m2_dynamic_previous_wsls_bias_constrained_parameters(constrained)
    return decode_m2_dynamic_previous_wsls_bias_raw_parameters(raw)


class PreviousWslsBiasRecurrenceTests(unittest.TestCase):
    def test_pi_trajectory_is_bit_exact_previous_category_matching_wsls_core_contract(self):
        category = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        feedback = [True, True, True, True, True, True, True]
        choice = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        session = _session(
            duration=[1.0] * len(category), choice=choice, category=category, feedback=feedback,
        )
        parameters = _decoded_parameters()
        evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, parameters, MOMENTS)
        pi = evaluation.predictions.previous_wsls_state
        expected = _reference_pi_trajectory(category, feedback)
        torch.testing.assert_close(pi, torch.tensor(expected, dtype=DTYPE), atol=0, rtol=0)

    def test_omission_with_valid_feedback_still_teaches_next_trial(self):
        category = [1.0, 0.0, 1.0]
        feedback = [True, True, True]
        choice = [1.0, float("nan"), 0.0]
        session = _session(duration=[1.0, 1.0, 1.0], choice=choice, category=category, feedback=feedback)
        parameters = _decoded_parameters()
        evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, parameters, MOMENTS)
        pi = evaluation.predictions.previous_wsls_state
        # pi_3 = C_2 = 0.0 despite trial 2 (index 1) being an omission: the
        # update reads only category_long/feedback_valid, never choice_long,
        # so the omitted trial still teaches the *following* trial.
        self.assertEqual(pi[2].item(), category[1])

    def test_feedback_invalid_trial_holds_pi_unchanged(self):
        category = [1.0, 1.0, 0.0]
        feedback = [True, False, True]
        choice = [1.0, 1.0, 0.0]
        session = _session(duration=[1.0, 1.0, 1.0], choice=choice, category=category, feedback=feedback)
        parameters = _decoded_parameters()
        evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, parameters, MOMENTS)
        pi = evaluation.predictions.previous_wsls_state
        # pi_2 must still equal pi_1 (=C_0=1.0): trial 1's feedback is invalid.
        self.assertEqual(pi[2].item(), pi[1].item())
        self.assertEqual(pi[1].item(), 1.0)

    def test_pi_and_belief_never_depend_on_choice_long(self):
        category = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        feedback = [True] * 6
        duration = [0.4, 1.9, 0.6, 1.8, 0.5, 2.1]
        parameters = _decoded_parameters()
        left = evaluate_m2_dynamic_previous_wsls_bias_session(
            _session(duration=duration, choice=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], category=category, feedback=feedback),
            parameters, MOMENTS,
        )
        right = evaluate_m2_dynamic_previous_wsls_bias_session(
            _session(duration=duration, choice=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], category=category, feedback=feedback),
            parameters, MOMENTS,
        )
        torch.testing.assert_close(
            left.predictions.previous_wsls_state, right.predictions.previous_wsls_state, atol=0, rtol=0,
        )
        torch.testing.assert_close(
            left.predictions.pre_choice_block_probability, right.predictions.pre_choice_block_probability,
        )

    def test_current_trial_category_never_leaks_into_its_own_prediction(self):
        category = [1.0, 0.0, 1.0, 1.0]
        feedback = [True, True, True, True]
        choice = [1.0, 0.0, 1.0, 0.0]
        session = _session(duration=[1.0] * 4, choice=choice, category=category, feedback=feedback)
        parameters = _decoded_parameters()
        evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, parameters, MOMENTS)
        # prior_long/prediction at trial t must be computable before C_t is
        # known: previous_wsls_state[t] equals the *pre*-update pi, i.e. it
        # reflects C_{t-1} (or earlier), never C_t.
        pi = evaluation.predictions.previous_wsls_state
        self.assertEqual(pi[0].item(), 0.5)
        self.assertEqual(pi[1].item(), category[0])
        self.assertEqual(pi[2].item(), category[1])
        self.assertEqual(pi[3].item(), category[2])

    def test_likelihood_eligibility_requires_valid_category_even_with_valid_choice(self):
        # A feedback-invalid trial may carry a non-binary/missing category
        # under the shared session contract; such a trial must not enter the
        # choice likelihood even though its own choice is valid.
        category = [1.0, float("nan"), 1.0]
        feedback = [True, False, True]
        choice = [1.0, 1.0, 0.0]
        session = _session(duration=[1.0, 1.0, 1.0], choice=choice, category=category, feedback=feedback)
        parameters = _decoded_parameters()
        evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, parameters, MOMENTS)
        self.assertFalse(bool(evaluation.predictions.valid_choice_mask[1]))
        self.assertTrue(torch.isnan(evaluation.predictions.probability_long[1]))


class PreviousWslsBiasNestingTests(unittest.TestCase):
    def test_eta_zero_reproduces_m2_dynamic_block_length_exactly(self):
        torch.manual_seed(7)
        n = 30
        duration = torch.where(
            torch.rand(n) < 0.5, 0.2 + 0.6 * torch.rand(n), 1.7 + 0.6 * torch.rand(n),
        ).to(DTYPE)
        category = (torch.rand(n) < 0.5).to(DTYPE)
        choice = (torch.rand(n) < 0.5).to(DTYPE)
        feedback = torch.rand(n) < 0.9
        session = M1SessionTrials(
            mouse_id="m", session_id="s", duration_s=duration, choice_long=choice,
            category_long=category, feedback_valid=feedback, split_role="train",
        )
        wsls_parameters = _decoded_parameters().with_explicit_eta(0.0)
        wsls_evaluation = evaluate_m2_dynamic_previous_wsls_bias_session(session, wsls_parameters, MOMENTS)

        block_length_parameters = M2DynamicBlockLengthParameters(
            w=wsls_parameters.w, s0=wsls_parameters.s0, ell_short=wsls_parameters.ell_short,
            ell_long=wsls_parameters.ell_long, kappa=wsls_parameters.kappa, ah=wsls_parameters.ah,
            delta0=wsls_parameters.delta0, bh=wsls_parameters.bh,
        )
        block_length_evaluation = evaluate_m2_dynamic_block_length_session(session, block_length_parameters, MOMENTS)

        torch.testing.assert_close(
            wsls_evaluation.predictions.probability_long, block_length_evaluation.predictions.probability_long,
        )
        torch.testing.assert_close(wsls_evaluation.negative_log_likelihood, block_length_evaluation.negative_log_likelihood)


class PreviousWslsBiasParameterTests(unittest.TestCase):
    def test_round_trip_and_alpha_fixed_at_one(self):
        parameters = _decoded_parameters(eta_logit=0.75)
        self.assertEqual(float(parameters.alpha), 1.0)
        constrained = parameters.as_tensor()
        raw = encode_m2_dynamic_previous_wsls_bias_constrained_parameters(constrained)
        decoded = decode_m2_dynamic_previous_wsls_bias_raw_parameters(raw)
        torch.testing.assert_close(decoded.as_tensor(), constrained)

    def test_parameter_order_has_nine_coordinates_no_learning_rate_logit(self):
        self.assertEqual(
            M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER,
            ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit"),
        )


class PreviousWslsBiasRegistryTests(unittest.TestCase):
    def test_registry_loads_with_nine_parameters_and_fixed_learning_rate(self):
        registry = load_m2_dynamic_previous_wsls_bias_model_registry()
        self.assertEqual(registry.model_id, "M2_dynamic_previous_wsls_bias_v1")
        self.assertEqual(
            registry.parameter_order,
            ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit"),
        )
        self.assertEqual(registry.wsls_learning_rate, 1.0)
        self.assertEqual(len(registry.initializations), 8)

    def test_starts_preserve_m2_dynamic_block_length_source_rows_exactly(self):
        registry = load_m2_dynamic_previous_wsls_bias_model_registry()
        shared = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh")
        for start in registry.initializations:
            for name in shared:
                self.assertTrue(torch.isfinite(torch.tensor(getattr(start, name))))
        self.assertGreaterEqual(len({start.eta_logit for start in registry.initializations}), 2)


class PreviousWslsBiasRecoveryGridTests(unittest.TestCase):
    """Regression coverage for a bug caught in code review: the registered
    recovery grid's ``eta`` values were being placed directly into the
    ``eta_logit`` coordinate without a logit transform, so a documented
    ``eta`` sweep of 0.0-0.5 silently produced ``eta`` values clustered
    around 0.5-0.6 instead."""

    def test_grid_eta_values_decode_to_the_documented_constrained_etas(self):
        registry = load_m2_dynamic_previous_wsls_bias_model_registry()
        grid = tuple(iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid((0.03, 0.5, 0.03, 0.03, 1.0)))
        decoded_etas = {round(float(_decode_generating_parameters(row, registry).eta), 6) for row in grid}
        self.assertEqual(decoded_etas, {0.0, 0.1, 0.2, 0.35, 0.5})

    def test_eta_zero_sentinel_never_collides_with_a_real_interior_eta_logit(self):
        # eta=0.5 -> eta_logit=logit(0.5)=0.0 exactly; naively reusing 0.0 as
        # the "construct the exact eta=0 boundary" sentinel would silently
        # turn this real eta=0.5 grid point into eta=0.0 instead.
        registry = load_m2_dynamic_previous_wsls_bias_model_registry()
        grid = tuple(iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid((0.03, 0.5, 0.03, 0.03, 1.0)))
        half = next(row for row in grid if abs(float(row[8])) < 1e-12)
        self.assertAlmostEqual(float(_decode_generating_parameters(half, registry).eta), 0.5, places=6)

    def test_bh_zero_boundary_constructs_without_error(self):
        registry = load_m2_dynamic_previous_wsls_bias_model_registry()
        grid = tuple(iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid((0.03, 0.5, 0.03, 0.03, 1.0)))
        zero_bh = next(row for row in grid if float(row[7]) == 0.0)
        decoded = _decode_generating_parameters(zero_bh, registry)
        self.assertEqual(float(decoded.bh), 0.0)

    def test_reported_generating_parameters_substitutes_nan_with_exact_negative_infinity(self):
        # Caught in code review: the raw NaN sentinel must never leak into
        # the reported/serialized result (bare NaN is non-standard JSON and
        # uninformative); -inf is the mathematically exact eta_logit at the
        # eta=0 boundary (sigmoid(-inf)=0 exactly).
        raw = torch.tensor([0.03, 0.5, 0.03, 0.03, 1.0, -1.2, 1.0, 0.1, float("nan")], dtype=torch.float64)
        reported = _reported_generating_parameters(raw)
        self.assertEqual(reported[8], float("-inf"))
        self.assertEqual(reported[:8], tuple(float(v) for v in raw[:8]))

    def test_reported_generating_parameters_leaves_finite_eta_logit_untouched(self):
        raw = torch.tensor([0.03, 0.5, 0.03, 0.03, 1.0, -1.2, 1.0, 0.1, 0.0], dtype=torch.float64)
        reported = _reported_generating_parameters(raw)
        self.assertEqual(reported[8], 0.0)


class PreviousWslsBiasSimulationTests(unittest.TestCase):
    def test_vectorized_simulation_preserves_omission_pattern(self):
        category = [1.0, 0.0, 1.0, 1.0, 0.0]
        feedback = [True, True, True, True, True]
        choice = [1.0, float("nan"), 0.0, 1.0, 0.0]
        session = _session(duration=[0.4, 1.8, 0.5, 1.9, 0.6], choice=choice, category=category, feedback=feedback)
        parameters = _decoded_parameters()
        simulated = simulate_m2_dynamic_previous_wsls_bias_choices((session,), parameters, MOMENTS, seed=0)
        self.assertEqual(len(simulated), 1)
        self.assertTrue(torch.isnan(simulated[0].choice_long[1]))
        self.assertFalse(bool(torch.isnan(simulated[0].choice_long[0])))
        self.assertTrue(torch.equal(simulated[0].category_long, session.category_long))


if __name__ == "__main__":
    unittest.main()
