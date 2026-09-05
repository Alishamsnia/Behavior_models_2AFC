"""Tests for M2_dynamic_block_length_v1 (independent of m4_hsmm_delta_v1)."""

from __future__ import annotations

import math
import unittest

import torch

from m0_sensory_v1.model.m1_session import M1SessionTrials
from m0_sensory_v1.model.m2_session import evaluate_m2_session
from m2_dynamic_block_length_v1.fit import fit_m2_dynamic_block_length_single_initialization
from m2_dynamic_block_length_v1.parameters import M2DynamicBlockLengthParameters
from m2_dynamic_block_length_v1.registry import load_m2_dynamic_block_length_model_registry
from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_session

DTYPE = torch.float64


def _session(n=75, seed=0):
    g = torch.Generator().manual_seed(seed)
    duration = torch.where(
        torch.rand(n, generator=g) < 0.5,
        0.2 + 0.6 * torch.rand(n, generator=g),
        1.7 + 0.6 * torch.rand(n, generator=g),
    ).to(DTYPE)
    category = (torch.rand(n, generator=g) < 0.5).to(DTYPE)
    choice = (torch.rand(n, generator=g) < 0.5).to(DTYPE)
    feedback = torch.rand(n, generator=g) < 0.9
    return M1SessionTrials(
        mouse_id="mouse", session_id="sess", duration_s=duration,
        choice_long=choice, category_long=category, feedback_valid=feedback, split_role="train",
    )


class M2DynamicBlockLengthModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_m2_dynamic_block_length_model_registry()

    def test_registry_preserves_all_eight_m2_starts_and_adds_bh(self):
        self.assertEqual(len(self.registry.initializations), 8)
        self.assertEqual(self.registry.run_length_center, 23)
        self.assertEqual(self.registry.maximum_represented_run_length, 65)
        for start in self.registry.initializations:
            self.assertGreaterEqual(start.bh, 0.0)
            self.assertGreater(start.delta0, 0.0)

    def test_registry_declares_its_own_nine_required_checks(self):
        self.assertEqual(
            set(self.registry.convergence.required_checks),
            {
                "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
                "stable_fresh_training_nll", "no_parameter_transform_errors",
                "no_probability_normalization_errors", "no_optimizer_runtime_errors",
                "tail_occupancy_acceptable",
            },
        )

    def test_exact_bh_zero_reproduces_m2_beyond_cap(self):
        session = _session(n=75)  # longer than Rmax=65 to exercise the tail state
        moments = self.registry.task_moments
        params = M2DynamicBlockLengthParameters(
            w=torch.tensor(0.03, dtype=DTYPE), s0=torch.tensor(0.5, dtype=DTYPE),
            ell_short=torch.tensor(0.03, dtype=DTYPE), ell_long=torch.tensor(0.03, dtype=DTYPE),
            kappa=torch.tensor(1.0, dtype=DTYPE), ah=torch.tensor(-1.2, dtype=DTYPE),
            delta0=torch.tensor(math.log(4.0), dtype=DTYPE), bh=torch.tensor(0.3, dtype=DTYPE),
        ).with_explicit_bh(0.0)
        mine = evaluate_m2_dynamic_block_length_session(session, params, moments)
        real = evaluate_m2_session(session, params.as_m2(), moments)
        for name in ("probability_long", "prior_long", "emission_normalizer"):
            torch.testing.assert_close(
                getattr(mine.predictions, name), getattr(real.predictions, name),
                rtol=0, atol=1e-12, equal_nan=True,
            )
        torch.testing.assert_close(
            mine.predictions.pre_choice_block_probability,
            real.predictions.pre_choice_block_probability, rtol=0, atol=1e-12,
        )
        torch.testing.assert_close(mine.negative_log_likelihood, real.negative_log_likelihood, rtol=0, atol=1e-11)

    def test_causal_no_lookahead(self):
        moments = self.registry.task_moments
        params = M2DynamicBlockLengthParameters(
            w=torch.tensor(0.03, dtype=DTYPE), s0=torch.tensor(0.5, dtype=DTYPE),
            ell_short=torch.tensor(0.03, dtype=DTYPE), ell_long=torch.tensor(0.03, dtype=DTYPE),
            kappa=torch.tensor(1.0, dtype=DTYPE), ah=torch.tensor(-1.2, dtype=DTYPE),
            delta0=torch.tensor(1.0, dtype=DTYPE), bh=torch.tensor(0.1, dtype=DTYPE),
        )
        session_a = _session(n=10, seed=1)
        session_b = M1SessionTrials(
            mouse_id=session_a.mouse_id, session_id=session_a.session_id,
            duration_s=session_a.duration_s.clone(), choice_long=session_a.choice_long.clone(),
            category_long=1.0 - session_a.category_long, feedback_valid=session_a.feedback_valid.clone(),
            split_role="train",
        )
        eval_a = evaluate_m2_dynamic_block_length_session(session_a, params, moments)
        eval_b = evaluate_m2_dynamic_block_length_session(session_b, params, moments)
        torch.testing.assert_close(
            eval_a.predictions.prior_long[0], eval_b.predictions.prior_long[0], rtol=0, atol=1e-14,
        )

    def test_transition_mass_conservation(self):
        moments = self.registry.task_moments
        params = M2DynamicBlockLengthParameters(
            w=torch.tensor(0.03, dtype=DTYPE), s0=torch.tensor(0.5, dtype=DTYPE),
            ell_short=torch.tensor(0.03, dtype=DTYPE), ell_long=torch.tensor(0.03, dtype=DTYPE),
            kappa=torch.tensor(1.0, dtype=DTYPE), ah=torch.tensor(-1.2, dtype=DTYPE),
            delta0=torch.tensor(1.0, dtype=DTYPE), bh=torch.tensor(0.2, dtype=DTYPE),
        )
        session = _session(n=30, seed=2)
        result = evaluate_m2_dynamic_block_length_session(session, params, moments)
        errors = result.predictions.transition_normalization_error[:-1]
        self.assertTrue(bool((errors.abs() < 1e-10).all()))

    def test_tail_gate_rejects_convergence(self):
        registry = load_m2_dynamic_block_length_model_registry()
        from dataclasses import replace
        strict_registry = replace(registry, maximum_accepted_mean_tail_probability=-1.0)
        train = [_session(n=20, seed=10)]
        val = [replace(_session(n=20, seed=11), split_role="validation")]
        run = fit_m2_dynamic_block_length_single_initialization(
            train, val, strict_registry, strict_registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=2, allow_short_convergence_budget=True,
        )
        self.assertFalse(run.tail_occupancy_acceptable)
        self.assertFalse(run.converged)

    def test_all_coordinate_gradients_match_finite_differences(self):
        moments = self.registry.task_moments
        session = _session(n=12, seed=3)
        raw = torch.tensor([-3.6, -0.6, -3.5, -3.5, 0.0, -1.2, 0.5, -2.2], dtype=DTYPE, requires_grad=True)
        from m2_dynamic_block_length_v1.session import evaluate_m2_dynamic_block_length_mouse_objective

        def loss_fn(vector):
            return evaluate_m2_dynamic_block_length_mouse_objective([session], vector, moments).loss

        loss = loss_fn(raw)
        loss.backward()
        analytic = raw.grad.clone()
        epsilon = 1e-6
        for index in range(8):
            bump = torch.zeros(8, dtype=DTYPE)
            bump[index] = epsilon
            with torch.no_grad():
                plus = float(loss_fn(raw.detach() + bump))
                minus = float(loss_fn(raw.detach() - bump))
            numeric = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(float(analytic[index]), numeric, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
