"""Acceptance tests for the additive capped-tail Model 4 extension."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.session import evaluate_m3_session  # noqa: E402
from m0_sensory_v1.model.fit import LBFGSSettings  # noqa: E402
from m4_hsmm_delta_v1.analysis import (  # noqa: E402
    build_m4_hazard_curve, build_m4_m3_validation_comparison,
    build_m4_trial_predictions, compare_m4_to_m3_predictions,
)
from m4_hsmm_delta_v1.diagnostics import (  # noqa: E402
    build_m4_implied_duration_distribution, evaluate_m4_truncation_sensitivity,
    summarize_m4_hazard_duration,
)
from m4_hsmm_delta_v1.fit import fit_m4_all_initializations, fit_m4_single_initialization  # noqa: E402
from m4_hsmm_delta_v1.identifiability import evaluate_m4_dynamic_surface  # noqa: E402
from m4_hsmm_delta_v1.m0_style_figures import generate_m4_m0_style_figures  # noqa: E402
from m4_hsmm_delta_v1.parameters import decode_m4_raw_parameters, encode_m4_constrained_parameters  # noqa: E402
from m4_hsmm_delta_v1.prior_state import (  # noqa: E402
    CappedTailHSMMPriorStateProvider,
    ConstantHazardPriorStateProvider,
)
from m4_hsmm_delta_v1.recovery import M4_RECOVERY_BH_GRID, iter_registered_m4_recovery_grid  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402
from m4_hsmm_delta_v1.session import (  # noqa: E402
    M4SessionTrials, capped_tail_hsmm_transition, evaluate_m4_mouse_objective,
    evaluate_m4_session,
)


DTYPE = torch.float64


def _session(*, n=12, session_id="s1", role="train", changed_index=None):
    categories = torch.tensor(([0, 0, 1, 1, 0, 1] * ((n + 5) // 6))[:n], dtype=DTYPE)
    durations = torch.where(categories.bool(), categories.new_tensor(2.0), categories.new_tensor(0.5))
    if changed_index is not None:
        categories[changed_index] = 1 - categories[changed_index]
    return M4SessionTrials(
        mouse_id="mouse", session_id=session_id, duration_s=durations,
        choice_long=categories.clone(), category_long=categories,
        feedback_valid=torch.ones(n, dtype=torch.bool), split_role=role,
    )


class M4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_m4_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor()
        cls.parameters = decode_m4_raw_parameters(cls.raw)

    def test_registry_extends_all_eight_m3_rows_exactly(self):
        self.assertEqual(self.registry.model_id, "M4_hsmm_delta_v1")
        self.assertEqual(self.registry.run_length_center, 23)
        self.assertEqual(self.registry.maximum_represented_run_length, 65)
        self.assertEqual(len(self.registry.initializations), 8)
        self.assertEqual(
            tuple(item.alpha_theta for item in self.registry.initializations),
            (0.02, 0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1),
        )
        for m4, m3 in zip(
            self.registry.initializations,
            self.registry.source_m3_registry.initializations,
            strict=True,
        ):
            self.assertEqual(m4.source_m3_start_id, m3.start_id)
            self.assertEqual(
                tuple(getattr(m4, name) for name in self.registry.parameter_order[:-1]),
                tuple(getattr(m3, name) for name in self.registry.parameter_order[:-1]),
            )
        for filename, expected_short_count in (
            ("m4_hsmm_delta_v1_mc06.json", 7225),
            ("m4_hsmm_delta_v1_mc07.json", 6852),
        ):
            subject = load_m4_model_registry(ROOT / "config" / filename)
            self.assertEqual(subject.task_moments.n_short, expected_short_count)

    def test_parameter_round_trip_and_centered_monotone_hazard(self):
        constrained = torch.tensor((.03, .55, .04, .6, 1.4, -3.2, .9, .08, .2), dtype=DTYPE)
        decoded = decode_m4_raw_parameters(encode_m4_constrained_parameters(constrained))
        torch.testing.assert_close(decoded.as_tensor(), constrained)
        run = torch.tensor((1., 23., 65.), dtype=DTYPE)
        hazard = decoded.hazard(run)
        self.assertTrue(bool(torch.all(hazard[1:] > hazard[:-1])))
        self.assertAlmostEqual(float(hazard[1]), float(torch.sigmoid(constrained[5])))

    def test_hazard_export_marks_center_and_capped_tail(self):
        curve = build_m4_hazard_curve(self.raw, self.registry)
        self.assertEqual(len(curve), 65)
        self.assertEqual(curve.loc[curve.is_center, "represented_run_length"].item(), 23)
        self.assertEqual(curve.loc[curve.is_capped_tail, "represented_run_length"].item(), 65)
        torch.testing.assert_close(
            torch.tensor(curve.neutral_hazard.to_numpy()),
            torch.tensor(curve.biased_hazard.to_numpy()) / 2.0,
        )

    def test_transition_conserves_mass_and_retains_tail_survival(self):
        joint = torch.zeros((3, 65), dtype=DTYPE)
        joint[2, -1] = 1.0
        hazard = torch.full((65,), .2, dtype=DTYPE)
        advanced = capped_tail_hsmm_transition(joint, hazard)
        self.assertAlmostEqual(float(advanced.sum()), 1.0)
        self.assertAlmostEqual(float(advanced[2, -1]), .8)
        self.assertAlmostEqual(float(advanced[1, 0]), .2)
        self.assertEqual(float(advanced[0].sum()), 0.0)

    def test_transition_endpoint_routing_and_constant_provider(self):
        posterior = torch.tensor((.2, .3, .5), dtype=DTYPE)
        provider = ConstantHazardPriorStateProvider(torch.tensor(.2, dtype=DTYPE))
        expected = posterior @ __import__(
            "m0_sensory_v1.model.m1_session", fromlist=["constant_hazard_transition_matrix"]
        ).constant_hazard_transition_matrix(torch.tensor(.2, dtype=DTYPE))
        torch.testing.assert_close(provider.transition(posterior), expected, rtol=0, atol=0)
        joint = torch.zeros((3, 4), dtype=DTYPE)
        joint[:, 2] = posterior
        no_switch = capped_tail_hsmm_transition(joint, torch.zeros(4, dtype=DTYPE))
        torch.testing.assert_close(no_switch[:, 3], posterior)
        all_switch = capped_tail_hsmm_transition(joint, torch.ones(4, dtype=DTYPE))
        torch.testing.assert_close(all_switch[:, 0], torch.tensor((0., .55, .35), dtype=DTYPE))
        torch.testing.assert_close(all_switch[0, 3], torch.tensor(.1, dtype=DTYPE))
        self.assertAlmostEqual(float(all_switch.sum()), 1.0)

    def test_reachable_support_and_final_trial_has_no_transition(self):
        evaluation = evaluate_m4_session(_session(n=8), self.parameters, self.registry.task_moments)
        prediction = evaluation.predictions
        self.assertEqual(prediction.maximum_reachable_run_length.tolist(), list(range(1, 9)))
        self.assertEqual(float(prediction.unreachable_support_mass_pre.max()), 0.0)
        self.assertEqual(float(prediction.unreachable_support_mass_post.max()), 0.0)
        self.assertTrue(math.isnan(float(prediction.transition_normalization_error[-1])))
        self.assertTrue(bool(torch.all(prediction.transition_normalization_error[:-1] < 1e-14)))

    def test_exact_bh_zero_reproduces_m3_beyond_cap(self):
        session = _session(n=75)
        parameters = self.parameters.with_explicit_bh(0.0)
        m4 = evaluate_m4_session(session, parameters, self.registry.task_moments)
        m3 = evaluate_m3_session(session, parameters.as_m3(), self.registry.task_moments)
        for left, right in (
            (m4.predictions.probability_long, m3.predictions.probability_long),
            (m4.predictions.prior_long, m3.predictions.prior_long),
            (m4.predictions.pre_choice_block_probability, m3.predictions.pre_choice_block_probability),
            (m4.predictions.post_feedback_block_probability, m3.predictions.post_feedback_block_probability),
            (m4.predictions.delta_pre, m3.predictions.delta_pre),
            (m4.predictions.delta_post, m3.predictions.delta_post),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=3e-15)
        self.assertGreater(float(m4.predictions.tail_probability_pre[-1]), 0)

    def test_current_category_selects_lapse_rate_and_feedback_changes_future_prior(self):
        parameters = decode_m4_raw_parameters(
            self.registry.initializations[1].raw_tensor()
        )
        left = evaluate_m4_session(_session(changed_index=None), parameters, self.registry.task_moments)
        right = evaluate_m4_session(_session(changed_index=4), parameters, self.registry.task_moments)
        self.assertEqual(float(left.predictions.prior_long[4]), float(right.predictions.prior_long[4]))
        self.assertEqual(float(left.predictions.non_lapse_probability_long[4]), float(right.predictions.non_lapse_probability_long[4]))
        self.assertNotEqual(float(left.predictions.probability_long[4]), float(right.predictions.probability_long[4]))
        self.assertNotEqual(float(left.predictions.prior_long[5]), float(right.predictions.prior_long[5]))

    def test_session_reset_and_order_invariant_loss_and_gradient(self):
        first = _session(session_id="first")
        second = _session(session_id="second", changed_index=3)

        def value(records):
            raw = self.raw.clone().requires_grad_()
            loss = evaluate_m4_mouse_objective(records, raw, self.registry.task_moments).loss
            return loss.detach(), torch.autograd.grad(loss, raw)[0]

        forward = value((first, second))
        reverse = value((second, first))
        torch.testing.assert_close(forward[0], reverse[0], rtol=0, atol=2e-15)
        torch.testing.assert_close(forward[1], reverse[1], rtol=1e-11, atol=2e-14)

    def test_all_nine_gradients_match_central_differences(self):
        raw = self.raw.clone().requires_grad_()

        def objective(value):
            return evaluate_m4_mouse_objective([_session()], value, self.registry.task_moments).loss

        gradient = torch.autograd.grad(objective(raw), raw)[0]
        finite_difference = []
        step = 1e-5
        for index in range(raw.numel()):
            offset = torch.zeros_like(raw)
            offset[index] = step
            finite_difference.append(float((objective(raw.detach()+offset)-objective(raw.detach()-offset))/(2*step)))
        torch.testing.assert_close(gradient, torch.tensor(finite_difference, dtype=DTYPE), rtol=2e-4, atol=2e-7)

    def test_one_step_adam_fit_smoke(self):
        run = fit_m4_single_initialization(
            [_session()], [_session(session_id="validation", role="validation")],
            self.registry, self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=1,
            allow_short_convergence_budget=True,
        )
        self.assertEqual(run.model_id, "M4_hsmm_delta_v1")
        self.assertEqual(len(run.final_constrained_parameters), 9)
        self.assertEqual(run.final_step, 1)
        self.assertTrue(torch.isfinite(torch.tensor(run.fresh_final_training_mean_nll)))

    def test_tail_gate_is_required_for_convergence_and_winner_eligibility(self):
        registry = replace(self.registry, maximum_accepted_mean_tail_probability=-1.0)
        run = fit_m4_single_initialization(
            [_session()], [_session(session_id="validation", role="validation")],
            registry, registry.initializations[0], optimizer_strategy="adam",
            max_epochs_override=1, allow_short_convergence_budget=True,
        )
        checks = dict(run.convergence_checks)
        self.assertFalse(run.tail_occupancy_acceptable)
        self.assertFalse(checks["tail_occupancy_acceptable"])
        self.assertFalse(run.converged)

    def test_parallel_multistart_matches_sequential_reference(self):
        registry = replace(self.registry, initializations=self.registry.initializations[:2])
        training = [_session(session_id="train")]
        validation = [_session(session_id="validation", role="validation")]
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            arguments = dict(
                optimizer_strategy="lbfgs",
                lbfgs_settings=LBFGSSettings(maximum_steps=1),
                allow_short_convergence_budget=True,
                cpu_threads_per_worker=1,
            )
            sequential = fit_m4_all_initializations(
                training, validation, registry, parallel_workers=1, **arguments
            )
            parallel = fit_m4_all_initializations(
                training, validation, registry, parallel_workers=2, **arguments
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual([run.start_id for run in parallel.runs], [1, 2])
        for left, right in zip(sequential.runs, parallel.runs, strict=True):
            self.assertEqual(left.final_raw_parameters, right.final_raw_parameters)
            self.assertEqual(
                left.recalculated_best_training_mean_nll,
                right.recalculated_best_training_mean_nll,
            )

    def test_paired_m4_m3_comparison_uses_only_aligned_validation_trials(self):
        base = pd.DataFrame({
            "mouse_id": ["m", "m"], "split_role": ["validation", "validation"],
            "session_id": ["s", "s"], "trial_index": [1, 2],
            "model_evaluated": [True, True],
        })
        m4 = base.assign(per_trial_negative_log_likelihood=[.2, .3])
        m3 = base.assign(per_trial_negative_log_likelihood=[.4, .5])
        result = compare_m4_to_m3_predictions(m4, m3)
        self.assertEqual(result.n_valid_trials.item(), 2)
        self.assertAlmostEqual(
            result.total_delta_bits_m4_vs_m3.item(), .4 / math.log(2)
        )

    def test_complete_comparison_uses_one_se_simpler_model_rule(self):
        base = pd.DataFrame({
            "mouse_id": ["m"] * 4, "split_role": ["validation"] * 4,
            "session_id": ["s1", "s1", "s2", "s2"], "trial_index": [1, 2, 1, 2],
            "choice_long": [0., 1., 0., 1.], "category_long": [0., 1., 1., 0.],
            "true_block": ["S", "S", "L", "L"], "model_evaluated": [True] * 4,
        })
        m4 = base.assign(model_probability_long=[.2, .8, .4, .6], per_trial_negative_log_likelihood=[.2]*4)
        m3 = base.assign(model_probability_long=[.25, .75, .45, .55], per_trial_negative_log_likelihood=[.3]*4)
        result = build_m4_m3_validation_comparison(m4, m3)
        self.assertEqual(result.overall.selected_model.item(), "M4_hsmm_delta_v1")
        self.assertEqual(len(result.per_session), 2)
        self.assertFalse(result.calibration.empty)

    def test_complete_comparison_accepts_matching_nullable_integer_observations(self):
        base = pd.DataFrame({
            "mouse_id": ["m"] * 3,
            "split_role": ["validation"] * 3,
            "session_id": ["s"] * 3,
            "trial_index": [1, 2, 3],
            "choice_long": pd.Series([0, pd.NA, 1], dtype="Int64"),
            "category_long": pd.Series([0, pd.NA, 1], dtype="Int64"),
            "true_block": ["S", "S", "L"],
            "model_evaluated": [True, False, True],
        })
        m4 = base.assign(
            model_probability_long=[0.2, float("nan"), 0.8],
            per_trial_negative_log_likelihood=[0.2, float("nan"), 0.2],
        )
        m3 = base.assign(
            model_probability_long=[0.25, float("nan"), 0.75],
            per_trial_negative_log_likelihood=[0.3, float("nan"), 0.3],
        )
        result = build_m4_m3_validation_comparison(m4, m3)
        self.assertEqual(result.overall.n_valid_trials.item(), 2)

    def test_compact_export_and_opt_in_full_marginals(self):
        n = 6
        session = _session(n=n)
        frame = pd.DataFrame({
            "mouse_id": ["mouse"]*n, "session_id": ["s1"]*n,
            "trial_index": range(n), "duration_s": session.duration_s.numpy(),
            "choice_long": session.choice_long.numpy(), "category_long": session.category_long.numpy(),
            "feedback_valid": [True]*n, "true_block": ["N"]*n, "split_role": ["train"]*n,
        })
        compact = build_m4_trial_predictions(frame, self.raw, self.registry)
        full = build_m4_trial_predictions(
            frame, self.raw, self.registry, include_full_run_length_distributions=True
        )
        self.assertNotIn("model_pre_choice_run_probability_r01", compact)
        self.assertIn("model_pre_choice_run_probability_r01", full)
        self.assertIn("model_run_length_entropy_pre", compact)
        self.assertEqual(len(full.columns) - len(compact.columns), 130)

    def test_m0_style_figures_include_all_five_families_and_bh(self):
        n = 48
        session = _session(n=n)
        frame = pd.DataFrame({
            "mouse_id": ["mouse"] * n,
            "session_id": ["s1"] * n,
            "trial_index": range(1, n + 1),
            "session_date": pd.Timestamp("2020-01-01"),
            "duration_s": session.duration_s.numpy(),
            "choice_long": session.choice_long.numpy(),
            "category_long": session.category_long.numpy(),
            "feedback_valid": [True] * n,
            "true_block": ["N"] * 12 + ["S"] * 12 + ["L"] * 12 + ["S"] * 12,
            "split_role": ["train"] * n,
        })
        predictions = build_m4_trial_predictions(frame, self.raw, self.registry)
        constrained = self.registry.initializations[0].constrained_tensor().tolist()
        fit_summary = {
            "model_id": "M4_hsmm_delta_v1",
            "mouse_id": "mouse",
            "winner_start_id": 1,
            "runs": [{
                "start_id": 1,
                "initial_constrained_parameters": constrained,
                "initial_training_mean_nll": 0.6,
                "initial_validation_mean_nll": 0.7,
                "history": [{
                    "step": 1,
                    "constrained_parameters": constrained,
                    "fresh_training_mean_nll": 0.5,
                    "validation_mean_nll": 0.65,
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_m4_m0_style_figures(
                predictions, fit_summary, output, test_unlocked=False
            )
            parameter_history = pd.read_csv(output / "parameter_history.csv")
            self.assertEqual(len(manifest["files"]), 5)
            self.assertIn("bh", parameter_history.columns)
            self.assertTrue((output / "psychometrics_mouse_vs_model.png").is_file())
            self.assertTrue((output / "m4_m0_style_figure_manifest.json").is_file())

    def test_low_hazard_duration_distribution_exceeds_default_cap_without_raising(self):
        """A valid, low fitted hazard can require far more than 5000 trials
        to reach the tail tolerance (survival decays as (1-h)^n, so small h
        needs n ~ 27.6/h steps). The distribution builder must extend past
        its default row-count cap analytically instead of raising on a
        legitimate fit."""
        low_hazard_constrained = torch.tensor(
            (.03, .55, .04, .6, 1.4, -8.0, .9, .08, 1e-6), dtype=DTYPE
        )
        raw = encode_m4_constrained_parameters(low_hazard_constrained)
        hazard_r23 = float(torch.sigmoid(low_hazard_constrained[5]))
        self.assertLess(hazard_r23, 1e-3)
        distribution = build_m4_implied_duration_distribution(raw, self.registry)
        self.assertGreater(distribution.duration_trials.max(), 5000)
        self.assertGreater(distribution.probability.sum(), 1 - 1e-11)
        self.assertTrue(distribution.uses_frozen_tail_hazard.iloc[-1])

    def test_hazard_by_run_length_is_cached_not_recomputed_per_trial(self):
        """transition() re-reads hazard_by_run_length on every trial; it
        must be the identical cached tensor each time, not a fresh sigmoid
        evaluation, since it depends only on this (frozen, one-per-session)
        provider's fixed ah/bh."""
        provider = CappedTailHSMMPriorStateProvider(parameters=self.parameters)
        first = provider.hazard_by_run_length
        second = provider.hazard_by_run_length
        self.assertIs(first, second)
        self.assertIs(provider.run_lengths, provider.run_lengths)
        self.assertIs(provider.state_hazard_by_run_length, provider.state_hazard_by_run_length)

    def test_duration_summary_truncation_sensitivity_and_identifiability(self):
        distribution = build_m4_implied_duration_distribution(self.raw, self.registry)
        neutral_distribution = build_m4_implied_duration_distribution(
            self.raw, self.registry, block_state="neutral"
        )
        self.assertGreater(distribution.probability.sum(), 1 - 1e-11)
        self.assertGreater(neutral_distribution.probability.sum(), 1 - 1e-11)
        summary = summarize_m4_hazard_duration(self.raw, self.registry)
        self.assertEqual(summary.hazard_r23, float(torch.sigmoid(self.parameters.ah)))
        self.assertEqual(summary.neutral_hazard_r23, summary.hazard_r23 / 2.0)
        self.assertGreater(summary.neutral_implied_mean_duration, summary.implied_mean_duration)
        sensitivity = evaluate_m4_truncation_sensitivity(
            [_session()], self.raw, self.registry, maximum_run_lengths=(65, 75)
        )
        self.assertEqual(sensitivity.maximum_run_length.tolist(), [65, 75])
        surface = evaluate_m4_dynamic_surface(
            [_session()], self.raw, self.registry, pair=("ah", "bh"), grid_points=3
        )
        self.assertEqual(len(surface), 9)
        self.assertTrue(torch.isfinite(torch.tensor(surface.mean_nll.to_numpy())).all())

    def test_recovery_grid_contains_registered_boundaries(self):
        cases = list(iter_registered_m4_recovery_grid((.03, .5, .05, .5, 1.0)))
        self.assertEqual(M4_RECOVERY_BH_GRID, (0.0, .05, .10, .20, .40))
        self.assertTrue(any(float(case[7]) == 0.0 for case in cases))
        self.assertTrue(any(float(case[8]) == 0.0 for case in cases))
        self.assertTrue(any(abs(float(torch.sigmoid(case[5])) - 1/23) < 1e-14 for case in cases))


if __name__ == "__main__":
    unittest.main()
