"""Scientific and integration tests for the M1 true-rate HMM."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.m1_analysis import build_m1_trial_predictions
from m0_sensory_v1.m1_figures import generate_m1_diagnostic_figures
from m0_sensory_v1.m1_m0_style_figures import generate_m1_m0_style_figures
from m0_sensory_v1.m1_registry import load_m1_model_registry
from m0_sensory_v1.model_comparison import compare_m1_to_m0_predictions
from m0_sensory_v1.registry import load_model_registry
from m0_sensory_v1.model import (
    LogOddsCoefficients,
    M1SessionTrials,
    constant_hazard_transition_matrix,
    decode_m1_raw_parameters,
    encode_m1_constrained_parameters,
    evaluate_m1_mouse_objective,
    evaluate_m1_session,
    fit_single_initialization,
    integrate_general_decision_region,
    simulate_m1_choices,
    simulate_m1_hmm_sessions,
)


DTYPE = torch.float64


def _session(
    categories=(0.0, 1.0, 0.0),
    choices=(0.0, 1.0, 0.0),
    feedback=(True, True, True),
    durations=(0.5, 2.0, 0.5),
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


class M1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max,
            epsilon=cls.registry.parameter_epsilon,
        )
        cls.parameters = decode_m1_raw_parameters(
            cls.raw,
            lapse_max=cls.registry.lapse_max,
            epsilon=cls.registry.parameter_epsilon,
        )

    def _evaluate(self, session: M1SessionTrials):
        return evaluate_m1_session(
            session,
            self.parameters,
            self.registry.task_moments,
            first_long_probability=self.registry.first_long_probability,
            prior_probability_epsilon=self.registry.prior_probability_epsilon,
            quadratic_tolerance=self.registry.quadratic_tolerance,
            discriminant_tolerance=self.registry.discriminant_tolerance,
        )

    def test_registry_and_subject_overlays(self) -> None:
        self.assertEqual(self.registry.state_order, ("N", "S", "L"))
        self.assertEqual(self.registry.true_rate_prototypes, (0.5, 0.2, 0.8))
        self.assertEqual(len(self.registry.initializations), 8)
        m0_starts = load_model_registry().initializations
        self.assertEqual(
            [(x.w, x.s0, x.ell_short, x.ell_long, x.kappa) for x in self.registry.initializations],
            [(x.w, x.s0, x.ell_short, x.ell_long, x.kappa) for x in m0_starts],
        )
        np.testing.assert_allclose(
            [x.hazard for x in self.registry.initializations],
            [1/30, .01, .02, .05, .08, .12, .20, .35],
        )
        for name, expected in (
            ("m1_hmm_true_v1_mc06.json", "chronological_70_15_15_mc06_v1"),
            ("m1_hmm_true_v1_mc07.json", "chronological_70_15_15_mc07_v1"),
        ):
            loaded = load_m1_model_registry(ROOT / "config" / name)
            self.assertEqual(loaded.split.version, expected)
            self.assertEqual(loaded.initializations, self.registry.initializations)

    def test_parameter_round_trip_and_hazard(self) -> None:
        constrained = torch.tensor([0.03, 0.55, 0.04, 0.6, 1.4, -3.2], dtype=DTYPE)
        raw = encode_m1_constrained_parameters(constrained)
        decoded = decode_m1_raw_parameters(raw)
        torch.testing.assert_close(decoded.as_tensor(), constrained)
        self.assertGreater(float(decoded.hazard), 0.0)
        self.assertLess(float(decoded.hazard), 1.0)
        self.assertAlmostEqual(float(decoded.hazard), 1 / (1 + math.exp(3.2)))

    def test_transition_kernel_endpoints_and_probability_conservation(self) -> None:
        zero = constant_hazard_transition_matrix(torch.tensor(0.0, dtype=DTYPE))
        torch.testing.assert_close(zero, torch.eye(3, dtype=DTYPE))
        one = constant_hazard_transition_matrix(torch.tensor(1.0, dtype=DTYPE))
        expected = torch.tensor([[.5, .25, .25], [0, 0, 1], [0, 1, 0]], dtype=DTYPE)
        torch.testing.assert_close(one, expected)
        middle = constant_hazard_transition_matrix(torch.tensor(.13, dtype=DTYPE))
        torch.testing.assert_close(middle.sum(1), torch.ones(3, dtype=DTYPE))
        self.assertAlmostEqual(float(middle[0, 1] + middle[0, 2]), .13 / 2.0)
        self.assertAlmostEqual(1.0 / float(middle[0, 1] + middle[0, 2]), 2.0 / .13)

    def test_general_decision_regions_match_normal_probabilities(self) -> None:
        duration = torch.tensor(0.0, dtype=DTYPE)
        sd = torch.tensor(1.0, dtype=DTYPE)
        outside, region = integrate_general_decision_region(
            duration, sd, LogOddsCoefficients(
                torch.tensor(1.0, dtype=DTYPE),
                torch.tensor(0.0, dtype=DTYPE),
                torch.tensor(-1.0, dtype=DTYPE),
            )
        )
        expected_outside = math.erfc(1 / math.sqrt(2))
        self.assertAlmostEqual(float(outside), expected_outside, places=13)
        self.assertEqual(region.long_region, "x < lower_boundary or x > upper_boundary")
        inside, _ = integrate_general_decision_region(
            duration, sd, LogOddsCoefficients(
                torch.tensor(-1.0, dtype=DTYPE),
                torch.tensor(0.0, dtype=DTYPE),
                torch.tensor(1.0, dtype=DTYPE),
            )
        )
        self.assertAlmostEqual(float(inside), 1.0 - expected_outside, places=13)
        upper, _ = integrate_general_decision_region(
            duration, sd, LogOddsCoefficients(
                torch.tensor(0.0, dtype=DTYPE),
                torch.tensor(1.0, dtype=DTYPE),
                torch.tensor(-1.0, dtype=DTYPE),
            )
        )
        self.assertAlmostEqual(float(upper), expected_outside / 2, places=13)

    def test_general_solver_handles_double_no_root_and_constant_cases(self) -> None:
        x = torch.tensor(0.3, dtype=DTYPE)
        sd = torch.tensor(0.7, dtype=DTYPE)
        cases = (
            ((1.0, 0.0, 0.0), 1.0),
            ((-1.0, 0.0, 0.0), 0.0),
            ((1.0, 0.0, 1.0), 1.0),
            ((-1.0, 0.0, -1.0), 0.0),
            ((0.0, 0.0, 1.0), 1.0),
            ((0.0, 0.0, -1.0), 0.0),
            ((0.0, 0.0, 0.0), 0.0),
        )
        for coefficients, expected in cases:
            q, _ = integrate_general_decision_region(
                x,
                sd,
                LogOddsCoefficients(*(torch.tensor(v, dtype=DTYPE) for v in coefficients)),
            )
            self.assertEqual(float(q), expected)

    def test_current_category_selects_lapse_rate_but_updates_belief_only_after_choice(self) -> None:
        parameters = decode_m1_raw_parameters(
            self.registry.initializations[1].raw_tensor()
        )
        first_short = evaluate_m1_session(
            _session(categories=(0, 0, 0)), parameters, self.registry.task_moments
        )
        first_long = evaluate_m1_session(
            _session(categories=(0, 1, 0)), parameters, self.registry.task_moments
        )
        p_short = first_short.predictions
        p_long = first_long.predictions
        self.assertEqual(float(p_short.prior_long[1]), float(p_long.prior_long[1]))
        self.assertEqual(
            float(p_short.non_lapse_probability_long[1]),
            float(p_long.non_lapse_probability_long[1]),
        )
        self.assertNotEqual(float(p_short.probability_long[1]), float(p_long.probability_long[1]))
        self.assertNotEqual(float(p_short.prior_long[2]), float(p_long.prior_long[2]))

    def test_omission_has_no_likelihood_but_still_transitions(self) -> None:
        evaluation = self._evaluate(
            _session(
                categories=(float("nan"), 1, 0),
                choices=(float("nan"), 1, 0),
                feedback=(False, True, True),
                durations=(float("nan"), 2, .5),
            )
        )
        prediction = evaluation.predictions
        self.assertTrue(torch.isnan(prediction.probability_long[0]))
        self.assertEqual(evaluation.valid_trial_count, 2)
        expected = torch.tensor(
            [1 - float(self.parameters.hazard) / 2,
             float(self.parameters.hazard) / 4,
             float(self.parameters.hazard) / 4],
            dtype=DTYPE,
        )
        torch.testing.assert_close(prediction.pre_choice_block_probability[1], expected)

    def test_feedback_valid_omission_updates_future_state(self) -> None:
        short = self._evaluate(
            _session(
                categories=(0, 0, 0), choices=(0, float("nan"), 0),
                feedback=(True, True, True), durations=(.5, 2, .5)
            )
        )
        long = self._evaluate(
            _session(
                categories=(0, 1, 0), choices=(0, float("nan"), 0),
                feedback=(True, True, True), durations=(.5, 2, .5)
            )
        )
        self.assertTrue(torch.isnan(short.predictions.probability_long[1]))
        self.assertNotEqual(
            float(short.predictions.prior_long[2]),
            float(long.predictions.prior_long[2]),
        )

    def test_objective_resets_sessions_is_order_invariant_and_has_gradients(self) -> None:
        sessions = (_session(session_id="a"), _session(session_id="b"))
        raw = self.raw.clone().requires_grad_()
        first = evaluate_m1_mouse_objective(
            sessions, raw, self.registry.task_moments,
            first_long_probability=self.registry.first_long_probability,
            prior_probability_epsilon=self.registry.prior_probability_epsilon,
            quadratic_tolerance=self.registry.quadratic_tolerance,
            discriminant_tolerance=self.registry.discriminant_tolerance,
        )
        first.loss.backward()
        self.assertTrue(bool(torch.all(torch.isfinite(raw.grad))))
        second = evaluate_m1_mouse_objective(
            tuple(reversed(sessions)), self.raw, self.registry.task_moments,
            first_long_probability=self.registry.first_long_probability,
            prior_probability_epsilon=self.registry.prior_probability_epsilon,
            quadratic_tolerance=self.registry.quadratic_tolerance,
            discriminant_tolerance=self.registry.discriminant_tolerance,
        )
        torch.testing.assert_close(first.loss.detach(), second.loss)
        for evaluation in first.session_evaluations:
            torch.testing.assert_close(
                evaluation.predictions.pre_choice_block_probability[0],
                torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
            )

    def test_all_six_objective_gradients_match_finite_differences(self) -> None:
        session = _session(
            categories=(0, 0, 1, 1, 0, 1),
            choices=(0, 0, 1, 1, 0, 1),
            feedback=(True,) * 6,
            durations=(.4, .7, 1.8, 2.1, .6, 2.0),
        )

        def objective(raw):
            return evaluate_m1_mouse_objective(
                [session], raw, self.registry.task_moments,
                first_long_probability=self.registry.first_long_probability,
                prior_probability_epsilon=self.registry.prior_probability_epsilon,
                quadratic_tolerance=self.registry.quadratic_tolerance,
                discriminant_tolerance=self.registry.discriminant_tolerance,
            ).loss

        raw = self.raw.clone().requires_grad_()
        analytical = torch.autograd.grad(objective(raw), raw)[0]
        numerical = torch.empty_like(raw)
        step = 1e-6
        for index in range(raw.numel()):
            plus, minus = raw.detach().clone(), raw.detach().clone()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (objective(plus) - objective(minus)) / (2 * step)
        torch.testing.assert_close(analytical, numerical, rtol=2e-4, atol=2e-7)

    def test_shared_optimizer_executes_bounded_m1_fit(self) -> None:
        train = _session(session_id="train")
        validation = _session(session_id="validation", role="validation")
        result = fit_single_initialization(
            [train], [validation], self.registry, self.registry.initializations[0],
            max_epochs_override=1,
        )
        self.assertEqual(result.model_id, "M1_hmm_true_v1")
        self.assertEqual(len(result.best_raw_parameters), 6)
        self.assertEqual(result.optimizer_step_count, 1)
        self.assertIsNone(result.exception_message)

    def test_design_preserving_simulation_is_reproducible(self) -> None:
        template = _session(choices=(0, 1, float("nan")), durations=(.5, 2, float("nan")))
        first = simulate_m1_choices(
            [template], self.parameters, self.registry.task_moments, seed=71
        )[0]
        second = simulate_m1_choices(
            [template], self.parameters, self.registry.task_moments, seed=71
        )[0]
        torch.testing.assert_close(first.choice_long, second.choice_long, equal_nan=True)
        self.assertTrue(torch.isnan(first.choice_long[-1]))
        torch.testing.assert_close(first.category_long, template.category_long)

    def test_trial_export_contains_causal_state_diagnostics(self) -> None:
        frame = pd.DataFrame({
            "mouse_id": ["mouse"] * 3,
            "session_id": ["s1"] * 3,
            "trial_index": [1, 2, 3],
            "session_date": pd.to_datetime(["2020-01-01"] * 3),
            "duration_s": [.5, 2., .5],
            "choice_long": [0, 1, 0],
            "category_long": [0, 1, 0],
            "feedback_valid": [1, 1, 1],
            "true_block": ["N", "S", "S"],
            "outcome": ["Reward"] * 3,
            "split_role": ["train"] * 3,
        })
        result = build_m1_trial_predictions(frame, self.raw, self.registry)
        expected = {
            "model_prior_long", "model_pre_choice_probability_N",
            "model_pre_choice_probability_S", "model_pre_choice_probability_L",
            "model_probability_true_block", "fitted_ah", "fitted_hazard",
            "model_emission_normalizer", "model_decision_regime",
            "fitted_subjective_mean_block_duration_trials",
        }
        self.assertTrue(expected.issubset(result.columns))
        np.testing.assert_allclose(
            result[[
                "model_pre_choice_probability_N",
                "model_pre_choice_probability_S",
                "model_pre_choice_probability_L",
            ]].sum(axis=1),
            1.0,
        )
        self.assertAlmostEqual(float(result["model_emission_normalizer"].iloc[0]), .5)
        self.assertTrue(np.isfinite(result["model_emission_normalizer"]).all())
        self.assertTrue(result["model_decision_regime"].str.len().gt(0).all())

    def test_m1_boundary_diagnostics_include_hazard(self) -> None:
        result = fit_single_initialization(
            [_session(session_id="train")],
            [_session(session_id="validation", role="validation")],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=1,
        )
        boundary = result.boundary_proximity
        self.assertTrue(hasattr(boundary, "hazard_fraction_to_nearest_boundary"))
        self.assertGreater(boundary.subjective_mean_block_duration_trials, 0)

    def test_generative_hmm_and_figure_workflow_smoke(self) -> None:
        duration = torch.tensor([.3,.6,1.8,2.1] * 8, dtype=DTYPE)
        generated, truth = simulate_m1_hmm_sessions(
            [duration], self.registry.initializations[0].constrained_tensor(),
            self.registry.task_moments, random_seed=77,
        )
        self.assertEqual(truth[0].shape, duration.shape)
        self.assertTrue(bool(torch.all((generated[0].choice_long == 0) | (generated[0].choice_long == 1))))
        frame = pd.DataFrame({
            "mouse_id":"mouse", "session_id":"s", "trial_index":np.arange(len(duration)),
            "session_date":pd.Timestamp("2020-01-01"), "duration_s":duration.numpy(),
            "choice_long":generated[0].choice_long.numpy(), "category_long":generated[0].category_long.numpy(),
            "feedback_valid":1, "true_block":np.asarray([("N","S","L")[int(x)] for x in truth[0]]),
            "outcome":"synthetic", "split_role":"train", "omission":0,
        })
        predictions = build_m1_trial_predictions(frame, self.raw, self.registry)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_m1_diagnostic_figures(predictions, temporary)
            self.assertIn("belief_trajectory", manifest["files"])
            self.assertTrue((Path(temporary) / "m1_figure_manifest.json").is_file())

    def test_m0_style_m1_figures_include_all_five_families_and_ah(self) -> None:
        duration = torch.tensor([.3, .6, 1.8, 2.1] * 8, dtype=DTYPE)
        generated, truth = simulate_m1_hmm_sessions(
            [duration],
            self.registry.initializations[0].constrained_tensor(),
            self.registry.task_moments,
            random_seed=91,
        )
        frame = pd.DataFrame({
            "mouse_id": "mouse",
            "session_id": "s",
            "trial_index": np.arange(len(duration)),
            "session_date": pd.Timestamp("2020-01-01"),
            "duration_s": duration.numpy(),
            "choice_long": generated[0].choice_long.numpy(),
            "category_long": generated[0].category_long.numpy(),
            "feedback_valid": 1,
            "true_block": np.asarray([
                ("N", "S", "L")[int(value)] for value in truth[0]
            ]),
            "outcome": "synthetic",
            "split_role": "train",
            "omission": 0,
        })
        predictions = build_m1_trial_predictions(frame, self.raw, self.registry)
        constrained = self.registry.initializations[0].constrained_tensor().tolist()
        fit_summary = {
            "model_id": "M1_hmm_true_v1",
            "mouse_id": "mouse",
            "winner_start_id": 1,
            "runs": [{
                "start_id": 1,
                "initial_constrained_parameters": constrained,
                "initial_training_mean_nll": 0.6,
                "initial_validation_mean_nll": 0.7,
                "history": [{
                    "epoch": 1,
                    "constrained_parameters_after_step": constrained,
                    "fresh_training_mean_nll": 0.5,
                    "validation_mean_nll": 0.65,
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_m1_m0_style_figures(
                predictions,
                fit_summary,
                temporary,
                test_unlocked=False,
            )
            parameter_history = pd.read_csv(
                Path(temporary) / "parameter_history.csv"
            )
            self.assertEqual(len(manifest["files"]), 5)
            self.assertIn("ah", parameter_history.columns)
            self.assertTrue(
                (Path(temporary) / "psychometrics_mouse_vs_model.png").is_file()
            )
            self.assertTrue(
                (Path(temporary) / "m1_m0_style_figure_manifest.json").is_file()
            )

    def test_paired_model_comparison_reports_bits_per_trial(self) -> None:
        common = {
            "mouse_id": ["m", "m"], "session_id": ["s", "s"],
            "trial_index": [1, 2], "split_role": ["validation", "validation"],
            "choice_long": [0, 1], "model_evaluated": [True, True],
        }
        m0 = pd.DataFrame({**common, "per_trial_negative_log_likelihood": [0.8, 0.6]})
        m1 = pd.DataFrame({**common, "per_trial_negative_log_likelihood": [0.6, 0.4]})
        result = compare_m1_to_m0_predictions(m0, m1)
        self.assertAlmostEqual(result.delta_elpd_m1_minus_m0, 0.4)
        self.assertAlmostEqual(
            result.delta_bits_per_trial_m1_minus_m0, 0.4 / (2 * math.log(2))
        )


if __name__ == "__main__":
    unittest.main()
