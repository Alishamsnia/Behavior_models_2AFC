"""Scientific, optimizer, export, and figure tests for M2."""

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

from m0_sensory_v1.m1_registry import load_m1_model_registry
from m0_sensory_v1.m2_analysis import build_m2_trial_predictions
from m0_sensory_v1.m2_figures import (
    generate_m2_diagnostic_figures,
    generate_m2_identifiability_figure,
)
from m0_sensory_v1.m2_m0_style_figures import generate_m2_m0_style_figures
from m0_sensory_v1.m2_registry import load_m2_model_registry
from m0_sensory_v1.registry import load_model_registry
from m0_sensory_v1.model import (
    LBFGSSettings,
    M2DecodedParameters,
    M1SessionTrials,
    SessionTrials,
    decode_m1_raw_parameters,
    decode_m2_raw_parameters,
    encode_m2_constrained_parameters,
    evaluate_m1_mouse_objective,
    evaluate_m1_session,
    evaluate_m2_hazard_contrast_surface,
    evaluate_m2_mouse_objective,
    evaluate_m2_session,
    evaluate_session,
    fit_single_initialization_with_strategy,
    simulate_m2_choices,
    simulate_m2_hmm_sessions,
    simulate_m2_real_design_sessions,
)


DTYPE = torch.float64


def _session(
    categories=(0.0, 1.0, 0.0, 1.0),
    choices=(0.0, 1.0, 0.0, 1.0),
    feedback=(True, True, True, True),
    durations=(0.5, 2.0, 0.6, 1.9),
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


class M2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m2_model_registry()
        cls.m1_registry = load_m1_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max,
            epsilon=cls.registry.parameter_epsilon,
        )
        cls.parameters = decode_m2_raw_parameters(
            cls.raw,
            lapse_max=cls.registry.lapse_max,
            epsilon=cls.registry.parameter_epsilon,
        )

    def test_registry_preserves_all_eight_m1_starts_and_adds_log4(self) -> None:
        self.assertEqual(self.registry.model_id, "M2_hmm_static_v1")
        self.assertEqual(len(self.registry.initializations), 8)
        for m2, m1 in zip(
            self.registry.initializations,
            self.m1_registry.initializations,
            strict=True,
        ):
            self.assertEqual(
                (m2.start_id, m2.w, m2.s0, m2.ell_short, m2.ell_long, m2.kappa, m2.ah),
                (m1.start_id, m1.w, m1.s0, m1.ell_short, m1.ell_long, m1.kappa, m1.ah),
            )
            self.assertEqual(m2.delta0, math.log(4.0))
        for filename, expected in (
            ("m2_hmm_static_v1_mc06.json", "chronological_70_15_15_mc06_v1"),
            ("m2_hmm_static_v1_mc07.json", "chronological_70_15_15_mc07_v1"),
        ):
            loaded = load_m2_model_registry(ROOT / "config" / filename)
            self.assertEqual(loaded.split.version, expected)
            self.assertEqual(loaded.initializations, self.registry.initializations)

    def test_parameter_round_trip_and_static_symmetric_prototypes(self) -> None:
        constrained = torch.tensor([0.03, 0.55, 0.04, 0.6, 1.4, -3.2, 0.9], dtype=DTYPE)
        decoded = decode_m2_raw_parameters(encode_m2_constrained_parameters(constrained))
        torch.testing.assert_close(decoded.as_tensor(), constrained)
        theta = decoded.prototypes()
        self.assertEqual(float(theta[0]), 0.5)
        self.assertLess(float(theta[1]), 0.5)
        self.assertGreater(float(theta[2]), 0.5)
        self.assertAlmostEqual(float(theta[1] + theta[2]), 1.0)

    def test_exact_log4_restriction_matches_m1_trialwise(self) -> None:
        session = _session(categories=(0, 0, 1, 1, 0, 1), choices=(0, 0, 1, 1, 0, 1), feedback=(True,) * 6, durations=(.4, .7, 1.8, 2.1, .6, 2.0))
        m1_start = self.m1_registry.initializations[0]
        m1_parameters = decode_m1_raw_parameters(m1_start.raw_tensor())
        m2_parameters = decode_m2_raw_parameters(
            encode_m2_constrained_parameters(
                torch.cat((m1_start.constrained_tensor(), torch.tensor([math.log(4.0)], dtype=DTYPE)))
            )
        )
        m1 = evaluate_m1_session(session, m1_parameters, self.registry.task_moments)
        m2 = evaluate_m2_session(session, m2_parameters, self.registry.task_moments)
        torch.testing.assert_close(m2.predictions.subjective_prototypes, torch.tensor([.5, .2, .8], dtype=DTYPE), rtol=0, atol=2e-11)
        torch.testing.assert_close(m2.predictions.probability_long, m1.predictions.probability_long, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.predictions.non_lapse_probability_long, m1.predictions.non_lapse_probability_long, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.predictions.prior_long, m1.predictions.prior_long, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.predictions.pre_choice_block_probability, m1.predictions.pre_choice_block_probability, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.predictions.post_feedback_block_probability, m1.predictions.post_feedback_block_probability, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.predictions.emission_normalizer, m1.predictions.emission_normalizer, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.per_trial_negative_log_likelihood, m1.per_trial_negative_log_likelihood, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(m2.negative_log_likelihood, m1.negative_log_likelihood, rtol=1e-11, atol=1e-12)

        second = _session(
            categories=(1, 0, 1, 0), choices=(1, 0, 1, 0),
            session_id="s2", durations=(2.0, .5, 1.9, .7),
        )
        m1_objective = evaluate_m1_mouse_objective(
            [session, second], m1_start.raw_tensor(), self.registry.task_moments
        )
        m2_objective = evaluate_m2_mouse_objective(
            [session, second],
            encode_m2_constrained_parameters(
                torch.cat((m1_start.constrained_tensor(), torch.tensor([math.log(4.0)], dtype=DTYPE)))
            ),
            self.registry.task_moments,
        )
        torch.testing.assert_close(m2_objective.loss, m1_objective.loss, rtol=1e-11, atol=1e-12)

    def test_exact_zero_contrast_reduces_to_m0(self) -> None:
        m0_registry = load_model_registry()
        common = decode_m1_raw_parameters(
            self.m1_registry.initializations[0].raw_tensor()
        ).common()
        ah = torch.tensor(-2.7, dtype=DTYPE)
        m2_parameters = M2DecodedParameters(
            w=common.w,
            s0=common.s0,
            ell_short=common.ell_short,
            ell_long=common.ell_long,
            kappa=common.kappa,
            ah=ah,
            delta0=torch.tensor(0.0, dtype=DTYPE),
            hazard=torch.sigmoid(ah),
        )
        m2_session = _session(
            categories=(0, 1, 1, 0, 1, 0),
            choices=(0, 1, 1, 0, 1, 0),
            feedback=(True,) * 6,
            durations=(.4, 1.8, 2.1, .6, 1.9, .7),
        )
        m0_session = SessionTrials(
            mouse_id=m2_session.mouse_id,
            session_id=m2_session.session_id,
            duration_s=m2_session.duration_s,
            category_long=m2_session.category_long,
            choice_long=m2_session.choice_long,
            split_role="train",
        )
        m0 = evaluate_session(m0_session, common, m0_registry.task_moments)
        m2 = evaluate_m2_session(m2_session, m2_parameters, m0_registry.task_moments)
        torch.testing.assert_close(
            m2.predictions.subjective_prototypes,
            torch.full((3,), .5, dtype=DTYPE),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            m2.predictions.prior_long,
            torch.full((6,), .5, dtype=DTYPE),
            rtol=0,
            atol=2e-15,
        )
        torch.testing.assert_close(
            m2.predictions.probability_long,
            m0.predictions.probability_long,
            rtol=1e-12,
            atol=1e-13,
        )
        torch.testing.assert_close(
            m2.per_trial_negative_log_likelihood,
            m0.per_trial_negative_log_likelihood,
            rtol=1e-12,
            atol=1e-13,
        )
        torch.testing.assert_close(
            m2.negative_log_likelihood,
            m0.negative_log_likelihood,
            rtol=1e-12,
            atol=1e-13,
        )

    def test_non_log4_prototype_is_shared_by_prior_and_emission(self) -> None:
        delta0 = torch.tensor(.7, dtype=DTYPE)
        parameters = M2DecodedParameters(
            w=self.parameters.w,
            s0=self.parameters.s0,
            ell_short=self.parameters.ell_short,
            ell_long=self.parameters.ell_long,
            kappa=self.parameters.kappa,
            ah=self.parameters.ah,
            delta0=delta0,
            hazard=self.parameters.hazard,
        )
        evaluation = evaluate_m2_session(
            _session(categories=(1, 0, 1, 0)), parameters, self.registry.task_moments
        )
        theta = torch.stack((
            torch.tensor(.5, dtype=DTYPE),
            torch.sigmoid(-delta0),
            torch.sigmoid(delta0),
        ))
        torch.testing.assert_close(evaluation.predictions.subjective_prototypes, theta)
        pre = evaluation.predictions.pre_choice_block_probability[0]
        torch.testing.assert_close(evaluation.predictions.prior_long[0], pre @ theta)
        emission = theta
        normalizer = torch.sum(pre * emission)
        torch.testing.assert_close(evaluation.predictions.emission_normalizer[0], normalizer)
        torch.testing.assert_close(
            evaluation.predictions.post_feedback_block_probability[0],
            pre * emission / normalizer,
        )
        second_session = evaluate_m2_session(
            _session(
                categories=(0, 1, 0, 1),
                session_id="different_history",
            ),
            parameters,
            self.registry.task_moments,
        )
        torch.testing.assert_close(
            second_session.predictions.subjective_prototypes, theta, rtol=0, atol=0
        )

    def test_current_category_selects_lapse_rate_and_changes_future_belief(self) -> None:
        parameters = decode_m2_raw_parameters(
            self.registry.initializations[1].raw_tensor()
        )
        short = evaluate_m2_session(_session(categories=(0, 0, 0, 1)), parameters, self.registry.task_moments)
        long = evaluate_m2_session(_session(categories=(0, 1, 0, 1)), parameters, self.registry.task_moments)
        self.assertEqual(float(short.predictions.prior_long[1]), float(long.predictions.prior_long[1]))
        self.assertEqual(float(short.predictions.non_lapse_probability_long[1]), float(long.predictions.non_lapse_probability_long[1]))
        self.assertNotEqual(float(short.predictions.probability_long[1]), float(long.predictions.probability_long[1]))
        self.assertNotEqual(float(short.predictions.prior_long[2]), float(long.predictions.prior_long[2]))

    def test_sessions_reset_and_do_not_carry_previous_belief(self) -> None:
        previous_short = _session(
            categories=(0, 0, 0, 0), session_id="previous",
        )
        previous_long = _session(
            categories=(1, 1, 1, 1), session_id="previous",
        )
        current = _session(
            categories=(1, 0, 1, 0), session_id="current",
        )
        left = evaluate_m2_mouse_objective(
            [previous_short, current], self.raw, self.registry.task_moments
        ).session_evaluations[1]
        right = evaluate_m2_mouse_objective(
            [previous_long, current], self.raw, self.registry.task_moments
        ).session_evaluations[1]
        torch.testing.assert_close(
            left.predictions.pre_choice_block_probability,
            right.predictions.pre_choice_block_probability,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            left.predictions.probability_long,
            right.predictions.probability_long,
            rtol=0,
            atol=0,
        )

    def test_session_order_preserves_loss_and_gradient(self) -> None:
        first = _session(
            categories=(0, 0, 1, 1), session_id="first",
        )
        second = _session(
            categories=(1, 0, 1, 0), session_id="second",
            durations=(2.1, .6, 1.8, .5),
        )

        def loss_and_gradient(sessions):
            raw = self.raw.clone().requires_grad_()
            loss = evaluate_m2_mouse_objective(
                sessions, raw, self.registry.task_moments
            ).loss
            return loss.detach(), torch.autograd.grad(loss, raw)[0]

        forward_loss, forward_gradient = loss_and_gradient([first, second])
        reverse_loss, reverse_gradient = loss_and_gradient([second, first])
        torch.testing.assert_close(forward_loss, reverse_loss, rtol=0, atol=2e-15)
        torch.testing.assert_close(
            forward_gradient, reverse_gradient, rtol=1e-12, atol=2e-15
        )

    def test_all_seven_gradients_match_finite_differences(self) -> None:
        session = _session(categories=(0, 0, 1, 1, 0, 1), choices=(0, 0, 1, 1, 0, 1), feedback=(True,) * 6, durations=(.4, .7, 1.8, 2.1, .6, 2.0))

        def objective(raw):
            return evaluate_m2_mouse_objective([session], raw, self.registry.task_moments).loss

        raw = self.raw.clone().requires_grad_()
        analytical = torch.autograd.grad(objective(raw), raw)[0]
        numerical = torch.empty_like(raw)
        step = 1e-6
        for index in range(raw.numel()):
            plus, minus = raw.detach().clone(), raw.detach().clone()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (objective(plus) - objective(minus)) / (2 * step)
        torch.testing.assert_close(analytical, numerical, rtol=3e-4, atol=3e-7)

    def test_all_three_optimizers_execute_on_m2(self) -> None:
        train = [_session(session_id="train")]
        validation = [_session(session_id="validation", role="validation")]
        settings = LBFGSSettings(maximum_steps=1, maximum_evaluations_per_step=4)
        for strategy in ("adam", "lbfgs", "hybrid"):
            result = fit_single_initialization_with_strategy(
                train,
                validation,
                self.registry,
                self.registry.initializations[0],
                optimizer_strategy=strategy,
                max_epochs_override=1 if strategy == "adam" else None,
                adam_warmup_epochs=1,
                lbfgs_settings=settings,
                allow_short_convergence_budget=True,
            )
            self.assertEqual(result.model_id, "M2_hmm_static_v1")
            self.assertEqual(len(result.best_raw_parameters), 7)
            self.assertIsNone(result.exception_message)

    def test_design_preserving_and_generative_simulation(self) -> None:
        template = _session(
            choices=(0, 1, float("nan"), 1),
            durations=(.5, 2., float("nan"), 1.9),
        )
        first = simulate_m2_choices(
            [template], self.parameters, self.registry.task_moments, seed=71
        )[0]
        second = simulate_m2_choices(
            [template], self.parameters, self.registry.task_moments, seed=71
        )[0]
        torch.testing.assert_close(first.choice_long, second.choice_long, equal_nan=True)
        self.assertTrue(torch.isnan(first.choice_long[2]))
        generated, truth = simulate_m2_hmm_sessions(
            [torch.tensor([.3, .6, 1.8, 2.1] * 4, dtype=DTYPE)],
            self.registry.initializations[0].constrained_tensor(),
            self.registry.task_moments,
            random_seed=77,
        )
        self.assertEqual(generated[0].choice_long.shape, truth[0].shape)

    def test_real_design_simulation_preserves_design_and_uses_block_prototypes(self) -> None:
        n = 600
        template = _session(
            categories=(0,) * n,
            choices=tuple(float("nan") if index % 19 == 0 else 0.0 for index in range(n)),
            feedback=(True,) * n,
            durations=tuple(.5 if index % 2 == 0 else 2.0 for index in range(n)),
            session_id="real_design",
            role="validation",
        )
        labels = ("S",) * (n // 2) + ("L",) * (n // 2)
        constrained = self.registry.initializations[0].constrained_tensor().clone()
        constrained[5] = torch.logit(torch.tensor(1 / 23, dtype=DTYPE))
        constrained[6] = math.log(4.0)
        first = simulate_m2_real_design_sessions(
            [template], {"real_design": labels}, constrained,
            self.registry.task_moments, random_seed=91,
        )[0]
        second = simulate_m2_real_design_sessions(
            [template], {"real_design": labels}, constrained,
            self.registry.task_moments, random_seed=91,
        )[0]
        torch.testing.assert_close(first.duration_s, template.duration_s)
        torch.testing.assert_close(first.feedback_valid, template.feedback_valid)
        torch.testing.assert_close(first.choice_long, second.choice_long, equal_nan=True)
        torch.testing.assert_close(first.category_long, second.category_long)
        self.assertTrue(torch.equal(torch.isnan(first.choice_long), torch.isnan(template.choice_long)))
        self.assertLess(
            float(first.category_long[: n // 2].mean()),
            float(first.category_long[n // 2 :].mean()),
        )

    def test_identifiability_surface_is_post_fit_and_has_optimum_grid_point(self) -> None:
        raw_before = self.raw.clone()
        session = _session(
            categories=(0, 0, 1, 1, 0, 1),
            choices=(0, 0, 1, 1, 0, 1),
            feedback=(True,) * 6,
            durations=(.4, .7, 1.8, 2.1, .6, 2.0),
        )
        surface = evaluate_m2_hazard_contrast_surface(
            [session], self.raw, self.registry,
            grid_points=3, ah_half_width=1.0, delta0_half_width=.5,
        )
        self.assertEqual(len(surface), 9)
        self.assertEqual(surface["delta_mean_nll_from_grid_minimum"].min(), 0.0)
        self.assertEqual(
            set(surface["surface_type"]),
            {"fixed_other_four_post_fit_objective_slice"},
        )
        torch.testing.assert_close(self.raw, raw_before, rtol=0, atol=0)
        center = surface.loc[
            np.isclose(surface["ah"], surface["fitted_ah"])
            & np.isclose(surface["delta0"], surface["fitted_delta0"])
        ].iloc[0]
        direct = evaluate_m2_mouse_objective(
            [session], self.raw, self.registry.task_moments
        ).loss
        self.assertAlmostEqual(float(center["mean_nll"]), float(direct), places=13)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_m2_identifiability_figure(surface, temporary)
            self.assertFalse(manifest["changes_fitted_loss_or_parameters"])
            self.assertTrue(Path(temporary, "m2_ah_delta0_loss_surface.png").exists())

    def test_export_and_both_figure_families(self) -> None:
        n = 32
        frame = pd.DataFrame({
            "mouse_id": "mouse",
            "session_id": "s",
            "trial_index": np.arange(n),
            "session_date": pd.Timestamp("2020-01-01"),
            "duration_s": np.tile([.3, .6, 1.8, 2.1], n // 4),
            "choice_long": np.tile([0, 0, 1, 1], n // 4),
            "category_long": np.tile([0, 0, 1, 1], n // 4),
            "feedback_valid": 1,
            "true_block": np.tile(["N"] * 8 + ["S"] * 8 + ["L"] * 8 + ["S"] * 8, 1),
            "outcome": "synthetic",
            "split_role": "validation",
            "omission": 0,
        })
        predictions = build_m2_trial_predictions(frame, self.raw, self.registry)
        self.assertTrue({"fitted_delta0", "model_subjective_prototype_S", "model_subjective_prototype_L"}.issubset(predictions.columns))
        m1_predictions = predictions.copy()
        m1_predictions["per_trial_negative_log_likelihood"] += .01
        m1_predictions["model_probability_long"] = np.clip(
            m1_predictions["model_probability_long"] + .01, 0, 1
        )
        m0_predictions = predictions.copy()
        m0_predictions["per_trial_negative_log_likelihood"] += .02
        m0_predictions["model_probability_long"] = np.clip(
            m0_predictions["model_probability_long"] + .02, 0, 1
        )
        constrained = self.registry.initializations[0].constrained_tensor().tolist()
        fit_summary = {
            "model_id": "M2_hmm_static_v1",
            "mouse_id": "mouse",
            "winner_start_id": 1,
            "runs": [{
                "start_id": 1,
                "initial_constrained_parameters": constrained,
                "initial_training_mean_nll": .6,
                "initial_validation_mean_nll": .7,
                "history": [{
                    "epoch": 1,
                    "constrained_parameters_after_step": constrained,
                    "fresh_training_mean_nll": .5,
                    "validation_mean_nll": .65,
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostic = generate_m2_diagnostic_figures(
                predictions,
                root / "diagnostic",
                m1_predictions=m1_predictions,
                m0_predictions=m0_predictions,
            )
            style = generate_m2_m0_style_figures(
                predictions, fit_summary, root / "m0_style", test_unlocked=False
            )
            self.assertIn("prototypes_and_hazard", diagnostic["files"])
            self.assertIn("delta0_reference", diagnostic["files"])
            self.assertIn("m1_m2_prior_overlay", diagnostic["files"])
            self.assertIn("m1_m2_belief_overlay", diagnostic["files"])
            self.assertIn("validation_nll_brier_comparison", diagnostic["files"])
            self.assertIn("m1_m2_validation_elpd", diagnostic["files"])
            self.assertEqual(len(style["files"]), 5)
            self.assertTrue(
                (root / "diagnostic" / "m0_m1_m2_validation_metrics.csv").exists()
            )
            parameter_history = pd.read_csv(root / "m0_style" / "parameter_history.csv")
            self.assertIn("delta0", parameter_history.columns)


if __name__ == "__main__":
    unittest.main()
