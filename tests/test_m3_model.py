"""Acceptance tests for the additive Phase-I Model 3 implementation."""

from __future__ import annotations

import math
import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model.m2_parameters import M2DecodedParameters  # noqa: E402
from m0_sensory_v1.model.m2_session import evaluate_m2_session  # noqa: E402
from m0_sensory_v1.model.fit import LBFGSSettings  # noqa: E402
from m3_hmm_delta_v1.fit import (  # noqa: E402
    fit_m3_all_initializations,
    fit_m3_single_initialization,
)
from m3_hmm_delta_v1.identifiability import evaluate_m3_delta_alpha_surface  # noqa: E402
from m3_hmm_delta_v1.parameters import (  # noqa: E402
    decode_m3_raw_parameters,
    encode_m3_constrained_parameters,
)
from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402
from m3_hmm_delta_v1.session import (  # noqa: E402
    M3SessionTrials,
    evaluate_m3_mouse_objective,
    evaluate_m3_session,
    m3_delta_update_signal,
)


DTYPE = torch.float64


def _session(
    categories=(0, 0, 1, 1, 0, 1),
    choices=(0, 0, 1, 1, 0, 1),
    feedback=(True, True, True, True, True, True),
    durations=(.4, .7, 1.8, 2.1, .6, 2.0),
    *,
    session_id="s1",
    role="train",
):
    return M3SessionTrials(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


class M3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_m3_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor()
        cls.parameters = decode_m3_raw_parameters(cls.raw)

    def test_registry_preserves_eight_m2_rows_and_adds_learning_rate(self):
        self.assertEqual(self.registry.model_id, "M3_hmm_delta_v1")
        self.assertEqual(len(self.registry.initializations), 8)
        self.assertEqual(
            tuple(item.alpha_theta for item in self.registry.initializations),
            (0.02, 0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1),
        )
        self.assertEqual(
            tuple(item.source_m2_start_id for item in self.registry.initializations),
            tuple(range(1, 9)),
        )
        self.assertEqual(self.registry.empirical_median_block_length_trials, 23)
        self.assertAlmostEqual(self.registry.empirical_inverse_median_hazard, 1 / 23)
        self.assertEqual(self.registry.fitting.default_optimizer, "adam")
        self.assertEqual(
            self.registry.fitting.supported_optimizers,
            ("adam", "lbfgs", "hybrid"),
        )
        self.assertEqual(self.registry.fitting.lbfgs_line_search_fn, "strong_wolfe")
        for filename, count in (
            ("m3_hmm_delta_v1_mc06.json", 7225),
            ("m3_hmm_delta_v1_mc07.json", 6852),
        ):
            loaded = load_m3_model_registry(ROOT / "config" / filename)
            self.assertEqual(loaded.task_moments.n_short, count)

    def test_parameter_round_trip(self):
        constrained = torch.tensor(
            [.03, .55, .04, .6, 1.4, -3.2, .9, .08], dtype=DTYPE
        )
        decoded = decode_m3_raw_parameters(
            encode_m3_constrained_parameters(constrained)
        )
        torch.testing.assert_close(decoded.as_tensor(), constrained)
        self.assertAlmostEqual(float(decoded.hazard), float(torch.sigmoid(constrained[5])))

    def test_exact_alpha_zero_reproduces_m2_trialwise(self):
        session = _session()
        m3_parameters = self.parameters.with_explicit_alpha(0.0)
        m2_parameters = M2DecodedParameters(
            w=m3_parameters.w,
            s0=m3_parameters.s0,
            ell_short=m3_parameters.ell_short,
            ell_long=m3_parameters.ell_long,
            kappa=m3_parameters.kappa,
            ah=m3_parameters.ah,
            delta0=m3_parameters.delta0,
            hazard=m3_parameters.hazard,
        )
        m3 = evaluate_m3_session(session, m3_parameters, self.registry.task_moments)
        m2 = evaluate_m2_session(session, m2_parameters, self.registry.task_moments)
        for left, right in (
            (m3.predictions.probability_long, m2.predictions.probability_long),
            (m3.predictions.prior_long, m2.predictions.prior_long),
            (m3.predictions.pre_choice_block_probability, m2.predictions.pre_choice_block_probability),
            (m3.predictions.post_feedback_block_probability, m2.predictions.post_feedback_block_probability),
            (m3.predictions.emission_normalizer, m2.predictions.emission_normalizer),
            (m3.per_trial_negative_log_likelihood, m2.per_trial_negative_log_likelihood),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(
            m3.predictions.subjective_prototypes_pre,
            m2.predictions.subjective_prototypes.expand(6, -1),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            m3.predictions.delta_pre,
            torch.full((6,), float(m3_parameters.delta0), dtype=DTYPE),
        )

    def test_current_category_selects_lapse_rate_and_updates_future_learning_state(self):
        parameters = decode_m3_raw_parameters(
            self.registry.initializations[1].raw_tensor()
        )
        left = evaluate_m3_session(
            _session(categories=(0, 0, 0, 1, 0, 1)),
            parameters,
            self.registry.task_moments,
        )
        right = evaluate_m3_session(
            _session(categories=(0, 1, 0, 1, 0, 1)),
            parameters,
            self.registry.task_moments,
        )
        self.assertEqual(float(left.predictions.prior_long[1]), float(right.predictions.prior_long[1]))
        self.assertEqual(float(left.predictions.non_lapse_probability_long[1]), float(right.predictions.non_lapse_probability_long[1]))
        self.assertNotEqual(float(left.predictions.probability_long[1]), float(right.predictions.probability_long[1]))
        self.assertNotEqual(float(left.predictions.prior_long[2]), float(right.predictions.prior_long[2]))
        self.assertNotEqual(float(left.predictions.delta_post[2]), float(right.predictions.delta_post[2]))

    def test_missing_feedback_skips_learning_but_transition_continues(self):
        evaluation = evaluate_m3_session(
            _session(feedback=(True, False, True, True, True, True)),
            self.parameters,
            self.registry.task_moments,
        )
        prediction = evaluation.predictions
        self.assertTrue(torch.isnan(prediction.delta_update_signal[1]))
        self.assertEqual(float(prediction.delta_post[1]), float(prediction.delta_pre[1]))
        self.assertFalse(bool(prediction.projection_activated[1]))
        self.assertFalse(torch.equal(
            prediction.pre_choice_block_probability[1],
            prediction.pre_choice_block_probability[2],
        ))

    def test_valid_feedback_omission_updates_without_choice_likelihood(self):
        evaluation = evaluate_m3_session(
            _session(
                choices=(0, 0, float("nan"), 1, 0, 1),
                feedback=(True, True, True, True, True, True),
            ),
            self.parameters,
            self.registry.task_moments,
        )
        self.assertFalse(bool(evaluation.predictions.valid_choice_mask[2]))
        self.assertTrue(bool(evaluation.predictions.feedback_update_mask[2]))
        self.assertTrue(torch.isnan(evaluation.per_trial_negative_log_likelihood[2]))
        self.assertTrue(torch.isfinite(evaluation.predictions.delta_update_signal[2]))
        self.assertNotEqual(
            float(evaluation.predictions.delta_post[2]),
            float(evaluation.predictions.delta_pre[2]),
        )

    def test_projection_records_proposal_and_one_based_location(self):
        ah = torch.tensor(-10.0, dtype=DTYPE)
        parameters = replace(
            self.parameters,
            ah=ah,
            hazard=torch.sigmoid(ah),
            delta0=torch.tensor(1e-6, dtype=DTYPE),
            alpha_theta=torch.tensor(1.0, dtype=DTYPE),
        )
        evaluation = evaluate_m3_session(
            _session(
                categories=(1, 1, 0), choices=(1, 1, 0), feedback=(True,) * 3,
                durations=(2.0, 2.0, .5),
            ),
            parameters,
            self.registry.task_moments,
        )
        self.assertEqual(evaluation.projection_locations, (3,))
        self.assertLess(float(evaluation.predictions.delta_unprojected_proposal[2]), 1e-6)
        self.assertEqual(float(evaluation.predictions.delta_post[2]), 1e-6)
        self.assertEqual(evaluation.projection_events[0].trial_index, 3)
        self.assertEqual(evaluation.projection_events[0].mouse_id, "mouse")
        self.assertEqual(evaluation.projection_events[0].session_id, "s1")

    def test_four_one_hot_learning_signs_and_values(self):
        theta = torch.tensor((0.5, 0.2, 0.8), dtype=DTYPE)
        short = torch.tensor((0.0, 1.0, 0.0), dtype=DTYPE)
        long = torch.tensor((0.0, 0.0, 1.0), dtype=DTYPE)
        cases = (
            (torch.tensor(0.0, dtype=DTYPE), short, 0.2),
            (torch.tensor(1.0, dtype=DTYPE), short, -0.8),
            (torch.tensor(0.0, dtype=DTYPE), long, -0.8),
            (torch.tensor(1.0, dtype=DTYPE), long, 0.2),
        )
        for category, responsibility, expected in cases:
            self.assertAlmostEqual(
                float(m3_delta_update_signal(category, responsibility, theta)), expected
            )

    def test_weak_delta_identifiability_grid_respects_registered_floor(self):
        weak = torch.tensor((.03, .55, .04, .6, 1.0, -3.2, .05, .08), dtype=DTYPE)
        raw = encode_m3_constrained_parameters(weak)
        surface = evaluate_m3_delta_alpha_surface(
            [_session()], raw, self.registry, grid_points=3, delta0_half_width=1.0
        )
        self.assertGreaterEqual(float(surface.delta0.min()), self.registry.delta_min)
        self.assertEqual(len(surface), 9)

    def test_session_reset_and_order_invariant_loss_and_gradient(self):
        first = _session(session_id="first")
        second = _session(
            categories=(1, 1, 0, 0, 1, 0),
            choices=(1, 1, 0, 0, 1, 0),
            durations=(2.1, 1.9, .5, .7, 2.0, .4),
            session_id="second",
        )

        def value(records):
            raw = self.raw.clone().requires_grad_()
            loss = evaluate_m3_mouse_objective(records, raw, self.registry.task_moments).loss
            return loss.detach(), torch.autograd.grad(loss, raw)[0]

        forward = value([first, second])
        reverse = value([second, first])
        torch.testing.assert_close(forward[0], reverse[0], rtol=0, atol=2e-15)
        torch.testing.assert_close(forward[1], reverse[1], rtol=1e-11, atol=2e-14)

    def test_all_eight_autograd_coordinates_match_central_differences(self):
        raw = self.raw.clone().requires_grad_()

        def objective(value):
            return evaluate_m3_mouse_objective(
                [_session()], value, self.registry.task_moments
            ).loss

        gradient = torch.autograd.grad(objective(raw), raw)[0]
        step = 1e-5
        finite = []
        for index in range(raw.numel()):
            plus = raw.detach().clone()
            minus = raw.detach().clone()
            plus[index] += step
            minus[index] -= step
            finite.append(float((objective(plus) - objective(minus)) / (2 * step)))
        torch.testing.assert_close(
            gradient,
            torch.tensor(finite, dtype=DTYPE),
            rtol=3e-4,
            atol=2e-7,
        )

    def test_fitter_uses_fresh_full_batch_objective(self):
        run = fit_m3_single_initialization(
            [_session(session_id="train")],
            [_session(session_id="validation", role="validation")],
            self.registry,
            self.registry.initializations[0],
            optimizer_strategy="lbfgs",
            lbfgs_settings=LBFGSSettings(maximum_steps=1),
            allow_short_convergence_budget=True,
        )
        fresh = evaluate_m3_mouse_objective(
            [_session(session_id="train")],
            torch.tensor(run.final_raw_parameters, dtype=DTYPE),
            self.registry.task_moments,
        )
        self.assertAlmostEqual(run.fresh_final_training_mean_nll, float(fresh.loss), places=14)
        self.assertEqual(run.training_session_ids, ("train",))
        self.assertEqual(run.validation_session_ids, ("validation",))
        json.dumps(run.as_dict())

    def test_evaluated_checks_must_match_declared_registry_checks(self) -> None:
        """M3's fit.py evaluates 8 checks (M2's inherited 7 plus
        projection_activation_acceptable) but registry.convergence.required_checks
        only ever reports the inherited 7 (M3's own config doesn't override
        it). fit_m3_single_initialization validates the two against a
        documented expected set internally; this proves that validation is
        actually wired by forcing a real mismatch."""
        tampered_convergence = replace(
            self.registry.convergence,
            required_checks=self.registry.convergence.required_checks + ("made_up_check",),
        )
        tampered_registry = replace(self.registry, convergence=tampered_convergence)
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            fit_m3_single_initialization(
                [_session(session_id="train")],
                [_session(session_id="validation", role="validation")],
                tampered_registry,
                tampered_registry.initializations[0],
                optimizer_strategy="adam",
                max_epochs_override=1,
                allow_short_convergence_budget=True,
            )

    def test_adam_pure_lbfgs_and_hybrid_match_m2_strategy_contract(self):
        training = [_session(session_id="train")]
        validation = [_session(session_id="validation", role="validation")]
        adam = fit_m3_single_initialization(
            training, validation, self.registry, self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=2,
        )
        lbfgs = fit_m3_single_initialization(
            training, validation, self.registry, self.registry.initializations[0],
            optimizer_strategy="lbfgs",
            lbfgs_settings=LBFGSSettings(maximum_steps=2),
            allow_short_convergence_budget=True,
        )
        hybrid = fit_m3_single_initialization(
            training, validation, self.registry, self.registry.initializations[0],
            optimizer_strategy="hybrid", adam_warmup_epochs=2,
            lbfgs_settings=LBFGSSettings(maximum_steps=2),
            allow_short_convergence_budget=True,
        )
        self.assertEqual((adam.adam_step_count, adam.lbfgs_step_count), (2, 0))
        self.assertEqual((lbfgs.adam_step_count, lbfgs.lbfgs_step_count), (0, 2))
        self.assertEqual((hybrid.adam_step_count, hybrid.lbfgs_step_count), (2, 2))
        self.assertEqual([item.optimizer_phase for item in adam.history], ["adam"] * 2)
        self.assertEqual([item.optimizer_phase for item in lbfgs.history], ["lbfgs"] * 2)
        self.assertEqual(
            [item.optimizer_phase for item in hybrid.history],
            ["adam", "adam", "lbfgs", "lbfgs"],
        )
        self.assertEqual(lbfgs.optimizer_settings["backend"], "torch")
        self.assertTrue(all(item.closure_evaluation_count >= 1 for item in lbfgs.history))

    def test_hybrid_training_updates_do_not_depend_on_validation_choices(self):
        training = [_session(session_id="train")]
        first_validation = [_session(session_id="validation_a", role="validation")]
        second_validation = [_session(
            categories=(0, 0, 1, 1, 0, 1),
            choices=(1, 1, 0, 0, 1, 0),
            session_id="validation_b",
            role="validation",
        )]
        options = dict(
            optimizer_strategy="hybrid",
            adam_warmup_epochs=2,
            lbfgs_settings=LBFGSSettings(maximum_steps=2),
            allow_short_convergence_budget=True,
        )
        first = fit_m3_single_initialization(
            training, first_validation, self.registry, self.registry.initializations[0], **options
        )
        second = fit_m3_single_initialization(
            training, second_validation, self.registry, self.registry.initializations[0], **options
        )
        self.assertEqual(first.best_raw_parameters, second.best_raw_parameters)
        self.assertEqual(
            [item.fresh_training_mean_nll for item in first.history],
            [item.fresh_training_mean_nll for item in second.history],
        )
        self.assertNotEqual(first.final_validation_mean_nll, second.final_validation_mean_nll)

    def test_parallel_multistart_matches_sequential_start_results_and_order(self):
        registry = replace(
            self.registry, initializations=self.registry.initializations[:2]
        )
        training = [_session(session_id="train")]
        validation = [_session(session_id="validation", role="validation")]
        previous_threads = torch.get_num_threads()
        initial_events = []
        step_events = []
        try:
            torch.set_num_threads(1)
            sequential = fit_m3_all_initializations(
                training, validation, registry,
                optimizer_strategy="lbfgs",
                lbfgs_settings=LBFGSSettings(maximum_steps=1),
                allow_short_convergence_budget=True,
                parallel_workers=1, cpu_threads_per_worker=1,
            )
            parallel = fit_m3_all_initializations(
                training, validation, registry,
                optimizer_strategy="lbfgs",
                lbfgs_settings=LBFGSSettings(maximum_steps=1),
                allow_short_convergence_budget=True,
                parallel_workers=2, cpu_threads_per_worker=1,
                initial_callback=lambda start_id, train, validation: initial_events.append(
                    (start_id, train, validation)
                ),
                epoch_callback=lambda start_id, record: step_events.append(
                    (start_id, record.step, record.optimizer_phase)
                ),
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual([run.start_id for run in parallel.runs], [1, 2])
        self.assertEqual(sorted(event[0] for event in initial_events), [1, 2])
        self.assertEqual(sorted(event[0] for event in step_events), [1, 2])
        self.assertTrue(all(event[1:] == (1, "lbfgs") for event in step_events))
        self.assertEqual(parallel.winner_start_id, sequential.winner_start_id)
        for left, right in zip(sequential.runs, parallel.runs, strict=True):
            self.assertEqual(left.start_id, right.start_id)
            self.assertEqual(left.random_seed, right.random_seed)
            self.assertEqual(left.final_raw_parameters, right.final_raw_parameters)
            self.assertEqual(
                left.recalculated_best_training_mean_nll,
                right.recalculated_best_training_mean_nll,
            )
            self.assertEqual(left.convergence_checks, right.convergence_checks)


if __name__ == "__main__":
    unittest.main()
