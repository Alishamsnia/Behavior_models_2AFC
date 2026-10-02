"""Fitting/registry/pipeline tests for M2_known_switch_v1, on synthetic data
only. Mirrors ``test_m1_known_switch_fit.py``."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m0_sensory_v1.model.fit import LBFGSSettings
from m2_known_switch_v1.analysis import evaluate_m2_known_switch_split_metrics
from m2_known_switch_v1.fit import (
    _detect_identifiability_ridge,
    fit_m2_known_switch_all_initializations,
    fit_m2_known_switch_single_initialization,
)
from m2_known_switch_v1.registry import load_m2_known_switch_model_registry
from m2_known_switch_v1.session import (
    M2KnownSwitchSessionTrials,
    evaluate_m2_known_switch_session,
)


DTYPE = torch.float64


def _session(true_block, categories, choices, feedback, durations, *, session_id="s1", role="train"):
    is_switch = [False] + [
        true_block[index] != true_block[index - 1] for index in range(1, len(true_block))
    ]
    return M2KnownSwitchSessionTrials(
        mouse_id="mouse", session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        is_switch=torch.tensor(is_switch, dtype=torch.bool),
        split_role=role,
    )


def _synthetic_sessions():
    training = [
        _session(
            true_block=["N", "S", "S", "L", "L", "S", "S"],
            categories=[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            choices=[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            feedback=[True] * 7,
            durations=[0.5, 0.5, 0.55, 2.0, 1.9, 0.45, 0.5],
            session_id=f"train_{index}",
        )
        for index in range(3)
    ]
    validation = [
        _session(
            true_block=["N", "N", "L", "L", "S"],
            categories=[1.0, 1.0, 1.0, 1.0, 0.0],
            choices=[1.0, 1.0, 1.0, 1.0, 0.0],
            feedback=[True] * 5,
            durations=[2.0, 2.1, 1.9, 2.05, 0.5],
            session_id="val_1",
            role="validation",
        )
    ]
    return training, validation


class M2KnownSwitchRegistryTests(unittest.TestCase):
    def test_loads_and_preserves_m0_starts_plus_delta0(self) -> None:
        registry = load_m2_known_switch_model_registry()
        self.assertEqual(registry.model_id, "M2_known_switch_v1")
        self.assertEqual(len(registry.initializations), 8)
        for start in registry.initializations:
            self.assertAlmostEqual(start.delta0, 1.3862943611198906, places=9)

    def test_subject_overlays_proxy_distinct_task_moments(self) -> None:
        mc06 = load_m2_known_switch_model_registry("config/m2_known_switch_v1_mc06.json")
        mc07 = load_m2_known_switch_model_registry("config/m2_known_switch_v1_mc07.json")
        self.assertNotEqual(mc06.task_moments.mu_short_s, mc07.task_moments.mu_short_s)


class M2KnownSwitchFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m2_known_switch_model_registry()
        cls.training, cls.validation = _synthetic_sessions()

    def test_adam_produces_a_well_formed_result(self) -> None:
        result = fit_m2_known_switch_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=15,
        )
        self.assertEqual(result.start_id, 1)
        self.assertEqual(len(result.history), 15)
        self.assertEqual(len(result.best_constrained_parameters), 6)
        self.assertEqual(len(result.initial_constrained_parameters), 6)
        self.assertTrue(all(math.isfinite(value) for value in result.initial_constrained_parameters))
        self.assertTrue(
            all(len(step.constrained_parameters_after_step) == 6 for step in result.history)
        )
        self.assertNotEqual(
            result.history[0].constrained_parameters_after_step,
            result.history[-1].constrained_parameters_after_step,
        )

    def test_lbfgs_produces_a_well_formed_result(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=8, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m2_known_switch_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="lbfgs", lbfgs_settings=settings,
        )
        self.assertTrue(all(step.optimizer_phase == "lbfgs" for step in result.history))

    def test_hybrid_runs_adam_warmup_then_lbfgs(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=5, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m2_known_switch_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="hybrid", adam_warmup_epochs=4,
            lbfgs_settings=settings,
        )
        phases = [step.optimizer_phase for step in result.history]
        self.assertEqual(phases.count("adam_warmup"), 4)
        self.assertEqual(phases.count("lbfgs"), 5)

    def test_restored_best_is_not_necessarily_the_final_step(self) -> None:
        result = fit_m2_known_switch_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=20,
        )
        best_recorded = min(step.fresh_training_mean_nll for step in result.history)
        self.assertAlmostEqual(
            result.recalculated_best_training_mean_nll, best_recorded, places=6
        )

    def test_failure_is_contained_not_raised(self) -> None:
        from m2_known_switch_v1.registry import M2KnownSwitchInitializationSpec

        bad = M2KnownSwitchInitializationSpec(
            start_id=99, w=-1.0, s0=0.5, ell_short=0.05, ell_long=0.05,
            kappa=1.0, delta0=1.0, purpose="deliberately invalid",
        )
        result = fit_m2_known_switch_single_initialization(
            self.training, self.validation, self.registry, bad,
            optimizer_strategy="adam", max_epochs_override=5,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "numerical_or_transform_failure")
        self.assertIsNotNone(result.exception_message)

    def test_all_initializations_selects_training_only_winner(self) -> None:
        result = fit_m2_known_switch_all_initializations(
            self.training, self.validation, self.registry,
            optimizer_strategy="adam", max_epochs_override=10,
        )
        self.assertEqual(len(result.runs), 8)
        self.assertIsInstance(result.identifiability_ridge_detected, bool)
        self.assertIsInstance(result.additional_starts_recommended, bool)
        for start_id in result.ridge_start_ids:
            self.assertIn(start_id, {run.start_id for run in result.runs})
        if result.winner_start_id is not None:
            winner = next(run for run in result.runs if run.start_id == result.winner_start_id)
            self.assertTrue(winner.converged)
        else:
            self.assertTrue(result.additional_starts_recommended)

    def test_multistart_summary_json_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_directory = Path(tmp) / "starts"
            result = fit_m2_known_switch_all_initializations(
                self.training, self.validation, self.registry,
                optimizer_strategy="adam", max_epochs_override=5,
                output_directory=output_directory,
            )
            summary_path = output_directory / "multistart_summary.json"
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["winner_start_id"], result.winner_start_id)
            self.assertEqual(
                summary["identifiability_ridge_detected"], result.identifiability_ridge_detected
            )

    def test_no_converged_run_reports_none_winner(self) -> None:
        from dataclasses import replace

        strict_registry = replace(
            self.registry,
            convergence=replace(self.registry.convergence, gradient_norm_tolerance=1e-300),
        )
        result = fit_m2_known_switch_all_initializations(
            self.training, self.validation, strict_registry,
            optimizer_strategy="adam", max_epochs_override=3,
        )
        self.assertIsNone(result.winner_start_id)
        self.assertEqual(result.selection_status, "no_converged_run")

    def test_start_order_independence_of_winner_selection(self) -> None:
        from dataclasses import replace

        forward = fit_m2_known_switch_all_initializations(
            self.training, self.validation, self.registry,
            optimizer_strategy="adam", max_epochs_override=10,
        )
        reversed_registry = replace(
            self.registry, initializations=tuple(reversed(self.registry.initializations))
        )
        backward = fit_m2_known_switch_all_initializations(
            self.training, self.validation, reversed_registry,
            optimizer_strategy="adam", max_epochs_override=10,
        )
        if forward.winner_start_id is not None and backward.winner_start_id is not None:
            self.assertEqual(forward.winner_start_id, backward.winner_start_id)

    def test_parallel_multistart_preserves_every_registered_start(self) -> None:
        from dataclasses import replace

        registry = replace(self.registry, initializations=self.registry.initializations[:2])
        result = fit_m2_known_switch_all_initializations(
            self.training, self.validation, registry,
            optimizer_strategy="adam", max_epochs_override=2,
            parallel_workers=2, cpu_threads_per_worker=1,
        )
        self.assertEqual([run.start_id for run in result.runs], [1, 2])


class M2KnownSwitchIdentifiabilityRidgeTests(unittest.TestCase):
    """Deterministic, hand-crafted cases for the pure ridge-detection helper
    (not exercised by real fits above, since the optimizer's landing point
    isn't controllable): confirms the port from m1_dynamic_block_length_v1
    actually distinguishes tied-and-close, tied-and-far, and not-tied.
    """

    @classmethod
    def setUpClass(cls) -> None:
        multi = load_m2_known_switch_model_registry().multi_start_selection
        cls.tolerance = multi.near_equal_training_nll_tolerance
        cls.scale = multi.parameter_distance_scale
        cls.threshold = multi.identifiability_parameter_distance_threshold
        cls.winner_values = (0.03, 0.5, 0.03, 0.03, 1.0, 0.0)
        cls.winner_nll = 0.5

    def _ridge(self, candidate_nll: float, candidate_values: tuple[float, ...]):
        eligible = [(1, self.winner_nll, self.winner_values), (2, candidate_nll, candidate_values)]
        return _detect_identifiability_ridge(
            eligible, 1,
            near_equal_training_nll_tolerance=self.tolerance,
            parameter_distance_scale=self.scale,
            identifiability_parameter_distance_threshold=self.threshold,
        )

    def test_tied_and_far_is_a_ridge(self) -> None:
        far_values = (0.03, 0.5, 0.03, 0.03, 1.35, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, far_values)
        self.assertTrue(detected)
        self.assertEqual(ridge_ids, (1, 2))

    def test_tied_and_close_is_not_a_ridge(self) -> None:
        close_values = (0.03, 0.5, 0.03, 0.03, 1.05, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, close_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())

    def test_far_parameters_but_not_tied_is_not_a_ridge(self) -> None:
        far_values = (0.03, 0.5, 0.03, 0.03, 1.35, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + 1.0, far_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())


class M2KnownSwitchAnalysisTests(unittest.TestCase):
    def test_split_metrics_match_direct_evaluation(self) -> None:
        registry = load_m2_known_switch_model_registry()
        training, validation = _synthetic_sessions()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        from m2_known_switch_v1.parameters import decode_m2_known_switch_raw_parameters

        parameters = decode_m2_known_switch_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        direct = [
            evaluate_m2_known_switch_session(session, parameters, registry.task_moments)
            for session in validation
        ]
        expected_mean_nll = float(
            sum(record.negative_log_likelihood for record in direct)
            / sum(record.valid_trial_count for record in direct)
        )
        metrics = evaluate_m2_known_switch_split_metrics(
            validation, raw, registry, required_split_role="validation"
        )
        self.assertAlmostEqual(metrics.mean_nll_per_valid_trial, expected_mean_nll, places=9)


if __name__ == "__main__":
    unittest.main()
