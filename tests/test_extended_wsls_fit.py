"""Fitting tests for Extended_WSLS_v1, on synthetic data only.

Mirrors the fit-specific tests in ``test_pure_wsls_fit.py`` (that model's
``fit.py`` is the structural template ``extended_wsls_v1/fit.py`` is a
near-verbatim copy of, plus the fitted learning-rate logit ``a``); the
registry/parameter/session/objective contract itself is already covered by
``test_extended_wsls_model.py``.
"""

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
from extended_wsls_v1 import M1SessionTrials, load_extended_wsls_model_registry
from extended_wsls_v1.fit import (
    _detect_identifiability_ridge,
    fit_extended_wsls_all_initializations,
    fit_extended_wsls_single_initialization,
)


DTYPE = torch.float64


def _session(categories, choices, feedback, *, session_id="s1", role="train") -> M1SessionTrials:
    n = len(categories)
    return M1SessionTrials(
        mouse_id="mouse",
        session_id=session_id,
        duration_s=torch.ones(n, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


def _synthetic_sessions():
    training = [
        _session(
            categories=[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0] * 3,
            choices=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0] * 3,
            feedback=[True] * 21,
            session_id=f"train_{index}",
        )
        for index in range(2)
    ]
    validation = [
        _session(
            categories=[1.0, 0.0, 1.0, 1.0, 0.0] * 2,
            choices=[1.0, 1.0, 1.0, 0.0, 0.0] * 2,
            feedback=[True] * 10,
            session_id="val_1",
            role="validation",
        )
    ]
    return training, validation


class ExtendedWslsFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_extended_wsls_model_registry()
        cls.training, cls.validation = _synthetic_sessions()

    def test_adam_produces_a_well_formed_result(self) -> None:
        result = fit_extended_wsls_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=15,
        )
        self.assertEqual(result.start_id, 1)
        self.assertEqual(result.optimizer_strategy, "adam")
        self.assertEqual(len(result.history), 15)
        self.assertEqual(result.optimizer_step_count, 15)
        self.assertEqual(len(result.initial_constrained_parameters), 3)
        self.assertTrue(all(math.isfinite(value) for value in result.initial_constrained_parameters))
        self.assertTrue(
            all(len(step.constrained_parameters_after_step) == 3 for step in result.history)
        )
        self.assertNotEqual(
            result.history[0].constrained_parameters_after_step,
            result.history[-1].constrained_parameters_after_step,
        )

    def test_hybrid_records_constrained_parameters_across_both_phases(self) -> None:
        # Hybrid mode numbers each phase's `step` independently from 1 (see
        # extended_wsls_sliding_window_v1.analysis's history_index docstring
        # for why the trajectory plot can't use `step` directly); this only
        # checks that constrained_parameters_after_step is still recorded
        # correctly for every row of both phases, not the plot's x-axis.
        settings = LBFGSSettings(
            maximum_steps=5, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_extended_wsls_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="hybrid", adam_warmup_epochs=4, lbfgs_settings=settings,
        )
        phases = [record.optimizer_phase for record in result.history]
        self.assertIn("adam_warmup", phases)
        self.assertIn("lbfgs", phases)
        self.assertTrue(
            all(len(record.constrained_parameters_after_step) == 3 for record in result.history)
        )
        self.assertTrue(
            all(
                math.isfinite(value)
                for record in result.history
                for value in record.constrained_parameters_after_step
            )
        )

    def test_failure_is_contained_not_raised(self) -> None:
        from dataclasses import replace

        bad_initialization = replace(self.registry.initializations[0], ell_short=float("inf"))
        result = fit_extended_wsls_single_initialization(
            self.training, self.validation, self.registry, bad_initialization,
            optimizer_strategy="adam", max_epochs_override=3,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "numerical_or_transform_failure")
        self.assertIsNotNone(result.exception_message)

    def test_all_initializations_selects_training_only_winner(self) -> None:
        result = fit_extended_wsls_all_initializations(
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
            converged_nlls = [
                run.recalculated_best_training_mean_nll for run in result.runs if run.converged
            ]
            self.assertAlmostEqual(
                winner.recalculated_best_training_mean_nll, min(converged_nlls), places=9
            )
        else:
            self.assertTrue(result.additional_starts_recommended)

    def test_multistart_summary_json_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_directory = Path(tmp) / "starts"
            result = fit_extended_wsls_all_initializations(
                self.training, self.validation, self.registry,
                optimizer_strategy="adam", max_epochs_override=5,
                output_directory=output_directory,
            )
            for run in result.runs:
                path = output_directory / f"start_{run.start_id:02d}" / "fit_result.json"
                self.assertTrue(path.is_file())
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["start_id"], run.start_id)
            summary_path = output_directory / "multistart_summary.json"
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["winner_start_id"], result.winner_start_id)
            self.assertEqual(
                summary["identifiability_ridge_detected"], result.identifiability_ridge_detected
            )


class ExtendedWslsIdentifiabilityRidgeTests(unittest.TestCase):
    """Deterministic, hand-crafted cases for the pure ridge-detection helper
    (not exercised by real fits above, since the optimizer's landing point
    isn't controllable): confirms the port from m1_dynamic_block_length_v1
    actually distinguishes tied-and-close, tied-and-far, and not-tied.
    """

    @classmethod
    def setUpClass(cls) -> None:
        multi = load_extended_wsls_model_registry().multi_start_selection
        cls.tolerance = multi.near_equal_training_nll_tolerance
        cls.scale = multi.parameter_distance_scale
        cls.threshold = multi.identifiability_parameter_distance_threshold
        cls.winner_values = (0.03, 0.03, 0.0)
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
        # a (scale 2.0) differs by 0.7 -> scaled 0.35 >= threshold 0.25.
        far_values = (0.03, 0.03, 0.7)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, far_values)
        self.assertTrue(detected)
        self.assertEqual(ridge_ids, (1, 2))

    def test_tied_and_close_is_not_a_ridge(self) -> None:
        # a differs by 0.1 -> scaled 0.05, well under threshold 0.25.
        close_values = (0.03, 0.03, 0.1)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, close_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())

    def test_far_parameters_but_not_tied_is_not_a_ridge(self) -> None:
        far_values = (0.03, 0.03, 0.7)
        detected, ridge_ids = self._ridge(self.winner_nll + 1.0, far_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())


if __name__ == "__main__":
    unittest.main()
