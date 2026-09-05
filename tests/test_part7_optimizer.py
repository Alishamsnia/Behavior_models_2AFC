"""Synthetic tests for the registered Part VII Adam procedure."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    SessionTrials,
    evaluate_mouse_objective,
    fit_single_initialization,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def _session(role: str, session_id: str, choices: list[float]) -> SessionTrials:
    return SessionTrials(
        mouse_id="synthetic_mouse",
        session_id=session_id,
        duration_s=torch.tensor(
            [0.2, 0.5, 0.8, 1.7, 2.0, 2.3, float("nan")],
            dtype=torch.float64,
        ),
        category_long=torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, float("nan")],
            dtype=torch.float64,
        ),
        choice_long=torch.tensor(choices, dtype=torch.float64),
        split_role=role,  # type: ignore[arg-type]
    )


class PartSevenOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.training = _session(
            "train",
            "train_1",
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, float("nan")],
        )
        cls.validation = _session(
            "validation",
            "validation_1",
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, float("nan")],
        )

    def test_single_start_takes_one_step_per_epoch_and_restores_best(self) -> None:
        result = fit_single_initialization(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=3,
        )
        self.assertEqual(result.final_epoch, 3)
        self.assertEqual(result.optimizer_step_count, 3)
        self.assertEqual(len(result.history), 3)
        self.assertFalse(result.is_registered_duration_run)
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "maximum_epochs_without_convergence")
        raw = torch.tensor(result.best_raw_parameters, dtype=torch.float64)
        with torch.no_grad():
            objective = evaluate_mouse_objective(
                [self.training],
                raw,
                self.registry.task_moments,
                required_split_role="train",
                lapse_max=self.registry.lapse_max,
                parameter_epsilon=self.registry.parameter_epsilon,
                log_probability_epsilon=self.registry.log_probability_epsilon,
            )
        self.assertAlmostEqual(
            result.recalculated_best_training_mean_nll,
            float(objective.loss),
            places=13,
        )
        self.assertTrue(all(torch.isfinite(raw)))

    def test_validation_choices_do_not_change_training_updates(self) -> None:
        alternate_validation = _session(
            "validation",
            "validation_2",
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, float("nan")],
        )
        first = fit_single_initialization(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=2,
        )
        second = fit_single_initialization(
            [self.training],
            [alternate_validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=2,
        )
        self.assertEqual(first.best_raw_parameters, second.best_raw_parameters)
        self.assertEqual(
            tuple(row.fresh_training_mean_nll for row in first.history),
            tuple(row.fresh_training_mean_nll for row in second.history),
        )
        self.assertNotEqual(
            first.final_validation_mean_nll,
            second.final_validation_mean_nll,
        )

    def test_scheduler_monitors_fresh_training_nll_and_reduces_on_plateau(self) -> None:
        convergence = replace(
            self.registry.convergence,
            meaningful_improvement_tolerance=1.0,
            no_improvement_patience_epochs=100,
        )
        scheduler = replace(self.registry.scheduler, patience_epochs=0)
        registry = replace(
            self.registry,
            convergence=convergence,
            scheduler=scheduler,
        )
        result = fit_single_initialization(
            [self.training],
            [self.validation],
            registry,
            registry.initializations[0],
            max_epochs_override=3,
        )
        learning_rates = [row.learning_rate_after_scheduler for row in result.history]
        self.assertLess(learning_rates[-1], learning_rates[0])
        # The huge tolerance above makes every epoch count as "no meaningful
        # improvement" for the scheduler/patience machinery (asserted via
        # learning_rates above), but the checkpoint itself must still track
        # the true best fresh training NLL seen, independent of that
        # tolerance -- otherwise a run that is genuinely improving can be
        # left pointed at a worse (even the initial) parameter vector purely
        # because no single step cleared an unrelated patience threshold.
        best_nll_in_history = min(
            row.fresh_training_mean_nll for row in result.history
        )
        self.assertEqual(result.best_fresh_training_mean_nll, best_nll_in_history)
        if best_nll_in_history < result.initial_training_mean_nll:
            expected_epoch = next(
                row.epoch
                for row in result.history
                if row.fresh_training_mean_nll == best_nll_in_history
            )
            expected_raw = next(
                row.raw_parameters_after_step
                for row in result.history
                if row.epoch == expected_epoch
            )
            self.assertEqual(result.best_epoch, expected_epoch)
            self.assertEqual(result.best_raw_parameters, expected_raw)
        else:
            self.assertEqual(result.best_epoch, 0)
            self.assertEqual(
                result.best_raw_parameters, result.initial_raw_parameters
            )

    def test_runtime_epoch_and_learning_rate_overrides_are_recorded(self) -> None:
        result = fit_single_initialization(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=2,
            initial_learning_rate_override=0.02,
        )
        self.assertEqual(result.registered_max_epochs, 2000)
        self.assertEqual(result.executed_max_epochs, 2)
        self.assertFalse(result.is_registered_duration_run)
        self.assertEqual(result.registered_initial_learning_rate, 0.01)
        self.assertEqual(result.executed_initial_learning_rate, 0.02)
        self.assertFalse(result.is_registered_learning_rate_run)
        self.assertAlmostEqual(
            result.history[0].learning_rate_after_scheduler,
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
