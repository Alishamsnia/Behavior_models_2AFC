"""Regression tests for opt-in L-BFGS and Adam-to-L-BFGS fitting."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    AlternativeFitRunResult,
    FitRunResult,
    LBFGSSettings,
    SessionTrials,
    fit_all_initializations,
    fit_single_initialization,
    fit_single_initialization_with_strategy,
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


class AlternativeOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.training = _session(
            "train", "train", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, float("nan")]
        )
        cls.validation = _session(
            "validation",
            "validation",
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, float("nan")],
        )
        cls.alternate_validation = _session(
            "validation",
            "alternate_validation",
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, float("nan")],
        )

    def test_adam_dispatch_is_exactly_the_existing_implementation(self) -> None:
        direct = fit_single_initialization(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=2,
        )
        dispatched = fit_single_initialization_with_strategy(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam",
            max_epochs_override=2,
        )
        self.assertIs(type(dispatched), FitRunResult)
        self.assertEqual(direct.as_dict(), dispatched.as_dict())

    def test_pure_lbfgs_preserves_constraints_and_records_closures(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=4,
            maximum_evaluations_per_step=10,
        )
        result = fit_single_initialization_with_strategy(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            optimizer_strategy="lbfgs",
            lbfgs_settings=settings,
            allow_short_convergence_budget=True,
        )
        self.assertIsInstance(result, AlternativeFitRunResult)
        self.assertEqual(result.optimizer_strategy, "lbfgs")
        self.assertEqual(result.adam_step_count, 0)
        self.assertEqual(result.lbfgs_step_count, 4)
        self.assertTrue(all(row.optimizer_phase == "lbfgs" for row in result.history))
        self.assertGreater(
            result.training_objective_evaluation_count,
            result.lbfgs_step_count,
        )
        self.assertEqual(
            result.training_objective_evaluation_count,
            sum(row.closure_evaluation_count for row in result.history)
            + len(result.history)
            + 2,
        )
        self.assertTrue(
            all(
                row.closure_evaluation_count >= 1
                and row.gradient_norm_after_step is not None
                for row in result.history
            )
        )
        self.assertLess(
            result.recalculated_best_training_mean_nll,
            result.initial_training_mean_nll,
        )
        w, s0, ell_short, ell_long, kappa = result.best_constrained_parameters
        self.assertGreater(w, 0.0)
        self.assertGreater(s0, 0.0)
        self.assertGreater(ell_short, 0.0)
        self.assertLess(ell_short, self.registry.lapse_max)
        self.assertGreater(ell_long, 0.0)
        self.assertLess(ell_long, self.registry.lapse_max)
        self.assertGreater(kappa, 0.0)

    def test_hybrid_has_two_phases_and_training_only_updates(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=3,
            maximum_evaluations_per_step=10,
        )
        first = fit_single_initialization_with_strategy(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            optimizer_strategy="hybrid",
            adam_warmup_epochs=2,
            lbfgs_settings=settings,
            allow_short_convergence_budget=True,
        )
        second = fit_single_initialization_with_strategy(
            [self.training],
            [self.alternate_validation],
            self.registry,
            self.registry.initializations[0],
            optimizer_strategy="hybrid",
            adam_warmup_epochs=2,
            lbfgs_settings=settings,
            allow_short_convergence_budget=True,
        )
        self.assertEqual(first.adam_step_count, 2)
        self.assertEqual(first.lbfgs_step_count, 3)
        self.assertEqual(first.phase_switch_epoch, 2)
        self.assertEqual(
            [row.optimizer_phase for row in first.history],
            ["adam", "adam", "lbfgs", "lbfgs", "lbfgs"],
        )
        self.assertEqual(first.best_raw_parameters, second.best_raw_parameters)
        self.assertEqual(
            [row.fresh_training_mean_nll for row in first.history],
            [row.fresh_training_mean_nll for row in second.history],
        )
        self.assertNotEqual(
            first.final_validation_mean_nll,
            second.final_validation_mean_nll,
        )

    def test_multistart_accepts_pure_lbfgs_without_changing_selection(self) -> None:
        result = fit_all_initializations(
            [self.training],
            [self.validation],
            self.registry,
            optimizer_strategy="lbfgs",
            lbfgs_settings=LBFGSSettings(
                maximum_steps=1,
                maximum_evaluations_per_step=5,
            ),
            allow_short_convergence_budget=True,
        )
        self.assertEqual(len(result.runs), 8)
        self.assertTrue(
            all(
                isinstance(run, AlternativeFitRunResult)
                and run.optimizer_strategy == "lbfgs"
                for run in result.runs
            )
        )

    def test_parallel_multistart_matches_sequential_results_and_order(self) -> None:
        registry = replace(
            self.registry,
            initializations=self.registry.initializations[:2],
        )
        options = dict(
            optimizer_strategy="lbfgs",
            lbfgs_settings=LBFGSSettings(
                maximum_steps=1,
                maximum_evaluations_per_step=5,
            ),
            allow_short_convergence_budget=True,
            cpu_threads_per_worker=1,
        )
        sequential = fit_all_initializations(
            [self.training],
            [self.validation],
            registry,
            parallel_workers=1,
            **options,
        )
        initial_events = []
        epoch_events = []
        completed = []
        parallel = fit_all_initializations(
            [self.training],
            [self.validation],
            registry,
            parallel_workers=2,
            initial_callback=lambda start_id, train, validation: initial_events.append(
                (start_id, train, validation)
            ),
            epoch_callback=lambda start_id, record: epoch_events.append(
                (start_id, record.epoch)
            ),
            completion_callback=lambda run: completed.append(run.start_id),
            **options,
        )
        self.assertEqual([run.start_id for run in parallel.runs], [1, 2])
        self.assertEqual(sorted(event[0] for event in initial_events), [1, 2])
        self.assertEqual(sorted(event[0] for event in epoch_events), [1, 2])
        self.assertEqual(sorted(completed), [1, 2])
        self.assertEqual(parallel.winner_start_id, sequential.winner_start_id)
        for left, right in zip(sequential.runs, parallel.runs, strict=True):
            self.assertEqual(left.start_id, right.start_id)
            self.assertEqual(left.random_seed, right.random_seed)
            self.assertEqual(left.best_raw_parameters, right.best_raw_parameters)
            self.assertEqual(
                left.recalculated_best_training_mean_nll,
                right.recalculated_best_training_mean_nll,
            )
            self.assertEqual(left.convergence_checks, right.convergence_checks)

    def test_lbfgs_settings_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            LBFGSSettings(maximum_steps=0)
        with self.assertRaises(ValueError):
            LBFGSSettings(learning_rate=0.0)
        with self.assertRaises(ValueError):
            LBFGSSettings(line_search_fn="unsupported")

    def test_full_run_rejects_a_structurally_short_stability_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            fit_single_initialization_with_strategy(
                [self.training],
                [self.validation],
                self.registry,
                self.registry.initializations[0],
                optimizer_strategy="lbfgs",
                lbfgs_settings=LBFGSSettings(maximum_steps=4),
            )

    def test_hybrid_stability_uses_combined_adam_and_lbfgs_history(self) -> None:
        convergence = replace(
            self.registry.convergence,
            gradient_norm_tolerance=1e9,
            meaningful_improvement_tolerance=1.0,
            stable_training_window_epochs=25,
        )
        registry = replace(self.registry, convergence=convergence)
        result = fit_single_initialization_with_strategy(
            [self.training],
            [self.validation],
            registry,
            registry.initializations[0],
            optimizer_strategy="hybrid",
            adam_warmup_epochs=24,
            lbfgs_settings=LBFGSSettings(maximum_steps=1),
        )
        self.assertEqual(result.lbfgs_step_count, 1)
        self.assertLess(result.lbfgs_step_count, 25)
        self.assertEqual(result.stability_history_source, "combined_adam_lbfgs")
        self.assertTrue(dict(result.convergence_checks)["stable_fresh_training_nll"])
        self.assertTrue(result.converged)

    def test_v2_schema_explicitly_types_alternative_optimizer_fields(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "schemas"
            / "m0_fit_result_v2.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["result_schema_version"]["const"],
            "m0_fit_result_v2",
        )
        for field in (
            "optimizer_strategy",
            "adam_step_count",
            "lbfgs_step_count",
            "phase_switch_epoch",
            "gradient_norm_at_restored_best",
            "convergence_budget_sufficient",
            "stability_history_source",
        ):
            self.assertIn(field, schema["properties"])
        strategy_epoch = schema["$defs"]["strategyEpochRecord"]
        self.assertIn("closure_evaluation_count", strategy_epoch["properties"])
        self.assertIn("gradient_norm_after_step", strategy_epoch["properties"])
        self.assertEqual(
            self.registry.multi_start_selection.result_schema_version,
            "m0_fit_result_v2",
        )


if __name__ == "__main__":
    unittest.main()
