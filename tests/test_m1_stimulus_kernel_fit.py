"""Fitting/registry/analysis tests for M1_stimulus_kernel_v1, on synthetic
data only (no real MATLAB files, no long real fits -- see module docstring
of fit.py for why every start's failure is contained, and for the
sequential-vs-ProcessPoolExecutor dispatch exercised below)."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from m0_sensory_v1.model.fit import LBFGSSettings
from m0_sensory_v1.model.m1_session import M1SessionTrials
from m1_stimulus_kernel_v1.analysis import (
    build_m1_stimulus_kernel_trial_predictions,
    evaluate_m1_stimulus_kernel_split_metrics,
    save_m1_stimulus_kernel_analysis_outputs,
)
from m1_stimulus_kernel_v1.fit import (
    M1StimulusKernelFitRunResult,
    _apply_stage2_result,
    _detect_identifiability_ridge,
    _only_gradient_norm_check_failed,
    _relax_gradient_norm_tolerance,
    _warm_start_initialization,
    fit_m1_stimulus_kernel_all_initializations,
    fit_m1_stimulus_kernel_single_initialization,
)
from m1_stimulus_kernel_v1.initialization import M1StimulusKernelInitializationSpec
from m1_stimulus_kernel_v1.parameters import decode_m1_stimulus_kernel_raw_parameters
from m1_stimulus_kernel_v1.registry import load_m1_stimulus_kernel_model_registry
from m1_stimulus_kernel_v1.session import evaluate_m1_stimulus_kernel_session


DTYPE = torch.float64


def _session(categories, choices, feedback, durations, *, session_id="s1", role="train"):
    return M1SessionTrials(
        mouse_id="mouse", session_id=session_id,
        duration_s=torch.tensor(durations, dtype=DTYPE),
        choice_long=torch.tensor(choices, dtype=DTYPE),
        category_long=torch.tensor(categories, dtype=DTYPE),
        feedback_valid=torch.tensor(feedback, dtype=torch.bool),
        split_role=role,
    )


def _synthetic_sessions():
    training = [
        _session(
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
            categories=[1.0, 1.0, 1.0, 1.0, 0.0],
            choices=[1.0, 1.0, 1.0, 1.0, 0.0],
            feedback=[True] * 5,
            durations=[2.0, 2.1, 1.9, 2.05, 0.5],
            session_id="val_1",
            role="validation",
        )
    ]
    return training, validation


class M1StimulusKernelRegistryTests(unittest.TestCase):
    def test_loads_and_preserves_m0_starts(self) -> None:
        registry = load_m1_stimulus_kernel_model_registry()
        self.assertEqual(registry.model_id, "M1_stimulus_kernel_v1")
        self.assertEqual(len(registry.initializations), 8)
        self.assertEqual(registry.initial_pi, 0.5)

    def test_subject_overlays_proxy_distinct_task_moments(self) -> None:
        mc06 = load_m1_stimulus_kernel_model_registry("config/m1_stimulus_kernel_v1_mc06.json")
        mc07 = load_m1_stimulus_kernel_model_registry("config/m1_stimulus_kernel_v1_mc07.json")
        self.assertNotEqual(mc06.task_moments.mu_short_s, mc07.task_moments.mu_short_s)
        self.assertEqual(mc06.source_m0_registry.model_id, "M0_sensory_v1")


class M1StimulusKernelFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_stimulus_kernel_model_registry()
        cls.training, cls.validation = _synthetic_sessions()

    def test_adam_produces_a_well_formed_result(self) -> None:
        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=15,
        )
        self.assertEqual(result.start_id, 1)
        self.assertEqual(result.optimizer_strategy, "adam")
        self.assertEqual(len(result.history), 15)
        self.assertEqual(result.optimizer_step_count, 15)
        self.assertTrue(all(step.optimizer_phase == "adam" for step in result.history))
        self.assertEqual(len(result.initial_constrained_parameters), 6)
        self.assertTrue(all(math.isfinite(value) for value in result.initial_constrained_parameters))
        self.assertTrue(
            all(len(step.constrained_parameters_after_step) == 6 for step in result.history)
        )
        first_step_values = result.history[0].constrained_parameters_after_step
        last_step_values = result.history[-1].constrained_parameters_after_step
        self.assertNotEqual(first_step_values, last_step_values)

    def test_lbfgs_produces_a_well_formed_result(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=8, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="lbfgs", lbfgs_settings=settings,
        )
        self.assertEqual(result.optimizer_strategy, "lbfgs")
        self.assertTrue(all(step.optimizer_phase == "lbfgs" for step in result.history))
        self.assertLessEqual(len(result.history), 8)

    def test_hybrid_runs_adam_warmup_then_lbfgs(self) -> None:
        settings = LBFGSSettings(
            maximum_steps=5, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="hybrid", adam_warmup_epochs=4,
            lbfgs_settings=settings,
        )
        phases = [step.optimizer_phase for step in result.history]
        self.assertEqual(phases.count("adam_warmup"), 4)
        self.assertEqual(phases.count("lbfgs"), 5)
        self.assertEqual(phases[:4], ["adam_warmup"] * 4)
        self.assertEqual(phases[4:], ["lbfgs"] * 5)

    def test_restored_best_is_not_necessarily_the_final_step(self) -> None:
        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=20,
        )
        best_recorded = min(step.fresh_training_mean_nll for step in result.history)
        self.assertAlmostEqual(
            result.recalculated_best_training_mean_nll, best_recorded, places=6
        )

    def test_failure_is_contained_not_raised(self) -> None:
        # An initialization whose encoded raw vector is immediately invalid
        # (negative w -- unreachable through the frozen registry, but a
        # direct unit-level check that fit_single_initialization degrades
        # to a non-converged result rather than propagating).
        bad = M1StimulusKernelInitializationSpec(
            start_id=99, w=-1.0, s0=0.5, ell_short=0.05, ell_long=0.05,
            kappa=1.0, a=0.0, purpose="deliberately invalid for failure-containment test",
        )
        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry, bad,
            optimizer_strategy="adam", max_epochs_override=5,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.termination_reason, "numerical_or_transform_failure")
        self.assertIsNotNone(result.exception_message)
        # The registry checksum must still be populated on the early-failure
        # return path, not only the normal one -- a resume check comparing
        # a saved failed run against today's registry needs it too.
        self.assertTrue(result.registry_config_sha256)

    def test_all_initializations_selects_training_only_winner(self) -> None:
        result = fit_m1_stimulus_kernel_all_initializations(
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

    def test_parallel_workers_matches_sequential_results(self) -> None:
        # parallel_workers>1 dispatches across a ProcessPoolExecutor instead
        # of the in-process loop; the fitted results (not wall-clock) must
        # be identical since every start uses a fresh, independent raw
        # tensor either way.
        sequential = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, self.registry,
            optimizer_strategy="adam", max_epochs_override=8,
        )
        parallel = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, self.registry,
            optimizer_strategy="adam", max_epochs_override=8,
            parallel_workers=4, cpu_threads_per_worker=1,
        )
        self.assertEqual(len(parallel.runs), 8)
        self.assertEqual(
            [run.start_id for run in parallel.runs],
            sorted(run.start_id for run in parallel.runs),
        )
        sequential_by_id = {run.start_id: run for run in sequential.runs}
        for run in parallel.runs:
            other = sequential_by_id[run.start_id]
            self.assertAlmostEqual(
                run.recalculated_best_training_mean_nll,
                other.recalculated_best_training_mean_nll,
                places=9,
            )
            self.assertEqual(run.converged, other.converged)

    def test_parallel_workers_writes_atomic_run_json_per_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_directory = Path(tmp) / "starts"
            result = fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry,
                optimizer_strategy="adam", max_epochs_override=3,
                parallel_workers=4, cpu_threads_per_worker=1,
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

    def test_rejects_nonpositive_parallel_settings(self) -> None:
        with self.assertRaises(ValueError):
            fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry, parallel_workers=0,
            )
        with self.assertRaises(ValueError):
            fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry, cpu_threads_per_worker=0,
            )

    def test_fit_run_result_carries_registry_checksum_for_resume(self) -> None:
        # scripts/_resume_staleness.py's docstring states every model's fit
        # result "already records a registry_config_sha256 per run" -- this
        # was not true for this model until fixed (mirroring a real gap
        # found in m1_known_switch_v1/fit.py, which also lacks this field
        # despite having its own ProcessPoolExecutor-based sliding-window
        # CLI --resume flag). Without it, result_registry_is_stale always
        # treats a saved result as unverifiable, so --resume can never skip
        # a completed fit.
        from _resume_staleness import result_registry_is_stale

        result = fit_m1_stimulus_kernel_single_initialization(
            self.training, self.validation, self.registry,
            self.registry.initializations[0],
            optimizer_strategy="adam", max_epochs_override=3,
        )
        self.assertTrue(result.registry_config_sha256)
        summary = {"runs": [result.as_dict()]}
        self.assertFalse(result_registry_is_stale(summary, self.registry))

        other_registry = load_m1_stimulus_kernel_model_registry(
            "config/m1_stimulus_kernel_v1_mc06.json"
        )
        self.assertTrue(result_registry_is_stale(summary, other_registry))

    def test_no_converged_run_reports_none_winner(self) -> None:
        strict_registry = replace(
            self.registry,
            convergence=replace(self.registry.convergence, gradient_norm_tolerance=1e-300),
        )
        result = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, strict_registry,
            optimizer_strategy="adam", max_epochs_override=3,
        )
        self.assertIsNone(result.winner_start_id)
        self.assertEqual(result.selection_status, "no_converged_run")

    def test_start_order_independence_of_winner_selection(self) -> None:
        forward = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, self.registry,
            optimizer_strategy="adam", max_epochs_override=10,
        )
        reversed_registry = replace(
            self.registry, initializations=tuple(reversed(self.registry.initializations))
        )
        backward = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, reversed_registry,
            optimizer_strategy="adam", max_epochs_override=10,
        )
        if forward.winner_start_id is not None and backward.winner_start_id is not None:
            self.assertEqual(forward.winner_start_id, backward.winner_start_id)


def _fit_run_result(**overrides) -> M1StimulusKernelFitRunResult:
    """Build a minimal, well-formed run result for escalation-helper tests
    that don't need a real optimization -- only specific fields matter per
    test, the rest are filler."""

    checks = overrides.pop(
        "checks",
        (
            ("finite_objective", True),
            ("finite_parameters", True),
            ("finite_gradients", True),
            ("small_gradient_norm", False),
            ("stable_fresh_training_nll", True),
            ("no_parameter_transform_errors", True),
            ("no_optimizer_runtime_errors", True),
        ),
    )
    defaults = dict(
        result_schema_version="m1_stimulus_kernel_fit_result_v1",
        model_id="M1_stimulus_kernel_v1",
        mouse_id="mouse",
        registry_config_sha256="deadbeef",
        start_id=1,
        optimizer_strategy="lbfgs",
        initial_raw_parameters=(0.0,) * 6,
        initial_constrained_parameters=(0.1, 0.5, 0.05, 0.05, 1.0, 0.0),
        best_raw_parameters=(0.0,) * 6,
        best_constrained_parameters=(0.1, 0.5, 0.05, 0.05, 1.0, 0.0),
        history=(),
        recalculated_best_training_mean_nll=0.5,
        final_validation_mean_nll=0.5,
        gradient_norm_at_restored_best=1.69e-5,
        optimizer_step_count=250,
        termination_reason="maximum_epochs_without_convergence",
        converged=False,
        checks=checks,
        exception_message=None,
    )
    defaults.update(overrides)
    return M1StimulusKernelFitRunResult(**defaults)


class M1StimulusKernelConvergenceEscalationTests(unittest.TestCase):
    """Deterministic, hand-crafted cases for the three pure escalation
    helpers, plus integration checks that the ladder in
    fit_m1_stimulus_kernel_all_initializations is actually wired up -- see
    fit.py's "Convergence escalation" docstring section. The real-world
    trigger this was added for (M1_stimulus_kernel_v1 MC06 rolling window 3:
    all eight starts failed only small_gradient_norm, one at
    gradient_norm_at_restored_best=1.69e-5 against a 1e-5 registry tolerance,
    the rest around 0.0025-0.0054) is reproduced by _fit_run_result's
    defaults above.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_stimulus_kernel_model_registry()
        cls.training, cls.validation = _synthetic_sessions()

    def test_relax_gradient_norm_tolerance_recovers_a_near_miss(self) -> None:
        run = _fit_run_result(gradient_norm_at_restored_best=1.69e-5)
        relaxed = _relax_gradient_norm_tolerance(run, tolerance=2e-5)
        self.assertTrue(relaxed.converged)
        self.assertEqual(dict(relaxed.checks)["small_gradient_norm"], True)
        self.assertEqual(
            relaxed.termination_reason,
            "maximum_epochs_convergence_checks_passed_relaxed_gradient_tolerance",
        )

    def test_relax_gradient_norm_tolerance_still_rejects_a_bad_run(self) -> None:
        run = _fit_run_result(gradient_norm_at_restored_best=0.0028)
        relaxed = _relax_gradient_norm_tolerance(run, tolerance=2e-5)
        self.assertFalse(relaxed.converged)

    def test_relax_gradient_norm_tolerance_leaves_exceptions_untouched(self) -> None:
        run = _fit_run_result(
            gradient_norm_at_restored_best=float("nan"),
            exception_message="RuntimeError: boom",
            termination_reason="numerical_or_transform_failure",
        )
        relaxed = _relax_gradient_norm_tolerance(run, tolerance=1e300)
        self.assertIs(relaxed, run)
        self.assertFalse(relaxed.converged)

    def test_warm_start_initialization_uses_best_checkpoint(self) -> None:
        run = _fit_run_result(
            start_id=7, best_constrained_parameters=(0.2, 0.6, 0.04, 0.06, 1.5, -0.3)
        )
        warm_start = _warm_start_initialization(run)
        self.assertEqual(warm_start.start_id, 7)
        self.assertEqual(
            (warm_start.w, warm_start.s0, warm_start.ell_short, warm_start.ell_long,
             warm_start.kappa, warm_start.a),
            (0.2, 0.6, 0.04, 0.06, 1.5, -0.3),
        )
        self.assertIn("escalation", warm_start.purpose)

    def test_apply_stage2_result_keeps_original_when_continuation_unconverged(self) -> None:
        # The real-world reason this matters: a targeted run is only ever
        # selected because every check except small_gradient_norm already
        # passed. A stage-2 continuation restarts with an empty history, so
        # replacing the original with a continuation that itself doesn't
        # converge could regress stable_fresh_training_nll from True to
        # False for no benefit -- worse than not running stage 2 at all.
        original = _fit_run_result(start_id=3, converged=False)
        stage2 = _fit_run_result(
            start_id=3, converged=False,
            checks=(
                ("finite_objective", True),
                ("finite_parameters", True),
                ("finite_gradients", True),
                ("small_gradient_norm", False),
                ("stable_fresh_training_nll", False),  # regressed vs. original's True
                ("no_parameter_transform_errors", True),
                ("no_optimizer_runtime_errors", True),
            ),
        )
        runs = [_fit_run_result(start_id=1), original]
        result = _apply_stage2_result(runs, original, stage2)
        self.assertIs(result[1], original)
        self.assertTrue(dict(result[1].checks)["stable_fresh_training_nll"])

    def test_apply_stage2_result_replaces_when_continuation_converged(self) -> None:
        original = _fit_run_result(start_id=3, converged=False)
        stage2 = _fit_run_result(
            start_id=3, converged=True, gradient_norm_at_restored_best=1e-7,
            checks=tuple(
                (name, True) for name, _ in _fit_run_result().checks
            ),
        )
        runs = [_fit_run_result(start_id=1), original]
        result = _apply_stage2_result(runs, original, stage2)
        self.assertIs(result[1], stage2)
        self.assertTrue(result[1].converged)

    def test_only_gradient_norm_check_failed_is_true_when_sole_failure(self) -> None:
        run = _fit_run_result()  # default checks: only small_gradient_norm is False
        self.assertTrue(_only_gradient_norm_check_failed(run))

    def test_only_gradient_norm_check_failed_is_false_with_another_failing_check(self) -> None:
        run = _fit_run_result(
            checks=(
                ("finite_objective", True),
                ("finite_parameters", True),
                ("finite_gradients", True),
                ("small_gradient_norm", False),
                ("stable_fresh_training_nll", False),  # a second, unrelated failure
                ("no_parameter_transform_errors", True),
                ("no_optimizer_runtime_errors", True),
            )
        )
        self.assertFalse(_only_gradient_norm_check_failed(run))

    def test_only_gradient_norm_check_failed_is_false_on_exception(self) -> None:
        run = _fit_run_result(exception_message="RuntimeError: boom")
        self.assertFalse(_only_gradient_norm_check_failed(run))

    def test_only_gradient_norm_check_failed_is_false_when_check_missing(self) -> None:
        # A malformed/future-schema record with no small_gradient_norm entry
        # at all must not be silently treated as "gradient norm is the sole
        # failure" just because every present check happens to pass.
        run = _fit_run_result(
            checks=(
                ("finite_objective", True),
                ("finite_parameters", True),
                ("finite_gradients", True),
                ("stable_fresh_training_nll", True),
                ("no_parameter_transform_errors", True),
                ("no_optimizer_runtime_errors", True),
            )
        )
        self.assertFalse(_only_gradient_norm_check_failed(run))

    def test_only_gradient_norm_check_failed_is_false_when_already_true(self) -> None:
        run = _fit_run_result(
            checks=(
                ("finite_objective", True),
                ("finite_parameters", True),
                ("finite_gradients", True),
                ("small_gradient_norm", True),
                ("stable_fresh_training_nll", True),
                ("no_parameter_transform_errors", True),
                ("no_optimizer_runtime_errors", True),
            )
        )
        self.assertFalse(_only_gradient_norm_check_failed(run))

    def test_rejects_invalid_escalation_settings(self) -> None:
        with self.assertRaises(ValueError):
            fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry,
                escalated_lbfgs_additional_steps=0,
            )
        with self.assertRaises(ValueError):
            fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry,
                gradient_norm_tolerance_relaxation_factor=0.5,
            )
        with self.assertRaises(ValueError):
            fit_m1_stimulus_kernel_all_initializations(
                self.training, self.validation, self.registry,
                gradient_norm_tolerance_relaxation_factor=float("nan"),
            )

    def test_escalation_ladder_engages_and_is_reported_when_unrecovered(self) -> None:
        # An unreachable gradient-norm tolerance guarantees stage 1 fails on
        # this tiny fixture regardless of step count, and (since 1e-300 is
        # unreachable at any relaxation factor tried here) stage 3 cannot
        # rescue it either -- this exercises the full ladder's wiring and
        # bookkeeping without depending on a fragile numerical near-miss.
        strict_registry = replace(
            self.registry,
            convergence=replace(self.registry.convergence, gradient_norm_tolerance=1e-300),
        )
        settings = LBFGSSettings(
            maximum_steps=3, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, strict_registry,
            optimizer_strategy="lbfgs", lbfgs_settings=settings,
            escalated_lbfgs_additional_steps=5,
        )
        self.assertIsNone(result.winner_start_id)
        self.assertEqual(result.convergence_escalation_stage, "unrecovered")
        self.assertIsNotNone(result.convergence_escalation_detail)
        self.assertEqual(
            result.convergence_escalation_detail["stage2_additional_max_steps"], 5
        )
        self.assertEqual(
            result.convergence_escalation_detail["stage2_targeting_rule"],
            "sole_failing_check_is_small_gradient_norm",
        )
        self.assertEqual(len(result.runs), 8)

    def test_escalation_can_be_disabled(self) -> None:
        strict_registry = replace(
            self.registry,
            convergence=replace(self.registry.convergence, gradient_norm_tolerance=1e-300),
        )
        settings = LBFGSSettings(
            maximum_steps=3, learning_rate=1.0, history_size=10,
            maximum_evaluations_per_step=20, tolerance_grad=1e-9,
            tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        result = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, strict_registry,
            optimizer_strategy="lbfgs", lbfgs_settings=settings,
            enable_convergence_escalation=False,
        )
        self.assertIsNone(result.winner_start_id)
        self.assertEqual(result.convergence_escalation_stage, "stage1_baseline")
        self.assertIsNone(result.convergence_escalation_detail)

    def test_adam_strategy_is_not_escalated(self) -> None:
        strict_registry = replace(
            self.registry,
            convergence=replace(self.registry.convergence, gradient_norm_tolerance=1e-300),
        )
        result = fit_m1_stimulus_kernel_all_initializations(
            self.training, self.validation, strict_registry,
            optimizer_strategy="adam", max_epochs_override=3,
        )
        self.assertIsNone(result.winner_start_id)
        self.assertEqual(result.convergence_escalation_stage, "stage1_baseline")


class M1StimulusKernelIdentifiabilityRidgeTests(unittest.TestCase):
    """Deterministic, hand-crafted cases for the pure ridge-detection helper
    (not exercised by real fits above, since the optimizer's landing point
    isn't controllable): confirms the port from m1_dynamic_block_length_v1
    actually distinguishes tied-and-close, tied-and-far, and not-tied.
    """

    @classmethod
    def setUpClass(cls) -> None:
        multi = load_m1_stimulus_kernel_model_registry().multi_start_selection
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
        # Only kappa (scale 1.0) differs by 0.35 >= threshold 0.25.
        far_values = (0.03, 0.5, 0.03, 0.03, 1.35, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, far_values)
        self.assertTrue(detected)
        self.assertEqual(ridge_ids, (1, 2))

    def test_tied_and_close_is_not_a_ridge(self) -> None:
        # kappa differs by only 0.05, well under threshold 0.25.
        close_values = (0.03, 0.5, 0.03, 0.03, 1.05, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + self.tolerance / 2, close_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())

    def test_far_parameters_but_not_tied_is_not_a_ridge(self) -> None:
        # Same large kappa gap as the ridge case, but the candidate's NLL is
        # far outside near_equal_training_nll_tolerance, so it never qualifies.
        far_values = (0.03, 0.5, 0.03, 0.03, 1.35, 0.0)
        detected, ridge_ids = self._ridge(self.winner_nll + 1.0, far_values)
        self.assertFalse(detected)
        self.assertEqual(ridge_ids, ())


class M1StimulusKernelAnalysisTests(unittest.TestCase):
    def test_split_metrics_match_direct_evaluation(self) -> None:
        registry = load_m1_stimulus_kernel_model_registry()
        training, validation = _synthetic_sessions()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        parameters = decode_m1_stimulus_kernel_raw_parameters(
            raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        direct = [
            evaluate_m1_stimulus_kernel_session(session, parameters, registry.task_moments)
            for session in validation
        ]
        expected_mean_nll = float(
            sum(record.negative_log_likelihood for record in direct)
            / sum(record.valid_trial_count for record in direct)
        )
        metrics = evaluate_m1_stimulus_kernel_split_metrics(
            validation, raw, registry, required_split_role="validation"
        )
        self.assertAlmostEqual(metrics.mean_nll_per_valid_trial, expected_mean_nll, places=9)

    def test_rejects_test_split_role(self) -> None:
        registry = load_m1_stimulus_kernel_model_registry()
        _, validation = _synthetic_sessions()
        raw = registry.initializations[0].raw_tensor(
            lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        )
        with self.assertRaises(ValueError):
            evaluate_m1_stimulus_kernel_split_metrics(
                validation, raw, registry, required_split_role="test"
            )


def _canonical_frame(true_block, *, session_id="s1", split_role="train") -> pd.DataFrame:
    n = len(true_block)
    return pd.DataFrame({
        "mouse_id": ["mouse"] * n,
        "session_id": [session_id] * n,
        "trial_index": list(range(1, n + 1)),
        "session_date": ["2026-01-01"] * n,
        "duration_s": ([0.5, 2.0] * n)[:n],
        "choice_long": ([0.0, 1.0] * n)[:n],
        "category_long": ([0.0, 1.0] * n)[:n],
        "feedback_valid": [1.0] * n,
        "true_block": true_block,
        "split_role": [split_role] * n,
    })


class M1StimulusKernelTrialPredictionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_stimulus_kernel_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max, epsilon=cls.registry.parameter_epsilon
        )

    def test_matches_direct_session_evaluation(self) -> None:
        frame = _canonical_frame(["N", "N", "S", "S", "L", "L"])
        predictions = build_m1_stimulus_kernel_trial_predictions(frame, self.raw, self.registry)
        parameters = decode_m1_stimulus_kernel_raw_parameters(
            self.raw, lapse_max=self.registry.lapse_max, epsilon=self.registry.parameter_epsilon
        )
        from m0_sensory_v1.model.m1_session import m1_session_trials_from_frame

        session = m1_session_trials_from_frame(frame, split_role="train")
        direct = evaluate_m1_stimulus_kernel_session(session, parameters, self.registry.task_moments)
        torch.testing.assert_close(
            torch.tensor(predictions["model_prior_long"].to_numpy(), dtype=torch.float64),
            direct.predictions.prior_long,
        )
        torch.testing.assert_close(
            torch.tensor(predictions["model_probability_long"].to_numpy(), dtype=torch.float64),
            direct.predictions.probability_long,
        )

    def test_block_type_derived_from_true_block(self) -> None:
        frame = _canonical_frame(["N", "S", "L"])
        predictions = build_m1_stimulus_kernel_trial_predictions(frame, self.raw, self.registry)
        self.assertEqual(
            predictions["block_type"].tolist(), ["neutral", "short_block", "long_block"]
        )

    def test_rejects_test_role_without_allow_test(self) -> None:
        frame = _canonical_frame(["N", "S"], split_role="test")
        with self.assertRaises(PermissionError):
            build_m1_stimulus_kernel_trial_predictions(frame, self.raw, self.registry)
        predictions = build_m1_stimulus_kernel_trial_predictions(
            frame, self.raw, self.registry, allow_test=True
        )
        self.assertEqual(len(predictions), 2)

    def test_rejects_wrong_parameter_count(self) -> None:
        frame = _canonical_frame(["N", "S"])
        with self.assertRaises(ValueError):
            build_m1_stimulus_kernel_trial_predictions(
                frame, self.raw[:5], self.registry
            )

    def test_save_analysis_outputs_writes_expected_files(self) -> None:
        frame = _canonical_frame(["N", "N", "S", "S", "L", "L"])
        predictions = build_m1_stimulus_kernel_trial_predictions(frame, self.raw, self.registry)
        fit_summary = {
            "model_id": "M1_stimulus_kernel_v1",
            "mouse_id": "mouse",
            "winner_start_id": 1,
            "runs": [{
                "start_id": 1,
                "best_constrained_parameters": [
                    float(value) for value in self.registry.initializations[0].constrained_tensor()
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = save_m1_stimulus_kernel_analysis_outputs(
                predictions, fit_summary, tmp, test_unlocked=False
            )
            self.assertTrue((Path(tmp) / "trial_predictions.csv").is_file())
            self.assertTrue((Path(tmp) / "performance_summary.csv").is_file())
            self.assertTrue((Path(tmp) / "model_output_summary.json").is_file())
        self.assertEqual(summary["saved_trial_count"], len(predictions))
        self.assertIn("w", summary["fitted_parameters"])
        self.assertIn("a", summary["fitted_parameters"])


if __name__ == "__main__":
    unittest.main()
