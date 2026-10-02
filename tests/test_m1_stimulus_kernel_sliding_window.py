"""Unit tests for M1_stimulus_kernel_sliding_window_v1's model-specific
aggregation logic (pure Python/numpy, no real data, no fitting)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import unittest
import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from m0_sliding_window_v1.experiment import SlidingWindow
from m1_stimulus_kernel_sliding_window_v1 import (
    M1_STIMULUS_KERNEL_PARAMETER_NAMES,
    aggregate_m1_stimulus_kernel_session_initializations,
)
from run_m1_stimulus_kernel_sliding_window import (
    _find_winner_run,
    _optimizer_provenance,
    _record_winner,
    single_session_run_failed,
)


def _window(window_id: int, training_session_ids, test_session_id: str) -> SlidingWindow:
    return SlidingWindow(
        window_id=window_id,
        training_session_ids=tuple(training_session_ids),
        training_ranks=tuple(range(1, len(training_session_ids) + 1)),
        validation_session_ids=(),
        validation_ranks=(),
        test_session_id=test_session_id,
        test_rank=len(training_session_ids) + 1,
    )


class M1StimulusKernelAggregationTests(unittest.TestCase):
    def test_parameter_names_include_a(self) -> None:
        self.assertEqual(
            M1_STIMULUS_KERNEL_PARAMETER_NAMES,
            ("w", "s0", "ell_short", "ell_long", "kappa", "a"),
        )

    def test_alpha_averaged_in_probability_space_not_logit_space(self) -> None:
        # Two windows contribute the same session with wildly different
        # alpha (0.02 and 0.98). A naive arithmetic mean of the raw logits
        # would return exactly 0 (alpha=0.5) since logit(0.02)=-logit(0.98).
        # Averaging in probability space instead gives alpha=0.5 too in
        # this specific symmetric case -- so use an asymmetric pair to
        # actually distinguish the two conventions.
        a_low = math.log(0.1 / 0.9)
        a_high = math.log(0.9 / 0.1)
        # symmetric case: both conventions agree at alpha=0.5, so use an
        # asymmetric contributor set instead.
        a_mid = math.log(0.3 / 0.7)
        windows = [
            _window(1, ["s1"], "s2"),
            _window(2, ["s1"], "s3"),
        ]
        winners = {
            1: [0.01, 0.5, 0.05, 0.05, 1.0, a_low],
            2: [0.01, 0.5, 0.05, 0.05, 1.0, a_mid],
        }
        assignments = pd.DataFrame({
            "mouse_id": ["m", "m", "m"],
            "session_id": ["s1", "s2", "s3"],
            "chronological_rank": [1, 2, 3],
        })
        result = aggregate_m1_stimulus_kernel_session_initializations(
            assignments, windows, winners
        )
        row = result.loc[result["session_id"] == "s1"].iloc[0]
        expected_alpha = (0.1 + 0.3) / 2.0
        expected_logit_mean_alpha = 1.0 / (1.0 + math.exp(-(a_low + a_mid) / 2.0))
        self.assertAlmostEqual(row["initial_alpha"], expected_alpha, places=9)
        # The probability-space mean must differ from a naive logit mean
        # whenever contributors are asymmetric around zero.
        self.assertNotAlmostEqual(row["initial_alpha"], expected_logit_mean_alpha, places=6)
        expected_a = math.log(expected_alpha / (1.0 - expected_alpha))
        self.assertAlmostEqual(row["initial_a"], expected_a, places=9)

    def test_kappa_still_averaged_geometrically(self) -> None:
        windows = [_window(1, ["s1"], "s2"), _window(2, ["s1"], "s3")]
        winners = {
            1: [0.01, 0.5, 0.05, 0.05, 2.0, 0.0],
            2: [0.01, 0.5, 0.05, 0.05, 8.0, 0.0],
        }
        assignments = pd.DataFrame({
            "mouse_id": ["m", "m", "m"],
            "session_id": ["s1", "s2", "s3"],
            "chronological_rank": [1, 2, 3],
        })
        result = aggregate_m1_stimulus_kernel_session_initializations(
            assignments, windows, winners
        )
        row = result.loc[result["session_id"] == "s1"].iloc[0]
        self.assertAlmostEqual(row["initial_kappa"], 4.0, places=9)  # geometric mean of 2,8

    def test_fallback_session_uses_final_window_winner(self) -> None:
        windows = [_window(1, ["s1"], "s2")]
        winners = {1: [0.01, 0.5, 0.05, 0.05, 1.0, 0.3]}
        assignments = pd.DataFrame({
            "mouse_id": ["m", "m"],
            "session_id": ["s1", "s2"],
            "chronological_rank": [1, 2],
        })
        result = aggregate_m1_stimulus_kernel_session_initializations(
            assignments, windows, winners
        )
        fallback_row = result.loc[result["session_id"] == "s2"].iloc[0]
        self.assertEqual(
            fallback_row["initialization_policy"],
            "final_window_winner_fallback_no_training_membership",
        )
        self.assertAlmostEqual(fallback_row["initial_a"], 0.3, places=9)

    def test_rejects_nonpositive_kappa(self) -> None:
        windows = [_window(1, ["s1"], "s2")]
        winners = {1: [0.01, 0.5, 0.05, 0.05, -1.0, 0.0]}
        assignments = pd.DataFrame({
            "mouse_id": ["m", "m"],
            "session_id": ["s1", "s2"],
            "chronological_rank": [1, 2],
        })
        with self.assertRaises(ValueError):
            aggregate_m1_stimulus_kernel_session_initializations(assignments, windows, winners)

    def test_requires_nonempty_windows(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_m1_stimulus_kernel_session_initializations(
                pd.DataFrame({"mouse_id": [], "session_id": [], "chronological_rank": []}),
                [],
                {},
            )


class SingleSessionRunFailedTests(unittest.TestCase):
    def test_clean_but_unconverged_run_is_not_failed(self) -> None:
        run_record = {
            "best_raw_parameters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "exception_message": None,
            "converged": False,
        }
        self.assertFalse(single_session_run_failed(run_record))

    def test_nonfinite_best_raw_parameters_is_failed(self) -> None:
        run_record = {
            "best_raw_parameters": [float("nan")] * 6,
            "exception_message": None,
            "converged": False,
        }
        self.assertTrue(single_session_run_failed(run_record))

    def test_exception_with_finite_parameters_is_still_failed(self) -> None:
        # A restore-step failure can leave the last-good (finite) checkpoint
        # in best_raw_parameters while still recording an exception -- this
        # must be treated as failed even though the numbers look fine.
        run_record = {
            "best_raw_parameters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "exception_message": "RuntimeError: restored-best gradient recomputation failed",
            "converged": False,
        }
        self.assertTrue(single_session_run_failed(run_record))


def _summary(runs, winner_start_id) -> dict:
    return {"winner_start_id": winner_start_id, "runs": runs}


def _run(start_id: int, best_constrained_parameters=(0.0,) * 6) -> dict:
    return {
        "start_id": start_id,
        "best_constrained_parameters": list(best_constrained_parameters),
    }


class WinnerLookupTests(unittest.TestCase):
    """The graceful-skip lookup that replaced a raising ``_winner()`` --
    the crash-containment fix this test file was missing before (a real
    MC06 rolling window crash, all eight starts failing only
    small_gradient_norm even after the escalation ladder, is what this
    protects against). Purely dict-based, no real fitting needed.
    """

    def test_returns_winner_run_when_present(self) -> None:
        summary = _summary([_run(1), _run(2, (1.0,) * 6)], winner_start_id=2)
        winner = _find_winner_run(summary)
        self.assertEqual(winner["start_id"], 2)

    def test_returns_none_for_explicit_null_winner(self) -> None:
        summary = _summary([_run(1), _run(2)], winner_start_id=None)
        self.assertIsNone(_find_winner_run(summary))

    def test_raises_on_missing_winner_start_id_key(self) -> None:
        summary = {"runs": [_run(1)]}
        with self.assertRaises(RuntimeError):
            _find_winner_run(summary)

    def test_raises_on_dangling_winner_start_id(self) -> None:
        summary = _summary([_run(1), _run(2)], winner_start_id=99)
        with self.assertRaises(RuntimeError):
            _find_winner_run(summary)

    def test_record_winner_adds_entry_when_present(self) -> None:
        summary = _summary([_run(5, (2.0,) * 6)], winner_start_id=5)
        winner_parameters: dict[int, list[float]] = {}
        _record_winner(_window(3, ["s1"], "s2"), summary, winner_parameters)
        self.assertEqual(winner_parameters, {3: [2.0] * 6})

    def test_record_winner_warns_and_skips_when_absent(self) -> None:
        summary = _summary([_run(1)], winner_start_id=None)
        winner_parameters: dict[int, list[float]] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _record_winner(_window(4, ["s1"], "s2"), summary, winner_parameters)
        self.assertEqual(winner_parameters, {})
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))


def _optimizer_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        optimizer="lbfgs", max_epochs=None, learning_rate=None,
        adam_warmup_epochs=1000, lbfgs_max_steps=250, lbfgs_learning_rate=1.0,
        lbfgs_history_size=10, lbfgs_max_evaluations=20,
        lbfgs_tolerance_grad=1e-9, lbfgs_tolerance_change=1e-12,
        lbfgs_escalated_additional_steps=250,
        gradient_norm_tolerance_relaxation_factor=2.0,
        disable_convergence_escalation=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class OptimizerProvenanceTests(unittest.TestCase):
    """The resume-staleness fields a saved fit must be checked against --
    added because a real gap was found where changing the new escalation
    CLI flags between runs and re-running with --resume would have
    silently reused a stale fit_result.json.
    """

    def test_includes_convergence_escalation_settings(self) -> None:
        provenance = _optimizer_provenance(_optimizer_args())
        self.assertIn("convergence_escalation", provenance)
        self.assertEqual(
            provenance["convergence_escalation"],
            {
                "enabled": True,
                "escalated_lbfgs_additional_steps": 250,
                "gradient_norm_tolerance_relaxation_factor": 2.0,
            },
        )

    def test_changes_when_escalation_settings_change(self) -> None:
        baseline = _optimizer_provenance(_optimizer_args())
        changed = _optimizer_provenance(
            _optimizer_args(gradient_norm_tolerance_relaxation_factor=3.0)
        )
        self.assertNotEqual(baseline, changed)

    def test_changes_when_escalation_disabled(self) -> None:
        baseline = _optimizer_provenance(_optimizer_args())
        disabled = _optimizer_provenance(
            _optimizer_args(disable_convergence_escalation=True)
        )
        self.assertNotEqual(baseline, disabled)


if __name__ == "__main__":
    unittest.main()
