"""Regression tests for the matched-only M1 action-kernel workflow."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import m1_action_kernel_sliding_window_v1 as action_sliding
from m0_sliding_window_v1.experiment import (
    SlidingWindow,
    prepare_full_session_matched_data,
)
from run_m1_action_kernel_sliding_window import (
    PARAMETER_NAMES,
    WORKFLOW,
    WORKFLOW_OUTPUT_SUFFIX,
    _build_single_session_jobs,
    _generate_single_global_figures,
    _parse_args,
    _result_roots,
    _sessions,
    _window_canonical,
    single_session_run_failed,
)


def _window() -> SlidingWindow:
    return SlidingWindow(
        window_id=1,
        training_session_ids=("train",),
        training_ranks=(1,),
        validation_session_ids=("validation",),
        validation_ranks=(2,),
        test_session_id="test",
        test_rank=3,
    )


def _cli_args(*extra: str):
    return _parse_args(
        [
            "--raw-data-root",
            "/tmp/raw",
            "--registry",
            "/tmp/registry.json",
            *extra,
        ]
    )


class MatchedOnlyWorkflowTests(unittest.TestCase):
    def test_action_api_has_no_legacy_workflow_or_aggregation(self) -> None:
        self.assertFalse(hasattr(action_sliding, "LEGACY_PARTITIONED_WORKFLOW"))
        self.assertFalse(hasattr(action_sliding, "WORKFLOW_CHOICES"))
        self.assertFalse(
            hasattr(action_sliding, "aggregate_m1_action_kernel_session_initializations")
        )

    def test_four_windows_times_eight_workers_times_one_thread_is_accepted(self) -> None:
        args = _cli_args(
            "--window-parallel-workers",
            "4",
            "--rolling-workers",
            "8",
            "--rolling-threads-per-worker",
            "1",
        )
        self.assertEqual(
            args.window_parallel_workers
            * min(args.rolling_workers, 8)
            * args.rolling_threads_per_worker,
            32,
        )
        self.assertFalse(hasattr(args, "workflow"))

    def test_removed_workflow_option_is_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _cli_args("--workflow", "legacy_partitioned")

    def test_cpu_budget_rejects_four_windows_times_eight_workers_times_two_threads(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _cli_args(
                "--window-parallel-workers",
                "4",
                "--rolling-workers",
                "8",
                "--rolling-threads-per-worker",
                "2",
            )

    def test_rolling_window_wiring_uses_full_sessions_for_every_role(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class Prepared:
            def frame(self, session_id: str, scope: str, role: str) -> pd.DataFrame:
                calls.append((session_id, scope, role))
                return pd.DataFrame({"session_id": [session_id], "split_role": [role]})

        result = _window_canonical(Prepared(), _window())
        self.assertEqual(len(result), 3)
        self.assertEqual(
            calls,
            [
                ("train", "full", "train"),
                ("validation", "full", "validation"),
                ("test", "full", "test"),
            ],
        )

    def test_optimizer_session_wiring_uses_full_scope(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class Prepared:
            def frame(self, session_id: str, scope: str, role: str) -> pd.DataFrame:
                calls.append((session_id, scope, role))
                return pd.DataFrame({"session_id": [session_id]})

        with patch(
            "run_m1_action_kernel_sliding_window.m1_session_trials_from_frame",
            side_effect=lambda frame, split_role: (frame, split_role),
        ):
            sessions = _sessions(Prepared(), ("s1", "s2"), "train")
        self.assertEqual(len(sessions), 2)
        self.assertEqual(calls, [("s1", "full", "train"), ("s2", "full", "train")])

    def test_single_session_jobs_use_literal_final_two_scopes(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class Prepared:
            mouse_id = "MC08"
            source_sha256 = ("abc",)

            def frame(self, session_id: str, scope: str, role: str) -> pd.DataFrame:
                calls.append((session_id, scope, role))
                return pd.DataFrame({"session_id": [session_id], "split_role": [role]})

        initialization = {
            "mouse_id": "MC08",
            "session_id": "s13",
            "chronological_rank": 13,
            **{f"initial_{name}": float(index + 1) for index, name in enumerate(PARAMETER_NAMES)},
        }
        args = SimpleNamespace(
            optimizer="lbfgs",
            max_epochs=None,
            learning_rate=None,
            adam_warmup_epochs=1000,
            lbfgs_max_steps=250,
            lbfgs_learning_rate=1.0,
            lbfgs_history_size=10,
            lbfgs_max_evaluations=20,
            lbfgs_tolerance_grad=1e-9,
            lbfgs_tolerance_change=1e-12,
            single_session_threads_per_worker=1,
            resume=False,
        )
        jobs = _build_single_session_jobs(
            Prepared(), object(), args, Path("/tmp/results"), pd.DataFrame([initialization])
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            calls,
            [
                ("s13", "final_two_prefix", "train"),
                ("s13", "final_two_heldout", "test"),
            ],
        )
        self.assertNotIn("workflow", jobs[0])

    def test_result_roots_are_always_isolated_under_matched_suffix(self) -> None:
        args = SimpleNamespace(
            sliding_output_dir=Path("/tmp/sliding"),
            single_output_dir=Path("/tmp/single"),
        )
        sliding, single = _result_roots(args, "M1_action_kernel_v1", "MC08")
        self.assertEqual(sliding.name, WORKFLOW_OUTPUT_SUFFIX)
        self.assertEqual(single.name, WORKFLOW_OUTPUT_SUFFIX)
        self.assertEqual(sliding.parent.name, "MC08")
        self.assertEqual(WORKFLOW, "full_session_matched")

    def test_global_figures_exclude_failed_session_even_if_stale_predictions_exist(self) -> None:
        initializations = pd.DataFrame(
            {
                "session_id": ["complete", "failed"],
                "chronological_rank": [13, 14],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank, session_id, status in (
                (13, "complete", "complete"),
                (14, "failed", "failed"),
            ):
                directory = root / f"session_{rank:03d}_{session_id}"
                directory.mkdir()
                (directory / "single_session_manifest.json").write_text(
                    json.dumps({"status": status}), encoding="utf-8"
                )
                (directory / "trial_predictions.csv").write_text(
                    "stale_or_current\n1\n", encoding="utf-8"
                )
            with patch(
                "run_m1_action_kernel_sliding_window.generate_global_prediction_figures",
                return_value={"source_count": 1},
            ) as generate:
                _generate_single_global_figures(initializations, root)
        sources = generate.call_args.args[0]
        self.assertEqual([source_id for source_id, _ in sources], ["session_013_complete"])

    def test_matched_preparation_accepts_three_blocks_without_legacy_partitioner(self) -> None:
        assignments = pd.DataFrame(
            {"mouse_id": ["MC08"], "session_id": ["s1"], "chronological_rank": [1]}
        )
        canonical = pd.DataFrame(
            {
                "session_id": ["s1", "s1", "s1"],
                "trial_index": [1, 2, 3],
                "block_index": [1, 2, 3],
                "true_block": ["short", "neutral", "long"],
                "choice_long": [0.0, 1.0, 1.0],
            }
        )
        loaded = SimpleNamespace(
            sessions=pd.DataFrame(
                {
                    "session_id": ["s1"],
                    "included_in_model_trials": [True],
                    "source_file": ["s1.mat"],
                    "source_sha256": ["abc"],
                }
            ),
            canonical_trials=canonical,
        )
        registry = SimpleNamespace(split=SimpleNamespace(assignment_path=Path("assign.csv")))
        with (
            patch("m0_sliding_window_v1.experiment.pd.read_csv", return_value=assignments),
            patch("m0_sliding_window_v1.experiment._discover_paths", return_value=(Path("s1.mat"),)),
            patch("m0_sliding_window_v1.experiment.prepare_dataset", return_value=loaded),
            patch(
                "m0_sliding_window_v1.experiment.partition_session_blocks",
                side_effect=AssertionError("legacy partitioner must not run"),
            ),
        ):
            prepared = prepare_full_session_matched_data("/tmp/raw", registry)
        self.assertEqual(prepared.partitions["s1"].heldout_block_indices, (2, 3))
        self.assertIsNone(prepared.partitions["s1"].ignored_terminal_block_index)


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
        run_record = {
            "best_raw_parameters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "exception_message": "RuntimeError: restored-best gradient recomputation failed",
            "converged": False,
        }
        self.assertTrue(single_session_run_failed(run_record))


if __name__ == "__main__":
    unittest.main()
