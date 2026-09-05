"""Regression tests for the command-line fitting callbacks and overrides."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.fit_m0 import (  # noqa: E402
    _print_initial,
    _print_runtime_settings,
    _validate_convergence_budget,
    parse_args as parse_fit_args,
)
from scripts.generate_m0_figures import _resolve_registry_path  # noqa: E402
from scripts import generate_m1_figures  # noqa: E402
from scripts.generate_m1_figures import (  # noqa: E402
    _generate_from_completed_fit,
    _resolve_m1_registry_path,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class FitCliTests(unittest.TestCase):
    def test_shared_parallel_worker_controls_are_exposed(self) -> None:
        argv = [
            "fit_m0.py",
            "--raw-data-root", "/tmp/raw",
            "--output-dir", "/tmp/out",
            "--parallel-workers", "4",
            "--cpu-threads-per-worker", "2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_fit_args()
        self.assertEqual(args.parallel_workers, 4)
        self.assertEqual(args.cpu_threads_per_worker, 2)

    def test_shared_parallel_worker_controls_reject_nonpositive_values(self) -> None:
        argv = [
            "fit_m0.py",
            "--raw-data-root", "/tmp/raw",
            "--output-dir", "/tmp/out",
            "--parallel-workers", "0",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_fit_args()

    def test_initial_callback_has_no_hidden_cli_scope_dependency(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            _print_initial(1, 0.5, 0.6)
        self.assertIn("start=01 epoch=0000", output.getvalue())

    def test_runtime_settings_print_explicit_full_run_overrides(self) -> None:
        args = argparse.Namespace(
            smoke_test=False,
            smoke_epochs=2,
            max_epochs=5000,
            learning_rate=None,
        )
        output = StringIO()
        with redirect_stdout(output):
            _print_runtime_settings(args, load_model_registry())
        text = output.getvalue()
        self.assertIn("effective_max_epochs=5000", text)
        self.assertIn("effective_initial_learning_rate=0.01", text)
        self.assertIn("epoch_override=1", text)

    def test_runtime_settings_report_hybrid_phase_controls(self) -> None:
        args = argparse.Namespace(
            optimizer="hybrid",
            smoke_test=False,
            smoke_epochs=2,
            max_epochs=None,
            learning_rate=None,
            adam_warmup_epochs=1000,
            lbfgs_max_steps=250,
            lbfgs_learning_rate=1.0,
            lbfgs_history_size=10,
            lbfgs_max_evaluations=20,
            lbfgs_tolerance_grad=1e-9,
            lbfgs_tolerance_change=1e-12,
        )
        output = StringIO()
        with redirect_stdout(output):
            _print_runtime_settings(args, load_model_registry())
        text = output.getvalue()
        self.assertIn("optimizer_strategy=hybrid", text)
        self.assertIn("adam_warmup_epochs=1000", text)
        self.assertIn("lbfgs_max_steps=250", text)
        self.assertIn("lbfgs_history_size=10", text)

    def test_full_alternative_run_rejects_short_convergence_budget(self) -> None:
        args = argparse.Namespace(
            optimizer="lbfgs",
            smoke_test=False,
            adam_warmup_epochs=1000,
            lbfgs_max_steps=24,
        )
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            _validate_convergence_budget(args, load_model_registry())

    def test_hybrid_budget_combines_both_phase_lengths(self) -> None:
        args = argparse.Namespace(
            optimizer="hybrid",
            smoke_test=False,
            adam_warmup_epochs=24,
            lbfgs_max_steps=1,
        )
        _validate_convergence_budget(args, load_model_registry())

    def test_figure_registry_is_inferred_from_fitted_subject(self) -> None:
        mc07 = _resolve_registry_path(None, {"mouse_id": "MC07"})
        mc08 = _resolve_registry_path(None, {"mouse_id": "MC08"})
        self.assertEqual(mc07.name, "m0_sensory_v1_mc07.json")
        self.assertEqual(mc08.name, "m0_sensory_v1.json")

    def test_m1_figure_registry_is_inferred_from_fitted_subject(self) -> None:
        mc06 = _resolve_m1_registry_path(None, {"mouse_id": "MC06"})
        mc08 = _resolve_m1_registry_path(None, {"mouse_id": "MC08"})
        self.assertEqual(mc06.name, "m1_hmm_true_v1_mc06.json")
        self.assertEqual(mc08.name, "m1_hmm_true_v1.json")

    def test_m1_completed_fit_mode_unlocks_test_without_refitting(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            (results / "starts").mkdir(parents=True)
            registry_path = root / "m1_registry.json"
            registry_path.write_text('{"model_id": "M1_hmm_true_v1"}\n')
            registry_sha = sha256(registry_path.read_bytes()).hexdigest()
            summary = {
                "mouse_id": "MC06",
                "winner_start_id": 3,
                "runs": [{
                    "start_id": 3,
                    "registry_config_sha256": registry_sha,
                    "best_raw_parameters": [0.0, 0.0, 0.0, 0.0, -2.0],
                }],
            }
            (results / "starts" / "multistart_summary.json").write_text(
                json.dumps(summary)
            )
            args = argparse.Namespace(
                results_dir=results,
                fit_summary=None,
                registry=registry_path,
                raw_data_root=root / "raw",
                unlock_test=True,
                output_dir=None,
                m0_predictions=None,
            )
            prepared = SimpleNamespace(
                canonical_trials=pd.DataFrame({"source": ["development"]})
            )
            locked = pd.DataFrame({"source": ["test"]})
            predictions = pd.DataFrame({"model_evaluated": [True, True]})
            with (
                patch.object(
                    generate_m1_figures,
                    "load_m1_model_registry",
                    return_value=object(),
                ),
                patch.object(
                    generate_m1_figures,
                    "prepare_registered_m1_fit_data",
                    return_value=prepared,
                ),
                patch.object(
                    generate_m1_figures,
                    "prepare_locked_test_analysis_trials",
                    return_value=locked,
                ) as load_test,
                patch.object(
                    generate_m1_figures,
                    "build_m1_trial_predictions",
                    return_value=predictions,
                ) as build,
                patch.object(
                    generate_m1_figures,
                    "save_m1_analysis_outputs",
                    return_value={"saved_valid_decision_count": 2},
                ) as save,
                patch.object(
                    generate_m1_figures,
                    "generate_m1_diagnostic_figures",
                    return_value={"m0_comparison_included": False},
                ),
                patch.object(
                    generate_m1_figures,
                    "generate_m1_m0_style_figures",
                    return_value={"test_unlocked": True},
                ) as m0_style,
            ):
                output = _generate_from_completed_fit(args)

            load_test.assert_called_once()
            combined = build.call_args.args[0]
            self.assertEqual(combined["source"].tolist(), ["development", "test"])
            self.assertTrue(build.call_args.kwargs["allow_test"])
            self.assertTrue(save.call_args.kwargs["test_unlocked"])
            self.assertTrue(m0_style.call_args.kwargs["test_unlocked"])
            self.assertTrue(output["test_unlocked"])


if __name__ == "__main__":
    unittest.main()
