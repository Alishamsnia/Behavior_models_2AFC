"""Runner-level contract shared by the five new models' sliding-window CLIs.

Every new runner continues past a window with no converged start (the
``run_m2_dynamic_previous_action_bias_sliding_window.py`` contract) instead of
aborting, and must keep the module-level hooks that
``extra_tools/forced_best_train_nll.py`` monkeypatches. Also covers the
tenth coordinate of M2_dynamic_action_kernel_bias_v1 through the runner's
fine-tune initialization and its opt-in identifiability surfaces.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "extra_tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

RUNNERS = {
    "pure_action": "run_pure_action_sliding_window",
    "extended_action": "run_extended_action_sliding_window",
    "previous_action_bias": "run_m1_previous_action_bias_sliding_window",
    "previous_wsls_bias": "run_m1_previous_wsls_bias_sliding_window",
    "action_kernel_bias": "run_m2_dynamic_action_kernel_bias_sliding_window",
}


class _Window:
    window_id = 7


class NoWinnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modules = {key: importlib.import_module(name) for key, name in RUNNERS.items()}

    def test_find_winner_run_contract(self) -> None:
        runs = [{"start_id": 1, "best_constrained_parameters": [0.1]},
                {"start_id": 2, "best_constrained_parameters": [0.2]}]
        for key, module in self.modules.items():
            with self.subTest(runner=key):
                self.assertIsNone(module._find_winner_run({"winner_start_id": None, "runs": runs}))
                self.assertIs(module._find_winner_run({"winner_start_id": 2, "runs": runs}), runs[1])
                with self.assertRaises(RuntimeError):
                    module._find_winner_run({"runs": runs})
                with self.assertRaises(RuntimeError):
                    module._find_winner_run({"winner_start_id": 9, "runs": runs})

    def test_record_winner_skips_no_winner_windows_without_raising(self) -> None:
        for key, module in self.modules.items():
            with self.subTest(runner=key):
                recorded: dict[int, list[float]] = {}
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    module._record_winner(_Window(), {"winner_start_id": None, "runs": []}, recorded)
                self.assertEqual(recorded, {})
                module._record_winner(
                    _Window(),
                    {"winner_start_id": 1, "runs": [{"start_id": 1, "best_constrained_parameters": [0.5]}]},
                    recorded,
                )
                self.assertEqual(recorded, {7: [0.5]})

    def test_forced_best_tool_hooks_and_registration(self) -> None:
        forced = importlib.import_module("forced_best_train_nll")
        for key, name in RUNNERS.items():
            with self.subTest(runner=key):
                self.assertEqual(forced.RUNNERS[key], name)
                module = self.modules[key]
                for hook in ("_run_windows", "_fit_window", "main", "manifest_fingerprint_is_stale",
                             "result_registry_is_stale", "_generate_sliding_global_figures"):
                    self.assertTrue(callable(getattr(module, hook)), hook)
                patched = [n for n in dir(module) if re.fullmatch(r"fit_(.*_)?all_initializations", n)]
                self.assertEqual(len(patched), 1, "the never-retrain guard must find the fitter by name")

    def test_wsls_style_runners_default_to_full_session_matched(self) -> None:
        for key in ("pure_action", "extended_action"):
            module = self.modules[key]
            saved_argv = sys.argv
            sys.argv = ["runner", "--raw-data-root", "/tmp", "--registry", "x.json"]
            try:
                args = module._parse_args()
            finally:
                sys.argv = saved_argv
            self.assertEqual(args.workflow, module.FULL_SESSION_MATCHED_WORKFLOW)


class ActionKernelBiasTenthCoordinateTests(unittest.TestCase):
    def test_fine_tune_clamp_preserves_a(self) -> None:
        runner = importlib.import_module(RUNNERS["action_kernel_bias"])
        from m2_dynamic_action_kernel_bias_v1.registry import load_m2_dynamic_action_kernel_bias_model_registry

        registry = load_m2_dynamic_action_kernel_bias_model_registry()
        self.assertEqual(runner.PARAMETER_NAMES[-1], "a")
        for a_value in (-3.5, 0.0, 2.2):
            values = [float(v) for v in registry.initializations[0].constrained_tensor()]
            values[9] = a_value
            clamped, adjusted = runner.clamp_fine_tune_initialization(
                values, runner.PARAMETER_NAMES,
                parameter_epsilon=registry.parameter_epsilon, lapse_max=registry.lapse_max,
            )
            self.assertEqual(clamped[9], a_value)
            self.assertNotIn("a", adjusted)

    def test_identifiability_outputs_write_both_surfaces(self) -> None:
        from m0_sensory_v1.model.m1_session import M1SessionTrials
        from m2_dynamic_action_kernel_bias_sliding_window_v1 import (
            generate_identifiability_outputs,
            identifiability_outputs_are_current,
        )
        from m2_dynamic_action_kernel_bias_v1.registry import load_m2_dynamic_action_kernel_bias_model_registry

        registry = load_m2_dynamic_action_kernel_bias_model_registry()
        session = M1SessionTrials(
            mouse_id="m", session_id="s",
            duration_s=torch.tensor([0.5, 2.0, 0.6, 2.1, 0.4, 1.9], dtype=torch.float64),
            choice_long=torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float64),
            category_long=torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64),
            feedback_valid=torch.ones(6, dtype=torch.bool), split_role="train",
        )
        raw = registry.initializations[2].raw_tensor(lapse_max=registry.lapse_max)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_identifiability_outputs((session,), raw, registry, tmp, grid_points=3)
            self.assertEqual(manifest["surface_count"], 2)
            self.assertEqual(set(manifest["files"]), {"eta_logit_by_bh", "eta_logit_by_a"})
            for name in manifest["files"].values():
                self.assertTrue((Path(tmp) / name).is_file())
            self.assertTrue(identifiability_outputs_are_current(tmp, grid_points=3))
            saved = json.loads(
                (Path(tmp) / "m2_dynamic_action_kernel_bias_identifiability_manifest.json").read_text()
            )
            self.assertEqual(saved["total_grid_evaluations"], 18)


if __name__ == "__main__":
    unittest.main()
