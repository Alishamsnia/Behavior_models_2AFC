"""Contracts for the isolated M4 chronological sliding experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m4_hsmm_delta_v1.analysis import build_m4_trial_predictions  # noqa: E402
from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402
from m4_sliding_window_v1 import (  # noqa: E402
    aggregate_m4_session_initializations,
    build_sliding_windows,
)
from scripts.run_m4_sliding_window import (  # noqa: E402
    CPU_BUDGET,
    _continuous_single_session_predictions,
    _generate_single_global_figures,
    _is_duration_tail_tolerance_error,
    _manifest_eligible_for_global_analysis,
    _parse_args,
    _single_session_ineligibility_reasons,
)


def _assignments(count: int = 16) -> pd.DataFrame:
    return pd.DataFrame({
        "mouse_id": ["MCXX"] * count,
        "session_id": [f"session_{rank:02d}" for rank in range(1, count + 1)],
        "chronological_rank": range(1, count + 1),
    })


def _canonical_session() -> pd.DataFrame:
    count = 8
    return pd.DataFrame({
        "mouse_id": ["mouse"] * count,
        "session_id": ["session"] * count,
        "session_date": pd.to_datetime(["2026-01-01"] * count),
        "trial_index": range(1, count + 1),
        "duration_s": [2.0, 2.15, 1.85, 2.0, 0.5, 2.3, 0.35, 1.7],
        "choice_long": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "category_long": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "feedback_valid": [True] * count,
        "true_block": ["N", "N", "N", "L", "L", "L", "S", "S"],
        "outcome": ["Reward"] * count,
        "experiment_partition": ["prefix"] * 4 + ["heldout"] * 4,
        "split_role": ["train"] * 4 + ["test"] * 4,
    })


class M4SlidingWindowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m4_model_registry(
            ROOT / "config" / "m4_hsmm_delta_v1_mc06.json"
        )

    def test_registry_retains_exactly_eight_m4_initializations(self) -> None:
        self.assertEqual(len(self.registry.initializations), 8)
        self.assertEqual([item.start_id for item in self.registry.initializations], list(range(1, 9)))

    def test_single_session_acceptance_uses_all_registered_m4_gates(self) -> None:
        accepted = {
            "converged": True,
            "projection_acceptable": True,
            "tail_occupancy_acceptable": True,
        }
        self.assertEqual(_single_session_ineligibility_reasons(accepted), [])
        rejected = {
            "converged": False,
            "projection_acceptable": True,
            "tail_occupancy_acceptable": False,
        }
        self.assertEqual(
            _single_session_ineligibility_reasons(rejected),
            ["not_converged", "tail_occupancy_unacceptable"],
        )

    def test_duration_tail_failure_is_the_only_recoverable_runtime_error(self) -> None:
        observed = RuntimeError(
            "Duration distribution did not reach the requested tail tolerance"
        )
        self.assertTrue(_is_duration_tail_tolerance_error(observed))
        self.assertFalse(_is_duration_tail_tolerance_error(RuntimeError("other")))

    def test_legacy_complete_manifest_eligibility_is_inferred(self) -> None:
        manifest = {
            "status": "complete",
            "converged": True,
            "projection_acceptable": True,
            "tail_occupancy_acceptable": True,
        }
        self.assertTrue(_manifest_eligible_for_global_analysis(manifest))
        manifest["tail_occupancy_acceptable"] = False
        self.assertFalse(_manifest_eligible_for_global_analysis(manifest))

    def test_global_pool_excludes_ineligible_single_sessions(self) -> None:
        initializations = pd.DataFrame(
            {
                "session_id": ["eligible", "ineligible"],
                "chronological_rank": [13, 14],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eligible = root / "session_013_eligible"
            ineligible = root / "session_014_ineligible"
            eligible.mkdir()
            ineligible.mkdir()
            (eligible / "trial_predictions.csv").write_text("x\n1\n")
            (ineligible / "trial_predictions.csv").write_text("x\n2\n")
            (eligible / "single_session_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "converged": True,
                        "projection_acceptable": True,
                        "tail_occupancy_acceptable": True,
                    }
                )
            )
            (ineligible / "single_session_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "ineligible",
                        "eligible_for_global_analysis": False,
                        "ineligibility_reasons": ["tail_occupancy_unacceptable"],
                    }
                )
            )
            with patch(
                "scripts.run_m4_sliding_window.generate_global_prediction_figures",
                return_value={"source_count": 1},
            ) as generate:
                _generate_single_global_figures(initializations, root)
            sources = generate.call_args.args[0]
            self.assertEqual([source_id for source_id, _ in sources], ["session_013_eligible"])
            selection = json.loads(
                (root / "single_session_global_selection.json").read_text()
            )
            self.assertEqual(selection["eligible_source_count"], 1)
            self.assertEqual(selection["excluded_source_count"], 1)

    def test_windows_remain_exact_9_3_1_with_step_one(self) -> None:
        windows = build_sliding_windows(_assignments())
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0].training_ranks, tuple(range(1, 10)))
        self.assertEqual(windows[0].validation_ranks, (10, 11, 12))
        self.assertEqual(windows[0].test_rank, 13)
        self.assertEqual(windows[-1].test_rank, 16)

    def test_m4_dynamic_coordinates_use_registered_aggregation_scales(self) -> None:
        assignments = _assignments()
        windows = build_sliding_windows(assignments)
        hazards = (0.10, 0.20, 0.30, 0.40)
        theta_long = (0.60, 0.70, 0.80, 0.90)
        alphas = (0.01, 0.03, 0.05, 0.07)
        slopes = (0.02, 0.04, 0.06, 0.08)
        kappas = (0.5, 1.0, 2.0, 4.0)
        winners = {
            window.window_id: [
                float(window.window_id), 0.2, 0.1, 0.5,
                kappas[window.window_id - 1],
                math.log(hazards[window.window_id - 1] / (1.0 - hazards[window.window_id - 1])),
                math.log(theta_long[window.window_id - 1] / (1.0 - theta_long[window.window_id - 1])),
                alphas[window.window_id - 1], slopes[window.window_id - 1],
            ]
            for window in windows
        }
        result = aggregate_m4_session_initializations(
            assignments, windows, winners
        ).set_index("chronological_rank")
        expected_hazard = float(np.mean(hazards))
        expected_theta = float(np.mean(theta_long))
        expected_kappa = float(np.exp(np.mean(np.log(kappas))))
        self.assertAlmostEqual(result.loc[4, "initial_kappa"], expected_kappa)
        self.assertAlmostEqual(result.loc[4, "initial_biased_hazard_at_center"], expected_hazard)
        self.assertAlmostEqual(result.loc[4, "initial_ah"], math.log(expected_hazard / (1 - expected_hazard)))
        self.assertAlmostEqual(result.loc[4, "initial_delta0"], math.log(expected_theta / (1 - expected_theta)))
        self.assertAlmostEqual(result.loc[4, "initial_alpha_theta"], np.mean(alphas))
        self.assertAlmostEqual(result.loc[4, "initial_bh"], np.mean(slopes))
        self.assertGreater(result.loc[4, "initial_bh"], 0.0)
        self.assertEqual(
            result.loc[16, "initialization_policy"],
            "final_window_winner_fallback_no_training_membership",
        )

    def test_single_session_test_continues_joint_hsmm_state_and_delta(self) -> None:
        frame = _canonical_session()
        prefix = frame.loc[frame["experiment_partition"].eq("prefix")].copy()
        heldout = frame.loc[frame["experiment_partition"].eq("heldout")].copy()
        initialization = self.registry.initializations[0]
        raw = initialization.raw_tensor(
            lapse_max=self.registry.lapse_max,
            epsilon=self.registry.parameter_epsilon,
        ).tolist()
        observed = _continuous_single_session_predictions(prefix, heldout, raw, self.registry)
        direct_frame = frame.copy()
        direct_frame["split_role"] = "train"
        direct = build_m4_trial_predictions(direct_frame, raw, self.registry).sort_values("trial_index")
        for column in (
            "model_prior_long", "model_delta_pre", "model_delta_post",
            "model_expected_capped_run_length_pre", "model_expected_hazard_pre",
            "model_tail_probability_pre", "model_pre_choice_probability_N",
            "model_pre_choice_probability_S", "model_pre_choice_probability_L",
        ):
            np.testing.assert_allclose(observed[column], direct[column])
        self.assertEqual(observed["split_role"].tolist(), ["train"] * 4 + ["test"] * 4)
        first_test = observed.loc[observed["split_role"].eq("test")].iloc[0]
        self.assertNotAlmostEqual(float(first_test["model_prior_long"]), 0.5)
        self.assertNotAlmostEqual(float(first_test["model_delta_pre"]), initialization.delta0)
        self.assertGreater(float(first_test["model_expected_capped_run_length_pre"]), 1.0)

    def test_full_run_length_export_remains_opt_in(self) -> None:
        frame = _canonical_session().copy()
        frame["split_role"] = "train"
        raw = self.registry.initializations[0].raw_tensor(
            lapse_max=self.registry.lapse_max,
            epsilon=self.registry.parameter_epsilon,
        ).tolist()
        compact = build_m4_trial_predictions(frame, raw, self.registry)
        full = build_m4_trial_predictions(
            frame, raw, self.registry, include_full_run_length_distributions=True
        )
        marginal = "model_pre_choice_run_probability_r65"
        self.assertNotIn(marginal, compact)
        self.assertIn(marginal, full)
        self.assertEqual(len(full.columns) - len(compact.columns), 130)

    def test_default_worker_allocations_each_use_32_cpu_threads(self) -> None:
        argv = ["run_m4_sliding_window.py", "--raw-data-root", "raw", "--registry", "registry.json"]
        with patch.object(sys, "argv", argv):
            args = _parse_args()
        self.assertEqual(CPU_BUDGET, 32)
        self.assertEqual(min(args.rolling_workers, 8) * args.rolling_threads_per_worker, CPU_BUDGET)
        self.assertEqual(args.single_session_workers * args.single_session_threads_per_worker, CPU_BUDGET)
        self.assertFalse(args.include_full_run_length_distributions)
        self.assertFalse(args.truncation_sensitivity)
        self.assertEqual(args.identifiability_grid_points, 0)


if __name__ == "__main__":
    unittest.main()
