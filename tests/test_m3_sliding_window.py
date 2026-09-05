"""Contracts for the isolated M3 chronological sliding experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3_hmm_delta_v1.analysis import build_m3_trial_predictions  # noqa: E402
from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402
from m3_sliding_window_v1 import (  # noqa: E402
    aggregate_m3_session_initializations,
    build_sliding_windows,
    figure_ready,
)
from scripts.run_m3_sliding_window import (  # noqa: E402
    CPU_BUDGET,
    _continuous_single_session_predictions,
    _parse_args,
)


def _assignments(count: int = 16) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mouse_id": ["MCXX"] * count,
            "session_id": [f"session_{rank:02d}" for rank in range(1, count + 1)],
            "chronological_rank": range(1, count + 1),
        }
    )


def _canonical_session() -> pd.DataFrame:
    count = 8
    return pd.DataFrame(
        {
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
        }
    )


class M3SlidingWindowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m3_model_registry(
            ROOT / "config" / "m3_hmm_delta_v1_mc06.json"
        )

    def test_registry_retains_exactly_eight_m3_initializations(self) -> None:
        self.assertEqual(len(self.registry.initializations), 8)
        self.assertEqual(
            [item.start_id for item in self.registry.initializations],
            list(range(1, 9)),
        )

    def test_windows_remain_exact_9_3_1_with_step_one(self) -> None:
        windows = build_sliding_windows(_assignments())
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0].training_ranks, tuple(range(1, 10)))
        self.assertEqual(windows[0].validation_ranks, (10, 11, 12))
        self.assertEqual(windows[0].test_rank, 13)
        self.assertEqual(windows[-1].test_rank, 16)

    def test_dynamic_parameters_are_averaged_on_interpretable_scales(self) -> None:
        assignments = _assignments()
        windows = build_sliding_windows(assignments)
        hazards = (0.10, 0.20, 0.30, 0.40)
        theta_long = (0.60, 0.70, 0.80, 0.90)
        learning_rates = (0.01, 0.03, 0.05, 0.07)
        kappas = (0.5, 1.0, 2.0, 4.0)
        winners = {
            window.window_id: [
                float(window.window_id),
                0.2,
                0.1,
                0.5,
                kappas[window.window_id - 1],
                math.log(hazards[window.window_id - 1] / (1.0 - hazards[window.window_id - 1])),
                math.log(theta_long[window.window_id - 1] / (1.0 - theta_long[window.window_id - 1])),
                learning_rates[window.window_id - 1],
            ]
            for window in windows
        }
        result = aggregate_m3_session_initializations(
            assignments, windows, winners
        ).set_index("chronological_rank")
        expected_hazard = float(np.mean(hazards))
        expected_theta_long = float(np.mean(theta_long))
        expected_kappa = float(np.exp(np.mean(np.log(kappas))))
        self.assertAlmostEqual(result.loc[4, "initial_kappa"], expected_kappa)
        self.assertAlmostEqual(result.loc[4, "initial_hazard"], expected_hazard)
        self.assertAlmostEqual(result.loc[4, "initial_theta_L"], expected_theta_long)
        self.assertAlmostEqual(
            result.loc[4, "initial_ah"],
            math.log(expected_hazard / (1.0 - expected_hazard)),
        )
        self.assertAlmostEqual(
            result.loc[4, "initial_delta0"],
            math.log(expected_theta_long / (1.0 - expected_theta_long)),
        )
        self.assertAlmostEqual(
            result.loc[4, "initial_alpha_theta"], np.mean(learning_rates)
        )
        self.assertEqual(
            result.loc[16, "initialization_policy"],
            "final_window_winner_fallback_no_training_membership",
        )

    def test_single_session_test_continues_belief_and_learned_delta(self) -> None:
        frame = _canonical_session()
        prefix = frame.loc[frame["experiment_partition"].eq("prefix")].copy()
        heldout = frame.loc[frame["experiment_partition"].eq("heldout")].copy()
        initialization = self.registry.initializations[0]
        raw = initialization.raw_tensor(
            lapse_max=self.registry.lapse_max,
            epsilon=self.registry.parameter_epsilon,
        ).tolist()
        observed = _continuous_single_session_predictions(
            prefix, heldout, raw, self.registry
        )
        direct_frame = frame.copy()
        direct_frame["split_role"] = "train"
        direct = build_m3_trial_predictions(
            direct_frame, raw, self.registry
        ).sort_values("trial_index")
        for column in (
            "model_prior_long",
            "model_delta_pre",
            "model_delta_post",
            "model_pre_choice_probability_N",
            "model_pre_choice_probability_S",
            "model_pre_choice_probability_L",
        ):
            np.testing.assert_allclose(observed[column], direct[column])
        self.assertEqual(
            observed["split_role"].tolist(), ["train"] * 4 + ["test"] * 4
        )
        first_test = observed.loc[observed["split_role"].eq("test")].iloc[0]
        self.assertNotAlmostEqual(float(first_test["model_prior_long"]), 0.5)
        self.assertNotAlmostEqual(
            float(first_test["model_delta_pre"]), initialization.delta0
        )

    def test_figure_adapter_derives_m3_behavioral_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "category_long": [0.0, 1.0],
                "true_block": ["S", "N"],
                "duration_s": [0.5, 2.0],
                "choice_long": [0.0, 1.0],
                "model_probability_long": [0.1, 0.9],
                "mouse_correct": [1.0, 1.0],
                "model_probability_correct": [0.9, 0.9],
            }
        )
        ready = figure_ready(frame)
        self.assertEqual(ready["trial_type"].tolist(), ["short", "long"])
        self.assertEqual(ready["block_type"].tolist(), ["short_block", "neutral"])

    def test_default_worker_allocations_each_use_32_cpu_threads(self) -> None:
        argv = [
            "run_m3_sliding_window.py",
            "--raw-data-root", "raw",
            "--registry", "registry.json",
        ]
        with patch.object(sys, "argv", argv):
            args = _parse_args()
        self.assertEqual(CPU_BUDGET, 32)
        self.assertEqual(
            min(args.rolling_workers, 8) * args.rolling_threads_per_worker,
            CPU_BUDGET,
        )
        self.assertEqual(
            args.single_session_workers * args.single_session_threads_per_worker,
            CPU_BUDGET,
        )


if __name__ == "__main__":
    unittest.main()
