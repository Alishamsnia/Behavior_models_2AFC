"""Contracts for the isolated M1 chronological sliding experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m0_sensory_v1.m1_analysis import build_m1_trial_predictions  # noqa: E402
from m0_sensory_v1.m1_registry import load_m1_model_registry  # noqa: E402
from m1_sliding_window_v1 import (  # noqa: E402
    aggregate_m1_session_initializations,
    build_sliding_windows,
)
from scripts.run_m1_sliding_window import (  # noqa: E402
    _continuous_single_session_predictions,
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


class M1SlidingWindowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_m1_model_registry(
            ROOT / "config" / "m1_hmm_true_v1_mc06.json"
        )

    def test_registry_retains_exactly_eight_m1_initializations(self) -> None:
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

    def test_hazard_is_averaged_on_probability_scale_then_logit_transformed(self) -> None:
        assignments = _assignments()
        windows = build_sliding_windows(assignments)
        hazards = (0.10, 0.20, 0.30, 0.40)
        kappas = (0.5, 1.0, 2.0, 4.0)
        winners = {
            window.window_id: [
                float(window.window_id),
                0.2,
                0.1,
                0.5,
                kappas[window.window_id - 1],
                math.log(hazards[window.window_id - 1] / (1.0 - hazards[window.window_id - 1])),
            ]
            for window in windows
        }
        result = aggregate_m1_session_initializations(
            assignments, windows, winners
        ).set_index("chronological_rank")
        expected_hazard = float(np.mean(hazards))
        expected_kappa = float(np.exp(np.mean(np.log(kappas))))
        self.assertAlmostEqual(result.loc[4, "initial_kappa"], expected_kappa)
        self.assertAlmostEqual(result.loc[4, "initial_hazard"], expected_hazard)
        self.assertAlmostEqual(
            result.loc[4, "initial_ah"],
            math.log(expected_hazard / (1.0 - expected_hazard)),
        )
        self.assertEqual(
            result.loc[16, "initialization_policy"],
            "final_window_winner_fallback_no_training_membership",
        )
        self.assertAlmostEqual(result.loc[16, "initial_hazard"], hazards[-1])

    def test_single_session_test_belief_continues_from_prefix(self) -> None:
        frame = _canonical_session()
        prefix = frame.loc[frame["experiment_partition"].eq("prefix")].copy()
        heldout = frame.loc[frame["experiment_partition"].eq("heldout")].copy()
        raw = self.registry.initializations[0].raw_tensor(
            lapse_max=self.registry.lapse_max,
            epsilon=self.registry.parameter_epsilon,
        ).tolist()
        observed = _continuous_single_session_predictions(
            prefix, heldout, raw, self.registry
        )
        direct_frame = frame.copy()
        direct_frame["split_role"] = "train"
        direct = build_m1_trial_predictions(
            direct_frame, raw, self.registry
        ).sort_values("trial_index")
        np.testing.assert_allclose(
            observed["model_prior_long"], direct["model_prior_long"]
        )
        self.assertEqual(
            observed["split_role"].tolist(), ["train"] * 4 + ["test"] * 4
        )
        first_test_prior = float(
            observed.loc[observed["split_role"].eq("test"), "model_prior_long"].iloc[0]
        )
        self.assertNotAlmostEqual(first_test_prior, 0.5)


if __name__ == "__main__":
    unittest.main()
