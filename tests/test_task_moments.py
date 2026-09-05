"""Tests for empirical truncated task-distribution moments."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data import TaskMoments, estimate_task_moments  # noqa: E402


class TaskMomentTests(unittest.TestCase):
    def test_estimate_task_moments_uses_population_variance_and_checks_support(self) -> None:
        trials = pd.DataFrame(
            {
                "category_long": [0, 0, 0, 1, 1],
                "duration_s": [0.2, 0.4, float("nan"), 1.7, 2.3],
                # These behavioral fields must not influence the moments.
                "choice_long": [0, 1, pd.NA, 0, 1],
                "outcome": ["Reward", "Punish", "DidNotChoose", "Reward", "Punish"],
            }
        )
        moments = estimate_task_moments(trials)
        self.assertAlmostEqual(moments.mu_short_s, 0.3)
        self.assertAlmostEqual(moments.mu_long_s, 2.0)
        self.assertAlmostEqual(moments.var_short_s2, 0.01)
        self.assertAlmostEqual(moments.var_long_s2, 0.09)
        self.assertEqual((moments.n_short, moments.n_long), (2, 2))

        overlapping = trials.copy()
        overlapping.loc[1, "duration_s"] = 1.8
        with self.assertRaisesRegex(ValueError, "supports overlap"):
            estimate_task_moments(overlapping)

        with self.assertRaisesRegex(ValueError, "must not overlap"):
            TaskMoments(
                mu_short_s=0.5,
                mu_long_s=1.0,
                var_short_s2=0.1,
                var_long_s2=0.1,
                n_short=2,
                n_long=2,
                min_short_s=0.2,
                max_short_s=1.1,
                min_long_s=0.9,
                max_long_s=1.2,
            )

    def test_registered_real_moments_have_positive_nonoverlapping_support(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "m0_sensory_v1.json"
        registered = json.loads(config_path.read_text(encoding="utf-8"))["task_moments"]
        self.assertGreater(registered["min_short_s"], 0.0)
        self.assertGreater(registered["min_long_s"], 0.0)
        self.assertLess(registered["max_short_s"], registered["min_long_s"])
        self.assertEqual(registered["n_short"] + registered["n_long"], 13_720)
        self.assertEqual(registered["variance_ddof"], 0)


if __name__ == "__main__":
    unittest.main()
