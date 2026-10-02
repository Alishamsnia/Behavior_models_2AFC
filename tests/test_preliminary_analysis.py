"""Tests for trial exports and notebook-compatible preliminary summaries."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.analysis import (  # noqa: E402
    build_trial_predictions,
    event_aligned_performance,
    four_bin_psychometric_summary,
    plot_psychometrics,
)
from m0_sensory_v1.pipeline import prepare_locked_test_analysis_trials  # noqa: E402
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def _canonical(
    role: str = "train",
    *,
    session_id: str = "synthetic_session",
    neutral_only: bool = False,
) -> pd.DataFrame:
    rows = []
    durations = np.r_[np.linspace(0.2, 0.8, 16), np.linspace(1.7, 2.3, 16)]
    categories = np.r_[np.zeros(16), np.ones(16)]
    blocks = (
        ["N"] * 32
        if neutral_only
        else ["S"] * 8 + ["L"] * 8 + ["L"] * 8 + ["S"] * 8
    )
    for index, (duration, category, block) in enumerate(
        zip(durations, categories, blocks), start=1
    ):
        rows.append(
            {
                "mouse_id": "synthetic",
                "session_id": session_id,
                "session_date": "2026-01-01",
                "trial_index": index,
                "duration_s": duration,
                "choice_long": category,
                "category_long": category,
                "true_block": block,
                "outcome": "Reward",
                "split_role": role,
            }
        )
    rows[3]["choice_long"] = np.nan
    rows[3]["outcome"] = "DidNotChoose"
    return pd.DataFrame(rows)


class PreliminaryAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor(
            lapse_max=cls.registry.lapse_max,
            epsilon=cls.registry.parameter_epsilon,
        ).tolist()

    def test_trial_export_retains_omission_and_skips_its_model_fields(self) -> None:
        predictions = build_trial_predictions(_canonical(), self.raw, self.registry)
        self.assertEqual(len(predictions), 32)
        omitted = predictions.loc[predictions["outcome"].eq("DidNotChoose")].iloc[0]
        self.assertFalse(omitted["model_evaluated"])
        for name in ("model_decision_lower_boundary_s", "model_decision_upper_boundary_s"):
            self.assertTrue(np.isnan(omitted[name]))
        self.assertEqual(omitted["model_decision_regime"], "not_evaluated_invalid_choice")
        np.testing.assert_allclose(
            predictions["model_positive_threshold_s"],
            predictions["model_decision_upper_boundary_s"], equal_nan=True,
        )
        self.assertTrue(np.isnan(omitted["model_probability_long"]))
        self.assertEqual(set(predictions["trial_type"]), {"short", "long"})

    def test_four_bins_are_created_separately_for_both_trial_types(self) -> None:
        predictions = build_trial_predictions(_canonical(), self.raw, self.registry)
        summary = four_bin_psychometric_summary(predictions)
        self.assertEqual(summary["bin_order"].tolist(), list(range(8)))
        self.assertEqual(summary.loc[summary["bin_order"] < 4, "trial_type"].unique().tolist(), ["short"])
        self.assertEqual(summary.loc[summary["bin_order"] >= 4, "trial_type"].unique().tolist(), ["long"])

    def test_event_alignment_finds_rare_and_directional_transition_events(self) -> None:
        predictions = build_trial_predictions(_canonical(), self.raw, self.registry)
        rare = event_aligned_performance(
            predictions, "rare_short_in_long", window=1
        )
        transition = event_aligned_performance(
            predictions, "short_to_long", window=1
        )
        self.assertIn(0, rare["relative_trial"].tolist())
        self.assertIn(0, transition["relative_trial"].tolist())
        self.assertGreater(int(rare.loc[rare["relative_trial"].eq(0), "n_events"].iloc[0]), 0)

    def test_psychometric_grid_adds_validation_row_and_neutral_column(self) -> None:
        train = build_trial_predictions(
            _canonical("train", session_id="train_session"),
            self.raw,
            self.registry,
        )
        validation = build_trial_predictions(
            _canonical(
                "validation",
                session_id="validation_session",
                neutral_only=True,
            ),
            self.raw,
            self.registry,
        )
        predictions = pd.concat([train, validation], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            outputs = plot_psychometrics(predictions, Path(directory))
            summary = pd.read_csv(Path(directory) / "psychometric_summary.csv")
            self.assertEqual(
                set(outputs),
                {"psychometrics_mouse_vs_model.png", "psychometrics_mouse_vs_model.pdf"},
            )
        neutral_validation = summary.loc[
            summary["data_subset"].eq("validation")
            & summary["block_subset"].eq("neutral")
        ]
        self.assertEqual(neutral_validation["bin_order"].tolist(), list(range(8)))

    def test_test_prediction_and_loading_both_require_explicit_unlock(self) -> None:
        with self.assertRaises(PermissionError):
            build_trial_predictions(_canonical("test"), self.raw, self.registry)
        with self.assertRaises(PermissionError):
            prepare_locked_test_analysis_trials("/unused", self.registry)


if __name__ == "__main__":
    unittest.main()
