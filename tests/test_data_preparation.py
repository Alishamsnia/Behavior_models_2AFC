"""Unit tests for the M0 data-preparation layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data.preparation import (  # noqa: E402
    DataPreparationResult,
    SessionFileMetadata,
    has_required_isi_bins,
    parse_behavior_file_path,
    prepare_session,
    prepare_dataset,
    select_legacy_model_trials,
    states_labeling,
)
from m0_sensory_v1.data.validation import validate_prepared_data  # noqa: E402


def _state(outcome: str, time: float = 1.0) -> dict[str, list[float]]:
    return {outcome: [time, time + 0.1]}


def _synthetic_session(contingency: int = 1) -> dict:
    outcomes = ["Reward", "Punish", "DidNotChoose", "RewardNaive"]
    trial_types = [1, 1, 2, 2]
    block_types = [0, 1, 1, 2]
    return {
        "nTrials": 4,
        "TrialSettings": [
            {
                "GUI": {
                    "Contingency": contingency,
                    "NumNaiveWarmup": 1,
                }
            }
        ],
        "TrialTypes": trial_types,
        "BlockTypes": block_types,
        "ProcessedSessionData": [
            {"trial_isi": {"PostMeanISI": value}}
            for value in (0.2, 0.5, 2.0, 2.3)
        ],
        "JitterFlag": [0, 1, 0, 0],
        "OptoType": [0, 0, 1, 0],
        "RawEvents": {
            "Trial": [
                {"States": _state(outcome), "Events": {}}
                for outcome in outcomes
            ]
        },
    }


class DataPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = SessionFileMetadata(
            mouse_id="MC08",
            session_id="synthetic_session",
            task_version="V_1",
            session_date=datetime(2026, 1, 1, 12, 0, 0),
        )

    def test_parse_behavior_file_path(self) -> None:
        parsed = parse_behavior_file_path(
            "MC08_SChR2_block_single_interval_discrimination_V_1_20250807_161922.mat"
        )
        self.assertEqual(parsed.mouse_id, "MC08")
        self.assertEqual(parsed.task_version, "V_1")
        self.assertEqual(parsed.session_date, datetime(2025, 8, 7, 16, 19, 22))

    def test_states_labeling_preserves_priority(self) -> None:
        outcome, timestamp = states_labeling(
            {"Reward": [2.0, 2.2], "Punish": [1.0, 1.2]}
        )
        self.assertEqual(outcome, "Punish")
        self.assertEqual(timestamp, 1.0)

    def test_prepare_session_reconstructs_legacy_choices_and_keeps_omission(self) -> None:
        canonical = prepare_session(
            _synthetic_session(),
            self.metadata,
            source_file="synthetic.mat",
        )
        self.assertEqual(len(canonical), 4)
        self.assertEqual(canonical["omission"].astype(int).tolist(), [0, 0, 1, 0])
        self.assertEqual(
            canonical["choice_long"].tolist(),
            [0, 1, pd.NA, 1],
        )

        model = select_legacy_model_trials(canonical)
        self.assertEqual(len(model), 3)
        self.assertEqual(model["source_trial_index"].astype(int).tolist(), [0, 1, 3])
        self.assertEqual(model["choice_long"].astype(int).tolist(), [0, 1, 1])

        canonical["session_index"] = pd.array([1] * len(canonical), dtype="Int64")
        model["session_index"] = pd.array([1] * len(model), dtype="Int64")
        sessions = pd.DataFrame([{"session_id": "synthetic_session"}])
        validate_prepared_data(DataPreparationResult(canonical, model, sessions))

    def test_reverse_contingency_swaps_category_and_block_codes(self) -> None:
        canonical = prepare_session(
            _synthetic_session(contingency=2),
            self.metadata,
            source_file="synthetic.mat",
        )
        self.assertEqual(canonical["category_long"].astype("Int64").tolist(), [1, 1, 0, 0])
        self.assertEqual(canonical["true_block"].tolist(), ["N", "L", "L", "S"])

    def test_has_required_isi_bins_requires_exact_counts(self) -> None:
        frame = pd.DataFrame(
            {
                "category_long": [0] * 5 + [1] * 5,
                "duration_s": [0.2, 0.35, 0.5, 0.65, 0.8, 1.7, 1.85, 2.0, 2.15, 2.3],
            }
        )
        passed, short_levels, long_levels = has_required_isi_bins(frame)
        self.assertTrue(passed)
        self.assertEqual(len(short_levels), 5)
        self.assertEqual(len(long_levels), 5)

        extra = pd.concat(
            [frame, pd.DataFrame({"category_long": [0], "duration_s": [0.95]})],
            ignore_index=True,
        )
        self.assertFalse(has_required_isi_bins(extra)[0])

    def test_prepare_dataset_sorts_sessions_chronologically_per_mouse(self) -> None:
        paths = [
            Path("MC08_task_block_V_1_20260103_120000.mat"),
            Path("MC09_task_block_V_1_20260102_120000.mat"),
            Path("MC08_task_block_V_1_20260101_120000.mat"),
        ]
        with (
            patch(
                "m0_sensory_v1.data.preparation.load_session_mat",
                side_effect=lambda _: _synthetic_session(),
            ),
            patch(
                "m0_sensory_v1.data.preparation._sha256_file",
                side_effect=lambda path: f"checksum-{path.stem}",
            ),
        ):
            result = prepare_dataset(paths, require_legacy_isi_bins=False)

        mc08_sessions = result.sessions.loc[result.sessions["mouse_id"].eq("MC08")]
        self.assertEqual(mc08_sessions["session_index"].tolist(), [1, 2])
        self.assertEqual(
            mc08_sessions["session_date"].dt.day.tolist(),
            [1, 3],
        )
        self.assertEqual(len(result.canonical_trials), 12)
        self.assertEqual(len(result.model_trials), 9)
        validate_prepared_data(result)


if __name__ == "__main__":
    unittest.main()
