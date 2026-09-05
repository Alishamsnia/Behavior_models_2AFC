"""Tests for frozen per-mouse chronological split assignments."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data import (  # noqa: E402
    create_chronological_splits,
    largest_remainder_counts,
    select_session_sources,
    validate_chronological_splits,
)


class PartThreeSplitTests(unittest.TestCase):
    def test_largest_remainder_counts_and_tie_priority(self) -> None:
        self.assertEqual(
            largest_remainder_counts(42),
            {"train": 30, "validation": 6, "test": 6},
        )
        self.assertEqual(
            largest_remainder_counts(2),
            {"train": 2, "validation": 0, "test": 0},
        )
        # For D=3, validation and test have equal 0.45 remainders, so the
        # registered tie priority allocates the remaining session to test.
        self.assertEqual(
            largest_remainder_counts(3),
            {"train": 2, "validation": 0, "test": 1},
        )
        self.assertEqual(sum(largest_remainder_counts(17).values()), 17)

    def test_chronological_split_is_per_mouse_stable_and_disjoint(self) -> None:
        sessions = pd.DataFrame(
            {
                "mouse_id": ["M2", "M1", "M1", "M2", "M1", "M2"],
                "session_id": ["z", "b", "a", "x", "c", "y"],
                "session_date": [
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                ],
            }
        )
        assignments = create_chronological_splits(sessions)
        validate_chronological_splits(assignments)

        m1 = assignments.loc[assignments["mouse_id"].eq("M1")]
        self.assertEqual(m1["session_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(m1["chronological_rank"].tolist(), [1, 2, 3])
        self.assertEqual(m1["split_role"].tolist(), ["train", "train", "test"])

        keys_by_role = {
            role: set(
                assignments.loc[assignments["split_role"].eq(role), "session_id"]
            )
            for role in ("train", "validation", "test")
        }
        self.assertTrue(keys_by_role["train"].isdisjoint(keys_by_role["test"]))
        self.assertTrue(keys_by_role["validation"].isdisjoint(keys_by_role["test"]))

    def test_source_selection_blocks_locked_test_before_loading(self) -> None:
        sessions = pd.DataFrame(
            {
                "mouse_id": ["M1", "M1", "M1"],
                "session_id": ["early", "middle", "late"],
                "session_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "source_file": [
                    "/tmp/early.mat",
                    "/tmp/middle.mat",
                    "/tmp/late.mat",
                ],
            }
        )
        assignments = create_chronological_splits(sessions)
        train_sources = select_session_sources(sessions, assignments, roles=("train",))
        self.assertEqual(
            [path.name for path in train_sources],
            ["early.mat", "middle.mat"],
        )
        with self.assertRaisesRegex(ValueError, "Locked test"):
            select_session_sources(sessions, assignments, roles=("test",))

    def test_frozen_mc08_assignments_have_expected_counts_and_boundaries(self) -> None:
        split_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "splits"
            / "chronological_70_15_15_v1.csv"
        )
        assignments = pd.read_csv(split_path)
        validate_chronological_splits(assignments)
        self.assertEqual(len(assignments), 42)
        self.assertEqual(
            assignments["split_role"].value_counts().to_dict(),
            {"train": 30, "validation": 6, "test": 6},
        )
        self.assertEqual(
            assignments.loc[assignments["chronological_rank"].eq(30), "split_role"].item(),
            "train",
        )
        self.assertEqual(
            assignments.loc[assignments["chronological_rank"].eq(31), "split_role"].item(),
            "validation",
        )
        self.assertEqual(
            assignments.loc[assignments["chronological_rank"].eq(37), "split_role"].item(),
            "test",
        )
        manifest = json.loads(split_path.with_suffix(".json").read_text())
        self.assertEqual(
            sha256(split_path.read_bytes()).hexdigest(),
            manifest["assignment_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
