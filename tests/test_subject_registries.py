"""Integrity tests for the notebook-derived MC06 and MC07 registries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class SubjectRegistryTests(unittest.TestCase):
    def test_notebook_subject_registries_and_source_lists_are_consistent(self) -> None:
        expected = {
            "MC06": (45, 44, 31, 6, 7),
            "MC07": (43, 42, 30, 6, 6),
        }
        for mouse, counts in expected.items():
            low = mouse.lower()
            registry = load_model_registry(
                REPOSITORY_ROOT / "config" / f"m0_sensory_v1_{low}.json"
            )
            source = json.loads(
                (
                    REPOSITORY_ROOT
                    / "config"
                    / "subjects"
                    / f"{low}_model_v24_commented_sessions.json"
                ).read_text()
            )
            listed, included, train, validation, test = counts
            self.assertEqual(source["mouse_id"], mouse)
            self.assertEqual(source["listed_session_count"], listed)
            self.assertEqual(len(source["session_ids"]), listed)
            self.assertEqual(source["included_session_count"], included)
            self.assertEqual(len(source["excluded_by_legacy_preparation"]), 1)
            assignments = registry.split.assignment_path.read_text()
            self.assertEqual(assignments.count("\n") - 1, included)
            self.assertEqual(assignments.count(",train,"), train)
            self.assertEqual(assignments.count(",validation,"), validation)
            self.assertEqual(assignments.count(",test,"), test)
            self.assertTrue(registry.split.locked_test)
            self.assertEqual(registry.lapse_max, 1.0)
            self.assertEqual(registry.task_moments.min_short_s, 0.2)
            self.assertEqual(registry.task_moments.max_short_s, 0.8)
            self.assertEqual(registry.task_moments.min_long_s, 1.7)
            self.assertEqual(registry.task_moments.max_long_s, 2.3)


if __name__ == "__main__":
    unittest.main()
