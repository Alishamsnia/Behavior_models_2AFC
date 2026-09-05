"""Tests for scripts/report_seven_model_coverage.py: the disk-inspection
inventory of what result coverage exists for each of the seven models."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import report_seven_model_coverage as report  # noqa: E402


class SevenModelCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sliding_root = Path(self._tmp.name) / "sliding"
        self.single_root = Path(self._tmp.name) / "single"
        self._patches = [
            patch.object(report, "SLIDING_ROOT", self.sliding_root),
            patch.object(report, "SINGLE_ROOT", self.single_root),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _make_window(self, model_dir: str, mouse: str, window_name: str, status: str | None) -> None:
        window_dir = self.sliding_root / model_dir / mouse / window_name
        window_dir.mkdir(parents=True)
        if status is not None:
            (window_dir / "window_manifest.json").write_text(
                json.dumps({"status": status}), encoding="utf-8"
            )

    def test_model_with_no_directories_reports_no_results(self) -> None:
        coverage = report.collect_coverage()
        self.assertFalse(coverage["M2_dynamic"]["has_any_results"])
        self.assertEqual(coverage["M2_dynamic"]["sliding_window"], {})
        self.assertEqual(coverage["M2_dynamic"]["single_session"], {})

    def test_counts_complete_vs_total_separately(self) -> None:
        self._make_window("M1_dynamic_block_length_v1", "MC06", "window_001", "complete")
        self._make_window("M1_dynamic_block_length_v1", "MC06", "window_002", "resume_stale_registry")
        self._make_window("M1_dynamic_block_length_v1", "MC06", "window_003", None)

        coverage = report.collect_coverage()
        m1_dynamic = coverage["M1_dynamic"]
        self.assertTrue(m1_dynamic["has_any_results"])
        self.assertEqual(m1_dynamic["sliding_window"]["MC06"], {"count": 3, "complete": 1})
        self.assertEqual(m1_dynamic["single_session"], {})

    def test_sliding_window_only_model_has_no_single_session_entry(self) -> None:
        self._make_window("M1_dynamic_block_length_v1", "MC06", "window_001", "complete")
        coverage = report.collect_coverage()
        self.assertEqual(coverage["M1_dynamic"]["single_session"], {})


if __name__ == "__main__":
    unittest.main()
