"""Tests for scripts/_resume_staleness.py: the shared --resume freshness
check used by every sliding-window/single-session runner script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _resume_staleness import (  # noqa: E402
    registry_config_sha256,
    result_registry_is_stale,
    saved_result_registry_checksum,
)


@dataclass(frozen=True)
class _FakeRegistry:
    config_path: Path


class ResumeStalenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self._tmp.name) / "registry.json"
        self.config_path.write_text('{"a": 1}', encoding="utf-8")
        self.registry = _FakeRegistry(config_path=self.config_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_checksum_changes_when_config_file_content_changes(self) -> None:
        original = registry_config_sha256(self.registry)
        self.config_path.write_text('{"a": 2}', encoding="utf-8")
        changed = registry_config_sha256(self.registry)
        self.assertNotEqual(original, changed)

    def test_saved_checksum_reads_multi_start_schema(self) -> None:
        result = {"runs": [{"registry_config_sha256": "abc"}, {"registry_config_sha256": "abc"}]}
        self.assertEqual(saved_result_registry_checksum(result), "abc")

    def test_saved_checksum_reads_single_run_schema(self) -> None:
        result = {"registry_config_sha256": "xyz"}
        self.assertEqual(saved_result_registry_checksum(result), "xyz")

    def test_missing_checksum_field_returns_none_not_stale(self) -> None:
        result = {"runs": [{"start_id": 0}]}
        self.assertIsNone(saved_result_registry_checksum(result))
        self.assertFalse(result_registry_is_stale(result, self.registry))

    def test_result_is_stale_only_when_checksum_actually_differs(self) -> None:
        current_checksum = registry_config_sha256(self.registry)
        fresh_result = {"registry_config_sha256": current_checksum}
        self.assertFalse(result_registry_is_stale(fresh_result, self.registry))

        self.config_path.write_text('{"a": 999}', encoding="utf-8")
        self.assertTrue(result_registry_is_stale(fresh_result, self.registry))

    def test_none_result_is_never_stale(self) -> None:
        self.assertFalse(result_registry_is_stale(None, self.registry))


if __name__ == "__main__":
    unittest.main()
