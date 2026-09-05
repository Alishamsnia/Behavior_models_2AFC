"""Tests for scripts/_resume_staleness.py: the shared --resume freshness
check used by every sliding-window/single-session runner script.

Two independent checks are tested: `result_registry_is_stale` (registry
config/initialization-table/split-assignment identity, retroactively
verifiable against every real saved result) and `manifest_fingerprint`/
`manifest_fingerprint_is_stale` (a newer source-code-tree check that can
only be compared once a manifest has recorded one)."""

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
    current_registry_identity,
    manifest_fingerprint,
    manifest_fingerprint_for_write,
    manifest_fingerprint_is_stale,
    registry_config_sha256,
    result_registry_is_stale,
    saved_registry_identity,
    source_code_sha256,
)


@dataclass(frozen=True)
class _FakeSplit:
    assignment_sha256: str


@dataclass(frozen=True)
class _FakeRegistry:
    config_path: Path
    initialization_table_sha256: str | None = None
    split: _FakeSplit | None = None


class ResumeStalenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self._tmp.name) / "registry.json"
        self.config_path.write_text('{"a": 1}', encoding="utf-8")
        self.registry = _FakeRegistry(config_path=self.config_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- registry_config_sha256 / current_registry_identity ---------------

    def test_checksum_changes_when_config_file_content_changes(self) -> None:
        original = registry_config_sha256(self.registry)
        self.config_path.write_text('{"a": 2}', encoding="utf-8")
        changed = registry_config_sha256(self.registry)
        self.assertNotEqual(original, changed)

    def test_current_identity_includes_artifact_hashes_when_present(self) -> None:
        plain = current_registry_identity(self.registry)
        self.assertEqual(set(plain), {"registry_config_sha256"})

        rich = _FakeRegistry(
            config_path=self.config_path,
            initialization_table_sha256="table123",
            split=_FakeSplit(assignment_sha256="split456"),
        )
        identity = current_registry_identity(rich)
        self.assertEqual(identity["initialization_table_sha256"], "table123")
        self.assertEqual(identity["split_assignment_sha256"], "split456")

    # -- saved_registry_identity / result_registry_is_stale ---------------

    def test_saved_identity_reads_multi_start_schema(self) -> None:
        result = {"runs": [{"registry_config_sha256": "abc"}, {"registry_config_sha256": "abc"}]}
        self.assertEqual(saved_registry_identity(result), {"registry_config_sha256": "abc"})

    def test_saved_identity_reads_single_run_schema(self) -> None:
        result = {"registry_config_sha256": "xyz"}
        self.assertEqual(saved_registry_identity(result), {"registry_config_sha256": "xyz"})

    def test_saved_identity_includes_nested_resolved_artifacts(self) -> None:
        result = {
            "runs": [{
                "registry_config_sha256": "abc",
                "model_registry_snapshot": {
                    "resolved_artifacts": {
                        "initialization_table_sha256": "table123",
                        "split_assignment_sha256": "split456",
                    },
                },
            }],
        }
        identity = saved_registry_identity(result)
        self.assertEqual(identity["initialization_table_sha256"], "table123")
        self.assertEqual(identity["split_assignment_sha256"], "split456")

    def test_one_run_missing_a_checksum_yields_no_verifiable_identity(self) -> None:
        """A result where only SOME runs recorded a checksum must not be
        treated as verified just because the ones that did happen to agree
        -- otherwise a single valid-looking run among several corrupted or
        incomplete ones would make the whole multi-start result look
        fresh."""
        result = {"runs": [
            {"registry_config_sha256": "abc"},
            {"start_id": 1},
            {"start_id": 2},
        ]}
        self.assertIsNone(saved_registry_identity(result))
        self.assertTrue(result_registry_is_stale(result, self.registry))

    def test_disagreeing_runs_yield_no_verifiable_identity(self) -> None:
        result = {"runs": [
            {"registry_config_sha256": "abc"},
            {"registry_config_sha256": "different"},
        ]}
        self.assertIsNone(saved_registry_identity(result))
        self.assertTrue(result_registry_is_stale(result, self.registry))

    def test_missing_checksum_field_is_treated_as_stale(self) -> None:
        """Unlike the pre-existing-schema fields this checks, a saved
        result that lacks even the top-level registry_config_sha256 cannot
        be verified at all and must be refit, not silently reused."""
        result = {"runs": [{"start_id": 0}]}
        self.assertIsNone(saved_registry_identity(result))
        self.assertTrue(result_registry_is_stale(result, self.registry))

    def test_result_is_stale_only_when_a_shared_dimension_differs(self) -> None:
        current_checksum = registry_config_sha256(self.registry)
        fresh_result = {"registry_config_sha256": current_checksum}
        self.assertFalse(result_registry_is_stale(fresh_result, self.registry))

        self.config_path.write_text('{"a": 999}', encoding="utf-8")
        self.assertTrue(result_registry_is_stale(fresh_result, self.registry))

    def test_result_is_stale_when_initialization_table_differs(self) -> None:
        rich = _FakeRegistry(
            config_path=self.config_path, initialization_table_sha256="new_table",
        )
        result = {
            "registry_config_sha256": registry_config_sha256(rich),
            "model_registry_snapshot": {
                "resolved_artifacts": {"initialization_table_sha256": "old_table"},
            },
        }
        self.assertTrue(result_registry_is_stale(result, rich))

    def test_extra_dimension_on_only_one_side_is_not_evidence_of_staleness(self) -> None:
        """A result saved by an older schema (no resolved_artifacts) is not
        stale just because the CURRENT registry happens to expose an
        initialization-table hash the saved result never recorded -- that
        dimension simply cannot be compared."""
        rich = _FakeRegistry(
            config_path=self.config_path, initialization_table_sha256="table123",
        )
        result = {"registry_config_sha256": registry_config_sha256(rich)}
        self.assertFalse(result_registry_is_stale(result, rich))

    def test_none_result_is_never_stale(self) -> None:
        self.assertFalse(result_registry_is_stale(None, self.registry))


class ManifestFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.source_root = Path(self._tmp.name) / "src"
        self.source_root.mkdir()
        (self.source_root / "module_a.py").write_text("x = 1\n", encoding="utf-8")
        self.config_path = Path(self._tmp.name) / "registry.json"
        self.config_path.write_text("{}", encoding="utf-8")
        self.registry = _FakeRegistry(config_path=self.config_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_source_hash_changes_when_any_py_file_changes(self) -> None:
        before = source_code_sha256(self.source_root)
        (self.source_root / "module_a.py").write_text("x = 2\n", encoding="utf-8")
        after = source_code_sha256(self.source_root)
        self.assertNotEqual(before, after)

    def test_source_hash_is_sensitive_to_file_path_not_just_content(self) -> None:
        same_content_elsewhere = source_code_sha256(self.source_root)
        (self.source_root / "module_a.py").rename(self.source_root / "module_b.py")
        renamed = source_code_sha256(self.source_root)
        self.assertNotEqual(same_content_elsewhere, renamed)

    def test_manifest_with_no_recorded_fingerprint_is_not_stale(self) -> None:
        """A manifest predating this check (or a fresh {}) has nothing to
        compare -- "not yet verifiable," not "stale"."""
        self.assertFalse(
            manifest_fingerprint_is_stale({}, self.registry, self.source_root)
        )
        self.assertFalse(
            manifest_fingerprint_is_stale(
                {"status": "complete"}, self.registry, self.source_root
            )
        )
        self.assertFalse(
            manifest_fingerprint_is_stale(None, self.registry, self.source_root)
        )

    def test_manifest_is_stale_once_source_code_changes(self) -> None:
        manifest = {"resume_fingerprint": manifest_fingerprint(self.registry, self.source_root)}
        self.assertFalse(
            manifest_fingerprint_is_stale(manifest, self.registry, self.source_root)
        )
        (self.source_root / "module_a.py").write_text("x = 999\n", encoding="utf-8")
        self.assertTrue(
            manifest_fingerprint_is_stale(manifest, self.registry, self.source_root)
        )

    def test_fingerprint_for_write_is_fresh_when_a_fit_actually_ran(self) -> None:
        recorded = manifest_fingerprint_for_write(
            False, {"resume_fingerprint": {"source_code_sha256": "stale_value"}},
            self.registry, self.source_root,
        )
        self.assertEqual(recorded, manifest_fingerprint(self.registry, self.source_root))
        self.assertNotEqual(recorded["source_code_sha256"], "stale_value")

    def test_fingerprint_for_write_is_carried_over_when_only_resumed(self) -> None:
        """A run that only reused an already-resumed fit (redoing
        post-processing, not the fit itself) must not stamp today's source
        code onto a manifest describing a fit that never actually ran under
        it."""
        previous_manifest = {"resume_fingerprint": {"source_code_sha256": "old_value"}}
        recorded = manifest_fingerprint_for_write(
            True, previous_manifest, self.registry, self.source_root,
        )
        self.assertEqual(recorded, {"source_code_sha256": "old_value"})

    def test_fingerprint_for_write_carries_over_absence_too(self) -> None:
        """A resumed fit whose manifest never recorded a fingerprint at all
        (legacy manifest) must not gain one just because this run happened
        to only redo post-processing."""
        recorded = manifest_fingerprint_for_write(True, {}, self.registry, self.source_root)
        self.assertIsNone(recorded)


if __name__ == "__main__":
    unittest.main()
