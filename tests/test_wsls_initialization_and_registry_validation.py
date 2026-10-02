"""Validation-edge-case tests for both new models' initialization tables
and checksum-pinned registries.

Mirrors the provenance/tamper-detection guarantees ``m1_stimulus_kernel_v1``
already has: both models' eight starts are chained to M0's own eight-row
table exactly like ``m1_stimulus_kernel_v1``'s are, so both the checksum
gate and the M0-source-row comparison are tested here.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from extended_wsls_v1.initialization import (
    ExtendedWslsInitializationSpec,
    validate_extended_wsls_initialization_table,
)
from extended_wsls_v1.registry import load_extended_wsls_model_registry
from pure_wsls_v1.initialization import (
    PureWslsInitializationSpec,
    validate_pure_wsls_initialization_table,
)
from pure_wsls_v1.registry import load_pure_wsls_model_registry


class PureWslsInitializationValidationTests(unittest.TestCase):
    def _valid_records(self):
        return [
            PureWslsInitializationSpec(1, 0.03, 0.03, "a"),
            PureWslsInitializationSpec(2, 0.1, 0.2, "b"),
        ]

    def test_accepts_valid_varied_table(self) -> None:
        records = validate_pure_wsls_initialization_table(self._valid_records(), expected_count=2)
        self.assertEqual(len(records), 2)

    def test_rejects_duplicate_vectors(self) -> None:
        records = [
            PureWslsInitializationSpec(1, 0.03, 0.03, "a"),
            PureWslsInitializationSpec(2, 0.03, 0.03, "b"),
        ]
        with self.assertRaises(ValueError):
            validate_pure_wsls_initialization_table(records, expected_count=2)

    def test_rejects_no_variation_in_ell_short(self) -> None:
        records = [
            PureWslsInitializationSpec(1, 0.03, 0.03, "a"),
            PureWslsInitializationSpec(2, 0.03, 0.2, "b"),
        ]
        with self.assertRaises(ValueError):
            validate_pure_wsls_initialization_table(records, expected_count=2)

    def test_rejects_out_of_bounds_lapse(self) -> None:
        records = [
            PureWslsInitializationSpec(1, 0.03, 0.03, "a"),
            PureWslsInitializationSpec(2, 1.5, 0.2, "b"),
        ]
        with self.assertRaises(ValueError):
            validate_pure_wsls_initialization_table(records, expected_count=2)

    def test_rejects_missing_purpose(self) -> None:
        records = [
            PureWslsInitializationSpec(1, 0.03, 0.03, ""),
            PureWslsInitializationSpec(2, 0.1, 0.2, "b"),
        ]
        with self.assertRaises(ValueError):
            validate_pure_wsls_initialization_table(records, expected_count=2)


class ExtendedWslsInitializationValidationTests(unittest.TestCase):
    def test_accepts_strict_bijection_between_lapse_config_and_alpha(self) -> None:
        """This is the actual production pattern: every one of M0's eight
        lapse configurations is paired with exactly one of the eight
        registered learning rates, by start_id -- the same one-to-one
        joint design M1_stimulus_kernel_v1 already uses. It must be
        accepted, not rejected."""

        records = [
            ExtendedWslsInitializationSpec(1, 0.03, 0.03, -3.0, "a"),
            ExtendedWslsInitializationSpec(2, 0.1, 0.1, -1.0, "b"),
            ExtendedWslsInitializationSpec(3, 0.2, 0.2, 1.0, "c"),
        ]
        validated = validate_extended_wsls_initialization_table(records, expected_count=3)
        self.assertEqual(len(validated), 3)

    def test_accepts_crossed_lapse_and_alpha_grid(self) -> None:
        """A non-bijective grid is also valid -- the validator does not
        require any particular pairing structure, only variation."""

        records = [
            ExtendedWslsInitializationSpec(1, 0.03, 0.03, -3.0, "a"),
            ExtendedWslsInitializationSpec(2, 0.03, 0.03, 1.0, "b"),
            ExtendedWslsInitializationSpec(3, 0.1, 0.1, -1.0, "c"),
        ]
        validated = validate_extended_wsls_initialization_table(records, expected_count=3)
        self.assertEqual(len(validated), 3)

    def test_rejects_nonfinite_logit(self) -> None:
        records = [
            ExtendedWslsInitializationSpec(1, 0.03, 0.03, float("inf"), "a"),
            ExtendedWslsInitializationSpec(2, 0.03, 0.03, 1.0, "b"),
            ExtendedWslsInitializationSpec(3, 0.1, 0.1, -1.0, "c"),
        ]
        with self.assertRaises(ValueError):
            validate_extended_wsls_initialization_table(records, expected_count=3)


class RegistryChecksumTamperDetectionTests(unittest.TestCase):
    def test_pure_wsls_artifact_loader_rejects_tampered_content(self) -> None:
        """Direct unit test of the registry's own checksum gate
        (``_artifact``): a byte-for-byte content change must be caught
        before the tampered file is ever parsed as JSON."""

        import json
        import tempfile

        from pure_wsls_v1.registry import _artifact

        table_path = Path("config/initializations/pure_wsls_v1_starts.json")
        real_sha256 = json.loads(Path("config/pure_wsls_v1.json").read_text())[
            "initialization_table"
        ]["sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tampered = json.loads(table_path.read_text())
            tampered["starts"][0]["ell_short"] = 0.9999
            (tmp_path / "tampered_starts.json").write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                _artifact(tmp_path, "tampered_starts.json", real_sha256)

    def test_extended_wsls_loads_real_frozen_registry(self) -> None:
        # Sanity companion to the tamper test above: the real, untouched
        # registry must load without any checksum complaint.
        registry = load_extended_wsls_model_registry()
        self.assertEqual(registry.model_id, "Extended_WSLS_v1")

    def test_pure_wsls_registry_rejects_an_ell_value_that_disagrees_with_m0(self) -> None:
        """End-to-end check of the M0-source-row provenance chain added to
        match ``m1_stimulus_kernel_v1/registry.py``: a start whose
        ``ell_short`` no longer matches M0's own row for that ``start_id``
        must be rejected at registry-load time, even though its own
        checksum is internally self-consistent (this is NOT the same
        failure mode as the checksum-tamper test above, which catches a
        content change that no longer matches its recorded checksum --
        here the checksum matches the tampered content perfectly, and only
        the cross-table M0 comparison catches the problem)."""

        import hashlib
        import json
        import shutil
        import tempfile

        from pure_wsls_v1.registry import load_pure_wsls_model_registry

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(ROOT / "config", tmp_root / "config")

            starts_path = tmp_root / "config" / "initializations" / "pure_wsls_v1_starts.json"
            table = json.loads(starts_path.read_text())
            table["starts"][0]["ell_short"] = 0.9999  # disagrees with M0's row 1 (0.03)
            starts_path.write_text(json.dumps(table))
            new_starts_sha256 = hashlib.sha256(starts_path.read_bytes()).hexdigest()

            config_path = tmp_root / "config" / "pure_wsls_v1.json"
            config = json.loads(config_path.read_text())
            config["initialization_table"]["sha256"] = new_starts_sha256
            config_path.write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "M0 source row"):
                load_pure_wsls_model_registry(config_path)


if __name__ == "__main__":
    unittest.main()
