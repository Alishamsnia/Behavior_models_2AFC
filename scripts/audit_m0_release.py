#!/usr/bin/env python3
"""Verify the frozen M0 implementation manifest and locked-test declaration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "config" / "releases" / "m0_sensory_v1_release.json"


def _tree_sha256(root: Path) -> str:
    """Hash relative names and contents of every Python file deterministically."""

    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    """Validate every declared checksum and return a shell status."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = REPOSITORY_ROOT / relative
        observed = sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if observed != expected:
            failures.append(f"{relative}: expected {expected}, observed {observed}")
    observed_source = _tree_sha256(REPOSITORY_ROOT / "src")
    observed_tests = _tree_sha256(REPOSITORY_ROOT / "tests")
    observed_scripts = _tree_sha256(REPOSITORY_ROOT / "scripts")
    if observed_source != manifest["source_tree_sha256"]:
        failures.append("source tree checksum mismatch")
    if observed_tests != manifest["tests_tree_sha256"]:
        failures.append("tests tree checksum mismatch")
    if observed_scripts != manifest.get("scripts_tree_sha256"):
        failures.append("scripts tree checksum mismatch")
    subject_status = manifest.get("test_evaluated_by_subject", {})
    expected_any_test_evaluated = any(subject_status.values())
    if manifest.get("locked_test_evaluated") is not expected_any_test_evaluated:
        failures.append("aggregate test-evaluation status disagrees with subjects")
    split_manifests = {
        "MC06": "chronological_70_15_15_mc06_v1.json",
        "MC07": "chronological_70_15_15_mc07_v1.json",
        "MC08": "chronological_70_15_15_v1.json",
    }
    for mouse_id, filename in split_manifests.items():
        split_manifest = json.loads(
            (REPOSITORY_ROOT / "config" / "splits" / filename).read_text(
                encoding="utf-8"
            )
        )
        if split_manifest.get("locked_test_evaluated") is not subject_status.get(
            mouse_id
        ):
            failures.append(f"{mouse_id} test status disagrees with split manifest")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print(
        f"Release audit passed: {manifest['release_id']} "
        f"({len(manifest['artifact_sha256'])} artifacts; test status audited)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
