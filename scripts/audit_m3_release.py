#!/usr/bin/env python3
"""Verify the frozen M3 release artifact hashes and registry."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402


def main() -> int:
    release_path = ROOT / "config/releases/m3_hmm_delta_v1_release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    failures = []
    references = dict(release["artifacts"])
    references[release["pre_m3_regression_reference"]["path"]] = release[
        "pre_m3_regression_reference"
    ]["sha256"]
    for relative, expected in references.items():
        path = ROOT / relative
        observed = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    registry = load_m3_model_registry()
    if len(registry.initializations) != 8:
        failures.append({"registry": "initialization_count", "observed": len(registry.initializations)})
    result = {
        "status": "passed" if not failures else "failed",
        "release_id": release["release_id"],
        "verified_artifact_count": len(references),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
