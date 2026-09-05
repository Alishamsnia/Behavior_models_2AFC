#!/usr/bin/env python3
"""Verify frozen M4 release hashes and core registry constants."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402


def main() -> int:
    release = json.loads(
        (ROOT / "config/releases/m4_hsmm_delta_v1_release.json").read_text(encoding="utf-8")
    )
    failures = []
    for relative, expected in release["artifacts"].items():
        path = ROOT / relative
        observed = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    registry = load_m4_model_registry()
    if registry.run_length_center != 23 or registry.maximum_represented_run_length != 65:
        failures.append({"registry": "duration_constants"})
    if len(registry.initializations) != 8:
        failures.append({"registry": "initialization_count", "observed": len(registry.initializations)})
    result = {
        "status": "passed" if not failures else "failed",
        "release_id": release["release_id"],
        "verified_artifact_count": len(release["artifacts"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
