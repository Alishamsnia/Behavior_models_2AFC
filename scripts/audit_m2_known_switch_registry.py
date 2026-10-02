#!/usr/bin/env python3
"""Load all three M2_known_switch_v1 subject registries without opening
behavioral data."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m2_known_switch_v1.registry import load_m2_known_switch_model_registry  # noqa: E402


def main() -> int:
    records = []
    for filename, mouse_id in (
        ("m2_known_switch_v1_mc06.json", "MC06"),
        ("m2_known_switch_v1_mc07.json", "MC07"),
        ("m2_known_switch_v1.json", "MC08"),
    ):
        registry = load_m2_known_switch_model_registry(ROOT / "config" / filename)
        records.append({
            "mouse_id": mouse_id,
            "model_id": registry.model_id,
            "parameter_order": registry.parameter_order,
            "initialization_count": len(registry.initializations),
            "initialization_table_version": registry.initialization_table_version,
            "initialization_table_sha256": registry.initialization_table_sha256,
            "split_version": registry.split.version,
            "locked_test": registry.split.locked_test,
            "source_m0_registry_model_id": registry.source_m0_registry.model_id,
        })
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
