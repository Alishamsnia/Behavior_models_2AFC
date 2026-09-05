#!/usr/bin/env python3
"""Load the complete M0 registry and print an integrity audit summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def main() -> None:
    """Validate all frozen artifacts and report registered component status."""

    registry = load_model_registry()
    print(
        json.dumps(
            {
                "model_id": registry.model_id,
                "software_version": registry.software_version,
                "implemented_scope": registry.implemented_scope,
                "precision": str(registry.precision),
                "initialization_count": len(registry.initializations),
                "initialization_table_sha256": registry.initialization_table_sha256,
                "split_version": registry.split.version,
                "split_assignment_sha256": registry.split.assignment_sha256,
                "objective_weighting": registry.objective_weighting,
                "optimizer_status": registry.optimizer.implementation_status,
                "scheduler_status": registry.scheduler.implementation_status,
                "locked_test": registry.split.locked_test,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
