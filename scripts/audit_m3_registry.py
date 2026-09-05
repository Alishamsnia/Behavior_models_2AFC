#!/usr/bin/env python3
"""Validate all M3 subject registries without opening behavioral data."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m3_hmm_delta_v1.registry import load_m3_model_registry  # noqa: E402


def main() -> int:
    records = []
    for filename, mouse_id in (
        ("m3_hmm_delta_v1_mc06.json", "MC06"),
        ("m3_hmm_delta_v1_mc07.json", "MC07"),
        ("m3_hmm_delta_v1.json", "MC08"),
    ):
        registry = load_m3_model_registry(ROOT / "config" / filename)
        records.append({
            "mouse_id": mouse_id,
            "model_id": registry.model_id,
            "parameter_order": registry.parameter_order,
            "initialization_count": len(registry.initializations),
            "learning_rate_starts": sorted({item.alpha_theta for item in registry.initializations}),
            "m2_source_start_ids": [item.source_m2_start_id for item in registry.initializations],
            "empirical_median_block_length_trials": registry.empirical_median_block_length_trials,
            "empirical_inverse_median_hazard": registry.empirical_inverse_median_hazard,
            "default_optimizer": registry.fitting.default_optimizer,
            "supported_optimizers": registry.fitting.supported_optimizers,
            "pdf_recommended_optimizer": registry.fitting.pdf_recommended_optimizer,
            "split_version": registry.split.version,
            "locked_test": registry.split.locked_test,
        })
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
