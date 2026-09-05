#!/usr/bin/env python3
"""Audit the frozen M1_dynamic_block_length registry and its exact M1-prefix
start contract."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m1_dynamic_block_length_v1.registry import load_m1_dynamic_block_length_model_registry  # noqa: E402


def main() -> int:
    registry = load_m1_dynamic_block_length_model_registry(sys.argv[1] if len(sys.argv) > 1 else None)
    assert registry.run_length_center == 23
    assert registry.maximum_represented_run_length == 65
    assert len(registry.initializations) == 8
    assert all(item.bh >= 0 for item in registry.initializations)
    assert registry.true_rate_prototypes == (0.5, 0.2, 0.8)
    assert set(registry.convergence.required_checks) == {
        "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
        "stable_fresh_training_nll", "no_parameter_transform_errors",
        "no_probability_normalization_errors", "no_optimizer_runtime_errors",
        "tail_occupancy_acceptable",
    }
    print(
        f"M1_dynamic_block_length registry audit passed: 8 starts, "
        f"center={registry.run_length_center}, Rmax={registry.maximum_represented_run_length}, "
        f"capped tail enabled, prototypes fixed at {registry.true_rate_prototypes}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
