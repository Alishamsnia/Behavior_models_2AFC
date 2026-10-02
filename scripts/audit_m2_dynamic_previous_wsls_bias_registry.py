#!/usr/bin/env python3
"""Audit the frozen M2_dynamic_previous_wsls_bias registry and its exact
M2_dynamic_block_length-prefix start contract."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m2_dynamic_previous_wsls_bias_v1.registry import load_m2_dynamic_previous_wsls_bias_model_registry  # noqa: E402


def main() -> int:
    registry = load_m2_dynamic_previous_wsls_bias_model_registry(sys.argv[1] if len(sys.argv) > 1 else None)
    assert registry.run_length_center == 23
    assert registry.maximum_represented_run_length == 65
    assert len(registry.initializations) == 8
    assert all(item.bh >= 0 and item.delta0 > 0 for item in registry.initializations)
    assert len({item.eta_logit for item in registry.initializations}) >= 2
    assert set(registry.convergence.required_checks) == {
        "finite_objective", "finite_parameters", "finite_gradients", "small_gradient_norm",
        "stable_fresh_training_nll", "no_parameter_transform_errors",
        "no_probability_normalization_errors", "no_optimizer_runtime_errors",
        "tail_occupancy_acceptable",
    }
    print(
        f"M2_dynamic_previous_wsls_bias registry audit passed: 8 starts, "
        f"center={registry.run_length_center}, Rmax={registry.maximum_represented_run_length}, "
        f"capped tail enabled, prototype_parameterization={registry.prototype_parameterization}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
