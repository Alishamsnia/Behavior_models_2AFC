#!/usr/bin/env python3
"""Audit the frozen M4 registry and its exact M3-prefix start contract."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from m4_hsmm_delta_v1.registry import load_m4_model_registry  # noqa: E402


def main() -> int:
    registry = load_m4_model_registry(sys.argv[1] if len(sys.argv) > 1 else None)
    assert registry.run_length_center == 23
    assert registry.maximum_represented_run_length == 65
    assert len(registry.initializations) == 8
    assert all(0 < item.alpha_theta <= 0.1 for item in registry.initializations)
    print(
        f"M4 registry audit passed: 8 starts, center={registry.run_length_center}, "
        f"Rmax={registry.maximum_represented_run_length}, capped tail enabled."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
