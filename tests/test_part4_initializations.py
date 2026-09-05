"""Tests for the frozen non-Cartesian eight-start Part IV table."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    decode_raw_parameters,
    validate_initialization_table,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class PartFourInitializationTests(unittest.TestCase):
    def test_frozen_table_has_exact_values_and_is_not_cartesian(self) -> None:
        starts = load_model_registry().initializations
        observed = [
            (item.start_id, item.w, item.s0, item.ell_short, item.ell_long)
            for item in starts
        ]
        expected = [
            (1, 0.0271, 0.555, 0.030, 0.030),
            (2, 0.0050, 0.700, 0.135, 0.015),
            (3, 0.0100, 1.200, 0.0105, 0.0195),
            (4, 0.0200, 0.250, 0.224, 0.056),
            (5, 0.0400, 0.900, 0.002, 0.008),
            (6, 0.0700, 0.150, 0.130, 0.070),
            (7, 0.1200, 0.450, 0.036, 0.324),
            (8, 0.2000, 0.350, 0.098, 0.002),
        ]
        self.assertEqual(observed, expected)

        marginal_product_size = 1
        for name in ("w", "s0", "ell_short", "ell_long"):
            marginal_product_size *= len({getattr(item, name) for item in starts})
        self.assertEqual(marginal_product_size, 4096)
        self.assertEqual(len(starts), 8)
        self.assertNotEqual(len(starts), marginal_product_size)

    def test_all_frozen_starts_round_trip_and_have_finite_raw_values(self) -> None:
        registry = load_model_registry()
        for start in registry.initializations:
            constrained = start.constrained_tensor()
            raw = start.raw_tensor(
                lapse_max=registry.lapse_max,
                epsilon=registry.parameter_epsilon,
            )
            decoded = decode_raw_parameters(
                raw,
                lapse_max=registry.lapse_max,
                epsilon=registry.parameter_epsilon,
            ).as_tensor()
            self.assertEqual(raw.dtype, torch.float64)
            self.assertTrue(torch.all(torch.isfinite(raw)))
            torch.testing.assert_close(
                decoded,
                constrained,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_invalid_initialization_tables_raise(self) -> None:
        registry = load_model_registry()
        starts = registry.initializations
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            validate_initialization_table(starts[:-1])
        with self.assertRaisesRegex(ValueError, "strictly"):
            validate_initialization_table(
                (*starts[:-1], replace(starts[-1], ell_short=1.0))
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_initialization_table(
                (*starts[:-1], replace(starts[0], start_id=8))
            )


if __name__ == "__main__":
    unittest.main()
