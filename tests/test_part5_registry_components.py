"""Tests for the Part V typed registry and component integration."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    LogOddsCoefficients,
    SessionTrials,
    evaluate_mouse_objective,
    solve_positive_decision_region,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class PartFiveRegistryComponentTests(unittest.TestCase):
    def test_registry_loads_all_required_part_five_fields(self) -> None:
        registry = load_model_registry()
        self.assertEqual(registry.model_id, "M0_sensory_v1")
        self.assertEqual(
            registry.parameter_order,
            ("w", "s0", "ell_short", "ell_long", "kappa"),
        )
        self.assertEqual(registry.prior_type, "fixed_neutral")
        self.assertEqual(registry.prior_long, 0.5)
        self.assertEqual(registry.precision, torch.float64)
        self.assertEqual(registry.duration_unit, "seconds")
        self.assertEqual(registry.lapse_max, 1.0)
        self.assertEqual(registry.objective_weighting, "pooled_trials")
        self.assertEqual(registry.optimizer.name, "adam")
        self.assertEqual(registry.scheduler.name, "reduce_on_plateau")
        self.assertEqual(registry.regularization_name, "none")
        self.assertEqual(registry.regularization_coefficient, 0.0)
        self.assertEqual(len(registry.initializations), 8)
        self.assertTrue(registry.split.locked_test)
        self.assertIn("elpd", registry.metrics.primary)
        self.assertEqual(registry.convergence.gradient_norm_tolerance, 1e-5)
        self.assertEqual(
            registry.optimizer.implementation_status,
            "implemented_not_run_full_real_fit",
        )

    def test_structured_positive_region_has_no_negative_root_output(self) -> None:
        simple = solve_positive_decision_region(
            LogOddsCoefficients(
                a2=torch.tensor(1.0, dtype=torch.float64),
                a1=torch.tensor(0.0, dtype=torch.float64),
                a0=torch.tensor(-1.0, dtype=torch.float64),
            )
        )
        self.assertEqual(simple.solver_regime, "quadratic_simple_root")
        self.assertEqual(simple.long_region, "x > threshold")
        self.assertEqual(simple.negative_root_policy, "discard")
        self.assertAlmostEqual(simple.threshold_s.item(), 1.0)
        self.assertFalse(hasattr(simple, "negative_root"))

        double = solve_positive_decision_region(
            LogOddsCoefficients(
                a2=torch.tensor(1.0, dtype=torch.float64),
                a1=torch.tensor(-2.0, dtype=torch.float64),
                a0=torch.tensor(1.0, dtype=torch.float64),
            )
        )
        self.assertEqual(double.solver_regime, "quadratic_double_root")

        linear = solve_positive_decision_region(
            LogOddsCoefficients(
                a2=torch.tensor(0.0, dtype=torch.float64),
                a1=torch.tensor(2.0, dtype=torch.float64),
                a0=torch.tensor(-3.0, dtype=torch.float64),
            )
        )
        self.assertEqual(linear.solver_regime, "linear")
        self.assertIsNone(linear.discriminant)

    def test_every_start_integrates_through_the_registered_objective(self) -> None:
        registry = load_model_registry()
        session = SessionTrials(
            mouse_id="synthetic",
            session_id="train_1",
            duration_s=torch.tensor(
                [0.2, 0.5, 0.8, 1.7, 2.0, 2.3],
                dtype=torch.float64,
            ),
            category_long=torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                dtype=torch.float64,
            ),
            choice_long=torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                dtype=torch.float64,
            ),
            split_role="train",
        )
        for start in registry.initializations:
            raw = start.raw_tensor().detach().requires_grad_(True)
            result = evaluate_mouse_objective(
                [session],
                raw,
                registry.task_moments,
                objective_weighting=registry.objective_weighting,
                lapse_max=registry.lapse_max,
                parameter_epsilon=registry.parameter_epsilon,
                log_probability_epsilon=registry.log_probability_epsilon,
            )
            gradient = torch.autograd.grad(result.loss, raw)[0]
            self.assertTrue(torch.isfinite(result.loss))
            self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_registry_rejects_modified_initialization_artifact(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            shutil.copytree(repository_root / "config", temporary_root / "config")
            table_path = (
                temporary_root
                / "config"
                / "initializations"
                / "m0_sensory_v1_starts_ell100.json"
            )
            table = json.loads(table_path.read_text(encoding="utf-8"))
            table["starts"][0]["w"] = 0.999
            table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                load_model_registry(temporary_root / "config" / "m0_sensory_v1.json")


if __name__ == "__main__":
    unittest.main()
