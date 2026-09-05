"""Remaining modified Part X mathematical, restriction, and recovery tests."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    LogOddsCoefficients,
    evaluate_mouse_objective,
    fit_single_initialization,
    predict_m0_trials,
    run_parameter_recovery_case,
    simulate_m0_sessions,
    solve_positive_decision_region,
)
from m0_sensory_v1.pipeline import discover_registered_session_paths  # noqa: E402
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class PartTenMandatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()

    def test_positive_root_override_rejects_unsupported_region_regimes(self) -> None:
        negative_discriminant = LogOddsCoefficients(
            a2=torch.tensor(1.0, dtype=torch.float64),
            a1=torch.tensor(0.0, dtype=torch.float64),
            a0=torch.tensor(1.0, dtype=torch.float64),
        )
        constant = LogOddsCoefficients(
            a2=torch.tensor(0.0, dtype=torch.float64),
            a1=torch.tensor(0.0, dtype=torch.float64),
            a0=torch.tensor(1.0, dtype=torch.float64),
        )
        linear_lower = LogOddsCoefficients(
            a2=torch.tensor(0.0, dtype=torch.float64),
            a1=torch.tensor(-1.0, dtype=torch.float64),
            a0=torch.tensor(1.0, dtype=torch.float64),
        )
        quadratic_lower = LogOddsCoefficients(
            a2=torch.tensor(-1.0, dtype=torch.float64),
            a1=torch.tensor(0.0, dtype=torch.float64),
            a0=torch.tensor(1.0, dtype=torch.float64),
        )
        with self.assertRaisesRegex(ValueError, "no real"):
            solve_positive_decision_region(negative_discriminant)
        with self.assertRaisesRegex(ValueError, "Constant"):
            solve_positive_decision_region(constant)
        with self.assertRaisesRegex(ValueError, "upper"):
            solve_positive_decision_region(linear_lower)
        with self.assertRaisesRegex(ValueError, "upper"):
            solve_positive_decision_region(quadratic_lower)

    def test_analytical_upper_tail_matches_numerical_quadrature(self) -> None:
        raw = self.registry.initializations[0].raw_tensor()
        durations = torch.tensor([0.2, 0.5, 0.8, 1.7, 2.0, 2.3], dtype=torch.float64)
        choices = torch.zeros_like(durations)
        categories = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64)
        prediction = predict_m0_trials(
            durations,
            categories,
            choices,
            raw,
            self.registry.task_moments,
        )
        ell_short = self.registry.initializations[0].ell_short
        ell_long = self.registry.initializations[0].ell_long
        numerical: list[float] = []
        for duration, category, threshold, sd in zip(
            durations.numpy(),
            categories.numpy(),
            prediction.positive_threshold_s.numpy(),
            prediction.sensory_sd_s.numpy(),
            strict=True,
        ):
            density = lambda x: np.exp(-0.5 * ((x - duration) / sd) ** 2) / (
                sd * np.sqrt(2.0 * np.pi)
            )
            upper_tail = quad(density, threshold, np.inf, epsabs=1e-12)[0]
            lapse_rate = ell_long if category == 1.0 else ell_short
            numerical.append((1.0 - lapse_rate) * upper_tail + 0.5 * lapse_rate)
        np.testing.assert_allclose(
            prediction.probability_long.numpy(),
            numerical,
            rtol=1e-10,
            atol=1e-11,
        )

    def test_pooled_objective_autograd_matches_finite_difference_all_parameters(self) -> None:
        duration = torch.tensor([0.25, 0.55, 0.75, 1.75, 2.05, 2.25], dtype=torch.float64)
        generating = self.registry.initializations[0].constrained_tensor()
        session = simulate_m0_sessions(
            [duration],
            generating,
            self.registry.task_moments,
            random_seed=81,
        )[0]

        def objective(raw: torch.Tensor) -> torch.Tensor:
            return evaluate_mouse_objective(
                [session],
                raw,
                self.registry.task_moments,
                required_split_role="train",
            ).loss

        raw = self.registry.initializations[1].raw_tensor().requires_grad_(True)
        analytical = torch.autograd.grad(objective(raw), raw)[0]
        numerical = torch.empty_like(raw)
        step = 1e-6
        for index in range(raw.numel()):
            plus = raw.detach().clone()
            minus = raw.detach().clone()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (objective(plus) - objective(minus)) / (2.0 * step)
        torch.testing.assert_close(analytical, numerical, rtol=5e-5, atol=2e-7)

    def test_simulation_is_seeded_and_uses_only_current_duration(self) -> None:
        duration = torch.tensor([0.3, 0.6, 1.8, 2.1] * 20, dtype=torch.float64)
        parameters = self.registry.initializations[0].constrained_tensor()
        first = simulate_m0_sessions(
            [duration],
            parameters,
            self.registry.task_moments,
            random_seed=92,
            mouse_id="first_metadata",
        )[0]
        second = simulate_m0_sessions(
            [duration],
            parameters,
            self.registry.task_moments,
            random_seed=92,
            mouse_id="changed_metadata",
        )[0]
        self.assertTrue(torch.equal(first.choice_long, second.choice_long))
        first_prediction = predict_m0_trials(
            first.duration_s,
            first.category_long,
            first.choice_long,
            self.registry.initializations[0].raw_tensor(),
            self.registry.task_moments,
        )
        second_prediction = predict_m0_trials(
            second.duration_s,
            second.category_long,
            1.0 - second.choice_long,
            self.registry.initializations[0].raw_tensor(),
            self.registry.task_moments,
        )
        torch.testing.assert_close(
            first_prediction.probability_long,
            second_prediction.probability_long,
        )

    def test_parameter_recovery_improves_training_nll(self) -> None:
        levels = torch.tensor(
            [0.2, 0.35, 0.5, 0.65, 0.8, 1.7, 1.85, 2.0, 2.15, 2.3],
            dtype=torch.float64,
        )
        result = run_parameter_recovery_case(
            self.registry.initializations[3].constrained_tensor(),
            [levels.repeat_interleave(50), levels.repeat_interleave(50)],
            [levels.repeat_interleave(20)],
            self.registry,
            self.registry.initializations[0],
            random_seed=123,
            maximum_epochs=80,
        )
        self.assertGreater(result.training_nll_improvement, 0.0)
        self.assertLess(result.probability_rmse_on_duration_grid, 0.05)
        self.assertLess(result.absolute_parameter_errors[0], 0.02)
        self.assertLess(result.absolute_parameter_errors[1], 0.25)
        self.assertLess(result.absolute_parameter_errors[2], 0.20)
        self.assertLess(result.absolute_parameter_errors[3], 0.20)

    def test_discovery_returns_only_registered_non_test_paths(self) -> None:
        assignments = pd.read_csv(self.registry.split.assignment_path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for session_id in assignments["session_id"].astype(str):
                (root / f"{session_id}.mat").touch()
            selected = discover_registered_session_paths(root, self.registry)
            expected = assignments.loc[
                assignments["split_role"].isin(["train", "validation"]),
                "session_id",
            ].astype(str)
            self.assertEqual(len(selected), 36)
            self.assertEqual([path.stem for path in selected], expected.tolist())
            test_ids = set(
                assignments.loc[assignments["split_role"].eq("test"), "session_id"].astype(str)
            )
            self.assertTrue(test_ids.isdisjoint({path.stem for path in selected}))

    def test_live_callbacks_receive_initial_and_every_epoch_nll(self) -> None:
        levels = torch.tensor([0.3, 0.6, 1.8, 2.1] * 10, dtype=torch.float64)
        parameters = self.registry.initializations[0].constrained_tensor()
        train = simulate_m0_sessions(
            [levels], parameters, self.registry.task_moments, random_seed=1
        )
        validation = simulate_m0_sessions(
            [levels],
            parameters,
            self.registry.task_moments,
            random_seed=2,
            split_role="validation",
        )
        initial_values: list[tuple[float, float]] = []
        epochs = []
        result = fit_single_initialization(
            train,
            validation,
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=3,
            initial_callback=lambda train_nll, validation_nll: initial_values.append(
                (train_nll, validation_nll)
            ),
            epoch_callback=epochs.append,
        )
        self.assertEqual(len(initial_values), 1)
        self.assertEqual([record.epoch for record in epochs], [1, 2, 3])
        self.assertEqual(result.initial_training_mean_nll, initial_values[0][0])
        self.assertEqual(result.initial_validation_mean_nll, initial_values[0][1])


if __name__ == "__main__":
    unittest.main()
