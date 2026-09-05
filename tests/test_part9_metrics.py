"""Tests for the registered Part IX evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    SessionTrials,
    calculate_evaluation_metrics,
    decode_raw_parameters,
    evaluate_session,
    evaluate_split_metrics,
    save_evaluation_metrics,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


class PartNineMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.raw = cls.registry.initializations[0].raw_tensor()
        cls.parameters = decode_raw_parameters(cls.raw)

    def _session(self, role: str = "validation") -> SessionTrials:
        return SessionTrials(
            mouse_id="synthetic_mouse",
            session_id=f"{role}_1",
            duration_s=torch.tensor(
                [0.2, 0.5, 0.8, 1.7, 2.0, 2.3, 1.0],
                dtype=torch.float64,
            ),
            category_long=torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                dtype=torch.float64,
            ),
            choice_long=torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, float("nan")],
                dtype=torch.float64,
            ),
            split_role=role,  # type: ignore[arg-type]
        )

    def test_part9_metrics_match_manual_values_and_skip_omissions(self) -> None:
        session = self._session()
        evaluation = evaluate_session(
            session,
            self.parameters,
            self.registry.task_moments,
        )
        metrics = calculate_evaluation_metrics(
            [evaluation],
            calibration_bins=5,
            block_labels_by_session={
                session.session_id: ["A", "A", "A", "B", "B", "B", "B"]
            },
        )
        valid = evaluation.predictions.valid_choice_mask
        probability = evaluation.predictions.probability_long[valid].numpy()
        observed = session.choice_long[valid].numpy()
        nll = evaluation.per_trial_negative_log_likelihood[valid].numpy()
        self.assertEqual(metrics.valid_trial_count, 6)
        self.assertAlmostEqual(metrics.elpd, -float(np.sum(nll)), places=13)
        self.assertAlmostEqual(metrics.mean_nll_per_valid_trial, np.mean(nll), places=13)
        self.assertAlmostEqual(
            metrics.brier_score,
            float(np.mean((observed - probability) ** 2)),
            places=13,
        )
        self.assertAlmostEqual(
            metrics.classification_accuracy,
            float(np.mean((probability > 0.5) == observed)),
            places=13,
        )
        self.assertEqual(sum(row.valid_trial_count for row in metrics.calibration_curve), 6)
        self.assertEqual(
            sum(row.valid_trial_count for row in metrics.overall_psychometric_curve),
            6,
        )
        self.assertEqual(metrics.block_diagnostic_status, "computed_diagnostic_only_not_used_by_model")
        self.assertEqual(sum(row.valid_trial_count for row in metrics.block_psychometric_curve), 6)

    def test_omission_only_session_has_null_session_metrics(self) -> None:
        valid = self._session()
        omissions = SessionTrials(
            mouse_id="synthetic_mouse",
            session_id="validation_omissions",
            duration_s=torch.tensor([0.4, 2.0], dtype=torch.float64),
            category_long=torch.tensor([0.0, 1.0], dtype=torch.float64),
            choice_long=torch.tensor([float("nan"), float("nan")], dtype=torch.float64),
            split_role="validation",
        )
        evaluations = [
            evaluate_session(item, self.parameters, self.registry.task_moments)
            for item in (valid, omissions)
        ]
        metrics = calculate_evaluation_metrics(evaluations)
        omission_metrics = metrics.per_session[1]
        self.assertEqual(omission_metrics.valid_trial_count, 0)
        self.assertIsNone(omission_metrics.elpd)
        self.assertIsNone(omission_metrics.brier_score)

    def test_part9_rejects_locked_test_evaluations(self) -> None:
        test_session = self._session("test")
        evaluation = evaluate_session(
            test_session,
            self.parameters,
            self.registry.task_moments,
        )
        with self.assertRaisesRegex(ValueError, "Locked test"):
            calculate_evaluation_metrics([evaluation])
        with self.assertRaisesRegex(ValueError, "train or validation"):
            evaluate_split_metrics(
                [test_session],
                self.raw,
                self.registry,
                required_split_role="test",
            )

    def test_evaluate_split_metrics_and_json_serialization(self) -> None:
        metrics = evaluate_split_metrics(
            [self._session()],
            self.raw.requires_grad_(True),
            self.registry,
            required_split_role="validation",
        )
        self.assertEqual(metrics.split_role, "validation")
        self.assertEqual(len(metrics.calibration_curve), 10)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "metrics.json"
            save_evaluation_metrics(metrics, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["valid_trial_count"], 6)
            self.assertIn("prediction_distribution", payload)


if __name__ == "__main__":
    unittest.main()
