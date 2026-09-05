"""Tests for Part VIII complete records and training-only selection."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.model import (  # noqa: E402
    SessionTrials,
    fit_all_initializations,
    fit_single_initialization,
    select_winning_start,
)
from m0_sensory_v1.registry import load_model_registry  # noqa: E402


def _session(role: str, session_id: str) -> SessionTrials:
    return SessionTrials(
        mouse_id="synthetic_mouse",
        session_id=session_id,
        duration_s=torch.tensor([0.3, 0.6, 1.8, 2.1], dtype=torch.float64),
        category_long=torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64),
        choice_long=torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64),
        split_role=role,  # type: ignore[arg-type]
    )


class PartEightMultiStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_model_registry()
        cls.training = _session("train", "train")
        cls.validation = _session("validation", "validation")

    def test_winner_uses_training_not_validation_and_flags_ridge(self) -> None:
        base = fit_single_initialization(
            [self.training],
            [self.validation],
            self.registry,
            self.registry.initializations[0],
            max_epochs_override=1,
        )
        run_1 = replace(
            base,
            start_id=1,
            converged=True,
            recalculated_best_training_mean_nll=0.20,
            final_validation_mean_nll=10.0,
            best_constrained_parameters=(0.01, 0.2, 0.02, 0.1, 1.0),
        )
        run_2 = replace(
            base,
            start_id=2,
            converged=True,
            recalculated_best_training_mean_nll=0.20 + 0.5e-6,
            final_validation_mean_nll=0.01,
            best_constrained_parameters=(0.20, 1.2, 0.18, 0.9, 2.0),
        )
        run_3 = replace(
            base,
            start_id=3,
            converged=False,
            recalculated_best_training_mean_nll=0.01,
            final_validation_mean_nll=0.001,
        )
        winner, ridge, ridge_ids = select_winning_start(
            [run_1, run_2, run_3],
            self.registry,
        )
        self.assertEqual(winner.start_id, 1)
        self.assertTrue(ridge)
        self.assertEqual(ridge_ids, (1, 2))

    def test_multistart_runs_all_frozen_initializations_and_saves_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = fit_all_initializations(
                [self.training],
                [self.validation],
                self.registry,
                output_directory=temporary,
                max_epochs_override=1,
            )
            self.assertEqual(len(result.runs), 8)
            self.assertEqual(
                {run.start_id for run in result.runs},
                set(range(1, 9)),
            )
            self.assertIsNone(result.winner_start_id)
            self.assertEqual(result.selection_status, "no_converged_run_no_winner")
            output_files = sorted(Path(temporary).glob("*.json"))
            self.assertEqual(len(output_files), 9)
            payload = json.loads(
                (Path(temporary) / "start_01.json").read_text(encoding="utf-8")
            )
            for field in (
                "registry_config_sha256",
                "model_registry_snapshot",
                "training_session_ids",
                "validation_session_ids",
                "training_valid_trial_count",
                "validation_valid_trial_count",
                "initial_constrained_parameters",
                "initial_raw_parameters",
                "best_raw_parameters",
                "best_constrained_parameters",
                "recalculated_best_training_mean_nll",
                "final_validation_mean_nll",
                "gradient_norm_at_restored_best",
                "history",
                "training_history",
                "validation_history",
                "boundary_proximity",
                "random_seed",
            ):
                self.assertIn(field, payload)
            policy = payload["model_registry_snapshot"]["binary_decision_data_policy"]
            self.assertIn("RewardNaive", policy)
            self.assertIn("PunishNaive", policy)


if __name__ == "__main__":
    unittest.main()
