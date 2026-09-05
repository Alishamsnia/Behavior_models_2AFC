"""Tests for independent session likelihoods and pooled mouse objectives."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from m0_sensory_v1.data import TaskMoments  # noqa: E402
from m0_sensory_v1.model import (  # noqa: E402
    SessionTrials,
    decode_raw_parameters,
    encode_constrained_parameters,
    evaluate_mouse_objective,
    evaluate_session,
    session_trials_from_frame,
)
from m0_sensory_v1.model.parameters import (  # noqa: E402
    decode_raw_parameters as actual_decoder,
)


DTYPE = torch.float64
MOMENTS = TaskMoments(
    mu_short_s=0.5,
    mu_long_s=2.0,
    var_short_s2=0.0296,
    var_long_s2=0.0296,
    n_short=5,
    n_long=5,
    min_short_s=0.2,
    max_short_s=0.8,
    min_long_s=1.7,
    max_long_s=2.3,
    source="Part II test",
)


def _raw(*, requires_grad: bool = False) -> torch.Tensor:
    constrained = torch.tensor([0.0271, 0.555, 0.03, 0.5, 1.0], dtype=DTYPE)
    return (
        encode_constrained_parameters(constrained)
        .detach()
        .requires_grad_(requires_grad)
    )


def _session(
    session_id: str,
    duration: list[float],
    choice: list[float],
    *,
    mouse_id: str = "MC08",
    split_role: str = "train",
) -> SessionTrials:
    category = [
        float("nan") if not torch.isfinite(torch.tensor(value)) else float(value >= 1.25)
        for value in duration
    ]
    return SessionTrials(
        mouse_id=mouse_id,
        session_id=session_id,
        duration_s=torch.tensor(duration, dtype=DTYPE),
        category_long=torch.tensor(category, dtype=DTYPE),
        choice_long=torch.tensor(choice, dtype=DTYPE),
        split_role=split_role,  # type: ignore[arg-type]
    )


class PartTwoSessionObjectiveTests(unittest.TestCase):
    def test_session_frame_adapter_preserves_missing_choices(self) -> None:
        frame = pd.DataFrame(
            {
                "mouse_id": ["MC08"] * 3,
                "session_id": ["s1"] * 3,
                "duration_s": [0.5, 1.0, 2.0],
                "category_long": pd.array([0, 0, 1], dtype="Int64"),
                "choice_long": pd.array([0, pd.NA, 1], dtype="Int64"),
            }
        )
        session = session_trials_from_frame(frame, split_role="train")
        self.assertEqual(session.duration_s.dtype, torch.float64)
        self.assertEqual(torch.isnan(session.choice_long).tolist(), [False, True, False])

    def test_session_evaluator_matches_manual_bernoulli_sum_and_skips_omissions(
        self,
    ) -> None:
        session = _session(
            "s1",
            [0.35, float("nan"), 1.85, 2.15],
            [0.0, float("nan"), 0.0, 1.0],
        )
        evaluation = evaluate_session(
            session,
            decode_raw_parameters(_raw()),
            MOMENTS,
        )
        mask = evaluation.predictions.valid_choice_mask
        probability = evaluation.predictions.probability_long[mask]
        observed = session.choice_long[mask]
        manual = -(
            observed * torch.log(probability)
            + (1.0 - observed) * torch.log1p(-probability)
        ).sum()
        torch.testing.assert_close(evaluation.negative_log_likelihood, manual)
        self.assertEqual(evaluation.valid_trial_count, 3)
        self.assertEqual(
            torch.isnan(evaluation.per_trial_negative_log_likelihood).tolist(),
            [False, True, False, False],
        )

    def test_pooled_and_equal_session_objectives_match_manual_formulas(self) -> None:
        sessions = (
            _session("s1", [0.2, 0.5, 2.0], [0.0, float("nan"), 1.0]),
            _session("s2", [0.35, 0.65, 1.85, 2.15], [0.0, 1.0, 0.0, 1.0]),
        )
        pooled = evaluate_mouse_objective(sessions, _raw(), MOMENTS)
        equal = evaluate_mouse_objective(
            sessions,
            _raw(),
            MOMENTS,
            objective_weighting="equal_sessions",
        )
        nll = torch.stack(
            [item.negative_log_likelihood for item in pooled.session_evaluations]
        )
        counts = torch.tensor([2.0, 4.0], dtype=DTYPE)
        torch.testing.assert_close(pooled.loss, nll.sum() / counts.sum())
        torch.testing.assert_close(equal.loss, torch.mean(nll / counts))
        self.assertEqual(pooled.total_valid_trial_count, 6)
        self.assertIsNone(_raw(requires_grad=True).grad)

    def test_reordering_sessions_preserves_loss_and_gradients(self) -> None:
        sessions = (
            _session("s1", [0.2, 0.5, 2.0], [0.0, 1.0, 1.0]),
            _session("s2", [0.35, 0.65, 1.85, 2.15], [0.0, 1.0, 0.0, 1.0]),
            _session("s3", [0.8, 1.7], [1.0, 0.0]),
        )

        def value_and_gradient(
            order: tuple[SessionTrials, ...],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raw = _raw(requires_grad=True)
            result = evaluate_mouse_objective(order, raw, MOMENTS)
            self.assertIsNone(raw.grad)
            gradient = torch.autograd.grad(result.loss, raw)[0]
            return result.loss.detach(), gradient

        first_loss, first_gradient = value_and_gradient(sessions)
        second_loss, second_gradient = value_and_gradient(tuple(reversed(sessions)))
        torch.testing.assert_close(first_loss, second_loss, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            first_gradient,
            second_gradient,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_objective_decodes_once_and_handles_zero_valid_session(self) -> None:
        sessions = (
            _session("empty", [0.5, 2.0], [float("nan"), float("nan")]),
            _session("valid", [0.5, 2.0], [0.0, 1.0]),
        )
        with patch(
            "m0_sensory_v1.model.session.decode_raw_parameters",
            wraps=actual_decoder,
        ) as decoder:
            pooled = evaluate_mouse_objective(sessions, _raw(), MOMENTS)
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(pooled.total_valid_trial_count, 2)
        self.assertEqual(pooled.session_evaluations[0].valid_trial_count, 0)
        with self.assertRaisesRegex(ValueError, "zero-valid"):
            evaluate_mouse_objective(
                sessions,
                _raw(),
                MOMENTS,
                objective_weighting="equal_sessions",
            )

    def test_locked_test_session_is_rejected(self) -> None:
        locked = _session("locked", [0.5, 2.0], [0.0, 1.0], split_role="test")
        with self.assertRaisesRegex(ValueError, "Locked test"):
            evaluate_mouse_objective(
                [locked],
                _raw(),
                MOMENTS,
                required_split_role="test",
            )

        validation = _session(
            "validation",
            [0.5, 2.0],
            [0.0, 1.0],
            split_role="validation",
        )
        result = evaluate_mouse_objective(
            [validation],
            _raw(),
            MOMENTS,
            required_split_role="validation",
        )
        self.assertEqual(result.total_valid_trial_count, 2)

    def test_pooled_parameters_cannot_span_mice(self) -> None:
        sessions = (
            _session("s1", [0.5], [0.0], mouse_id="MC08"),
            _session("s2", [2.0], [1.0], mouse_id="MC09"),
        )
        with self.assertRaisesRegex(ValueError, "only one mouse"):
            evaluate_mouse_objective(sessions, _raw(), MOMENTS)


if __name__ == "__main__":
    unittest.main()
