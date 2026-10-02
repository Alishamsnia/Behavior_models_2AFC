"""M1_stimulus_kernel-specific contracts for the isolated chronological
sliding experiment. The generic windowing mechanics are unchanged and reused
from ``m0_sliding_window_v1.experiment``."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from m0_sliding_window_v1.experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    PreparedSlidingData,
    SlidingWindow,
    WORKFLOW_CHOICES,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    prediction_performance,
    prepare_sliding_data,
)


M1_STIMULUS_KERNEL_PARAMETER_NAMES = ("w", "s0", "ell_short", "ell_long", "kappa", "a")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale M1_stimulus_kernel means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_m1_stimulus_kernel_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average M1_stimulus_kernel winners for session fine-tuning initialization.

    Only used by the legacy_partitioned workflow -- the full_session_matched
    workflow uses the generic ``match_window_winners_to_test_sessions``
    directly instead. The five shared sensory/lapse coordinates are averaged
    on their constrained scales exactly as
    ``m1_known_switch_sliding_window_v1.aggregate_m1_known_switch_session_initializations``
    does (arithmetic mean, geometric mean for the positive-scale ``kappa``).
    The learning-rate logit ``a`` is averaged in probability space -- mean
    ``alpha=sigmoid(a)`` across contributing windows, then converted back via
    ``logit`` -- the same convention
    ``m1_dynamic_block_length_sliding_window_v1`` uses for its hazard logit
    ``ah``, rather than a naive arithmetic mean of the raw logit (which does
    not commute with the sigmoid nonlinearity and would understate the mean
    rate whenever contributing values straddle zero asymmetrically).
    """

    if not windows:
        raise ValueError("At least one completed M1_stimulus_kernel window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every M1_stimulus_kernel window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (6,) or not np.all(np.isfinite(final_values)):
        raise ValueError("M1_stimulus_kernel window winners must contain six finite parameters")

    rows: list[dict[str, object]] = []
    for assignment in assignments.sort_values("chronological_rank").itertuples(
        index=False
    ):
        session_id = str(assignment.session_id)
        contributors = [
            window.window_id
            for window in windows
            if session_id in window.training_session_ids
        ]
        fallback = not contributors
        used = contributors or [final_window.window_id]
        matrix = np.asarray([winner_parameters[item] for item in used], dtype=float)
        if matrix.shape != (len(used), 6) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every M1_stimulus_kernel winner must contain six finite parameters")
        if np.any(matrix[:, 4] <= 0.0):
            raise ValueError("Every M1_stimulus_kernel winner requires positive kappa")

        mean = matrix.mean(axis=0)
        mean[4] = np.exp(np.log(matrix[:, 4]).mean())
        mean_alpha = float(_sigmoid(matrix[:, 5]).mean())
        mean[5] = _logit(mean_alpha)
        coordinate_sd = matrix.std(axis=0, ddof=0)
        row: dict[str, object] = {
            "mouse_id": str(assignment.mouse_id),
            "session_id": session_id,
            "chronological_rank": int(assignment.chronological_rank),
            "contributing_window_ids": ";".join(str(item) for item in used),
            "n_contributing_windows": len(used),
            "initialization_policy": (
                "final_window_winner_fallback_no_training_membership"
                if fallback
                else "mean_constrained_parameters_geometric_kappa_probability_space_alpha"
            ),
        }
        for index, name in enumerate(M1_STIMULUS_KERNEL_PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(coordinate_sd[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        row["initial_alpha"] = mean_alpha
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "LEGACY_PARTITIONED_WORKFLOW",
    "M1_STIMULUS_KERNEL_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_m1_stimulus_kernel_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
