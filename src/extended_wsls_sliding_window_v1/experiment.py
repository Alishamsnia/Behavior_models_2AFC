"""Extended_WSLS-specific contracts for the isolated chronological sliding
experiment. The generic windowing mechanics are unchanged and reused from
``m0_sliding_window_v1.experiment``, exactly as
``m1_stimulus_kernel_sliding_window_v1.experiment`` reuses them.
"""

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


EXTENDED_WSLS_PARAMETER_NAMES = ("ell_short", "ell_long", "a")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability-scale Extended_WSLS means must lie strictly inside (0,1)")
    return float(np.log(probability) - np.log1p(-probability))


def aggregate_extended_wsls_session_initializations(
    assignments: pd.DataFrame,
    windows: Sequence[SlidingWindow],
    winner_parameters: Mapping[int, Sequence[float]],
) -> pd.DataFrame:
    """Average Extended_WSLS winners for session fine-tuning initialization.

    Only used by the legacy_partitioned workflow -- the full_session_matched
    workflow uses the generic ``match_window_winners_to_test_sessions``
    directly instead. The two lapse rates are averaged arithmetically; the
    learning-rate logit ``a`` is averaged in probability space -- mean
    ``alpha=sigmoid(a)`` across contributing windows, then converted back
    via ``logit`` -- the same convention
    ``m1_stimulus_kernel_sliding_window_v1`` uses, rather than a naive
    arithmetic mean of the raw logit (which does not commute with the
    sigmoid nonlinearity).
    """

    if not windows:
        raise ValueError("At least one completed Extended_WSLS window is required")
    if set(winner_parameters) != {window.window_id for window in windows}:
        raise ValueError("Every Extended_WSLS window requires one winning parameter vector")
    final_window = windows[-1]
    final_values = np.asarray(
        winner_parameters[final_window.window_id], dtype=float
    )
    if final_values.shape != (3,) or not np.all(np.isfinite(final_values)):
        raise ValueError("Extended_WSLS window winners must contain three finite parameters")

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
        if matrix.shape != (len(used), 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Every Extended_WSLS winner must contain three finite parameters")

        mean = matrix.mean(axis=0)
        mean_alpha = float(_sigmoid(matrix[:, 2]).mean())
        mean[2] = _logit(mean_alpha)
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
                else "mean_lapse_rates_probability_space_alpha"
            ),
        }
        for index, name in enumerate(EXTENDED_WSLS_PARAMETER_NAMES):
            row[f"initial_{name}"] = float(mean[index])
            row[f"window_sd_{name}"] = float(coordinate_sd[index])
            row[f"window_min_{name}"] = float(matrix[:, index].min())
            row[f"window_max_{name}"] = float(matrix[:, index].max())
        row["initial_alpha"] = mean_alpha
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "BlockPartition",
    "EXTENDED_WSLS_PARAMETER_NAMES",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "LEGACY_PARTITIONED_WORKFLOW",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_extended_wsls_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
