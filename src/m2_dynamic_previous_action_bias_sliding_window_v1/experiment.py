"""M2_dynamic_previous_action_bias-specific contracts for the isolated
chronological sliding experiment. The generic windowing mechanics are
unchanged and reused from ``m0_sliding_window_v1.experiment``.

Scoped to the ``full_session_matched`` workflow only -- no ``--workflow``
option, no legacy block-partition/window-aggregation path -- matching
``m1_action_kernel_sliding_window_v1``'s own restriction, for the same
reason: the action-history state ``pi_t`` (and the bias-mixing weight
``eta`` that reads it) has no natural "average across several rolling
windows" story the way M2_dynamic_block_length_v1's own hazard/prototype
coordinates do. Window winners therefore map 1:1 to their following test
session via the generic ``match_window_winners_to_test_sessions`` rather
than being averaged.
"""

from __future__ import annotations

from m0_sliding_window_v1.experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    PreparedSlidingData,
    SlidingWindow,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    prediction_performance,
    prepare_full_session_matched_data as prepare_sliding_data,
)


M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit",
)


__all__ = [
    "BlockPartition",
    "FULL_SESSION_MATCHED_WORKFLOW",
    "M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
