"""Sliding-window and session-fine-tuning experiment for Extended_WSLS."""

from .analysis import (
    FIGURE_ANALYSIS_SCHEMA_VERSION,
    figure_outputs_are_current,
    figure_ready,
    generate_extended_wsls_experiment_figures,
    generate_global_prediction_figures,
)

from .experiment import (
    BlockPartition,
    EXTENDED_WSLS_PARAMETER_NAMES,
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    PreparedSlidingData,
    SlidingWindow,
    WORKFLOW_CHOICES,
    aggregate_extended_wsls_session_initializations,
    build_sliding_windows,
    clamp_fine_tune_initialization,
    match_window_winners_to_test_sessions,
    partition_final_two_blocks,
    partition_session_blocks,
    prediction_performance,
    prepare_sliding_data,
)

__all__ = [
    "FIGURE_ANALYSIS_SCHEMA_VERSION",
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
    "figure_outputs_are_current",
    "figure_ready",
    "generate_extended_wsls_experiment_figures",
    "generate_global_prediction_figures",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
