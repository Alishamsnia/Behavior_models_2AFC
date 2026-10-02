"""Sliding-window and session-fine-tuning experiment for M1_stimulus_kernel."""

from .analysis import (
    FIGURE_ANALYSIS_SCHEMA_VERSION,
    figure_outputs_are_current,
    figure_ready,
    generate_global_prediction_figures,
    generate_m1_stimulus_kernel_experiment_figures,
)

from .experiment import (
    BlockPartition,
    FULL_SESSION_MATCHED_WORKFLOW,
    LEGACY_PARTITIONED_WORKFLOW,
    M1_STIMULUS_KERNEL_PARAMETER_NAMES,
    PreparedSlidingData,
    SlidingWindow,
    WORKFLOW_CHOICES,
    aggregate_m1_stimulus_kernel_session_initializations,
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
    "FULL_SESSION_MATCHED_WORKFLOW",
    "LEGACY_PARTITIONED_WORKFLOW",
    "M1_STIMULUS_KERNEL_PARAMETER_NAMES",
    "PreparedSlidingData",
    "SlidingWindow",
    "WORKFLOW_CHOICES",
    "aggregate_m1_stimulus_kernel_session_initializations",
    "build_sliding_windows",
    "clamp_fine_tune_initialization",
    "figure_outputs_are_current",
    "figure_ready",
    "generate_global_prediction_figures",
    "generate_m1_stimulus_kernel_experiment_figures",
    "match_window_winners_to_test_sessions",
    "partition_final_two_blocks",
    "partition_session_blocks",
    "prediction_performance",
    "prepare_sliding_data",
]
