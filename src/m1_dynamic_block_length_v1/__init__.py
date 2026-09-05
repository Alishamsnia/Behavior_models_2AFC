"""M1_dynamic_block_length_v1: M1's fixed true-rate prototypes plus a
duration-dependent (capped-tail HSMM) hazard. Independent of
``m4_hsmm_delta_v1``.
"""

from .analysis import (
    M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES,
    build_m1_dynamic_block_length_hazard_curve,
    build_m1_dynamic_block_length_trial_predictions,
    save_m1_dynamic_block_length_analysis_outputs,
)
from .diagnostics import (
    M1DynamicBlockLengthHazardDurationSummary,
    build_m1_dynamic_block_length_implied_duration_distribution,
    evaluate_m1_dynamic_block_length_truncation_sensitivity,
    summarize_m1_dynamic_block_length_hazard_duration,
    summarize_m1_dynamic_block_length_prediction_diagnostics,
)
from .figures import (
    generate_m1_dynamic_block_length_diagnostic_figures,
    generate_m1_dynamic_block_length_duration_figures,
)
from .fit import (
    M1DynamicBlockLengthFitRunResult,
    M1DynamicBlockLengthMultiStartFitResult,
    fit_m1_dynamic_block_length_all_initializations,
    fit_m1_dynamic_block_length_single_initialization,
)
from .identifiability import (
    evaluate_m1_dynamic_block_length_dynamic_surface,
    profile_m1_dynamic_block_length_dynamic_surface,
)
from .initialization import (
    M1DynamicBlockLengthInitializationSpec,
    validate_m1_dynamic_block_length_initialization_table,
)
from .m0_style_figures import generate_m1_dynamic_block_length_m0_style_figures
from .parameters import (
    M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER,
    M1DynamicBlockLengthParameters,
    decode_m1_dynamic_block_length_raw_parameters,
    encode_m1_dynamic_block_length_constrained_parameters,
)
from .pipeline import prepare_registered_m1_dynamic_block_length_fit_data
from .recovery import (
    M1DynamicBlockLengthRecoveryResult,
    iter_registered_m1_dynamic_block_length_recovery_grid,
    run_m1_dynamic_block_length_design_preserving_recovery,
)
from .registry import M1DynamicBlockLengthRegistry, load_m1_dynamic_block_length_model_registry
from .session import (
    M1DynamicBlockLengthMouseObjectiveResult,
    M1DynamicBlockLengthPredictions,
    M1DynamicBlockLengthSessionEvaluation,
    M1DynamicBlockLengthSessionTrials,
    evaluate_m1_dynamic_block_length_mouse_objective,
    evaluate_m1_dynamic_block_length_session,
    m1_dynamic_block_length_session_trials_from_frame,
)
from .simulation import simulate_m1_dynamic_block_length_choices

__all__ = [
    "M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER",
    "M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES",
    "M1DynamicBlockLengthParameters",
    "decode_m1_dynamic_block_length_raw_parameters",
    "encode_m1_dynamic_block_length_constrained_parameters",
    "M1DynamicBlockLengthRegistry",
    "load_m1_dynamic_block_length_model_registry",
    "M1DynamicBlockLengthInitializationSpec",
    "validate_m1_dynamic_block_length_initialization_table",
    "prepare_registered_m1_dynamic_block_length_fit_data",
    "M1DynamicBlockLengthMouseObjectiveResult",
    "M1DynamicBlockLengthPredictions",
    "M1DynamicBlockLengthSessionEvaluation",
    "M1DynamicBlockLengthSessionTrials",
    "evaluate_m1_dynamic_block_length_mouse_objective",
    "evaluate_m1_dynamic_block_length_session",
    "m1_dynamic_block_length_session_trials_from_frame",
    "M1DynamicBlockLengthFitRunResult",
    "M1DynamicBlockLengthMultiStartFitResult",
    "fit_m1_dynamic_block_length_all_initializations",
    "fit_m1_dynamic_block_length_single_initialization",
    "simulate_m1_dynamic_block_length_choices",
    "M1DynamicBlockLengthRecoveryResult",
    "iter_registered_m1_dynamic_block_length_recovery_grid",
    "run_m1_dynamic_block_length_design_preserving_recovery",
    "build_m1_dynamic_block_length_hazard_curve",
    "build_m1_dynamic_block_length_trial_predictions",
    "save_m1_dynamic_block_length_analysis_outputs",
    "M1DynamicBlockLengthHazardDurationSummary",
    "build_m1_dynamic_block_length_implied_duration_distribution",
    "evaluate_m1_dynamic_block_length_truncation_sensitivity",
    "summarize_m1_dynamic_block_length_hazard_duration",
    "summarize_m1_dynamic_block_length_prediction_diagnostics",
    "generate_m1_dynamic_block_length_diagnostic_figures",
    "generate_m1_dynamic_block_length_duration_figures",
    "generate_m1_dynamic_block_length_m0_style_figures",
    "evaluate_m1_dynamic_block_length_dynamic_surface",
    "profile_m1_dynamic_block_length_dynamic_surface",
]
