"""M2_dynamic_block_length_v1: M2's fitted static prototypes plus a
duration-dependent (capped-tail HSMM) hazard. Independent of
``m4_hsmm_delta_v1``.
"""

from .analysis import (
    M2_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES,
    build_m2_dynamic_block_length_hazard_curve,
    build_m2_dynamic_block_length_trial_predictions,
    save_m2_dynamic_block_length_analysis_outputs,
)
from .diagnostics import (
    M2DynamicBlockLengthHazardDurationSummary,
    build_m2_dynamic_block_length_implied_duration_distribution,
    evaluate_m2_dynamic_block_length_truncation_sensitivity,
    summarize_m2_dynamic_block_length_hazard_duration,
    summarize_m2_dynamic_block_length_prediction_diagnostics,
)
from .figures import (
    generate_m2_dynamic_block_length_diagnostic_figures,
    generate_m2_dynamic_block_length_duration_figures,
)
from .fit import (
    M2DynamicBlockLengthFitRunResult,
    M2DynamicBlockLengthMultiStartFitResult,
    fit_m2_dynamic_block_length_all_initializations,
    fit_m2_dynamic_block_length_single_initialization,
)
from .identifiability import (
    evaluate_m2_dynamic_block_length_dynamic_surface,
    profile_m2_dynamic_block_length_dynamic_surface,
)
from .initialization import (
    M2DynamicBlockLengthInitializationSpec,
    validate_m2_dynamic_block_length_initialization_table,
)
from .m0_style_figures import generate_m2_dynamic_block_length_m0_style_figures
from .parameters import (
    M2_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER,
    M2DynamicBlockLengthParameters,
    decode_m2_dynamic_block_length_raw_parameters,
    encode_m2_dynamic_block_length_constrained_parameters,
)
from .pipeline import prepare_registered_m2_dynamic_block_length_fit_data
from .recovery import (
    M2DynamicBlockLengthRecoveryResult,
    iter_registered_m2_dynamic_block_length_recovery_grid,
    run_m2_dynamic_block_length_design_preserving_recovery,
)
from .registry import M2DynamicBlockLengthRegistry, load_m2_dynamic_block_length_model_registry
from .session import (
    M2DynamicBlockLengthMouseObjectiveResult,
    M2DynamicBlockLengthPredictions,
    M2DynamicBlockLengthSessionEvaluation,
    M2DynamicBlockLengthSessionTrials,
    evaluate_m2_dynamic_block_length_mouse_objective,
    evaluate_m2_dynamic_block_length_session,
    m2_dynamic_block_length_session_trials_from_frame,
)
from .simulation import simulate_m2_dynamic_block_length_choices

__all__ = [
    "M2_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER",
    "M2_DYNAMIC_BLOCK_LENGTH_PARAMETER_NAMES",
    "M2DynamicBlockLengthParameters",
    "decode_m2_dynamic_block_length_raw_parameters",
    "encode_m2_dynamic_block_length_constrained_parameters",
    "M2DynamicBlockLengthRegistry",
    "load_m2_dynamic_block_length_model_registry",
    "M2DynamicBlockLengthInitializationSpec",
    "validate_m2_dynamic_block_length_initialization_table",
    "prepare_registered_m2_dynamic_block_length_fit_data",
    "M2DynamicBlockLengthMouseObjectiveResult",
    "M2DynamicBlockLengthPredictions",
    "M2DynamicBlockLengthSessionEvaluation",
    "M2DynamicBlockLengthSessionTrials",
    "evaluate_m2_dynamic_block_length_mouse_objective",
    "evaluate_m2_dynamic_block_length_session",
    "m2_dynamic_block_length_session_trials_from_frame",
    "M2DynamicBlockLengthFitRunResult",
    "M2DynamicBlockLengthMultiStartFitResult",
    "fit_m2_dynamic_block_length_all_initializations",
    "fit_m2_dynamic_block_length_single_initialization",
    "simulate_m2_dynamic_block_length_choices",
    "M2DynamicBlockLengthRecoveryResult",
    "iter_registered_m2_dynamic_block_length_recovery_grid",
    "run_m2_dynamic_block_length_design_preserving_recovery",
    "build_m2_dynamic_block_length_hazard_curve",
    "build_m2_dynamic_block_length_trial_predictions",
    "save_m2_dynamic_block_length_analysis_outputs",
    "M2DynamicBlockLengthHazardDurationSummary",
    "build_m2_dynamic_block_length_implied_duration_distribution",
    "evaluate_m2_dynamic_block_length_truncation_sensitivity",
    "summarize_m2_dynamic_block_length_hazard_duration",
    "summarize_m2_dynamic_block_length_prediction_diagnostics",
    "generate_m2_dynamic_block_length_diagnostic_figures",
    "generate_m2_dynamic_block_length_duration_figures",
    "generate_m2_dynamic_block_length_m0_style_figures",
    "evaluate_m2_dynamic_block_length_dynamic_surface",
    "profile_m2_dynamic_block_length_dynamic_surface",
]
