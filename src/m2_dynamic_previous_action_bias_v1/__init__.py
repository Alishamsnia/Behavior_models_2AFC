"""M2_dynamic_previous_action_bias_v1: M2_dynamic_block_length_v1's fitted
static prototypes and duration-dependent hazard, unchanged, plus a fitted
previous-action bias (the fixed alpha=1 boundary of the action-history
Rescorla-Wagner recursion -- the mouse's last valid action) mixed between
the sensory readout and lapse. See
``docs/models/M2_dynamic_previous_action_bias_v1.md`` for the full
scientific contract.
"""

from .analysis import (
    M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES,
    build_m2_dynamic_previous_action_bias_hazard_curve,
    build_m2_dynamic_previous_action_bias_trial_predictions,
    evaluate_m2_dynamic_previous_action_bias_bias_evidence_dependence,
    save_m2_dynamic_previous_action_bias_analysis_outputs,
)
from .diagnostics import (
    M2DynamicPreviousActionBiasHazardDurationSummary,
    build_m2_dynamic_previous_action_bias_implied_duration_distribution,
    evaluate_m2_dynamic_previous_action_bias_truncation_sensitivity,
    summarize_m2_dynamic_previous_action_bias_hazard_duration,
    summarize_m2_dynamic_previous_action_bias_prediction_diagnostics,
)
from .figures import (
    generate_m2_dynamic_previous_action_bias_diagnostic_figures,
    generate_m2_dynamic_previous_action_bias_duration_figures,
)
from .fit import (
    M2DynamicPreviousActionBiasBoundaryProximity,
    M2DynamicPreviousActionBiasFitRunResult,
    M2DynamicPreviousActionBiasFitStepRecord,
    M2DynamicPreviousActionBiasMultiStartFitResult,
    fit_m2_dynamic_previous_action_bias_all_initializations,
    fit_m2_dynamic_previous_action_bias_single_initialization,
)
from .identifiability import (
    evaluate_m2_dynamic_previous_action_bias_dynamic_surface,
    profile_m2_dynamic_previous_action_bias_dynamic_surface,
)
from .initialization import (
    M2DynamicPreviousActionBiasInitializationSpec,
    validate_m2_dynamic_previous_action_bias_initialization_table,
)
from .parameters import (
    M2_DYNAMIC_PREVIOUS_ACTION_BIAS_LEARNING_RATE,
    M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER,
    M2DynamicPreviousActionBiasParameters,
    decode_m2_dynamic_previous_action_bias_raw_parameters,
    encode_m2_dynamic_previous_action_bias_constrained_parameters,
)
from .m0_style_figures import generate_m2_dynamic_previous_action_bias_m0_style_figures
from .pipeline import prepare_registered_m2_dynamic_previous_action_bias_fit_data
from .recovery import (
    M2DynamicPreviousActionBiasRecoveryResult,
    iter_registered_m2_dynamic_previous_action_bias_recovery_grid,
    run_m2_dynamic_previous_action_bias_design_preserving_recovery,
)
from .registry import M2DynamicPreviousActionBiasRegistry, load_m2_dynamic_previous_action_bias_model_registry
from .session import (
    M2DynamicPreviousActionBiasMouseObjectiveResult,
    M2DynamicPreviousActionBiasPredictions,
    M2DynamicPreviousActionBiasSessionEvaluation,
    M2DynamicPreviousActionBiasSessionTrials,
    evaluate_m2_dynamic_previous_action_bias_mouse_objective,
    evaluate_m2_dynamic_previous_action_bias_session,
    m2_dynamic_previous_action_bias_session_trials_from_frame,
)
from .simulation import simulate_m2_dynamic_previous_action_bias_choices

__all__ = [
    "M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER",
    "M2_DYNAMIC_PREVIOUS_ACTION_BIAS_LEARNING_RATE",
    "M2DynamicPreviousActionBiasParameters",
    "decode_m2_dynamic_previous_action_bias_raw_parameters",
    "encode_m2_dynamic_previous_action_bias_constrained_parameters",
    "M2DynamicPreviousActionBiasRegistry",
    "load_m2_dynamic_previous_action_bias_model_registry",
    "M2DynamicPreviousActionBiasInitializationSpec",
    "validate_m2_dynamic_previous_action_bias_initialization_table",
    "prepare_registered_m2_dynamic_previous_action_bias_fit_data",
    "M2DynamicPreviousActionBiasMouseObjectiveResult",
    "M2DynamicPreviousActionBiasPredictions",
    "M2DynamicPreviousActionBiasSessionEvaluation",
    "M2DynamicPreviousActionBiasSessionTrials",
    "evaluate_m2_dynamic_previous_action_bias_mouse_objective",
    "evaluate_m2_dynamic_previous_action_bias_session",
    "m2_dynamic_previous_action_bias_session_trials_from_frame",
    "M2DynamicPreviousActionBiasFitStepRecord",
    "M2DynamicPreviousActionBiasBoundaryProximity",
    "M2DynamicPreviousActionBiasFitRunResult",
    "M2DynamicPreviousActionBiasMultiStartFitResult",
    "fit_m2_dynamic_previous_action_bias_all_initializations",
    "fit_m2_dynamic_previous_action_bias_single_initialization",
    "simulate_m2_dynamic_previous_action_bias_choices",
    "M2DynamicPreviousActionBiasRecoveryResult",
    "iter_registered_m2_dynamic_previous_action_bias_recovery_grid",
    "run_m2_dynamic_previous_action_bias_design_preserving_recovery",
    "M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_NAMES",
    "build_m2_dynamic_previous_action_bias_hazard_curve",
    "build_m2_dynamic_previous_action_bias_trial_predictions",
    "evaluate_m2_dynamic_previous_action_bias_bias_evidence_dependence",
    "save_m2_dynamic_previous_action_bias_analysis_outputs",
    "M2DynamicPreviousActionBiasHazardDurationSummary",
    "build_m2_dynamic_previous_action_bias_implied_duration_distribution",
    "evaluate_m2_dynamic_previous_action_bias_truncation_sensitivity",
    "summarize_m2_dynamic_previous_action_bias_hazard_duration",
    "summarize_m2_dynamic_previous_action_bias_prediction_diagnostics",
    "generate_m2_dynamic_previous_action_bias_diagnostic_figures",
    "generate_m2_dynamic_previous_action_bias_duration_figures",
    "generate_m2_dynamic_previous_action_bias_m0_style_figures",
    "evaluate_m2_dynamic_previous_action_bias_dynamic_surface",
    "profile_m2_dynamic_previous_action_bias_dynamic_surface",
]
