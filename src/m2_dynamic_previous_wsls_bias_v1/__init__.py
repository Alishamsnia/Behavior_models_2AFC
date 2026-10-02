"""M2_dynamic_previous_wsls_bias_v1: M2_dynamic_block_length_v1's fitted
static prototypes and duration-dependent hazard, unchanged, plus a fitted
previous-category (win-stay/lose-shift) bias (the fixed alpha=1 boundary of
the previous-category delta-rule recursion -- the true category on the last
feedback-valid trial, gated by ``feedback_valid`` rather than choice
validity) mixed between the sensory readout and lapse. See
``docs/models/M2_dynamic_previous_wsls_bias_v1.md`` for the full
scientific contract.
"""

from .analysis import (
    M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES,
    build_m2_dynamic_previous_wsls_bias_hazard_curve,
    build_m2_dynamic_previous_wsls_bias_trial_predictions,
    evaluate_m2_dynamic_previous_wsls_bias_bias_evidence_dependence,
    save_m2_dynamic_previous_wsls_bias_analysis_outputs,
)
from .diagnostics import (
    M2DynamicPreviousWslsBiasHazardDurationSummary,
    build_m2_dynamic_previous_wsls_bias_implied_duration_distribution,
    evaluate_m2_dynamic_previous_wsls_bias_truncation_sensitivity,
    summarize_m2_dynamic_previous_wsls_bias_hazard_duration,
    summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics,
)
from .figures import (
    generate_m2_dynamic_previous_wsls_bias_diagnostic_figures,
    generate_m2_dynamic_previous_wsls_bias_duration_figures,
)
from .fit import (
    M2DynamicPreviousWslsBiasBoundaryProximity,
    M2DynamicPreviousWslsBiasFitRunResult,
    M2DynamicPreviousWslsBiasFitStepRecord,
    M2DynamicPreviousWslsBiasMultiStartFitResult,
    fit_m2_dynamic_previous_wsls_bias_all_initializations,
    fit_m2_dynamic_previous_wsls_bias_single_initialization,
)
from .identifiability import (
    evaluate_m2_dynamic_previous_wsls_bias_dynamic_surface,
    profile_m2_dynamic_previous_wsls_bias_dynamic_surface,
)
from .initialization import (
    M2DynamicPreviousWslsBiasInitializationSpec,
    validate_m2_dynamic_previous_wsls_bias_initialization_table,
)
from .parameters import (
    M2_DYNAMIC_PREVIOUS_WSLS_BIAS_LEARNING_RATE,
    M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER,
    M2DynamicPreviousWslsBiasParameters,
    decode_m2_dynamic_previous_wsls_bias_raw_parameters,
    encode_m2_dynamic_previous_wsls_bias_constrained_parameters,
)
from .m0_style_figures import generate_m2_dynamic_previous_wsls_bias_m0_style_figures
from .pipeline import prepare_registered_m2_dynamic_previous_wsls_bias_fit_data
from .recovery import (
    M2DynamicPreviousWslsBiasRecoveryResult,
    iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid,
    run_m2_dynamic_previous_wsls_bias_design_preserving_recovery,
)
from .registry import M2DynamicPreviousWslsBiasRegistry, load_m2_dynamic_previous_wsls_bias_model_registry
from .session import (
    M2DynamicPreviousWslsBiasMouseObjectiveResult,
    M2DynamicPreviousWslsBiasPredictions,
    M2DynamicPreviousWslsBiasSessionEvaluation,
    M2DynamicPreviousWslsBiasSessionTrials,
    evaluate_m2_dynamic_previous_wsls_bias_mouse_objective,
    evaluate_m2_dynamic_previous_wsls_bias_session,
    m2_dynamic_previous_wsls_bias_session_trials_from_frame,
)
from .simulation import simulate_m2_dynamic_previous_wsls_bias_choices

__all__ = [
    "M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER",
    "M2_DYNAMIC_PREVIOUS_WSLS_BIAS_LEARNING_RATE",
    "M2DynamicPreviousWslsBiasParameters",
    "decode_m2_dynamic_previous_wsls_bias_raw_parameters",
    "encode_m2_dynamic_previous_wsls_bias_constrained_parameters",
    "M2DynamicPreviousWslsBiasRegistry",
    "load_m2_dynamic_previous_wsls_bias_model_registry",
    "M2DynamicPreviousWslsBiasInitializationSpec",
    "validate_m2_dynamic_previous_wsls_bias_initialization_table",
    "prepare_registered_m2_dynamic_previous_wsls_bias_fit_data",
    "M2DynamicPreviousWslsBiasMouseObjectiveResult",
    "M2DynamicPreviousWslsBiasPredictions",
    "M2DynamicPreviousWslsBiasSessionEvaluation",
    "M2DynamicPreviousWslsBiasSessionTrials",
    "evaluate_m2_dynamic_previous_wsls_bias_mouse_objective",
    "evaluate_m2_dynamic_previous_wsls_bias_session",
    "m2_dynamic_previous_wsls_bias_session_trials_from_frame",
    "M2DynamicPreviousWslsBiasFitStepRecord",
    "M2DynamicPreviousWslsBiasBoundaryProximity",
    "M2DynamicPreviousWslsBiasFitRunResult",
    "M2DynamicPreviousWslsBiasMultiStartFitResult",
    "fit_m2_dynamic_previous_wsls_bias_all_initializations",
    "fit_m2_dynamic_previous_wsls_bias_single_initialization",
    "simulate_m2_dynamic_previous_wsls_bias_choices",
    "M2DynamicPreviousWslsBiasRecoveryResult",
    "iter_registered_m2_dynamic_previous_wsls_bias_recovery_grid",
    "run_m2_dynamic_previous_wsls_bias_design_preserving_recovery",
    "M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_NAMES",
    "build_m2_dynamic_previous_wsls_bias_hazard_curve",
    "build_m2_dynamic_previous_wsls_bias_trial_predictions",
    "evaluate_m2_dynamic_previous_wsls_bias_bias_evidence_dependence",
    "save_m2_dynamic_previous_wsls_bias_analysis_outputs",
    "M2DynamicPreviousWslsBiasHazardDurationSummary",
    "build_m2_dynamic_previous_wsls_bias_implied_duration_distribution",
    "evaluate_m2_dynamic_previous_wsls_bias_truncation_sensitivity",
    "summarize_m2_dynamic_previous_wsls_bias_hazard_duration",
    "summarize_m2_dynamic_previous_wsls_bias_prediction_diagnostics",
    "generate_m2_dynamic_previous_wsls_bias_diagnostic_figures",
    "generate_m2_dynamic_previous_wsls_bias_duration_figures",
    "generate_m2_dynamic_previous_wsls_bias_m0_style_figures",
    "evaluate_m2_dynamic_previous_wsls_bias_dynamic_surface",
    "profile_m2_dynamic_previous_wsls_bias_dynamic_surface",
]
