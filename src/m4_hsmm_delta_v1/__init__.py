"""Duration-aware capped-tail extension of ``M3_hmm_delta_v1``."""

from .parameters import M4DecodedParameters, decode_m4_raw_parameters, encode_m4_constrained_parameters
from .registry import M4HSMMRegistry, load_m4_model_registry
from .prior_state import (
    CappedTailHSMMPriorStateProvider,
    ConstantHazardPriorStateProvider,
    PriorStateProvider,
)
from .analysis import (
    M4M3ValidationComparison,
    build_m4_hazard_curve,
    build_m4_m3_validation_comparison,
    build_m4_trial_predictions,
)
from .diagnostics import (
    M4HazardDurationSummary,
    build_m4_implied_duration_distribution,
    evaluate_m4_truncation_sensitivity,
    summarize_m4_hazard_duration,
    summarize_m4_prediction_diagnostics,
)
from .fit import (
    M4FitRunResult,
    M4MultiStartFitResult,
    fit_m4_all_initializations,
    fit_m4_single_initialization,
)
from .identifiability import evaluate_m4_dynamic_surface, profile_m4_dynamic_surface
from .m0_style_figures import generate_m4_m0_style_figures
from .recovery import (
    M3M4ModelRecoveryResult,
    M4RecoveryResult,
    iter_registered_m4_recovery_grid,
    run_m3_m4_model_recovery,
    run_m4_design_preserving_recovery,
)
from .simulation import simulate_m4_choices
from .session import (
    M4MouseObjectiveResult,
    M4Predictions,
    M4SessionEvaluation,
    M4SessionTrials,
    capped_tail_hsmm_transition,
    evaluate_m4_mouse_objective,
    evaluate_m4_session,
)

__all__ = [
    "CappedTailHSMMPriorStateProvider", "ConstantHazardPriorStateProvider",
    "M3M4ModelRecoveryResult", "M4DecodedParameters", "M4FitRunResult",
    "M4HSMMRegistry", "M4HazardDurationSummary", "M4M3ValidationComparison",
    "M4MouseObjectiveResult", "M4MultiStartFitResult", "M4Predictions",
    "M4RecoveryResult", "PriorStateProvider",
    "M4SessionEvaluation", "M4SessionTrials", "capped_tail_hsmm_transition",
    "build_m4_hazard_curve", "build_m4_implied_duration_distribution",
    "build_m4_m3_validation_comparison", "build_m4_trial_predictions",
    "decode_m4_raw_parameters", "encode_m4_constrained_parameters",
    "evaluate_m4_dynamic_surface", "evaluate_m4_mouse_objective", "evaluate_m4_session",
    "evaluate_m4_truncation_sensitivity",
    "fit_m4_all_initializations", "fit_m4_single_initialization",
    "generate_m4_m0_style_figures",
    "iter_registered_m4_recovery_grid", "load_m4_model_registry",
    "profile_m4_dynamic_surface", "run_m3_m4_model_recovery",
    "run_m4_design_preserving_recovery", "simulate_m4_choices",
    "summarize_m4_hazard_duration", "summarize_m4_prediction_diagnostics",
]
