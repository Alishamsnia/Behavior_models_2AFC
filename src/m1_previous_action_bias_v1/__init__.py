"""M1_previous_action_bias_v1 (sensory_actionSingle_lapse): M0's unchanged
Bayesian sensory readout with its fixed neutral prior, mixed before lapse
with the mouse's own last valid choice, ``m_t=(1-eta)r_t+eta*pi_t`` (alpha
fixed at 1). ``M2_dynamic_previous_action_bias_v1`` without the HSMM. See
``docs/models/M1_previous_action_bias_v1.md`` for the full scientific
contract; the sibling ``m1_previous_wsls_bias_v1`` differs only in the bias
state's driver (last feedback-valid category).
"""

from .analysis import evaluate_m1_previous_action_bias_split_metrics
from .fit import (
    M1PreviousActionBiasEpochRecord,
    M1PreviousActionBiasFitRunResult,
    M1PreviousActionBiasMultiStartFitResult,
    fit_m1_previous_action_bias_all_initializations,
    fit_m1_previous_action_bias_single_initialization,
)
from .initialization import (
    M1PreviousActionBiasInitializationSpec,
    validate_m1_previous_action_bias_initialization_table,
)
from .parameters import (
    M1PreviousActionBiasDecodedParameters,
    M1_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER,
    decode_m1_previous_action_bias_raw_parameters,
    encode_m1_previous_action_bias_constrained_parameters,
)
from .pipeline import (
    M1PreviousActionBiasPreparedFitData,
    prepare_registered_m1_previous_action_bias_fit_data,
)
from .registry import M1PreviousActionBiasRegistry, load_m1_previous_action_bias_model_registry
from .session import (
    M1SessionTrials,
    M1PreviousActionBiasMouseObjectiveResult,
    M1PreviousActionBiasPredictions,
    M1PreviousActionBiasSessionEvaluation,
    evaluate_m1_previous_action_bias_mouse_objective,
    evaluate_m1_previous_action_bias_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "M1_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER",
    "decode_m1_previous_action_bias_raw_parameters",
    "encode_m1_previous_action_bias_constrained_parameters",
    "M1PreviousActionBiasDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1PreviousActionBiasMouseObjectiveResult",
    "M1PreviousActionBiasPredictions",
    "M1PreviousActionBiasSessionEvaluation",
    "evaluate_m1_previous_action_bias_mouse_objective",
    "evaluate_m1_previous_action_bias_session",
    "M1PreviousActionBiasRegistry",
    "load_m1_previous_action_bias_model_registry",
    "M1PreviousActionBiasPreparedFitData",
    "prepare_registered_m1_previous_action_bias_fit_data",
    "M1PreviousActionBiasInitializationSpec",
    "validate_m1_previous_action_bias_initialization_table",
    "M1PreviousActionBiasEpochRecord",
    "M1PreviousActionBiasFitRunResult",
    "M1PreviousActionBiasMultiStartFitResult",
    "fit_m1_previous_action_bias_all_initializations",
    "fit_m1_previous_action_bias_single_initialization",
    "evaluate_m1_previous_action_bias_split_metrics",
]
