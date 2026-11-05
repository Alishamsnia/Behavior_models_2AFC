"""M1_previous_wsls_bias_v1 (sensory_wslsSingle_lapse): M0's unchanged
Bayesian sensory readout with its fixed neutral prior, mixed before lapse
with the last feedback-valid true category (win-stay/lose-shift),
``m_t=(1-eta)r_t+eta*pi_t`` (alpha fixed at 1).
``M2_dynamic_previous_wsls_bias_v1`` without the HSMM. See
``docs/models/M1_previous_wsls_bias_v1.md`` for the full scientific
contract; the sibling ``m1_previous_action_bias_v1`` differs only in the
bias state's driver (the mouse's own last valid choice).
"""

from .analysis import evaluate_m1_previous_wsls_bias_split_metrics
from .fit import (
    M1PreviousWslsBiasEpochRecord,
    M1PreviousWslsBiasFitRunResult,
    M1PreviousWslsBiasMultiStartFitResult,
    fit_m1_previous_wsls_bias_all_initializations,
    fit_m1_previous_wsls_bias_single_initialization,
)
from .initialization import (
    M1PreviousWslsBiasInitializationSpec,
    validate_m1_previous_wsls_bias_initialization_table,
)
from .parameters import (
    M1PreviousWslsBiasDecodedParameters,
    M1_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER,
    decode_m1_previous_wsls_bias_raw_parameters,
    encode_m1_previous_wsls_bias_constrained_parameters,
)
from .pipeline import (
    M1PreviousWslsBiasPreparedFitData,
    prepare_registered_m1_previous_wsls_bias_fit_data,
)
from .registry import M1PreviousWslsBiasRegistry, load_m1_previous_wsls_bias_model_registry
from .session import (
    M1SessionTrials,
    M1PreviousWslsBiasMouseObjectiveResult,
    M1PreviousWslsBiasPredictions,
    M1PreviousWslsBiasSessionEvaluation,
    evaluate_m1_previous_wsls_bias_mouse_objective,
    evaluate_m1_previous_wsls_bias_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "M1_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER",
    "decode_m1_previous_wsls_bias_raw_parameters",
    "encode_m1_previous_wsls_bias_constrained_parameters",
    "M1PreviousWslsBiasDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1PreviousWslsBiasMouseObjectiveResult",
    "M1PreviousWslsBiasPredictions",
    "M1PreviousWslsBiasSessionEvaluation",
    "evaluate_m1_previous_wsls_bias_mouse_objective",
    "evaluate_m1_previous_wsls_bias_session",
    "M1PreviousWslsBiasRegistry",
    "load_m1_previous_wsls_bias_model_registry",
    "M1PreviousWslsBiasPreparedFitData",
    "prepare_registered_m1_previous_wsls_bias_fit_data",
    "M1PreviousWslsBiasInitializationSpec",
    "validate_m1_previous_wsls_bias_initialization_table",
    "M1PreviousWslsBiasEpochRecord",
    "M1PreviousWslsBiasFitRunResult",
    "M1PreviousWslsBiasMultiStartFitResult",
    "fit_m1_previous_wsls_bias_all_initializations",
    "fit_m1_previous_wsls_bias_single_initialization",
    "evaluate_m1_previous_wsls_bias_split_metrics",
]
