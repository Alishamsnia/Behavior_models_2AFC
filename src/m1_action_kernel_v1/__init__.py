"""M1_action_kernel_v1: M1's unchanged Bayesian sensory readout, evaluated
with a scalar delta-rule prior (a Rescorla-Wagner-style choice-history/
perseveration kernel driven by the mouse's own actions, not the true
category) instead of M1's hazard-driven three-state block belief. See
``docs/models/M1_action_kernel_v1.md`` for the full scientific contract and
its explicit comparison with the sibling ``M1_stimulus_kernel_v1``.
"""

from .analysis import evaluate_m1_action_kernel_split_metrics
from .fit import (
    M1ActionKernelEpochRecord,
    M1ActionKernelFitRunResult,
    M1ActionKernelMultiStartFitResult,
    fit_m1_action_kernel_all_initializations,
    fit_m1_action_kernel_single_initialization,
)
from .initialization import (
    M1ActionKernelInitializationSpec,
    validate_m1_action_kernel_initialization_table,
)
from .parameters import (
    M1ActionKernelDecodedParameters,
    M1_ACTION_KERNEL_PARAMETER_ORDER,
    decode_m1_action_kernel_raw_parameters,
    encode_m1_action_kernel_constrained_parameters,
)
from .pipeline import (
    M1ActionKernelPreparedFitData,
    prepare_registered_m1_action_kernel_fit_data,
)
from .registry import M1ActionKernelRegistry, load_m1_action_kernel_model_registry
from .session import (
    M1SessionTrials,
    M1ActionKernelMouseObjectiveResult,
    M1ActionKernelPredictions,
    M1ActionKernelSessionEvaluation,
    evaluate_m1_action_kernel_mouse_objective,
    evaluate_m1_action_kernel_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "M1_ACTION_KERNEL_PARAMETER_ORDER",
    "decode_m1_action_kernel_raw_parameters",
    "encode_m1_action_kernel_constrained_parameters",
    "M1ActionKernelDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1ActionKernelMouseObjectiveResult",
    "M1ActionKernelPredictions",
    "M1ActionKernelSessionEvaluation",
    "evaluate_m1_action_kernel_mouse_objective",
    "evaluate_m1_action_kernel_session",
    "M1ActionKernelRegistry",
    "load_m1_action_kernel_model_registry",
    "M1ActionKernelPreparedFitData",
    "prepare_registered_m1_action_kernel_fit_data",
    "M1ActionKernelInitializationSpec",
    "validate_m1_action_kernel_initialization_table",
    "M1ActionKernelEpochRecord",
    "M1ActionKernelFitRunResult",
    "M1ActionKernelMultiStartFitResult",
    "fit_m1_action_kernel_all_initializations",
    "fit_m1_action_kernel_single_initialization",
    "evaluate_m1_action_kernel_split_metrics",
]
