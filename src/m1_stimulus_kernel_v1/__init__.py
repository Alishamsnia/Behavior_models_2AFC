"""M1_stimulus_kernel_v1: M1's unchanged Bayesian sensory readout, evaluated
with a scalar delta-rule prior (a Rescorla-Wagner-style stimulus-history
kernel) instead of M1's hazard-driven three-state block belief. See
``docs/models/M1_stimulus_kernel_v1.md`` for the full scientific contract.
"""

from .analysis import evaluate_m1_stimulus_kernel_split_metrics
from .fit import (
    M1StimulusKernelEpochRecord,
    M1StimulusKernelFitRunResult,
    M1StimulusKernelMultiStartFitResult,
    fit_m1_stimulus_kernel_all_initializations,
    fit_m1_stimulus_kernel_single_initialization,
)
from .initialization import (
    M1StimulusKernelInitializationSpec,
    validate_m1_stimulus_kernel_initialization_table,
)
from .parameters import (
    M1StimulusKernelDecodedParameters,
    M1_STIMULUS_KERNEL_PARAMETER_ORDER,
    decode_m1_stimulus_kernel_raw_parameters,
    encode_m1_stimulus_kernel_constrained_parameters,
)
from .pipeline import (
    M1StimulusKernelPreparedFitData,
    prepare_registered_m1_stimulus_kernel_fit_data,
)
from .registry import M1StimulusKernelRegistry, load_m1_stimulus_kernel_model_registry
from .session import (
    M1SessionTrials,
    M1StimulusKernelMouseObjectiveResult,
    M1StimulusKernelPredictions,
    M1StimulusKernelSessionEvaluation,
    evaluate_m1_stimulus_kernel_mouse_objective,
    evaluate_m1_stimulus_kernel_session,
    m1_session_trials_from_frame,
)

__all__ = [
    "M1_STIMULUS_KERNEL_PARAMETER_ORDER",
    "decode_m1_stimulus_kernel_raw_parameters",
    "encode_m1_stimulus_kernel_constrained_parameters",
    "M1StimulusKernelDecodedParameters",
    "M1SessionTrials",
    "m1_session_trials_from_frame",
    "M1StimulusKernelMouseObjectiveResult",
    "M1StimulusKernelPredictions",
    "M1StimulusKernelSessionEvaluation",
    "evaluate_m1_stimulus_kernel_mouse_objective",
    "evaluate_m1_stimulus_kernel_session",
    "M1StimulusKernelRegistry",
    "load_m1_stimulus_kernel_model_registry",
    "M1StimulusKernelPreparedFitData",
    "prepare_registered_m1_stimulus_kernel_fit_data",
    "M1StimulusKernelInitializationSpec",
    "validate_m1_stimulus_kernel_initialization_table",
    "M1StimulusKernelEpochRecord",
    "M1StimulusKernelFitRunResult",
    "M1StimulusKernelMultiStartFitResult",
    "fit_m1_stimulus_kernel_all_initializations",
    "fit_m1_stimulus_kernel_single_initialization",
    "evaluate_m1_stimulus_kernel_split_metrics",
]
