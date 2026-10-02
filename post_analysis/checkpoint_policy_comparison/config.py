"""Frozen configuration for the MC08 checkpoint-policy comparison."""

from __future__ import annotations

from dataclasses import dataclass


VALIDATION_TOLERANCE_NLL = 0.001
NUMERICAL_TIE_TOLERANCE = 1e-12
PRACTICAL_TEST_TIE_NLL = 0.001
BOOTSTRAP_SEED = 20260921
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_BLOCK_LENGTH = 12
WORKFLOW = "full_session_matched_v1"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    directory: str
    label: str
    color: str
    loader: str | None
    predictor: str | None
    encoder: str | None = None
    oracle: bool = False


MODEL_SPECS = (
    ModelSpec("pure_wsls", "Pure_WSLS_v1", "WSLS single", "#7F7F7F", "pure_wsls_v1.registry:load_pure_wsls_model_registry", "pure_wsls_v1.analysis:build_pure_wsls_trial_predictions", "pure_wsls_v1.parameters:encode_pure_wsls_constrained_parameters"),
    ModelSpec("extended_wsls", "Extended_WSLS_v1", "WSLS history", "#4E79A7", "extended_wsls_v1.registry:load_extended_wsls_model_registry", "extended_wsls_v1.analysis:build_extended_wsls_trial_predictions", "extended_wsls_v1.parameters:encode_extended_wsls_constrained_parameters"),
    ModelSpec("m0_sensory", "M0_sensory_v1", "M0 sensory", "#59A14F", "m0_sensory_v1.registry:load_model_registry", "m0_sensory_v1.analysis:build_trial_predictions"),
    ModelSpec("m1_action_kernel", "M1_action_kernel_v1", "M1 action kernel", "#F28E2B", "m1_action_kernel_v1.registry:load_m1_action_kernel_model_registry", "m1_action_kernel_v1.analysis:build_m1_action_kernel_trial_predictions", "m1_action_kernel_v1.parameters:encode_m1_action_kernel_constrained_parameters"),
    ModelSpec("m1_stimulus_kernel", "M1_stimulus_kernel_v1", "M1 stimulus kernel", "#FFBE7D", None, None),
    ModelSpec("m1_known_switch", "M1_known_switch_v1", "M1 known switch", "#E15759", None, None, oracle=True),
    ModelSpec("m1_hmm_true", "M1_hmm_true_v1", "M1 HMM true", "#76B7B2", "m0_sensory_v1.m1_registry:load_m1_model_registry", "m0_sensory_v1.m1_analysis:build_m1_trial_predictions"),
    ModelSpec("m1_dynamic", "M1_dynamic_block_length_v1", "M1 dynamic", "#EDC948", "m1_dynamic_block_length_v1.registry:load_m1_dynamic_block_length_model_registry", "m1_dynamic_block_length_v1.analysis:build_m1_dynamic_block_length_trial_predictions"),
    ModelSpec("m2_known_switch", "M2_known_switch_v1", "M2 known switch", "#B07AA1", None, None, oracle=True),
    ModelSpec("m2_static", "M2_hmm_static_v1", "M2 static", "#FF9DA7", "m0_sensory_v1.m2_registry:load_m2_model_registry", "m0_sensory_v1.m2_analysis:build_m2_trial_predictions"),
    ModelSpec("m2_dynamic", "M2_dynamic_block_length_v1", "M2 dynamic", "#9C755F", "m2_dynamic_block_length_v1.registry:load_m2_dynamic_block_length_model_registry", "m2_dynamic_block_length_v1.analysis:build_m2_dynamic_block_length_trial_predictions"),
    ModelSpec("m2_dynamic_wsls_bias", "M2_dynamic_previous_wsls_bias_v1", "M2 dynamic + WSLS", "#D37295", "m2_dynamic_previous_wsls_bias_v1.registry:load_m2_dynamic_previous_wsls_bias_model_registry", "m2_dynamic_previous_wsls_bias_v1.analysis:build_m2_dynamic_previous_wsls_bias_trial_predictions"),
    ModelSpec("m2_dynamic_action_bias", "M2_dynamic_previous_action_bias_v1", "M2 dynamic + action", "#499894", "m2_dynamic_previous_action_bias_v1.registry:load_m2_dynamic_previous_action_bias_model_registry", "m2_dynamic_previous_action_bias_v1.analysis:build_m2_dynamic_previous_action_bias_trial_predictions"),
    ModelSpec("m3_hmm_delta", "M3_hmm_delta_v1", "M3 HMM delta", "#D55E00", "m3_hmm_delta_v1.registry:load_m3_model_registry", "m3_hmm_delta_v1.analysis:build_m3_trial_predictions"),
    ModelSpec("m4_hsmm_delta", "M4_hsmm_delta_v1", "M4 HSMM delta", "#CC79A7", "m4_hsmm_delta_v1.registry:load_m4_model_registry", "m4_hsmm_delta_v1.analysis:build_m4_trial_predictions"),
)

MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
POLICIES = ("current_train_mle", "validation_min", "validation_conservative")
