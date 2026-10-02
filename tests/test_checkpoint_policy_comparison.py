from types import SimpleNamespace

import torch

from post_analysis.checkpoint_policy_comparison.analysis import select_checkpoints


def _encoder(values, *, lapse_max, epsilon):
    return torch.as_tensor(values, dtype=torch.float64)


def test_validation_policies_use_history_order_and_fixed_tolerance():
    spec = SimpleNamespace(encoder="tests.test_checkpoint_policy_comparison:_encoder")
    registry = SimpleNamespace(lapse_max=1.0, parameter_epsilon=1e-10)
    history = [
        {"optimizer_phase": "lbfgs", "fresh_training_mean_nll": .5, "validation_mean_nll": .4010, "constrained_parameters": [1.]},
        {"optimizer_phase": "lbfgs", "fresh_training_mean_nll": .4, "validation_mean_nll": .4005, "constrained_parameters": [2.]},
        {"optimizer_phase": "lbfgs", "fresh_training_mean_nll": .3, "validation_mean_nll": .4000, "constrained_parameters": [3.]},
    ]
    selected = select_checkpoints(history, spec, registry)
    assert selected["validation_min"]["ordinal"] == 2
    assert selected["validation_conservative"]["ordinal"] == 0


def test_non_lbfgs_and_nonfinite_records_are_excluded():
    spec = SimpleNamespace(encoder="tests.test_checkpoint_policy_comparison:_encoder")
    registry = SimpleNamespace(lapse_max=1.0, parameter_epsilon=1e-10)
    history = [
        {"optimizer_phase": "adam", "fresh_training_mean_nll": .1, "validation_mean_nll": .1, "constrained_parameters": [1.]},
        {"optimizer_phase": "lbfgs", "fresh_training_mean_nll": .2, "validation_mean_nll": float("nan"), "constrained_parameters": [2.]},
        {"optimizer_phase": "lbfgs", "fresh_training_mean_nll": .3, "validation_mean_nll": .4, "constrained_parameters": [3.]},
    ]
    selected = select_checkpoints(history, spec, registry)
    assert selected["validation_min"]["ordinal"] == 2
