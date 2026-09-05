"""Synthetic hazard and M0/M1 recovery workflows for M1 acceptance testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .fit import FitRunResult, fit_single_initialization
from .m1_initialization import M1InitializationSpec
from .m1_parameters import decode_m1_raw_parameters, encode_m1_constrained_parameters
from .m1_session import M1SessionTrials, constant_hazard_transition_matrix, evaluate_m1_session


@dataclass(frozen=True)
class M1HazardRecoveryResult:
    """Known and recovered M1 values for one synthetic hazard condition."""

    generating_parameters: tuple[float, ...]
    recovered_parameters: tuple[float, ...]
    generating_hazard: float
    recovered_hazard: float
    absolute_hazard_error: float
    generating_mean_duration: float
    recovered_mean_duration: float
    initial_training_mean_nll: float
    recovered_training_mean_nll: float
    validation_mean_nll: float
    fit_result: FitRunResult


def simulate_m1_hmm_sessions(
    durations_by_session: Iterable[torch.Tensor],
    constrained_parameters: torch.Tensor,
    moments,
    *,
    random_seed: int,
    split_role: str = "train",
    mouse_id: str = "synthetic_m1_mouse",
    first_long_probability: float = 0.5,
) -> tuple[tuple[M1SessionTrials, ...], tuple[torch.Tensor, ...]]:
    """Generate latent states, categories, and choices from the registered M1."""

    durations = tuple(durations_by_session)
    if not durations or constrained_parameters.shape != (6,):
        raise ValueError("M1 simulation requires sessions and a six-vector")
    if split_role not in {"train", "validation"}:
        raise ValueError("Synthetic M1 split must be train or validation")
    parameters = decode_m1_raw_parameters(
        encode_m1_constrained_parameters(constrained_parameters)
    )
    transition = constant_hazard_transition_matrix(
        parameters.hazard, first_long_probability=first_long_probability
    )
    theta = torch.tensor([0.5, 0.2, 0.8], dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    sessions: list[M1SessionTrials] = []
    truths: list[torch.Tensor] = []
    for session_index, duration in enumerate(durations, start=1):
        if duration.ndim != 1 or duration.dtype != torch.float64 or duration.numel() == 0:
            raise TypeError("Synthetic M1 durations must be nonempty 1D float64")
        state = 0
        states: list[int] = []
        categories: list[float] = []
        for trial in range(duration.numel()):
            states.append(state)
            categories.append(
                float(torch.rand((), generator=generator) < theta[state])
            )
            if trial + 1 < duration.numel():
                state = int(torch.multinomial(transition[state], 1, generator=generator))
        category = torch.tensor(categories, dtype=torch.float64)
        template = M1SessionTrials(
            mouse_id=mouse_id,
            session_id=f"{split_role}_{session_index:03d}",
            duration_s=duration.clone(),
            choice_long=torch.zeros_like(duration),
            category_long=category,
            feedback_valid=torch.ones(duration.numel(), dtype=torch.bool),
            split_role=split_role,
        )
        with torch.no_grad():
            prediction = evaluate_m1_session(
                template, parameters, moments,
                first_long_probability=first_long_probability,
            ).predictions.probability_long
            choice = (torch.rand(duration.shape, generator=generator) < prediction).to(torch.float64)
        sessions.append(M1SessionTrials(
            mouse_id=mouse_id,
            session_id=template.session_id,
            duration_s=duration.clone(),
            choice_long=choice,
            category_long=category,
            feedback_valid=template.feedback_valid.clone(),
            split_role=split_role,
        ))
        truths.append(torch.tensor(states, dtype=torch.int64))
    return tuple(sessions), tuple(truths)


def run_m1_hazard_recovery_case(
    generating_parameters: torch.Tensor,
    training_durations: Iterable[torch.Tensor],
    validation_durations: Iterable[torch.Tensor],
    registry,
    initialization: M1InitializationSpec,
    *,
    random_seed: int,
    maximum_epochs: int,
) -> M1HazardRecoveryResult:
    """Simulate one M1 hazard condition and refit all five pooled parameters."""

    if maximum_epochs <= 0:
        raise ValueError("maximum_epochs must be positive")
    training, _ = simulate_m1_hmm_sessions(
        training_durations, generating_parameters, registry.task_moments,
        random_seed=random_seed, split_role="train",
        first_long_probability=registry.first_long_probability,
    )
    validation, _ = simulate_m1_hmm_sessions(
        validation_durations, generating_parameters, registry.task_moments,
        random_seed=random_seed + 1, split_role="validation",
        first_long_probability=registry.first_long_probability,
    )
    fit = fit_single_initialization(
        training, validation, registry, initialization,
        max_epochs_override=maximum_epochs,
    )
    recovered = torch.tensor(fit.best_constrained_parameters, dtype=torch.float64)
    generated_h = float(torch.sigmoid(generating_parameters[4]))
    recovered_h = float(torch.sigmoid(recovered[4]))
    return M1HazardRecoveryResult(
        generating_parameters=tuple(float(x) for x in generating_parameters),
        recovered_parameters=tuple(float(x) for x in recovered),
        generating_hazard=generated_h,
        recovered_hazard=recovered_h,
        absolute_hazard_error=abs(recovered_h - generated_h),
        generating_mean_duration=1.0 / generated_h,
        recovered_mean_duration=1.0 / recovered_h,
        initial_training_mean_nll=fit.initial_training_mean_nll,
        recovered_training_mean_nll=fit.recalculated_best_training_mean_nll,
        validation_mean_nll=fit.final_validation_mean_nll,
        fit_result=fit,
    )
