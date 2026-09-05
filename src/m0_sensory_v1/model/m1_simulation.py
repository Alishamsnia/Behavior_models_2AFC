"""Design-preserving choice simulation for M1 recovery checks."""

from __future__ import annotations

from typing import Iterable

import torch

from ..data.task_moments import TaskMoments
from .m1_parameters import M1DecodedParameters
from .m1_session import M1SessionTrials, evaluate_m1_session


def simulate_m1_choices(
    template_sessions: Iterable[M1SessionTrials],
    parameters: M1DecodedParameters,
    moments: TaskMoments,
    *,
    seed: int,
    first_long_probability: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> tuple[M1SessionTrials, ...]:
    """Simulate choices conditional on fixed stimulus/feedback timelines.

    Categories, omissions, session lengths, and feedback availability are held
    at their observed design values. Valid-choice locations receive Bernoulli
    draws from the causal M1 probability; all other choice entries remain NaN.
    This is the appropriate conditional simulation target for the outer choice
    likelihood and does not fit the task's category generator.
    """

    records = tuple(template_sessions)
    if not records:
        raise ValueError("At least one M1 template session is required")
    generator = torch.Generator(device=parameters.w.device)
    generator.manual_seed(seed)
    simulated: list[M1SessionTrials] = []
    for template in records:
        with torch.no_grad():
            evaluation = evaluate_m1_session(
                template,
                parameters,
                moments,
                first_long_probability=first_long_probability,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            )
        valid = evaluation.predictions.valid_choice_mask
        choices = torch.full_like(template.choice_long, torch.nan)
        draws = torch.rand(
            int(valid.sum().detach().cpu()),
            dtype=template.choice_long.dtype,
            device=template.choice_long.device,
            generator=generator,
        )
        choices[valid] = (
            draws < evaluation.predictions.probability_long[valid]
        ).to(template.choice_long.dtype)
        simulated.append(
            M1SessionTrials(
                mouse_id=template.mouse_id,
                session_id=template.session_id,
                duration_s=template.duration_s.clone(),
                choice_long=choices,
                category_long=template.category_long.clone(),
                feedback_valid=template.feedback_valid.clone(),
                split_role=template.split_role,
            )
        )
    return tuple(simulated)
