"""Design-preserving choice simulation for M1_dynamic_block_length recovery."""

from __future__ import annotations

from typing import Iterable

import torch

from m0_sensory_v1.data.task_moments import TaskMoments

from .parameters import M1DynamicBlockLengthParameters
from .session import M1DynamicBlockLengthSessionTrials, evaluate_m1_dynamic_block_length_session


def simulate_m1_dynamic_block_length_choices(
    template_sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    parameters: M1DynamicBlockLengthParameters,
    moments: TaskMoments,
    *,
    seed: int,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    first_long_probability: float = 0.5,
) -> tuple[M1DynamicBlockLengthSessionTrials, ...]:
    """Sample choices while preserving design, feedback, and omissions."""

    templates = tuple(template_sessions)
    if not templates:
        raise ValueError("At least one M1_dynamic_block_length template session is required")
    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    simulated: list[M1DynamicBlockLengthSessionTrials] = []
    for template in templates:
        with torch.no_grad():
            prediction = evaluate_m1_dynamic_block_length_session(
                template, parameters, moments,
                run_length_center=run_length_center, maximum_run_length=maximum_run_length,
                first_long_probability=first_long_probability,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = prediction.valid_choice_mask
        draws = torch.rand(
            int(valid.sum()), dtype=template.choice_long.dtype,
            device=template.choice_long.device, generator=generator,
        )
        choices[valid] = (draws < prediction.probability_long[valid]).to(template.choice_long.dtype)
        simulated.append(M1DynamicBlockLengthSessionTrials(
            mouse_id=template.mouse_id, session_id=template.session_id,
            duration_s=template.duration_s.clone(), choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(), split_role=template.split_role,
        ))
    return tuple(simulated)
