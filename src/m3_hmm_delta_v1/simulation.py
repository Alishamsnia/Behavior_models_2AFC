"""Design-preserving choice simulation for M3 recovery."""

from __future__ import annotations

from typing import Iterable

import torch

from m0_sensory_v1.data.task_moments import TaskMoments

from .parameters import M3DecodedParameters
from .session import M3SessionTrials, evaluate_m3_session


def simulate_m3_choices(
    template_sessions: Iterable[M3SessionTrials],
    parameters: M3DecodedParameters,
    moments: TaskMoments,
    *,
    seed: int,
    first_long_probability: float = 0.5,
    delta_min: float = 1e-6,
) -> tuple[M3SessionTrials, ...]:
    """Sample choices while preserving durations, categories, feedback, and omissions."""

    templates = tuple(template_sessions)
    if not templates:
        raise ValueError("At least one M3 template session is required")
    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    simulated: list[M3SessionTrials] = []
    for template in templates:
        with torch.no_grad():
            prediction = evaluate_m3_session(
                template,
                parameters,
                moments,
                first_long_probability=first_long_probability,
                delta_min=delta_min,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = prediction.valid_choice_mask
        draws = torch.rand(
            int(valid.sum()),
            dtype=template.choice_long.dtype,
            device=template.choice_long.device,
            generator=generator,
        )
        choices[valid] = (draws < prediction.probability_long[valid]).to(
            template.choice_long.dtype
        )
        simulated.append(M3SessionTrials(
            mouse_id=template.mouse_id,
            session_id=template.session_id,
            duration_s=template.duration_s.clone(),
            choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(),
            split_role=template.split_role,
        ))
    return tuple(simulated)
