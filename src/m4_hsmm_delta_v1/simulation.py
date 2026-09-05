"""Design-preserving choice simulation for M4 recovery."""

from __future__ import annotations

from typing import Iterable

import torch

from m0_sensory_v1.data.task_moments import TaskMoments

from .parameters import M4DecodedParameters
from .session import M4SessionTrials, evaluate_m4_session


def simulate_m4_choices(
    template_sessions: Iterable[M4SessionTrials],
    parameters: M4DecodedParameters,
    moments: TaskMoments,
    *,
    seed: int,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    first_long_probability: float = 0.5,
    delta_min: float = 1e-6,
) -> tuple[M4SessionTrials, ...]:
    r"""Sample choices while preserving design, feedback, and omissions.

    Purpose
    -------
    Generate recovery data on the exact observed trial timelines.
    Mathematical definition
    -----------------------
    .. math:: y_t\sim\operatorname{Bernoulli}(p_t).

    For valid trial :math:`t`, draw :math:`y_t\sim Bernoulli(p_t)` from M4.
    Parameter interpretation
    ------------------------
    Parameters and hazard constants have the same meanings as session evaluation.
    Input units and shapes
    ----------------------
    Templates contain ``(T,)`` seconds/category/mask vectors; ``seed`` is integer.
    Returned quantities and shapes
    ------------------------------
    Returns matching sessions with only the valid choices resampled.
    Constraints
    -----------
    At least one template is required.
    Numerical stability
    -------------------
    Sampling uses the evaluator's checked float64 probabilities.
    Causal restrictions
    -------------------
    True blocks and true run lengths are not simulation inputs.
    Important edge cases
    --------------------
    Omission locations remain NaN and valid feedback is preserved independently.
    Related tests
    -------------
    Recovery-grid and deterministic session-evaluation acceptance tests.
    """

    templates = tuple(template_sessions)
    if not templates:
        raise ValueError("At least one M4 template session is required")
    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    simulated: list[M4SessionTrials] = []
    for template in templates:
        with torch.no_grad():
            prediction = evaluate_m4_session(
                template, parameters, moments,
                run_length_center=run_length_center,
                maximum_run_length=maximum_run_length,
                first_long_probability=first_long_probability,
                delta_min=delta_min,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = prediction.valid_choice_mask
        draws = torch.rand(
            int(valid.sum()), dtype=template.choice_long.dtype,
            device=template.choice_long.device, generator=generator,
        )
        choices[valid] = (draws < prediction.probability_long[valid]).to(template.choice_long.dtype)
        simulated.append(M4SessionTrials(
            mouse_id=template.mouse_id, session_id=template.session_id,
            duration_s=template.duration_s.clone(), choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(), split_role=template.split_role,
        ))
    return tuple(simulated)
