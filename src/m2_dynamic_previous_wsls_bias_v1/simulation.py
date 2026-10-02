"""Vectorized generative choice simulation for
M2_dynamic_previous_wsls_bias recovery.

Unlike the sibling ``M2_dynamic_previous_action_bias_v1`` (whose ``pi_t`` is
driven by the mouse's own sampled/generated choice and therefore requires a
genuinely sequential predict->sample->update loop), this model's
previous-category bias state ``pi_t`` is driven only by ``category_long``
and gated only by ``feedback_valid`` -- both exogenous, task-design
quantities that are fixed by the template session and never depend on the
sampled choice (see ``session.py``'s module docstring). So, exactly like
``M2_dynamic_block_length_v1``'s own ``xi`` belief, BOTH causal states this
model carries are independent of the generated choice sequence: the entire
causal trajectory (``xi``, ``pi_t``, ``r_t``, ``m_t``, and hence
``probability_long``) can be computed once via
``evaluate_m2_dynamic_previous_wsls_bias_session`` on the template session
(its own ``choice_long`` is read only for scoring, never for state
evolution), and every valid trial's Bernoulli choice can then be sampled in
one vectorized pass from ``predictions.probability_long`` -- exactly
``m2_dynamic_block_length_v1.simulation``'s own pattern. Duration, category,
and feedback validity are held fixed from the template session
(design-preserving, exactly M2_dynamic_block_length_v1's own convention);
only the choice sequence is generated.
"""

from __future__ import annotations

from typing import Iterable

import torch

from m0_sensory_v1.data.task_moments import TaskMoments

from .parameters import M2DynamicPreviousWslsBiasParameters
from .session import M2DynamicPreviousWslsBiasSessionTrials, evaluate_m2_dynamic_previous_wsls_bias_session


def simulate_m2_dynamic_previous_wsls_bias_choices(
    template_sessions: Iterable[M2DynamicPreviousWslsBiasSessionTrials],
    parameters: M2DynamicPreviousWslsBiasParameters,
    moments: TaskMoments,
    *,
    seed: int,
    run_length_center: int = 23,
    maximum_run_length: int = 65,
    first_long_probability: float = 0.5,
    initial_pi: float = 0.5,
    prior_probability_epsilon: float = 1e-10,
    quadratic_tolerance: float = 1e-12,
    discriminant_tolerance: float = 1e-12,
) -> tuple[M2DynamicPreviousWslsBiasSessionTrials, ...]:
    """Evaluate the causal trajectory once per template session, then sample
    every valid trial's choice in one vectorized pass, while preserving
    design, feedback, and omissions."""

    templates = tuple(template_sessions)
    if not templates:
        raise ValueError("At least one M2_dynamic_previous_wsls_bias template session is required")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")

    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    simulated: list[M2DynamicPreviousWslsBiasSessionTrials] = []

    for template in templates:
        with torch.no_grad():
            prediction = evaluate_m2_dynamic_previous_wsls_bias_session(
                template, parameters, moments,
                run_length_center=run_length_center, maximum_run_length=maximum_run_length,
                first_long_probability=first_long_probability, initial_pi=initial_pi,
                prior_probability_epsilon=prior_probability_epsilon,
                quadratic_tolerance=quadratic_tolerance,
                discriminant_tolerance=discriminant_tolerance,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = prediction.valid_choice_mask
        draws = torch.rand(
            int(valid.sum()), dtype=template.choice_long.dtype,
            device=template.choice_long.device, generator=generator,
        )
        choices[valid] = (draws < prediction.probability_long[valid]).to(template.choice_long.dtype)
        simulated.append(M2DynamicPreviousWslsBiasSessionTrials(
            mouse_id=template.mouse_id, session_id=template.session_id,
            duration_s=template.duration_s.clone(), choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(), split_role=template.split_role,
        ))
    return tuple(simulated)
