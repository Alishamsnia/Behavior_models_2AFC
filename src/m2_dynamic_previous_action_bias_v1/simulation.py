"""Sequential generative choice simulation for
M2_dynamic_previous_action_bias recovery.

Unlike M2_dynamic_block_length_v1's own ``simulate_m2_dynamic_block_length_choices``
(which can sample every valid trial's choice independently in one vectorized
pass, because the joint block belief never depends on the sampled choice),
this model's action-history state ``pi_t`` causally depends on which
choices get generated -- so this simulator must be a genuinely sequential
per-trial loop: predict, mix, lapse-mix, sample the Bernoulli choice, then
feed *that sampled choice* (not any real historical choice) into the
``pi_t`` update before advancing to the next trial. This is the same
causal contract M1_action_kernel_v1's own docs describe as required for "a
fully-generative free-running simulation" (predict -> sample -> update ->
repeat) -- a mode that package explicitly has not built. Duration,
category, and feedback validity are held fixed from the template session
(design-preserving, exactly M2_dynamic_block_length_v1's own convention);
only the choice sequence is generated.
"""

from __future__ import annotations

from typing import Iterable

import torch

from m0_sensory_v1.data.task_moments import TaskMoments
from m0_sensory_v1.model.bayesian import lapse_mixture
from m0_sensory_v1.model.capped_tail_hsmm import (
    block_marginal,
    capped_tail_hsmm_transition,
    feedback_update,
    initialize_capped_tail_state,
)
from m0_sensory_v1.model.m1_bayesian import m1_bayesian_choice_probability

from .parameters import M2DynamicPreviousActionBiasParameters
from .session import M2DynamicPreviousActionBiasSessionTrials


def simulate_m2_dynamic_previous_action_bias_choices(
    template_sessions: Iterable[M2DynamicPreviousActionBiasSessionTrials],
    parameters: M2DynamicPreviousActionBiasParameters,
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
) -> tuple[M2DynamicPreviousActionBiasSessionTrials, ...]:
    """Sample choices sequentially, feeding each sampled choice into the
    causal action-history update, while preserving design and feedback."""

    templates = tuple(template_sessions)
    if not templates:
        raise ValueError("At least one M2_dynamic_previous_action_bias template session is required")
    if not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_pi must lie in (0,1)")

    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    alpha = parameters.alpha
    eta = parameters.eta
    theta = parameters.prototypes()
    simulated: list[M2DynamicPreviousActionBiasSessionTrials] = []

    for template in templates:
        run_lengths = torch.arange(
            1, maximum_run_length + 1, dtype=parameters.w.dtype, device=parameters.w.device
        )
        biased_hazard = parameters.hazard(run_lengths, center=run_length_center)
        xi_pre = initialize_capped_tail_state(parameters.w, maximum_run_length)
        pi_current = parameters.w.new_tensor(initial_pi)
        choices = torch.full_like(template.choice_long, torch.nan)
        # Samplable exactly where the template's own original trial would have
        # contributed a likelihood term (duration valid AND the template's own
        # observed choice was valid) -- not duration validity alone. This
        # preserves the template's omission pattern, matching
        # M2_dynamic_block_length_v1's own `simulate_m2_dynamic_block_length_choices`
        # (which samples only where `valid_choice_mask` was true for the
        # template). Gating on duration alone would inject a sampled action
        # -- and a pi_t update -- on trials the real mouse never responded on,
        # silently changing the omission structure "design-preserving"
        # simulation is supposed to hold fixed.
        samplable = (
            torch.isfinite(template.duration_s) & (template.duration_s > 0)
            & torch.isfinite(template.choice_long)
            & ((template.choice_long == 0) | (template.choice_long == 1))
        )

        with torch.no_grad():
            for index in range(template.duration_s.numel()):
                block_pre = block_marginal(xi_pre)
                prior = torch.dot(block_pre, theta)
                pi_before = pi_current

                if bool(samplable[index]):
                    result = m1_bayesian_choice_probability(
                        template.duration_s[index], parameters.common(), moments, prior,
                        prior_probability_epsilon=prior_probability_epsilon,
                        quadratic_tolerance=quadratic_tolerance,
                        discriminant_tolerance=discriminant_tolerance,
                    )
                    r_t = result.probability_long
                    m_t = (1.0 - eta) * r_t + eta * pi_before
                    probability = lapse_mixture(m_t, template.category_long[index], parameters.common())
                    draw = torch.rand((), dtype=template.choice_long.dtype, device=template.choice_long.device, generator=generator)
                    action = (draw < probability).to(template.choice_long.dtype)
                    choices[index] = action
                    pi_current = pi_before + alpha * (action - pi_before)
                else:
                    pi_current = pi_before

                if bool(template.feedback_valid[index]):
                    category = template.category_long[index]
                    emission = theta.pow(category) * (1 - theta).pow(1 - category)
                    xi_post, _ = feedback_update(xi_pre, emission)
                else:
                    xi_post = xi_pre

                if index + 1 < template.duration_s.numel():
                    xi_pre = capped_tail_hsmm_transition(
                        xi_post, biased_hazard, first_long_probability=first_long_probability
                    )
                    xi_pre = xi_pre / xi_pre.sum()

        simulated.append(M2DynamicPreviousActionBiasSessionTrials(
            mouse_id=template.mouse_id, session_id=template.session_id,
            duration_s=template.duration_s.clone(), choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(), split_role=template.split_role,
        ))
    return tuple(simulated)
