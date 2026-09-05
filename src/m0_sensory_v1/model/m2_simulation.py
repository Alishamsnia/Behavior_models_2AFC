"""Design-preserving and generative simulation for M2 recovery checks."""

from __future__ import annotations

from typing import Iterable

import torch

from ..data.task_moments import TaskMoments
from .m1_session import constant_hazard_transition_matrix
from .m2_parameters import (
    M2DecodedParameters,
    decode_m2_raw_parameters,
    encode_m2_constrained_parameters,
)
from .m2_session import M2SessionTrials, evaluate_m2_session


def simulate_m2_choices(
    template_sessions: Iterable[M2SessionTrials],
    parameters: M2DecodedParameters,
    moments: TaskMoments,
    *,
    seed: int,
    first_long_probability: float = 0.5,
) -> tuple[M2SessionTrials, ...]:
    r"""Simulate choices conditional on fixed category/feedback timelines.

    Purpose
    -------
    Generate choice-only replications while preserving an observed task design.

    Mathematical definition
    -----------------------
    .. math::

       Y_{d,t}^{sim}\sim\operatorname{Bernoulli}(p_{d,t}^{M2}).

    Parameter interpretation
    ------------------------
    M2 parameters determine predictions; categories and feedback remain fixed.

    Input units and shapes
    ----------------------
    Template sessions use float64 seconds and one Boolean feedback mask per trial.

    Returned quantities and shapes
    ------------------------------
    Returns sessions with identical timelines and newly sampled valid choices.

    Constraints
    -----------
    At least one template and an explicit deterministic seed are required.

    Numerical stability
    -------------------
    Probabilities come from the checked M2 evaluator and use Torch Bernoulli draws.

    Causal restrictions
    -------------------
    Only causal M2 probabilities generate choices; future values are never inputs.

    Important edge cases
    --------------------
    Original omission locations remain NaN and receive no simulated decision.

    Related tests
    -------------
    ``test_design_preserving_and_generative_simulation``.
    """

    records = tuple(template_sessions)
    if not records:
        raise ValueError("At least one M2 template session is required")
    generator = torch.Generator(device=parameters.w.device).manual_seed(seed)
    simulated: list[M2SessionTrials] = []
    for template in records:
        with torch.no_grad():
            predictions = evaluate_m2_session(
                template,
                parameters,
                moments,
                first_long_probability=first_long_probability,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = predictions.valid_choice_mask
        draws = torch.rand(
            int(valid.sum().detach().cpu()),
            dtype=template.choice_long.dtype,
            device=template.choice_long.device,
            generator=generator,
        )
        choices[valid] = (draws < predictions.probability_long[valid]).to(
            template.choice_long.dtype
        )
        simulated.append(M2SessionTrials(
            mouse_id=template.mouse_id,
            session_id=template.session_id,
            duration_s=template.duration_s.clone(),
            choice_long=choices,
            category_long=template.category_long.clone(),
            feedback_valid=template.feedback_valid.clone(),
            split_role=template.split_role,
        ))
    return tuple(simulated)


def simulate_m2_hmm_sessions(
    durations_by_session: Iterable[torch.Tensor],
    constrained_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    random_seed: int,
    split_role: str = "train",
    mouse_id: str = "synthetic_m2_mouse",
    first_long_probability: float = 0.5,
) -> tuple[tuple[M2SessionTrials, ...], tuple[torch.Tensor, ...]]:
    r"""Generate latent states, subjective categories, and choices from M2.

    Purpose
    -------
    Provide a fully generative synthetic HMM utility for mechanism smoke tests.

    Mathematical definition
    -----------------------
    .. math::

       Z_{t+1}\sim T(h)_{Z_t},\quad C_t\sim
       \operatorname{Bernoulli}(\theta_{Z_t}),\quad
       Y_t\sim\operatorname{Bernoulli}(p_t).

    Parameter interpretation
    ------------------------
    Hazard generates latent switches and ``delta0`` generates category contrast.

    Input units and shapes
    ----------------------
    Each duration tensor is nonempty one-dimensional float64 seconds.

    Returned quantities and shapes
    ------------------------------
    Returns simulated sessions and matching integer latent-state trajectories.

    Constraints
    -----------
    Requires a constrained seven-vector and train or validation split role.

    Numerical stability
    -------------------
    Parameters round-trip through stable registered transforms before generation.

    Causal restrictions
    -------------------
    Choice generation calls the same pre-choice/post-feedback M2 evaluator.

    Important edge cases
    --------------------
    This utility regenerates schedules and therefore is not the real-design
    recovery path used for acceptance analysis.

    Related tests
    -------------
    ``test_design_preserving_and_generative_simulation``.
    """

    durations = tuple(durations_by_session)
    if not durations or constrained_parameters.shape != (7,):
        raise ValueError("M2 simulation requires sessions and a seven-vector")
    if split_role not in {"train", "validation"}:
        raise ValueError("Synthetic M2 split must be train or validation")
    parameters = decode_m2_raw_parameters(
        encode_m2_constrained_parameters(constrained_parameters)
    )
    transition = constant_hazard_transition_matrix(
        parameters.hazard, first_long_probability=first_long_probability
    )
    theta = parameters.prototypes()
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    sessions: list[M2SessionTrials] = []
    truths: list[torch.Tensor] = []
    for session_index, duration in enumerate(durations, start=1):
        if duration.ndim != 1 or duration.dtype != torch.float64 or duration.numel() == 0:
            raise TypeError("Synthetic M2 durations must be nonempty 1D float64")
        state = 0
        states: list[int] = []
        categories: list[float] = []
        for trial in range(duration.numel()):
            states.append(state)
            categories.append(float(torch.rand((), generator=generator) < theta[state]))
            if trial + 1 < duration.numel():
                state = int(torch.multinomial(transition[state], 1, generator=generator))
        category = torch.tensor(categories, dtype=torch.float64)
        template = M2SessionTrials(
            mouse_id=mouse_id,
            session_id=f"{split_role}_{session_index:03d}",
            duration_s=duration.clone(),
            choice_long=torch.zeros_like(duration),
            category_long=category,
            feedback_valid=torch.ones(duration.numel(), dtype=torch.bool),
            split_role=split_role,
        )
        with torch.no_grad():
            probability = evaluate_m2_session(
                template, parameters, moments,
                first_long_probability=first_long_probability,
            ).predictions.probability_long
            choices = (torch.rand(duration.shape, generator=generator) < probability).to(torch.float64)
        sessions.append(M2SessionTrials(
            mouse_id=mouse_id,
            session_id=template.session_id,
            duration_s=duration.clone(),
            choice_long=choices,
            category_long=category,
            feedback_valid=template.feedback_valid.clone(),
            split_role=split_role,
        ))
        truths.append(torch.tensor(states, dtype=torch.int64))
    return tuple(sessions), tuple(truths)


def simulate_m2_real_design_sessions(
    template_sessions: Iterable[M2SessionTrials],
    true_blocks_by_session: dict[str, Iterable[object]],
    constrained_parameters: torch.Tensor,
    moments: TaskMoments,
    *,
    random_seed: int,
    first_long_probability: float = 0.5,
) -> tuple[M2SessionTrials, ...]:
    r"""Simulate M2 while preserving real timelines, masks, and block schedules.

    Purpose
    -------
    Generate acceptance-level recovery data on the observed experimental design.

    Mathematical definition
    -----------------------
    .. math::

       C_{d,t}^{sim}\sim\operatorname{Bernoulli}(\theta_{b_{d,t}}),
       \qquad Y_{d,t}^{sim}\sim\operatorname{Bernoulli}(p_{d,t}^{M2}),

    where the observed retrospective block schedule :math:`b_{d,t}` is fixed.

    Parameter interpretation
    ------------------------
    ``delta0`` controls generated block-category contrast and ``ah`` controls the
    simulated observer's inference, not the preserved experimenter schedule.

    Input units and shapes
    ----------------------
    Templates retain real float64 durations, session lengths, omission locations,
    and feedback masks; each block-label sequence must match its session length.

    Returned quantities and shapes
    ------------------------------
    Returns sessions with the same design and newly generated categories/choices.

    Constraints
    -----------
    State labels must be ``N``, ``S``, or ``L`` wherever feedback is valid.

    Numerical stability
    -------------------
    Parameters use registered transforms and choices use checked M2 probabilities.

    Causal restrictions
    -------------------
    True block is used only by this offline generator; it never enters the fitted
    evaluator or any current-trial prediction.

    Important edge cases
    --------------------
    Omission/invalid-choice positions remain NaN while their real feedback and
    transition semantics are preserved.

    Related tests
    -------------
    ``test_real_design_simulation_preserves_lengths_masks_and_block_schedule``.
    """

    records = tuple(template_sessions)
    if not records or constrained_parameters.shape != (7,):
        raise ValueError("Real-design M2 simulation requires templates and a seven-vector")
    parameters = decode_m2_raw_parameters(
        encode_m2_constrained_parameters(constrained_parameters)
    )
    theta = parameters.prototypes().detach().cpu()
    state_index = {"N": 0, "S": 1, "L": 2}
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    simulated: list[M2SessionTrials] = []
    for template in records:
        if template.session_id not in true_blocks_by_session:
            raise KeyError(f"Missing true-block schedule for {template.session_id}")
        labels = tuple(str(value) for value in true_blocks_by_session[template.session_id])
        if len(labels) != template.duration_s.numel():
            raise ValueError(f"True-block schedule length mismatch for {template.session_id}")
        category_values: list[float] = []
        for index, label in enumerate(labels):
            if label not in state_index:
                if bool(template.feedback_valid[index].detach().cpu()):
                    raise ValueError(
                        f"Feedback-valid trial has invalid true block {label!r}"
                    )
                category_values.append(float("nan"))
                continue
            category_values.append(
                float(torch.rand((), generator=generator) < theta[state_index[label]])
            )
        category = torch.tensor(
            category_values,
            dtype=template.category_long.dtype,
            device=template.category_long.device,
        )
        generation_template = M2SessionTrials(
            mouse_id=template.mouse_id,
            session_id=template.session_id,
            duration_s=template.duration_s.clone(),
            choice_long=template.choice_long.clone(),
            category_long=category,
            feedback_valid=template.feedback_valid.clone(),
            split_role=template.split_role,
        )
        with torch.no_grad():
            prediction = evaluate_m2_session(
                generation_template,
                parameters,
                moments,
                first_long_probability=first_long_probability,
            ).predictions
        choices = torch.full_like(template.choice_long, torch.nan)
        valid = prediction.valid_choice_mask
        draws = torch.rand(
            int(valid.sum().detach().cpu()),
            dtype=template.choice_long.dtype,
            generator=generator,
        ).to(template.choice_long.device)
        choices[valid] = (draws < prediction.probability_long[valid]).to(
            template.choice_long.dtype
        )
        simulated.append(M2SessionTrials(
            mouse_id=template.mouse_id,
            session_id=template.session_id,
            duration_s=template.duration_s.clone(),
            choice_long=choices,
            category_long=category,
            feedback_valid=template.feedback_valid.clone(),
            split_role=template.split_role,
        ))
    return tuple(simulated)
