"""Prior-state provider interfaces that preserve every registered model kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from m0_sensory_v1.model.m1_session import constant_hazard_transition_matrix

from .parameters import M4DecodedParameters


@runtime_checkable
class PriorStateProvider(Protocol):
    r"""Minimal causal interface shared by constant-hazard and HSMM priors.

    A provider owns only hidden-prior operations: initialization, block
    marginalization, feedback conditioning, and the post-feedback transition.
    It never receives stimulus duration, choice, true block, true run length,
    or prototype-learning state. Consequently it cannot change any sensory,
    lapse, likelihood, or learning calculation.
    """

    def initialize(self, reference: torch.Tensor) -> torch.Tensor: ...
    def block_marginal(self, state: torch.Tensor) -> torch.Tensor: ...
    def feedback_update(
        self, state: torch.Tensor, emission_by_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def transition(self, posterior: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ConstantHazardPriorStateProvider:
    r"""Non-invasive adapter around the unchanged M1--M3 HMM transition.

    The state is :math:`\xi(b)` with shape ``(3,)`` in ``(N,S,L)`` order.
    Initialization is :math:`\xi_1(N)=1`. Feedback uses
    :math:`\xi^+(b)\propto\xi^-(b)e(b)`, and transition delegates directly to
    the already-verified ``constant_hazard_transition_matrix``. M0--M3 do not
    call this adapter; it exists to prove that the common interface represents
    their existing logic without modifying it.
    """

    hazard: torch.Tensor
    first_long_probability: float = 0.5

    def initialize(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor((1.0, 0.0, 0.0))

    def block_marginal(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape != (3,):
            raise ValueError("Constant-hazard prior state must have shape (3,)")
        return state

    def feedback_update(
        self, state: torch.Tensor, emission_by_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != (3,) or emission_by_block.shape != (3,):
            raise ValueError("HMM feedback update requires two three-state vectors")
        unnormalized = state * emission_by_block
        normalizer = unnormalized.sum()
        if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
            raise FloatingPointError("HMM feedback update has invalid normalizer")
        return unnormalized / normalizer, normalizer

    def transition(self, posterior: torch.Tensor) -> torch.Tensor:
        matrix = constant_hazard_transition_matrix(
            self.hazard, first_long_probability=self.first_long_probability
        )
        return posterior @ matrix


@dataclass(frozen=True)
class CappedTailHSMMPriorStateProvider:
    r"""Provider for M4's joint block/run-length prior.

    The state has shape ``(3,Rmax)`` and entry ``state[b,r-1]`` represents
    :math:`P(Z=b,R=r)`. It starts at
    :math:`\xi^-_1(N,1)=1`. Feedback obeys
    :math:`\xi^+(b,r)\propto\xi^-(b,r)e(b)` because emissions depend on block
    but not run length. The biased-state hazard is
    :math:`h_B(r)=\sigma[a_h+b_h(r-r_0)]`; the neutral-state hazard is the
    fixed task-design tie :math:`h_N(r)=h_B(r)/2`.

    For ``r<Rmax``, survival moves ``r -> r+1``. At ``Rmax``, the final bin is
    the registered tail :math:`R\ge Rmax`; survival remains there and continues
    to use :math:`h(Rmax)`. Every switch enters run length one. Neutral exits to
    S/L according to the fixed first-long probability, S switches only to L,
    L switches only to S, and neither biased state can enter N.
    """

    parameters: M4DecodedParameters
    run_length_center: int = 23
    maximum_run_length: int = 65
    first_long_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.run_length_center <= 0:
            raise ValueError("run_length_center must be positive")
        if self.maximum_run_length <= 1:
            raise ValueError("maximum_run_length must exceed one")
        if not 0.0 <= self.first_long_probability <= 1.0:
            raise ValueError("first_long_probability must lie in [0,1]")

    @property
    def run_lengths(self) -> torch.Tensor:
        return torch.arange(
            1,
            self.maximum_run_length + 1,
            dtype=self.parameters.w.dtype,
            device=self.parameters.w.device,
        )

    @property
    def hazard_by_run_length(self) -> torch.Tensor:
        """Return the fitted biased-block hazard ``h_B(r)``."""

        return self.parameters.hazard(
            self.run_lengths, center=self.run_length_center
        )

    @property
    def state_hazard_by_run_length(self) -> torch.Tensor:
        """Return tied hazards in ``(N,S,L)`` row order."""

        biased = self.hazard_by_run_length
        return torch.stack((biased / 2.0, biased, biased), dim=0)

    def initialize(self, reference: torch.Tensor) -> torch.Tensor:
        first = reference.new_tensor((1.0, 0.0, 0.0)).unsqueeze(1)
        remainder = reference.new_zeros((3, self.maximum_run_length - 1))
        return torch.cat((first, remainder), dim=1)

    def block_marginal(self, state: torch.Tensor) -> torch.Tensor:
        self._validate_state_shape(state)
        return state.sum(dim=1)

    def run_length_marginal(self, state: torch.Tensor) -> torch.Tensor:
        self._validate_state_shape(state)
        return state.sum(dim=0)

    def feedback_update(
        self, state: torch.Tensor, emission_by_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_state_shape(state)
        if emission_by_block.shape != (3,):
            raise ValueError("M4 emissions must have shape (3,)")
        unnormalized = state * emission_by_block.unsqueeze(1)
        normalizer = unnormalized.sum()
        if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
            raise FloatingPointError("M4 feedback update has invalid normalizer")
        return unnormalized / normalizer, normalizer

    def transition(self, posterior: torch.Tensor) -> torch.Tensor:
        return capped_tail_hsmm_transition(
            posterior,
            self.hazard_by_run_length,
            first_long_probability=self.first_long_probability,
        )

    def _validate_state_shape(self, state: torch.Tensor) -> None:
        if state.shape != (3, self.maximum_run_length):
            raise ValueError(
                f"M4 state must have shape (3,{self.maximum_run_length})"
            )


def capped_tail_hsmm_transition(
    joint_posterior: torch.Tensor,
    hazard_by_run_length: torch.Tensor,
    *,
    first_long_probability: float = 0.5,
) -> torch.Tensor:
    r"""Route every posterior source state through M4's capped-tail kernel.

    Purpose
    -------
    Advance the causal joint block/run-length posterior by one trial.

    Mathematical definition
    -----------------------

    For source mass :math:`q_b(r)=\xi^+(b,r)`, let the supplied vector be the
    biased-state hazard :math:`h_B(r)` and derive
    :math:`h_N(r)=h_B(r)/2`. Survival and switching are
    :math:`q_b(r)[1-h_b(r)]` and :math:`q_b(r)h_b(r)`. For ``r<Rmax`` survival
    is placed at ``(b,r+1)``; tail survival at ``r=Rmax`` remains in that bin.
    Switch mass is routed as

    .. math::

       N\to S:(1-\omega)q_N(r)h_N(r),\quad
       N\to L:\omega q_N(r)h_N(r),

       S\to L:q_S(r)h_B(r),\qquad L\to S:q_L(r)h_B(r),

    and every switch targets ``r=1``. The construction is functional and
    conserves total probability, including at the tail. Direct ``h=0`` and
    ``h=1`` inputs are supported for endpoint tests and bypass the sigmoid.

    Parameter interpretation
    ------------------------
    ``first_long_probability`` controls only neutral exit direction.

    Input units and shapes
    ----------------------
    The posterior is ``(3,Rmax)`` probability mass and hazard is ``(Rmax,)``.

    Returned quantities and shapes
    ------------------------------
    Returns normalized ``(3,Rmax)`` next-trial probability mass.

    Constraints
    -----------
    Biased hazards and the neutral-exit probability lie in ``[0,1]``; the
    derived neutral hazards therefore lie in ``[0,0.5]``.

    Numerical stability
    -------------------
    Uses multiplication and addition only; no division or logarithm occurs.

    Causal restrictions
    -------------------
    It accepts no choice, stimulus, true-block, or true-run-length label.

    Important edge cases
    --------------------
    Tail survival remains at ``Rmax``; endpoint hazards zero and one are exact.

    Related tests
    -------------
    ``test_transition_endpoint_routing_and_constant_provider`` and the M3 nesting test.
    """

    if joint_posterior.ndim != 2 or joint_posterior.shape[0] != 3:
        raise ValueError("M4 joint posterior must have shape (3, Rmax)")
    if hazard_by_run_length.shape != (joint_posterior.shape[1],):
        raise ValueError("M4 hazard vector must have shape (Rmax,)")
    if not 0.0 <= first_long_probability <= 1.0:
        raise ValueError("first_long_probability must lie in [0,1]")
    if not bool(torch.all(torch.isfinite(hazard_by_run_length))):
        raise ValueError("M4 hazard vector must be finite")
    if not bool(torch.all((hazard_by_run_length >= 0) & (hazard_by_run_length <= 1))):
        raise ValueError("M4 hazards must lie in [0,1]")

    state_hazard = torch.stack(
        (hazard_by_run_length / 2.0, hazard_by_run_length, hazard_by_run_length),
        dim=0,
    )
    survival = joint_posterior * (1.0 - state_hazard)
    switch = joint_posterior * state_hazard
    zero_column = joint_posterior.new_zeros((3, 1))
    aged = torch.cat((zero_column, survival[:, :-1]), dim=1)
    tail_survival = torch.cat(
        (
            joint_posterior.new_zeros((3, joint_posterior.shape[1] - 1)),
            survival[:, -1:],
        ),
        dim=1,
    )
    switch_mass = switch.sum(dim=1)
    reset_by_state = torch.stack(
        (
            switch_mass.new_zeros(()),
            switch_mass[0] * (1.0 - first_long_probability) + switch_mass[2],
            switch_mass[0] * first_long_probability + switch_mass[1],
        )
    )
    reset = torch.cat(
        (
            reset_by_state.unsqueeze(1),
            joint_posterior.new_zeros((3, joint_posterior.shape[1] - 1)),
        ),
        dim=1,
    )
    return aged + tail_survival + reset
