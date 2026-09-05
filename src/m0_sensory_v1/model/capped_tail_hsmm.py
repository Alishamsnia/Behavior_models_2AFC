"""Independent capped-tail duration-dependent HSMM kernel.

This is a shared scientific primitive for the model family, not part of any
single model's frozen scientific contract. ``m0_sensory_v1/model/`` is
already the project's de-facto shared-primitives namespace -- ``m1_session.py``,
``m1_bayesian.py``, ``m2_parameters.py``, and similar files already live here
despite belonging to M1/M2, not M0. This module follows that precedent: it
implements a joint distribution over block state ``b in (N,S,L)`` and
represented run length ``r in (1,...,Rmax)``, with a duration-dependent
biased-state hazard ``h_B(r)=sigmoid(ah+bh*(r-center))`` and a retained tail
bin at ``r=Rmax`` representing ``R>=Rmax``.

This module is independently authored. It does not import anything from
``m4_hsmm_delta_v1`` and is never imported by it. It is cross-verified against
that package's own kernel only in the test suite
(``tests/test_capped_tail_hsmm_kernel.py``), never at runtime. Any model that
depends on this module must checksum it in its own release manifest.
"""

from __future__ import annotations

import torch


def capped_tail_hazard(
    ah: torch.Tensor,
    bh: torch.Tensor,
    run_lengths: torch.Tensor,
    *,
    center: int,
) -> torch.Tensor:
    r"""Return the biased-state hazard ``sigmoid(ah+bh*(r-center))``.

    Purpose
    -------
    Provide the single production definition of the duration-dependent
    biased-block hazard so every consumer decodes the identical formula.

    Mathematical definition
    -----------------------
    .. math::

       h_B(r)=\sigma\big(a_h+b_h\,(r-r_0)\big).

    Parameter interpretation
    ------------------------
    ``ah`` is the hazard logit at the registered center run length ``r0``;
    ``bh>=0`` is the run-length slope. ``bh=0`` recovers a constant hazard.

    Input units and shapes
    ----------------------
    ``run_lengths`` is a floating tensor of represented run lengths, any
    shape; ``ah``/``bh`` are scalars broadcastable against it.

    Returned quantities and shapes
    ------------------------------
    Returns a tensor of biased-state hazards with the shape of ``run_lengths``.

    Constraints
    -----------
    ``center`` must be a positive integer; ``run_lengths`` must be floating.

    Numerical stability
    -------------------
    ``sigmoid`` is evaluated directly; inputs and outputs preserve dtype and
    device and remain part of the autograd graph.

    Causal restrictions
    -------------------
    This is a pure function of the fitted hazard parameters and run length;
    it never receives true block or true run-length labels.

    Important edge cases
    --------------------
    ``bh=0`` gives a hazard constant in ``r``, equal to ``sigmoid(ah)`` for
    every represented run length.

    Related tests
    -------------
    ``tests/test_capped_tail_hsmm_kernel.py::test_hazard_matches_closed_form``.
    """

    if not isinstance(run_lengths, torch.Tensor) or not run_lengths.is_floating_point():
        raise TypeError("run_lengths must be a floating torch.Tensor")
    if center <= 0:
        raise ValueError("center must be a positive integer")
    return torch.sigmoid(ah + bh * (run_lengths - float(center)))


def initialize_capped_tail_state(
    reference: torch.Tensor, maximum_run_length: int
) -> torch.Tensor:
    """Return the one-hot ``P(N,r=1)=1`` joint state, shape ``(3,Rmax)``.

    ``reference`` supplies dtype/device only (matches every other model's
    ``parameters.w.new_tensor(...)`` convention).
    """

    if maximum_run_length <= 1:
        raise ValueError("maximum_run_length must exceed one")
    first = reference.new_tensor((1.0, 0.0, 0.0)).unsqueeze(1)
    remainder = reference.new_zeros((3, maximum_run_length - 1))
    return torch.cat((first, remainder), dim=1)


def block_marginal(state: torch.Tensor) -> torch.Tensor:
    """Sum the joint ``(3,Rmax)`` state over run length, returning ``(3,)``."""

    _validate_state_shape(state)
    return state.sum(dim=1)


def run_length_marginal(state: torch.Tensor) -> torch.Tensor:
    """Sum the joint ``(3,Rmax)`` state over block, returning ``(Rmax,)``."""

    _validate_state_shape(state)
    return state.sum(dim=0)


def feedback_update(
    state: torch.Tensor, emission_by_block: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition the joint state on a block-only emission.

    ``posterior(b,r) propto state(b,r) * emission_by_block(b)`` -- emissions
    depend on block, never on run length, so every column of ``state`` for a
    fixed block is scaled by the same factor and the conditional run-length
    distribution within each block is preserved (see
    ``tests/test_capped_tail_hsmm_kernel.py`` for the invariant test).
    """

    _validate_state_shape(state)
    if emission_by_block.shape != (3,):
        raise ValueError("emission_by_block must have shape (3,)")
    unnormalized = state * emission_by_block.unsqueeze(1)
    normalizer = unnormalized.sum()
    if not bool(torch.isfinite(normalizer) & (normalizer > 0.0)):
        raise FloatingPointError("capped-tail feedback update has an invalid normalizer")
    return unnormalized / normalizer, normalizer


def capped_tail_hsmm_transition(
    joint_posterior: torch.Tensor,
    hazard_by_run_length: torch.Tensor,
    *,
    first_long_probability: float = 0.5,
) -> torch.Tensor:
    r"""Advance the joint block/run-length posterior by one trial.

    Purpose
    -------
    Apply one capped-tail HSMM transition: survival ages run length by one
    (or is retained in the tail bin), and every switch resets run length to
    one and follows M1's fixed destination rule.

    Mathematical definition
    -----------------------
    For source mass :math:`q_b(r)=\xi^+(b,r)`, let ``hazard_by_run_length`` be
    the biased-state hazard :math:`h_B(r)` and derive :math:`h_N(r)=h_B(r)/2`.
    Survival and switch mass are

    .. math::

       q_b(r)\,[1-h_b(r)],\qquad q_b(r)\,h_b(r).

    For :math:`r<R_{max}` survival moves to :math:`(b,r+1)`; survival at
    :math:`r=R_{max}` remains in that bin. Switch mass is routed as

    .. math::

       N\to S:(1-\omega)\sum_r q_N(r)h_N(r),\quad
       N\to L:\omega\sum_r q_N(r)h_N(r),

       S\to L:\sum_r q_S(r)h_B(r),\qquad L\to S:\sum_r q_L(r)h_B(r),

    and every switch destination is run length one. Neutral is never
    re-entered.

    Parameter interpretation
    ------------------------
    ``first_long_probability`` (:math:`\omega`) controls only neutral exit
    direction and is a fixed task-design constant, not a fitted parameter.

    Input units and shapes
    ----------------------
    ``joint_posterior`` is a normalized ``(3,Rmax)`` probability mass;
    ``hazard_by_run_length`` has shape ``(Rmax,)``.

    Returned quantities and shapes
    ------------------------------
    Returns a normalized ``(3,Rmax)`` next-trial probability mass; total mass
    is conserved by construction (survival and switch mass partition the
    input exactly and are fully redistributed).

    Constraints
    -----------
    Hazards and ``first_long_probability`` lie in ``[0,1]``; ``Rmax`` must
    exceed one.

    Numerical stability
    -------------------
    The construction is purely functional (``torch.cat``/``torch.stack``, no
    in-place indexing), so it is safe under autograd and preserves dtype and
    device throughout.

    Causal restrictions
    -------------------
    This is a function of posterior mass and the fitted hazard vector only;
    it never receives true block or true run-length labels.

    Important edge cases
    --------------------
    A run-length-independent hazard vector reduces this kernel's block
    marginal exactly to M1's ``constant_hazard_transition_matrix`` (see the
    constant-hazard identity test in
    ``tests/test_capped_tail_hsmm_kernel.py``).

    Related tests
    -------------
    ``tests/test_capped_tail_hsmm_kernel.py`` (mass conservation, tail
    retention, switch routing, endpoint behavior, constant-hazard identity).
    """

    _validate_state_shape(joint_posterior)
    maximum_run_length = joint_posterior.shape[1]
    if maximum_run_length <= 1:
        raise ValueError("capped-tail HSMM requires Rmax to exceed one")
    if hazard_by_run_length.shape != (maximum_run_length,):
        raise ValueError("hazard_by_run_length must have shape (Rmax,)")
    if not 0.0 <= first_long_probability <= 1.0:
        raise ValueError("first_long_probability must lie in [0,1]")
    if not bool(
        torch.all(
            torch.isfinite(hazard_by_run_length)
            & (hazard_by_run_length >= 0.0)
            & (hazard_by_run_length <= 1.0)
        )
    ):
        raise ValueError("hazard_by_run_length must contain finite probabilities in [0,1]")

    biased = hazard_by_run_length
    state_hazard = torch.stack((biased / 2.0, biased, biased), dim=0)
    survival_mass = joint_posterior * (1.0 - state_hazard)
    switch_mass = joint_posterior * state_hazard

    shifted = torch.cat(
        (joint_posterior.new_zeros((3, 1)), survival_mass[:, :-1]), dim=1
    )
    tail_addition = torch.cat(
        (
            joint_posterior.new_zeros((3, maximum_run_length - 1)),
            survival_mass[:, -1:],
        ),
        dim=1,
    )
    survived = shifted + tail_addition

    omega = joint_posterior.new_tensor(first_long_probability)
    total_switch_from_n = switch_mass[0].sum()
    total_switch_from_s = switch_mass[1].sum()
    total_switch_from_l = switch_mass[2].sum()
    switch_into_s = (1.0 - omega) * total_switch_from_n + total_switch_from_l
    switch_into_l = omega * total_switch_from_n + total_switch_from_s

    first_column = torch.stack(
        (joint_posterior.new_zeros(()), switch_into_s, switch_into_l)
    ).unsqueeze(1)
    switch_addition = torch.cat(
        (first_column, joint_posterior.new_zeros((3, maximum_run_length - 1))), dim=1
    )
    return survived + switch_addition


def _validate_state_shape(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor) or not state.is_floating_point():
        raise TypeError("state must be a floating torch.Tensor")
    if state.ndim != 2 or state.shape[0] != 3:
        raise ValueError("capped-tail HSMM state must have shape (3,Rmax)")
