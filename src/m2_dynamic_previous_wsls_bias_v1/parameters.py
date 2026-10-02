"""Pooled parameter transformations for M2_dynamic_previous_wsls_bias_v1.

M2_dynamic_previous_wsls_bias_v1 keeps M2_dynamic_block_length_v1's eight
pooled values (fitted static prototypes plus a duration-dependent hazard
slope) completely unchanged and adds exactly one new pooled coordinate: a
bias-mixing-weight logit ``eta_logit`` whose sigmoid ``eta`` is the mixing
weight applied between the sensory-readout's non-lapse probability and the
previous-category (win-stay/lose-shift) bias state (see ``session.py``).
There is no learning-rate logit here: the bias learning rate ``alpha`` is
fixed to the registry constant ``1.0`` (see ``registry.py``), the exact
one-step "previous category" special case of the shared delta rule --
matching ``pure_wsls_v1``'s own fixed-``alpha=1`` convention
(``wsls_core_v1.evaluator``). The first eight coordinates
decode/encode through M2_dynamic_block_length_v1's own registered
transform (``m2_dynamic_block_length_v1.parameters``, reused unmodified) --
this model's true parent is M2_dynamic_block_length_v1, not M2 directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from m0_sensory_v1.model.parameters import DecodedParameters
from m2_dynamic_block_length_v1.parameters import (
    M2DynamicBlockLengthParameters,
    decode_m2_dynamic_block_length_raw_parameters,
    encode_m2_dynamic_block_length_constrained_parameters,
)


M2_DYNAMIC_PREVIOUS_WSLS_BIAS_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit",
)

M2_DYNAMIC_PREVIOUS_WSLS_BIAS_LEARNING_RATE = 1.0
"""The fixed, non-fitted previous-category bias learning rate ``alpha`` this
model uses -- the exact one-step "previous category" boundary of the shared
delta rule, at which ``pi_{t+1}`` collapses exactly to ``C_t`` on every
feedback-valid trial (see ``session.py``)."""


@dataclass(frozen=True)
class M2DynamicPreviousWslsBiasParameters:
    """M2_dynamic_block_length_v1's eight pooled values plus a fitted
    bias-mixing-weight logit; ``alpha`` is the fixed constant ``1.0``, not
    a fitted field."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    delta0: torch.Tensor
    bh: torch.Tensor
    eta_logit: torch.Tensor
    eta: torch.Tensor

    @property
    def alpha(self) -> torch.Tensor:
        """Return the fixed previous-category bias learning rate as a scalar tensor."""

        return self.w.new_tensor(M2_DYNAMIC_PREVIOUS_WSLS_BIAS_LEARNING_RATE)

    def common(self) -> DecodedParameters:
        """Return the unchanged five-parameter Bayesian sensory/lapse readout."""

        return DecodedParameters(
            w=self.w, s0=self.s0, ell_short=self.ell_short, ell_long=self.ell_long, kappa=self.kappa,
        )

    def prototypes(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Return static ``(0.5,sigmoid(-delta0),sigmoid(delta0))``, exactly
        M2_dynamic_block_length_v1's own formula."""

        active = self.delta0 if delta is None else delta
        neutral = torch.ones_like(active) * 0.5
        long = torch.sigmoid(active)
        short = torch.sigmoid(-active)
        return torch.stack((neutral, short, long), dim=-1)

    def hazard(self, run_length: torch.Tensor, *, center: int = 23) -> torch.Tensor:
        """Return ``sigmoid(ah+bh*(r-center))``, delegating to the shared kernel formula."""

        from m0_sensory_v1.model.capped_tail_hsmm import capped_tail_hazard

        return capped_tail_hazard(self.ah, self.bh, run_length, center=center)

    def as_tensor(self) -> torch.Tensor:
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa,
             self.ah, self.delta0, self.bh, self.eta_logit),
            dim=-1,
        )

    def as_m2_dynamic_block_length(self) -> M2DynamicBlockLengthParameters:
        """Return the eight-coordinate M2_dynamic_block_length_v1 restriction.

        Used only for the exact ``eta=0`` boundary test against real
        M2_dynamic_block_length_v1.
        """

        return M2DynamicBlockLengthParameters(
            w=self.w, s0=self.s0, ell_short=self.ell_short, ell_long=self.ell_long,
            kappa=self.kappa, ah=self.ah, delta0=self.delta0, bh=self.bh,
        )

    def with_explicit_eta(self, value: float | torch.Tensor) -> "M2DynamicPreviousWslsBiasParameters":
        """Return a copy with ``eta`` pinned to an explicit scalar in ``[0,1]``.

        ``sigmoid(eta_logit)`` can approach but never exactly reach ``0`` for
        finite ``eta_logit``, so the exact ``eta=0`` nesting boundary must be
        constructed directly.
        """

        eta_value = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if eta_value.ndim != 0 or not bool(torch.isfinite(eta_value) & (eta_value >= 0.0) & (eta_value <= 1.0)):
            raise ValueError("Explicit eta must be a finite scalar in [0,1]")
        return replace(self, eta=eta_value)

    def with_explicit_bh(self, value: float | torch.Tensor) -> "M2DynamicPreviousWslsBiasParameters":
        """Return a copy with ``bh`` pinned to an explicit nonnegative scalar.

        ``softplus(raw)+epsilon`` can approach but never exactly reach ``0``
        for finite raw input, so the exact ``bh=0`` boundary (nesting real
        M2 exactly, via ``M2_dynamic_block_length_v1``) must be constructed
        directly -- mirroring
        ``M2DynamicBlockLengthParameters.with_explicit_bh``, which this
        model's own eight shared coordinates otherwise decode/encode through
        unmodified.
        """

        slope = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if slope.ndim != 0 or not bool(torch.isfinite(slope) & (slope >= 0)):
            raise ValueError("Explicit bh must be a finite nonnegative scalar")
        return replace(self, bh=slope)


def decode_m2_dynamic_previous_wsls_bias_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2DynamicPreviousWslsBiasParameters:
    r"""Decode M2_dynamic_previous_wsls_bias's nine unconstrained coordinates.

    Purpose
    -------
    Reuse M2_dynamic_block_length_v1's own eight-coordinate decoder
    unchanged and add a bias-mixing-weight logit; the previous-category
    bias learning rate is a fixed constant, not decoded from a coordinate.

    Mathematical definition
    -----------------------
    .. math::

       \eta=\operatorname{sigmoid}(\widetilde\eta),\qquad \alpha\equiv 1.

    Parameter interpretation
    ------------------------
    The first eight coordinates are M2_dynamic_block_length_v1's unchanged
    sensory/lapse/hazard-anchor/static-prototype-contrast/hazard-slope
    values; ``eta_logit`` is the bias-mixing-weight logit.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension nine.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply ``0<eta<1``.

    Numerical stability
    -------------------
    Torch's stable sigmoid is used in the input dtype and on the input device.

    Causal restrictions
    -------------------
    Decoding accepts parameters only and cannot inspect behavioral history.

    Important edge cases
    --------------------
    Exact ``eta=0`` is intentionally unavailable through the free decoder
    and must be constructed via ``with_explicit_eta`` for the boundary test.

    Related tests
    -------------
    ``test_round_trip_and_alpha_fixed_at_one``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 9:
        raise ValueError("M2_dynamic_previous_wsls_bias raw_parameters must have final dimension 9")
    m2_dynamic = decode_m2_dynamic_block_length_raw_parameters(
        raw_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    eta_logit = raw_parameters[..., 8]
    return M2DynamicPreviousWslsBiasParameters(
        w=m2_dynamic.w, s0=m2_dynamic.s0, ell_short=m2_dynamic.ell_short, ell_long=m2_dynamic.ell_long,
        kappa=m2_dynamic.kappa, ah=m2_dynamic.ah, delta0=m2_dynamic.delta0, bh=m2_dynamic.bh,
        eta_logit=eta_logit, eta=torch.sigmoid(eta_logit),
    )


def encode_m2_dynamic_previous_wsls_bias_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2_dynamic_previous_wsls_bias values to raw coordinates.

    Purpose
    -------
    Initialize this model from scientific values while reusing
    M2_dynamic_block_length_v1's exact eight-coordinate encoder.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde\eta=\eta_{logit}.

    Parameter interpretation
    ------------------------
    Input order is
    ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit)``; the ninth
    value is a logit, not a probability.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension nine.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite ``eta_logit``; the first eight coordinates obey
    M2_dynamic_block_length_v1's own constraints.

    Numerical stability
    -------------------
    Inverse transforms are delegated to M2_dynamic_block_length_v1's encoder.

    Causal restrictions
    -------------------
    Initialization is independent of trials, outcomes, blocks, and feedback.

    Important edge cases
    --------------------
    Infinite bias-mixing-weight logits are rejected.

    Related tests
    -------------
    ``test_round_trip_and_alpha_fixed_at_one`` and the
    frozen eight-start registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 9:
        raise ValueError("M2_dynamic_previous_wsls_bias constrained_parameters must have final dimension 9")
    eta_logit = constrained_parameters[..., 8]
    if not bool(torch.all(torch.isfinite(eta_logit)).detach().cpu()):
        raise ValueError("M2_dynamic_previous_wsls_bias bias-mixing-weight logits must be finite")
    m2_dynamic_raw = encode_m2_dynamic_block_length_constrained_parameters(
        constrained_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((m2_dynamic_raw, eta_logit.unsqueeze(-1)), dim=-1)
