"""Pooled parameter transformations for M2_dynamic_previous_action_bias_v1.

M2_dynamic_previous_action_bias_v1 keeps M2_dynamic_block_length_v1's eight
pooled values (fitted static prototypes plus a duration-dependent hazard
slope) completely unchanged and adds exactly one new pooled coordinate: a
bias-mixing-weight logit ``eta_logit`` whose sigmoid ``eta`` is the mixing
weight applied between the sensory-readout's non-lapse probability and the
action-history state (see ``session.py``). Unlike its sibling
``M2_dynamic_action_kernel_bias_v1``, there is no learning-rate logit here:
the action-history learning rate ``alpha`` is fixed to the registry
constant ``1.0`` (see ``registry.py``), the exact one-step "previous
action" special case of the shared delta rule. The first eight coordinates
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


M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit",
)

M2_DYNAMIC_PREVIOUS_ACTION_BIAS_LEARNING_RATE = 1.0
"""The fixed, non-fitted action-history learning rate ``alpha`` this model
uses -- the exact one-step "previous action" boundary of the shared delta
rule, at which ``pi_{t+1}`` collapses exactly to ``A_t`` (see ``session.py``)."""


@dataclass(frozen=True)
class M2DynamicPreviousActionBiasParameters:
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
        """Return the fixed action-history learning rate as a scalar tensor."""

        return self.w.new_tensor(M2_DYNAMIC_PREVIOUS_ACTION_BIAS_LEARNING_RATE)

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

    def with_explicit_eta(self, value: float | torch.Tensor) -> "M2DynamicPreviousActionBiasParameters":
        """Return a copy with ``eta`` pinned to an explicit scalar in ``[0,1]``.

        ``sigmoid(eta_logit)`` can approach but never exactly reach ``0`` for
        finite ``eta_logit``, so the exact ``eta=0`` nesting boundary must be
        constructed directly.
        """

        eta_value = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if eta_value.ndim != 0 or not bool(torch.isfinite(eta_value) & (eta_value >= 0.0) & (eta_value <= 1.0)):
            raise ValueError("Explicit eta must be a finite scalar in [0,1]")
        return replace(self, eta=eta_value)


def decode_m2_dynamic_previous_action_bias_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2DynamicPreviousActionBiasParameters:
    r"""Decode M2_dynamic_previous_action_bias's nine unconstrained coordinates.

    Purpose
    -------
    Reuse M2_dynamic_block_length_v1's own eight-coordinate decoder
    unchanged and add a bias-mixing-weight logit; the action-history
    learning rate is a fixed constant, not decoded from a coordinate.

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
    ``test_m2_dynamic_previous_action_bias_parameter_round_trip``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 9:
        raise ValueError("M2_dynamic_previous_action_bias raw_parameters must have final dimension 9")
    m2_dynamic = decode_m2_dynamic_block_length_raw_parameters(
        raw_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    eta_logit = raw_parameters[..., 8]
    return M2DynamicPreviousActionBiasParameters(
        w=m2_dynamic.w, s0=m2_dynamic.s0, ell_short=m2_dynamic.ell_short, ell_long=m2_dynamic.ell_long,
        kappa=m2_dynamic.kappa, ah=m2_dynamic.ah, delta0=m2_dynamic.delta0, bh=m2_dynamic.bh,
        eta_logit=eta_logit, eta=torch.sigmoid(eta_logit),
    )


def encode_m2_dynamic_previous_action_bias_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2_dynamic_previous_action_bias values to raw coordinates.

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
    ``test_m2_dynamic_previous_action_bias_parameter_round_trip`` and the
    frozen eight-start registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 9:
        raise ValueError("M2_dynamic_previous_action_bias constrained_parameters must have final dimension 9")
    eta_logit = constrained_parameters[..., 8]
    if not bool(torch.all(torch.isfinite(eta_logit)).detach().cpu()):
        raise ValueError("M2_dynamic_previous_action_bias bias-mixing-weight logits must be finite")
    m2_dynamic_raw = encode_m2_dynamic_block_length_constrained_parameters(
        constrained_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((m2_dynamic_raw, eta_logit.unsqueeze(-1)), dim=-1)
