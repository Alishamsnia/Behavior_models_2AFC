"""Pooled parameter transformations for M2_dynamic_block_length_v1.

M2's fitted static prototypes (via ``delta0``) plus one nonnegative
duration-dependent hazard slope ``bh``. Independent of ``m4_hsmm_delta_v1``:
the eight shared coordinates decode through M2's own
``decode_m2_raw_parameters``/``encode_m2_constrained_parameters``
(``m0_sensory_v1.model.m2_parameters``), and the hazard formula delegates to
the shared, independently-authored ``capped_tail_hazard``
(``m0_sensory_v1.model.capped_tail_hsmm``) -- the single production
definition of that formula. There is no prototype-learning rate and no delta
update anywhere in this model: ``delta0`` is static for the whole session,
exactly as in M2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from m0_sensory_v1.model.capped_tail_hsmm import capped_tail_hazard
from m0_sensory_v1.model.m2_parameters import (
    M2DecodedParameters,
    decode_m2_raw_parameters,
    encode_m2_constrained_parameters,
)
from m0_sensory_v1.model.parameters import DecodedParameters


M2_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh"
)


@dataclass(frozen=True)
class M2DynamicBlockLengthParameters:
    """M2's seven pooled values plus a nonnegative run-length hazard slope."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    delta0: torch.Tensor
    bh: torch.Tensor

    def common(self) -> DecodedParameters:
        """Return the unchanged five-parameter Bayesian sensory/lapse readout."""

        return DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
        )

    def prototypes(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Return static ``(0.5,sigmoid(-delta0),sigmoid(delta0))`` in ``(N,S,L)`` order.

        Exactly M2's own formula (``M2DecodedParameters.prototypes``).
        Accepts an explicit ``delta`` override only for boundary-test symmetry
        with the wider family's convention; the free model always uses
        ``self.delta0``.
        """

        active = self.delta0 if delta is None else delta
        neutral = torch.ones_like(active) * 0.5
        long = torch.sigmoid(active)
        short = torch.sigmoid(-active)
        return torch.stack((neutral, short, long), dim=-1)

    def hazard(self, run_length: torch.Tensor, *, center: int = 23) -> torch.Tensor:
        """Return ``sigmoid(ah+bh*(r-center))`` via the shared kernel formula."""

        return capped_tail_hazard(self.ah, self.bh, run_length, center=center)

    def as_tensor(self) -> torch.Tensor:
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh)``."""

        return torch.stack(
            (
                self.w, self.s0, self.ell_short, self.ell_long, self.kappa,
                self.ah, self.delta0, self.bh,
            ),
            dim=-1,
        )

    def as_m2(self) -> M2DecodedParameters:
        """Return the ``bh``-dropped ``M2DecodedParameters`` boundary object.

        Used only by the exact ``bh=0`` boundary test against real M2 — the
        conceptual restriction this model nests exactly at that boundary.
        """

        return M2DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
            ah=self.ah,
            delta0=self.delta0,
            hazard=torch.sigmoid(self.ah),
        )

    def with_explicit_bh(self, value: float | torch.Tensor) -> "M2DynamicBlockLengthParameters":
        """Return a copy with ``bh`` pinned to an explicit nonnegative scalar."""

        slope = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if slope.ndim != 0 or not bool(torch.isfinite(slope) & (slope >= 0)):
            raise ValueError("Explicit bh must be a finite nonnegative scalar")
        return replace(self, bh=slope)


def decode_m2_dynamic_block_length_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2DynamicBlockLengthParameters:
    r"""Decode M2_dynamic_block_length's eight unconstrained coordinates.

    Purpose
    -------
    Reuse M2's own seven-coordinate decoder unchanged and add one nonnegative
    duration-dependent hazard slope.

    Mathematical definition
    -----------------------
    .. math::

       b_h=\operatorname{softplus}(\widetilde b_h)+\epsilon.

    Parameter interpretation
    ------------------------
    The first seven coordinates are M2's unchanged sensory/lapse/hazard-anchor/
    static-prototype-contrast values; ``bh`` is the nonnegative run-length
    hazard slope. ``delta0`` keeps M2's own domain (no artificial floor):
    there is no delta-update rule here to protect from a runaway negative
    proposal, so M2's ordinary ``softplus(raw)+epsilon`` positivity is
    sufficient on its own.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension eight.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply ``bh>0`` (strictly positive via the registered
    epsilon floor).

    Numerical stability
    -------------------
    Stable Torch softplus is used with the registered epsilon.

    Causal restrictions
    -------------------
    Decoding accepts parameters only and cannot inspect behavioral history.

    Important edge cases
    --------------------
    Exact ``bh=0`` is intentionally unavailable through the free decoder and
    must be constructed directly via ``with_explicit_bh(0.0)`` for the
    boundary test against real M2.

    Related tests
    -------------
    ``test_m2_dynamic_block_length_parameter_round_trip``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 8:
        raise ValueError("M2_dynamic_block_length raw_parameters must have final dimension 8")
    m2 = decode_m2_raw_parameters(
        raw_parameters[..., :7], lapse_max=lapse_max, epsilon=epsilon
    )
    bh = F.softplus(raw_parameters[..., 7]) + epsilon
    return M2DynamicBlockLengthParameters(
        w=m2.w, s0=m2.s0, ell_short=m2.ell_short, ell_long=m2.ell_long,
        kappa=m2.kappa, ah=m2.ah, delta0=m2.delta0, bh=bh,
    )


def encode_m2_dynamic_block_length_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2_dynamic_block_length values to raw coordinates.

    Purpose
    -------
    Initialize this model from scientific values while reusing M2's exact
    seven-coordinate encoder.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde b_h=\operatorname{softplus}^{-1}(b_h-\epsilon).

    Parameter interpretation
    ------------------------
    Input order is ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh)``.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension eight.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite ``ah``, ``delta0>epsilon``, and ``bh>epsilon``.

    Numerical stability
    -------------------
    Inverse softplus uses ``x + log(-expm1(-x))`` to avoid overflow/cancellation.

    Causal restrictions
    -------------------
    Initialization is independent of trials, outcomes, blocks, and feedback.

    Important edge cases
    --------------------
    The exact ``bh=0`` boundary is rejected because its raw value is infinite;
    use ``with_explicit_bh(0.0)`` for boundary tests instead.

    Related tests
    -------------
    ``test_m2_dynamic_block_length_parameter_round_trip`` and the frozen
    eight-start registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 8:
        raise ValueError(
            "M2_dynamic_block_length constrained_parameters must have final dimension 8"
        )
    bh = constrained_parameters[..., 7]
    if not bool(torch.all(torch.isfinite(bh) & (bh > epsilon)).detach().cpu()):
        raise ValueError("M2_dynamic_block_length bh must be finite and greater than epsilon")
    m2_raw = encode_m2_constrained_parameters(
        constrained_parameters[..., :7], lapse_max=lapse_max, epsilon=epsilon
    )
    shifted = bh - epsilon
    raw_bh = shifted + torch.log(-torch.expm1(-shifted))
    return torch.cat((m2_raw, raw_bh.unsqueeze(-1)), dim=-1)
