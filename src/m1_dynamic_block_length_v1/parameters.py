"""Pooled parameter transformations for M1_dynamic_block_length_v1.

M1's fixed true-rate prototypes ``(0.5,0.2,0.8)`` plus one nonnegative
duration-dependent hazard slope ``bh``. Independent of ``m4_hsmm_delta_v1``:
the seven shared coordinates decode through M1's own
``decode_m1_raw_parameters``/``encode_m1_constrained_parameters``
(``m0_sensory_v1.model.m1_parameters``), and the hazard formula delegates to
the shared, independently-authored ``capped_tail_hazard``
(``m0_sensory_v1.model.capped_tail_hsmm``) -- the single production
definition of that formula.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from m0_sensory_v1.model.capped_tail_hsmm import capped_tail_hazard
from m0_sensory_v1.model.m1_parameters import (
    M1DecodedParameters,
    decode_m1_raw_parameters,
    encode_m1_constrained_parameters,
)
from m0_sensory_v1.model.parameters import DecodedParameters


M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "bh"
)


@dataclass(frozen=True)
class M1DynamicBlockLengthParameters:
    """M1's six pooled values plus a nonnegative run-length hazard slope."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
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

    def hazard(self, run_length: torch.Tensor, *, center: int = 23) -> torch.Tensor:
        """Return ``sigmoid(ah+bh*(r-center))`` via the shared kernel formula."""

        return capped_tail_hazard(self.ah, self.bh, run_length, center=center)

    def as_tensor(self) -> torch.Tensor:
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,ah,bh)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah, self.bh),
            dim=-1,
        )

    def as_m1(self) -> M1DecodedParameters:
        """Return the ``bh``-dropped ``M1DecodedParameters`` boundary object.

        Used only by the exact ``bh=0`` boundary test against real M1 — the
        conceptual restriction this model nests exactly at that boundary.
        """

        return M1DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
            ah=self.ah,
            hazard=torch.sigmoid(self.ah),
        )

    def with_explicit_bh(self, value: float | torch.Tensor) -> "M1DynamicBlockLengthParameters":
        """Return a copy with ``bh`` pinned to an explicit nonnegative scalar."""

        slope = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if slope.ndim != 0 or not bool(torch.isfinite(slope) & (slope >= 0)):
            raise ValueError("Explicit bh must be a finite nonnegative scalar")
        return replace(self, bh=slope)


def decode_m1_dynamic_block_length_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M1DynamicBlockLengthParameters:
    r"""Decode M1_dynamic_block_length's seven unconstrained coordinates.

    Purpose
    -------
    Reuse M1's own six-coordinate decoder unchanged and add one nonnegative
    duration-dependent hazard slope.

    Mathematical definition
    -----------------------
    .. math::

       b_h=\operatorname{softplus}(\widetilde b_h)+\epsilon.

    Parameter interpretation
    ------------------------
    The first six coordinates are M1's unchanged sensory/lapse/hazard-anchor
    values; ``bh`` is the nonnegative run-length hazard slope.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension seven.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply ``bh>0`` (strictly positive via the registered
    epsilon floor, matching every other softplus-decoded slope in this
    family).

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
    boundary test against real M1.

    Related tests
    -------------
    ``test_m1_dynamic_block_length_parameter_round_trip``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 7:
        raise ValueError("M1_dynamic_block_length raw_parameters must have final dimension 7")
    m1 = decode_m1_raw_parameters(
        raw_parameters[..., :6], lapse_max=lapse_max, epsilon=epsilon
    )
    bh = F.softplus(raw_parameters[..., 6]) + epsilon
    return M1DynamicBlockLengthParameters(
        w=m1.w, s0=m1.s0, ell_short=m1.ell_short, ell_long=m1.ell_long,
        kappa=m1.kappa, ah=m1.ah, bh=bh,
    )


def encode_m1_dynamic_block_length_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M1_dynamic_block_length values to raw coordinates.

    Purpose
    -------
    Initialize this model from scientific values while reusing M1's exact
    six-coordinate encoder.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde b_h=\operatorname{softplus}^{-1}(b_h-\epsilon).

    Parameter interpretation
    ------------------------
    Input order is ``(w,s0,ell_short,ell_long,kappa,ah,bh)``.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension seven.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite ``ah`` and ``bh>epsilon``.

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
    ``test_m1_dynamic_block_length_parameter_round_trip`` and the frozen
    eight-start registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 7:
        raise ValueError(
            "M1_dynamic_block_length constrained_parameters must have final dimension 7"
        )
    bh = constrained_parameters[..., 6]
    if not bool(torch.all(torch.isfinite(bh) & (bh > epsilon)).detach().cpu()):
        raise ValueError("M1_dynamic_block_length bh must be finite and greater than epsilon")
    m1_raw = encode_m1_constrained_parameters(
        constrained_parameters[..., :6], lapse_max=lapse_max, epsilon=epsilon
    )
    shifted = bh - epsilon
    raw_bh = shifted + torch.log(-torch.expm1(-shifted))
    return torch.cat((m1_raw, raw_bh.unsqueeze(-1)), dim=-1)
