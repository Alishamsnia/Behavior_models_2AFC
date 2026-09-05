"""Pooled parameters for the capped-tail duration-aware M4 HSMM."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from m0_sensory_v1.model.parameters import DecodedParameters
from m3_hmm_delta_v1.parameters import (
    M3DecodedParameters,
    decode_m3_raw_parameters,
    encode_m3_constrained_parameters,
)


M4_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta", "bh"
)


@dataclass(frozen=True)
class M4DecodedParameters:
    """Eight pooled values for M3 plus a nonnegative run-length slope."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    delta0: torch.Tensor
    alpha_theta: torch.Tensor
    bh: torch.Tensor

    def common(self) -> DecodedParameters:
        return DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
        )

    def prototypes(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        active = self.delta0 if delta is None else delta
        return torch.stack(
            (torch.ones_like(active) * 0.5, torch.sigmoid(-active), torch.sigmoid(active)),
            dim=-1,
        )

    def hazard(self, run_length: torch.Tensor, *, center: int = 23) -> torch.Tensor:
        """Return ``sigmoid(ah + bh * (r-center))`` on represented run lengths."""

        if not isinstance(run_length, torch.Tensor) or not run_length.is_floating_point():
            raise TypeError("run_length must be a floating Torch tensor")
        if center <= 0:
            raise ValueError("hazard center must be positive")
        return torch.sigmoid(self.ah + self.bh * (run_length - float(center)))

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah, self.delta0,
             self.alpha_theta, self.bh),
            dim=-1,
        )

    def as_m3(self) -> M3DecodedParameters:
        """Return the first eight coordinates as an M3 parameter object."""

        return M3DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
            ah=self.ah,
            delta0=self.delta0,
            alpha_theta=self.alpha_theta,
            hazard=torch.sigmoid(self.ah),
        )

    def with_explicit_alpha(self, value: float | torch.Tensor) -> "M4DecodedParameters":
        alpha = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if alpha.ndim != 0 or not bool(torch.isfinite(alpha) & (alpha >= 0) & (alpha <= 1)):
            raise ValueError("Explicit alpha_theta must lie in [0,1]")
        return replace(self, alpha_theta=alpha)

    def with_explicit_bh(self, value: float | torch.Tensor) -> "M4DecodedParameters":
        slope = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if slope.ndim != 0 or not bool(torch.isfinite(slope) & (slope >= 0)):
            raise ValueError("Explicit bh must be a finite nonnegative scalar")
        return replace(self, bh=slope)


def decode_m4_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M4DecodedParameters:
    """Decode M3's first eight raw coordinates and positive ``bh``."""

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 9:
        raise ValueError("M4 raw_parameters must have final dimension 9")
    m3 = decode_m3_raw_parameters(
        raw_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    return M4DecodedParameters(
        w=m3.w,
        s0=m3.s0,
        ell_short=m3.ell_short,
        ell_long=m3.ell_long,
        kappa=m3.kappa,
        ah=m3.ah,
        delta0=m3.delta0,
        alpha_theta=m3.alpha_theta,
        bh=F.softplus(raw_parameters[..., 8]) + epsilon,
    )


def encode_m4_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Encode readable M4 coordinates into an unconstrained nine-vector."""

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 9:
        raise ValueError("M4 constrained_parameters must have final dimension 9")
    bh = constrained_parameters[..., 8]
    if not bool(torch.all(torch.isfinite(bh) & (bh > epsilon))):
        raise ValueError("Free M4 bh must be finite and greater than epsilon")
    m3_raw = encode_m3_constrained_parameters(
        constrained_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    shifted = bh - epsilon
    raw_bh = shifted + torch.log(-torch.expm1(-shifted))
    return torch.cat((m3_raw, raw_bh.unsqueeze(-1)), dim=-1)
