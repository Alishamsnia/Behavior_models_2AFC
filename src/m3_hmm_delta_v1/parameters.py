"""Pooled parameter transformations for ``M3_hmm_delta_v1``."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from m0_sensory_v1.model.m2_parameters import (
    decode_m2_raw_parameters,
    encode_m2_constrained_parameters,
)
from m0_sensory_v1.model.parameters import DecodedParameters


M3_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "alpha_theta"
)


@dataclass(frozen=True)
class M3DecodedParameters:
    """Seven pooled fitted values and the derived constant biased hazard."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    delta0: torch.Tensor
    alpha_theta: torch.Tensor
    hazard: torch.Tensor

    def common(self) -> DecodedParameters:
        """Return the unchanged sensory/lapse parameter object."""

        return DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
        )

    def prototypes(self, delta: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``(0.5, sigmoid(-delta), sigmoid(delta))`` in ``(N,S,L)`` order."""

        active_delta = self.delta0 if delta is None else delta
        neutral = torch.ones_like(active_delta) * 0.5
        return torch.stack(
            (neutral, torch.sigmoid(-active_delta), torch.sigmoid(active_delta)),
            dim=-1,
        )

    def as_tensor(self) -> torch.Tensor:
        """Stack fitted coordinates in the immutable registered order."""

        return torch.stack(
            (
                self.w,
                self.s0,
                self.ell_short,
                self.ell_long,
                self.kappa,
                self.ah,
                self.delta0,
                self.alpha_theta,
            ),
            dim=-1,
        )

    def with_explicit_alpha(self, alpha_theta: float | torch.Tensor) -> "M3DecodedParameters":
        """Construct an exact boundary model, including ``alpha_theta=0``.

        The free sigmoid decoder cannot attain zero at a finite raw coordinate.
        Exact M2-nesting tests therefore use this explicit constrained helper.
        """

        alpha = torch.as_tensor(alpha_theta, dtype=self.w.dtype, device=self.w.device)
        if alpha.ndim != 0 or not bool(
            torch.isfinite(alpha) & (alpha >= 0.0) & (alpha <= 1.0)
        ):
            raise ValueError("Explicit alpha_theta must be a finite scalar in [0,1]")
        return replace(self, alpha_theta=alpha)


def decode_m3_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M3DecodedParameters:
    """Decode the eight raw pooled coordinates without mutating M0–M2.

    The first seven transforms are exactly M2's transforms. The new scalar is
    ``alpha_theta=sigmoid(raw_alpha_theta)``, so finite free coordinates lie
    strictly in ``(0,1)``. Exact ``alpha_theta=0`` exists only through
    :meth:`M3DecodedParameters.with_explicit_alpha` for nesting and generating
    boundary cases. Inputs have final dimension eight and outputs preserve any
    leading batch dimensions.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 8:
        raise ValueError("M3 raw_parameters must have final dimension 8")
    m2 = decode_m2_raw_parameters(
        raw_parameters[..., :7], lapse_max=lapse_max, epsilon=epsilon
    )
    alpha_theta = torch.sigmoid(raw_parameters[..., 7])
    return M3DecodedParameters(
        w=m2.w,
        s0=m2.s0,
        ell_short=m2.ell_short,
        ell_long=m2.ell_long,
        kappa=m2.kappa,
        ah=m2.ah,
        delta0=m2.delta0,
        alpha_theta=alpha_theta,
        hazard=m2.hazard,
    )


def encode_m3_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """Encode readable M3 values into eight unconstrained optimizer coordinates."""

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 8:
        raise ValueError("M3 constrained_parameters must have final dimension 8")
    alpha = constrained_parameters[..., 7]
    if not bool(torch.all(torch.isfinite(alpha) & (alpha > 0.0) & (alpha < 1.0))):
        raise ValueError("Free M3 alpha_theta must lie strictly in (0,1)")
    m2_raw = encode_m2_constrained_parameters(
        constrained_parameters[..., :7], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((m2_raw, torch.logit(alpha).unsqueeze(-1)), dim=-1)
