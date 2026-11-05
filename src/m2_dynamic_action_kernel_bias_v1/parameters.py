"""Pooled parameter transformations for M2_dynamic_action_kernel_bias_v1.

M2_dynamic_action_kernel_bias_v1
(sensory_learnRatio_learnDynamicBlockSwitch_actionHistory_lapse) keeps
``M2_dynamic_previous_action_bias_v1``'s nine pooled values completely
unchanged -- M2_dynamic_block_length_v1's eight plus the bias-mixing-weight
logit ``eta_logit`` -- and appends exactly one new pooled coordinate: the
action-history learning-rate logit ``a``, with ``alpha = sigmoid(a)``. The
parent fixes ``alpha`` at the constant 1.0 (the one-trial-back "previous
action" special case); this model fits it, so ``pi_t`` becomes an
exponentially weighted action history. Appending ``a`` last keeps the first
nine coordinates byte-for-byte the parent's own order and transforms.

The first eight coordinates decode/encode through
M2_dynamic_block_length_v1's own registered transform
(``m2_dynamic_block_length_v1.parameters``, reused unmodified); ``eta_logit``
and ``a`` are stored raw (the logits themselves), the same convention the
parent uses for ``eta_logit`` and ``M1_action_kernel_v1`` uses for ``a``.
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


M2_DYNAMIC_ACTION_KERNEL_BIAS_PARAMETER_ORDER = (
    "w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0", "bh", "eta_logit", "a",
)


@dataclass(frozen=True)
class M2DynamicActionKernelBiasParameters:
    """M2_dynamic_previous_action_bias_v1's nine pooled values plus the fitted
    action-history learning-rate logit ``a`` (``alpha = sigmoid(a)``)."""

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
    a: torch.Tensor
    alpha: torch.Tensor

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
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit,a)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa,
             self.ah, self.delta0, self.bh, self.eta_logit, self.a),
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

    def with_explicit_eta(self, value: float | torch.Tensor) -> "M2DynamicActionKernelBiasParameters":
        """Return a copy with ``eta`` pinned to an explicit scalar in ``[0,1]``.

        ``sigmoid(eta_logit)`` never exactly reaches ``0`` for finite
        ``eta_logit``, so the exact ``eta=0`` nesting boundary must be
        constructed directly.
        """

        eta_value = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if eta_value.ndim != 0 or not bool(torch.isfinite(eta_value) & (eta_value >= 0.0) & (eta_value <= 1.0)):
            raise ValueError("Explicit eta must be a finite scalar in [0,1]")
        return replace(self, eta=eta_value)

    def with_explicit_bh(self, value: float | torch.Tensor) -> "M2DynamicActionKernelBiasParameters":
        """Return a copy with ``bh`` pinned to an explicit nonnegative scalar
        (the exact ``bh=0`` boundary is unreachable through softplus)."""

        bh_value = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if bh_value.ndim != 0 or not bool(torch.isfinite(bh_value) & (bh_value >= 0.0)):
            raise ValueError("Explicit bh must be a finite nonnegative scalar")
        return replace(self, bh=bh_value)

    def with_explicit_alpha(self, value: float | torch.Tensor) -> "M2DynamicActionKernelBiasParameters":
        """Return a copy with ``alpha`` pinned to an explicit scalar in ``[0,1]``.

        ``sigmoid(a)`` never exactly reaches ``1`` for finite ``a``, so the
        exact ``alpha=1`` boundary -- at which this model reproduces
        ``M2_dynamic_previous_action_bias_v1`` exactly -- must be constructed
        directly.
        """

        alpha_value = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if alpha_value.ndim != 0 or not bool(
            torch.isfinite(alpha_value) & (alpha_value >= 0.0) & (alpha_value <= 1.0)
        ):
            raise ValueError("Explicit alpha must be a finite scalar in [0,1]")
        return replace(self, alpha=alpha_value)


def decode_m2_dynamic_action_kernel_bias_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2DynamicActionKernelBiasParameters:
    r"""Decode M2_dynamic_action_kernel_bias's ten unconstrained coordinates.

    Mathematical definition
    -----------------------
    The first eight coordinates use M2_dynamic_block_length_v1's own
    decoder, and

    .. math::

       \eta=\operatorname{sigmoid}(\widetilde\eta),\qquad
       \alpha=\operatorname{sigmoid}(a).

    Constraints
    -----------
    Finite raw values imply ``0<eta<1`` and ``0<alpha<1``. Exact ``eta=0``
    and ``alpha=1`` are unavailable through the free decoder and must be
    constructed via ``with_explicit_eta``/``with_explicit_alpha``.

    Related tests
    -------------
    ``test_parameter_round_trip_and_bounds``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 10:
        raise ValueError("M2_dynamic_action_kernel_bias raw_parameters must have final dimension 10")
    m2_dynamic = decode_m2_dynamic_block_length_raw_parameters(
        raw_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    eta_logit = raw_parameters[..., 8]
    a = raw_parameters[..., 9]
    return M2DynamicActionKernelBiasParameters(
        w=m2_dynamic.w, s0=m2_dynamic.s0, ell_short=m2_dynamic.ell_short, ell_long=m2_dynamic.ell_long,
        kappa=m2_dynamic.kappa, ah=m2_dynamic.ah, delta0=m2_dynamic.delta0, bh=m2_dynamic.bh,
        eta_logit=eta_logit, eta=torch.sigmoid(eta_logit),
        a=a, alpha=torch.sigmoid(a),
    )


def encode_m2_dynamic_action_kernel_bias_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2_dynamic_action_kernel_bias values to raw coordinates.

    Input order is ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit,a)``;
    the last two values are logits (stored raw), not probabilities. The first
    eight use M2_dynamic_block_length_v1's exact encoder. Infinite logits are
    rejected.

    Related tests
    -------------
    ``test_parameter_round_trip_and_bounds``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 10:
        raise ValueError("M2_dynamic_action_kernel_bias constrained_parameters must have final dimension 10")
    logits = constrained_parameters[..., 8:10]
    if not bool(torch.all(torch.isfinite(logits)).detach().cpu()):
        raise ValueError("M2_dynamic_action_kernel_bias bias-weight and learning-rate logits must be finite")
    m2_dynamic_raw = encode_m2_dynamic_block_length_constrained_parameters(
        constrained_parameters[..., :8], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((m2_dynamic_raw, logits), dim=-1)
