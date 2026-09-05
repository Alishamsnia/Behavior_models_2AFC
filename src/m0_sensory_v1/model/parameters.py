"""Constrained M0 parameterization implemented with Torch autograd."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa")


@dataclass(frozen=True)
class DecodedParameters:
    """Named constrained M0 parameters, each retaining its autograd graph."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor

    @property
    def log_kappa(self) -> torch.Tensor:
        """Return the additive log-cost used by the Bayesian decision rule."""

        return torch.log(self.kappa)

    def as_tensor(self) -> torch.Tensor:
        """Stack parameters in order ``(w,s0,ell_short,ell_long,kappa)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa), dim=-1
        )


def decode_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> DecodedParameters:
    r"""Map unconstrained optimizer coordinates to scientific parameters.

    Purpose
    -------
    Apply every M0 parameter constraint in one differentiable location. Adam
    will optimize raw coordinates rather than projecting constrained values.

    Mathematical definition
    -----------------------
    For raw vector
    :math:`\widetilde\Theta=(\widetilde w,\widetilde s_0,
    \widetilde\ell_{\mathrm{short}},\widetilde\ell_{\mathrm{long}},
    \widetilde{\log\kappa})`, decode

    .. math::

       w &= \operatorname{softplus}(\widetilde w)+\epsilon,\\
       s_0 &= \operatorname{softplus}(\widetilde s_0)+\epsilon,\\
       \ell_{\mathrm{short}} &= \ell_{\max}
          \operatorname{sigmoid}(\widetilde\ell_{\mathrm{short}}),\\
       \ell_{\mathrm{long}} &= \ell_{\max}
          \operatorname{sigmoid}(\widetilde\ell_{\mathrm{long}}),\\
       \kappa &= \exp(\widetilde{\log\kappa}).

    Parameter interpretation
    ------------------------
    ``w`` scales duration-dependent variance, ``s0`` is baseline sensory SD in
    seconds. ``ell_short`` and ``ell_long`` are the lapse probabilities on
    short- and long-category stimulus trials, respectively. Conditional on a
    lapse, the response is uniform over the two alternatives. ``kappa`` is the
    positive internal decision-cost ratio: one is neutral, values above one
    favor short choices, and values below one favor long choices.

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating Torch tensor with final dimension five.
    Scalar fitting uses shape ``(5,)`` and preferably ``torch.float64``.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors with the batch shape preceding the final input
    dimension. Their autograd links to ``raw_parameters`` are preserved.

    Constraints
    -----------
    Guarantees :math:`w>0`, :math:`s_0>0`,
    :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`, and
    :math:`\kappa>0` for finite raw values. Unlike the lapse rates, ``kappa``
    has no registered upper bound: the exponential decoder is intentionally
    left unconstrained above zero, so the optimizer alone is responsible for
    keeping it finite. This is a deliberate design choice, not an omission.

    Numerical stability
    -------------------
    Torch's stable softplus and sigmoid implementations are used. ``epsilon``
    prevents exactly zero sensory scales.

    Causal restrictions
    -------------------
    This function receives parameters only; it cannot use trials, outcomes,
    blocks, histories, or session identities.

    Important edge cases
    --------------------
    ``lapse_max`` and ``epsilon`` must be positive. Non-floating and incorrectly
    shaped tensors raise before optimization.

    Related tests
    -------------
    ``test_parameter_round_trip_and_constraints`` and
    ``test_masked_prediction_autograd_matches_finite_difference``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 5:
        raise ValueError(
            "raw_parameters must have final dimension 5 in "
            "(w,s0,ell_short,ell_long,kappa) order"
        )
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")

    raw_w, raw_s0, raw_ell_short, raw_ell_long, raw_log_kappa = raw_parameters.unbind(dim=-1)
    return DecodedParameters(
        w=F.softplus(raw_w) + epsilon,
        s0=F.softplus(raw_s0) + epsilon,
        ell_short=lapse_max * torch.sigmoid(raw_ell_short),
        ell_long=lapse_max * torch.sigmoid(raw_ell_long),
        # Intentionally unbounded above (unlike ell_short/ell_long, which are
        # capped by lapse_max): kappa=exp(raw_log_kappa) only guarantees
        # kappa>0, with no registered ceiling on bias strength.
        kappa=torch.exp(raw_log_kappa),
    )


def encode_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Convert readable constrained initial values to raw coordinates.

    Purpose
    -------
    Initialize an unconstrained optimizer from scientifically interpretable
    values without duplicating transform conventions.

    Mathematical definition
    -----------------------
    For :math:`\Theta=(w,s_0,\ell_short,ell_long,\kappa)`, compute

    .. math::

       \widetilde w &= \operatorname{softplus}^{-1}(w-\epsilon),\\
       \widetilde s_0 &= \operatorname{softplus}^{-1}(s_0-\epsilon),\\
       \widetilde\ell_{\mathrm{short}} &=
          \operatorname{logit}(\ell_{\mathrm{short}}/\ell_{\max}),\\
       \widetilde\ell_{\mathrm{long}} &=
          \operatorname{logit}(\ell_{\mathrm{long}}/\ell_{\max}),\\
       \widetilde{\log\kappa} &= \log\kappa.

    Parameter interpretation
    ------------------------
    Input order is ``w, s0, ell_short, ell_long, kappa`` and matches
    :func:`decode_raw_parameters` exactly.

    Input units and shapes
    ----------------------
    Floating Torch tensor with final dimension five; ``s0`` is in seconds.

    Returned quantities and shapes
    ------------------------------
    Raw Torch tensor with the same shape, dtype, and device.

    Constraints
    -----------
    Requires :math:`w>\epsilon`, :math:`s_0>\epsilon`,
    :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`, and
    finite :math:`\kappa>0`. No upper bound on ``kappa`` is enforced or
    required here; only finiteness and positivity are checked, by design.

    Numerical stability
    -------------------
    The inverse softplus uses the stable identity
    :math:`x+\log(-\operatorname{expm1}(-x))`.

    Causal restrictions
    -------------------
    No behavioral data are accepted.

    Important edge cases
    --------------------
    Boundary initializations are rejected because their raw logits are
    infinite and unsuitable for Adam.

    Related tests
    -------------
    ``test_parameter_round_trip_and_constraints``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 5:
        raise ValueError("constrained_parameters must have final dimension 5")
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")

    w, s0, ell_short, ell_long, kappa = constrained_parameters.unbind(dim=-1)
    valid = (
        (w > epsilon)
        & (s0 > epsilon)
        & (ell_short > 0.0)
        & (ell_short < lapse_max)
        & (ell_long > 0.0)
        & (ell_long < lapse_max)
        & torch.isfinite(kappa)
        & (kappa > 0.0)
    )
    if not bool(torch.all(valid).detach().cpu()):
        raise ValueError("Constrained parameters lie outside their open bounds")

    def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        return value + torch.log(-torch.expm1(-value))

    raw_w = inverse_softplus(w - epsilon)
    raw_s0 = inverse_softplus(s0 - epsilon)
    raw_ell_short = torch.logit(ell_short / lapse_max)
    raw_ell_long = torch.logit(ell_long / lapse_max)
    raw_log_kappa = torch.log(kappa)
    return torch.stack(
        (raw_w, raw_s0, raw_ell_short, raw_ell_long, raw_log_kappa), dim=-1
    )
