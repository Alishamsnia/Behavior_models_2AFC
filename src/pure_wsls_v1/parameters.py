"""Pooled parameter transformations for Pure_WSLS_v1.

Pure_WSLS_v1 has no sensory readout at all, so it does not decode
``w``/``s0``/``kappa`` and does not depend on
``m0_sensory_v1.model.parameters.decode_raw_parameters`` -- unlike
``m1_stimulus_kernel_v1``, which still carries M0's full five-parameter
sensory/lapse vector even though its own hazard/hidden-state machinery is
gone. Only the two category-dependent lapse rates are fitted; ``alpha`` is
fixed at exactly ``1.0`` and is never a decoded coordinate (see
``wsls_core_v1.evaluator`` for why the exact literal float matters).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PURE_WSLS_PARAMETER_ORDER = ("ell_short", "ell_long")
PURE_WSLS_FIXED_ALPHA = 1.0


@dataclass(frozen=True)
class PureWslsDecodedParameters:
    """The two fitted lapse rates plus the fixed learning rate for export parity."""

    ell_short: torch.Tensor
    ell_long: torch.Tensor
    alpha: float

    def as_tensor(self) -> torch.Tensor:
        """Stack fitted parameters as ``(ell_short, ell_long)``."""

        return torch.stack((self.ell_short, self.ell_long), dim=-1)


def decode_pure_wsls_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> PureWslsDecodedParameters:
    r"""Decode Pure_WSLS's unconstrained two-parameter vector.

    Purpose
    -------
    Convert raw optimizer coordinates into the two category-dependent lapse
    rates, reproducing M0's own lapse transform exactly without carrying
    any sensory coordinate.

    Mathematical definition
    -----------------------
    .. math::

       \ell_k=\ell_{\max}\operatorname{sigmoid}(\widetilde\ell_k),\quad
       k\in\{\mathrm{short},\mathrm{long}\}.

    Parameter interpretation
    ------------------------
    ``epsilon`` is accepted for call-site parity with every sibling
    family's decoder signature but is unused: unlike ``w``/``s0``, a lapse
    rate needs no additive floor above zero (the open sigmoid range already
    keeps it in ``(0, lapse_max)``).

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating tensor with final dimension two, in
    ``(ell_short, ell_long)`` order.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors with the input's leading batch shape, plus the
    fixed Python float ``alpha=1.0``.

    Constraints
    -----------
    Finite raw values guarantee :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`.

    Numerical stability
    -------------------
    Torch's stable sigmoid is used in the input dtype and on the input device.

    Causal restrictions
    -------------------
    Decoding receives no trial, outcome, or session data.

    Related tests
    -------------
    ``test_pure_wsls_parameter_round_trip_and_bounds``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 2:
        raise ValueError("Pure_WSLS raw_parameters must have final dimension 2")
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")
    raw_ell_short, raw_ell_long = raw_parameters.unbind(dim=-1)
    return PureWslsDecodedParameters(
        ell_short=lapse_max * torch.sigmoid(raw_ell_short),
        ell_long=lapse_max * torch.sigmoid(raw_ell_long),
        alpha=PURE_WSLS_FIXED_ALPHA,
    )


def encode_pure_wsls_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable ``(ell_short, ell_long)`` values to raw coordinates.

    Purpose
    -------
    Initialize Pure_WSLS from scientific lapse values, mirroring M0's own
    ``logit(ell/lapse_max)`` transform exactly.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde\ell_k=\operatorname{logit}(\ell_k/\ell_{\max}).

    Input units and shapes
    ----------------------
    A floating tensor with final dimension two.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`.

    Related tests
    -------------
    ``test_pure_wsls_parameter_round_trip_and_bounds``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 2:
        raise ValueError("Pure_WSLS constrained_parameters must have final dimension 2")
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")
    ell_short, ell_long = constrained_parameters.unbind(dim=-1)
    valid = (
        (ell_short > 0.0) & (ell_short < lapse_max)
        & (ell_long > 0.0) & (ell_long < lapse_max)
    )
    if not bool(torch.all(valid).detach().cpu()):
        raise ValueError("Pure_WSLS constrained lapse rates lie outside their open bounds")
    raw_ell_short = torch.logit(ell_short / lapse_max)
    raw_ell_long = torch.logit(ell_long / lapse_max)
    return torch.stack((raw_ell_short, raw_ell_long), dim=-1)
