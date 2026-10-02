"""Pooled parameter transformations for Extended_WSLS_v1.

Extended_WSLS_v1 has no sensory readout at all, so it does not decode
``w``/``s0``/``kappa`` and does not depend on
``m0_sensory_v1.model.parameters.decode_raw_parameters``. Three coordinates
are fitted: the two category-dependent lapse rates plus a learning-rate
logit ``a`` whose sigmoid is the scalar delta-rule rate ``alpha`` -- the
same ``a``/``alpha`` convention ``m1_stimulus_kernel_v1`` already uses,
reproduced here without any sensory coordinate alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EXTENDED_WSLS_PARAMETER_ORDER = ("ell_short", "ell_long", "a")


@dataclass(frozen=True)
class ExtendedWslsDecodedParameters:
    """The two fitted lapse rates plus the fitted learning-rate logit/rate."""

    ell_short: torch.Tensor
    ell_long: torch.Tensor
    a: torch.Tensor
    alpha: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        """Stack fitted parameters as ``(ell_short, ell_long, a)``."""

        return torch.stack((self.ell_short, self.ell_long, self.a), dim=-1)


def decode_extended_wsls_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> ExtendedWslsDecodedParameters:
    r"""Decode Extended_WSLS's unconstrained three-parameter vector.

    Mathematical definition
    -----------------------
    .. math::

       \ell_k=\ell_{\max}\operatorname{sigmoid}(\widetilde\ell_k),\quad
       k\in\{\mathrm{short},\mathrm{long}\},\qquad
       a=\widetilde a,\qquad \alpha=\operatorname{sigmoid}(a).

    Parameter interpretation
    ------------------------
    ``epsilon`` is accepted for call-site parity with every sibling
    family's decoder signature but is unused (see
    ``pure_wsls_v1.parameters.decode_pure_wsls_raw_parameters`` for why a
    lapse rate needs no additive floor).

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating tensor with final dimension three, in
    ``(ell_short, ell_long, a)`` order.

    Constraints
    -----------
    Finite raw values guarantee :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`
    and :math:`0<\alpha<1` (open, since ``sigmoid`` never reaches its
    endpoints for finite input).

    Important edge cases
    --------------------
    Very large finite ``|a|`` can saturate ``alpha`` to a value that rounds
    to exactly ``0.0`` or ``1.0`` in floating point; unlike
    ``pure_wsls_v1``, this model never treats that saturated value as the
    literal fixed point -- it always uses the ordinary algebraic recurrence
    (see ``wsls_core_v1.evaluator``).

    Related tests
    -------------
    ``test_extended_wsls_parameter_round_trip_and_alpha_constraints``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 3:
        raise ValueError("Extended_WSLS raw_parameters must have final dimension 3")
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")
    raw_ell_short, raw_ell_long, a = raw_parameters.unbind(dim=-1)
    return ExtendedWslsDecodedParameters(
        ell_short=lapse_max * torch.sigmoid(raw_ell_short),
        ell_long=lapse_max * torch.sigmoid(raw_ell_long),
        a=a,
        alpha=torch.sigmoid(a),
    )


def encode_extended_wsls_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable ``(ell_short, ell_long, a)`` values to raw coordinates.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde\ell_k=\operatorname{logit}(\ell_k/\ell_{\max}),\qquad
       \widetilde a=a.

    Constraints
    -----------
    Requires :math:`0<\ell_{\mathrm{short}},\ell_{\mathrm{long}}<\ell_{\max}`
    and finite ``a``.

    Related tests
    -------------
    ``test_extended_wsls_parameter_round_trip_and_alpha_constraints``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 3:
        raise ValueError("Extended_WSLS constrained_parameters must have final dimension 3")
    if not 0.0 < lapse_max <= 1.0 or epsilon <= 0.0:
        raise ValueError("lapse_max must lie in (0, 1] and epsilon must be positive")
    ell_short, ell_long, a = constrained_parameters.unbind(dim=-1)
    if not bool(torch.all(torch.isfinite(a)).detach().cpu()):
        raise ValueError("Extended_WSLS learning-rate logits must be finite")
    valid = (
        (ell_short > 0.0) & (ell_short < lapse_max)
        & (ell_long > 0.0) & (ell_long < lapse_max)
    )
    if not bool(torch.all(valid).detach().cpu()):
        raise ValueError("Extended_WSLS constrained lapse rates lie outside their open bounds")
    raw_ell_short = torch.logit(ell_short / lapse_max)
    raw_ell_long = torch.logit(ell_long / lapse_max)
    return torch.stack((raw_ell_short, raw_ell_long, a), dim=-1)
