"""Pooled parameter transformations for the true-rate constant-hazard M1 HMM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M1_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa", "ah")


@dataclass(frozen=True)
class M1DecodedParameters:
    """Named M1 parameters and the derived constant switch hazard."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    hazard: torch.Tensor

    def common(self) -> DecodedParameters:
        """Return the five parameters shared with M0's Bayesian readout."""

        return DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
        )

    def as_tensor(self) -> torch.Tensor:
        """Stack parameters as ``(w,s0,ell_short,ell_long,kappa,ah)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah),
            dim=-1,
        )


def decode_m1_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M1DecodedParameters:
    r"""Decode M1's unconstrained pooled parameter vector.

    Purpose
    -------
    Extend the unchanged M0 sensory/lapse decoder with one constant-hazard
    logit shared by every session of a mouse.

    Mathematical definition
    -----------------------
    The first five coordinates use M0's transforms and the sixth obeys

    .. math::

       a_h=\widetilde a_h,\qquad h=\operatorname{sigmoid}(a_h).

    Parameter interpretation
    ------------------------
    ``ah`` is the fitted log switch odds per task trial and ``hazard`` is its
    probability-scale representation.

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating tensor with final dimension six.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors with the input's leading batch shape.

    Constraints
    -----------
    Finite raw values guarantee ``0 < hazard < 1``; common M0 constraints are
    delegated to :func:`decode_raw_parameters`.

    Numerical stability
    -------------------
    Torch's stable sigmoid is used in the input dtype and on the input device.

    Causal restrictions
    -------------------
    Decoding receives no trial, outcome, block, history, or session data.

    Important edge cases
    --------------------
    Exact kernel endpoints ``h=0`` and ``h=1`` are tested by passing hazards
    directly to the transition kernel, not through this finite-logit decoder.

    Related tests
    -------------
    ``test_m1_parameter_round_trip_and_hazard_constraints``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 6:
        raise ValueError("M1 raw_parameters must have final dimension 6")
    common = decode_raw_parameters(
        raw_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    ah = raw_parameters[..., 5]
    return M1DecodedParameters(
        w=common.w,
        s0=common.s0,
        ell_short=common.ell_short,
        ell_long=common.ell_long,
        kappa=common.kappa,
        ah=ah,
        hazard=torch.sigmoid(ah),
    )


def encode_m1_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M1 values ``(w,s0,ell_short,ell_long,kappa,ah)`` to raw coordinates.

    Purpose
    -------
    Initialize M1 from scientific values while reusing M0's exact common
    parameter transforms.

    Mathematical definition
    -----------------------
    The common coordinates are inverse-transformed as for M0 and

    .. math::

       \widetilde a_h=a_h.

    Parameter interpretation
    ------------------------
    The sixth value is a hazard logit, not a probability. Its interpretable
    probability is ``sigmoid(ah)``.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension six; ``s0`` is seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Common values obey M0's open bounds and ``ah`` must be finite.

    Numerical stability
    -------------------
    Stable inverse softplus/logit operations are delegated to M0's encoder.

    Causal restrictions
    -------------------
    No behavioral data are accepted.

    Important edge cases
    --------------------
    Infinite hazard logits are rejected; exact transition endpoints belong to
    direct transition-kernel tests only.

    Related tests
    -------------
    ``test_m1_parameter_round_trip_and_hazard_constraints``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 6:
        raise ValueError("M1 constrained_parameters must have final dimension 6")
    ah = constrained_parameters[..., 5]
    if not bool(torch.all(torch.isfinite(ah)).detach().cpu()):
        raise ValueError("M1 hazard logits must be finite")
    common_raw = encode_constrained_parameters(
        constrained_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((common_raw, ah.unsqueeze(-1)), dim=-1)
