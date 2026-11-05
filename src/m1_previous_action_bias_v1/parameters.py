"""Pooled parameter transformations for M1_previous_action_bias_v1.

M1_previous_action_bias_v1 keeps M0's unchanged five-parameter sensory/lapse
readout and adds exactly one new pooled coordinate, a logit ``eta_logit``
whose sigmoid is the previous-action bias-mixing weight ``eta`` (see
``session.py``). The first five coordinates decode/encode through M0's own
registered transform (reused, not reimplemented) and the sixth is stored raw
(the logit itself), exactly the shape ``M1_action_kernel_v1`` uses for ``a``
and ``M2_dynamic_previous_action_bias_v1`` uses for its own ``eta_logit``.
The action-history learning rate is NOT fitted: it is the fixed constant
``alpha = 1.0`` (carried on the decoded object for reporting only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from m0_sensory_v1.model.parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M1_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa", "eta_logit")
M1_PREVIOUS_ACTION_BIAS_FIXED_ALPHA = 1.0


@dataclass(frozen=True)
class M1PreviousActionBiasDecodedParameters:
    """M0's five pooled values plus the fitted bias-mixing-weight logit."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    eta_logit: torch.Tensor
    eta: torch.Tensor
    alpha: float = M1_PREVIOUS_ACTION_BIAS_FIXED_ALPHA

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
        """Stack parameters as ``(w,s0,ell_short,ell_long,kappa,eta_logit)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.eta_logit),
            dim=-1,
        )


def decode_m1_previous_action_bias_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M1PreviousActionBiasDecodedParameters:
    r"""Decode M1_previous_action_bias's unconstrained pooled parameter vector.

    Mathematical definition
    -----------------------
    The first five coordinates use M0's transforms and the sixth obeys

    .. math::

       \eta=\operatorname{sigmoid}(\widetilde\eta_{\mathrm{logit}}),
       \qquad \alpha\equiv 1\ \text{(fixed, not decoded)}.

    Parameter interpretation
    ------------------------
    ``eta`` is the weight of the previous-valid-action state in the mixture
    ``m_t=(1-eta)r_t+eta*pi_t`` applied before lapse.

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating tensor with final dimension six.

    Constraints
    -----------
    Finite raw values guarantee ``0 < eta < 1``; common M0 constraints are
    delegated to :func:`decode_raw_parameters`.

    Related tests
    -------------
    ``test_parameter_round_trip_and_eta_constraints``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 6:
        raise ValueError("M1_previous_action_bias raw_parameters must have final dimension 6")
    common = decode_raw_parameters(
        raw_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    eta_logit = raw_parameters[..., 5]
    return M1PreviousActionBiasDecodedParameters(
        w=common.w,
        s0=common.s0,
        ell_short=common.ell_short,
        ell_long=common.ell_long,
        kappa=common.kappa,
        eta_logit=eta_logit,
        eta=torch.sigmoid(eta_logit),
    )


def encode_m1_previous_action_bias_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable ``(w,s0,ell_short,ell_long,kappa,eta_logit)`` to raw coordinates.

    The sixth value is the bias-weight logit (stored raw, as the M2
    previous-action-bias model stores its own ``eta_logit``), not ``eta``
    itself. Common coordinates use M0's exact inverse transforms. Infinite
    logits are rejected.

    Related tests
    -------------
    ``test_parameter_round_trip_and_eta_constraints``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 6:
        raise ValueError("M1_previous_action_bias constrained_parameters must have final dimension 6")
    eta_logit = constrained_parameters[..., 5]
    if not bool(torch.all(torch.isfinite(eta_logit)).detach().cpu()):
        raise ValueError("M1_previous_action_bias bias-weight logits must be finite")
    common_raw = encode_constrained_parameters(
        constrained_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((common_raw, eta_logit.unsqueeze(-1)), dim=-1)
