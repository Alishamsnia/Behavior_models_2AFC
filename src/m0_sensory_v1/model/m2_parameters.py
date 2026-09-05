"""Pooled parameter transformations for the static-prototype M2 HMM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M2_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa", "ah", "delta0")


@dataclass(frozen=True)
class M2DecodedParameters:
    """Named M2 parameters and their derived hazard and static prototypes."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    ah: torch.Tensor
    delta0: torch.Tensor
    hazard: torch.Tensor

    def common(self) -> DecodedParameters:
        """Return the unchanged five-parameter Bayesian sensory/lapse readout."""

        return DecodedParameters(
            w=self.w,
            s0=self.s0,
            ell_short=self.ell_short,
            ell_long=self.ell_long,
            kappa=self.kappa,
        )

    def prototypes(self) -> torch.Tensor:
        r"""Return static subjective rates in registered ``(N,S,L)`` order.

        Purpose
        -------
        Construct M2's one pooled, session-invariant prototype vector.

        Mathematical definition
        -----------------------
        .. math::

           \boldsymbol\theta(\delta_0)=
           (0.5,\sigma(-\delta_0),\sigma(\delta_0)).

        Parameter interpretation
        ------------------------
        ``delta0`` is the mouse-level subjective block contrast. Symmetry gives
        :math:`\theta_S+\theta_L=1`; ``delta0=log(4)`` is M1 and ``delta0=0``
        is the functional M0 boundary.

        Input units and shapes
        ----------------------
        ``delta0`` is dimensionless and may have any leading batch shape.

        Returned quantities and shapes
        ------------------------------
        Returns the leading batch shape followed by a three-state axis.

        Constraints
        -----------
        Freely decoded M2 parameters have ``delta0>0``; direct boundary tests
        may explicitly construct ``delta0=0``.

        Numerical stability
        -------------------
        Torch's stable sigmoid is used without clipping the prototypes.

        Causal restrictions
        -------------------
        The function receives no trials or feedback. Its result is static and
        the evaluator must use the same tensor in both priors and emissions.

        Important edge cases
        --------------------
        At zero all entries equal 0.5 and hazard becomes choice-unidentifiable.

        Related tests
        -------------
        ``test_parameter_round_trip_and_static_symmetric_prototypes``,
        ``test_exact_log4_restriction_matches_m1_trialwise``, and
        ``test_exact_zero_contrast_reduces_to_m0``.
        """

        neutral = torch.ones_like(self.delta0) * 0.5
        long = torch.sigmoid(self.delta0)
        short = torch.sigmoid(-self.delta0)
        return torch.stack((neutral, short, long), dim=-1)

    def as_tensor(self) -> torch.Tensor:
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,ah,delta0)``."""

        return torch.stack(
            (
                self.w,
                self.s0,
                self.ell_short,
                self.ell_long,
                self.kappa,
                self.ah,
                self.delta0,
            ),
            dim=-1,
        )


def decode_m2_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2DecodedParameters:
    r"""Decode M2's seven unconstrained pooled optimizer coordinates.

    Purpose
    -------
    Apply M2's differentiable parameter constraints once per pooled objective.

    Mathematical definition
    -----------------------
    .. math::

       \delta_0=\operatorname{softplus}(\widetilde\delta_0)+\epsilon,
       \qquad h=\sigma(a_h).

    Parameter interpretation
    ------------------------
    The first five values are the unchanged sensory/lapse parameters, ``ah``
    is the constant-hazard logit, and ``delta0`` is static subjective contrast.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension seven; ``s0`` is in seconds.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply positive ``w``, ``s0``, and ``delta0`` plus bounded
    category-specific lapse and hazard probabilities.

    Numerical stability
    -------------------
    Stable Torch softplus and sigmoid operations are used with registered epsilon.

    Causal restrictions
    -------------------
    Decoding accepts parameters only and cannot inspect behavioral history.

    Important edge cases
    --------------------
    Exact ``delta0=0`` is intentionally unavailable through the free decoder
    and must be constructed directly for the M0 boundary test.

    Related tests
    -------------
    ``test_parameter_round_trip_and_static_symmetric_prototypes`` and
    ``test_all_six_gradients_match_finite_differences``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 7:
        raise ValueError("M2 raw_parameters must have final dimension 7")
    common = decode_raw_parameters(
        raw_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    ah = raw_parameters[..., 5]
    delta0 = torch.nn.functional.softplus(raw_parameters[..., 6]) + epsilon
    return M2DecodedParameters(
        w=common.w,
        s0=common.s0,
        ell_short=common.ell_short,
        ell_long=common.ell_long,
        kappa=common.kappa,
        ah=ah,
        delta0=delta0,
        hazard=torch.sigmoid(ah),
    )


def encode_m2_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2 values to seven raw optimizer coordinates.

    Purpose
    -------
    Convert interpretable starts into the exact raw coordinates optimized by M2.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde\delta_0=
       \operatorname{softplus}^{-1}(\delta_0-\epsilon).

    Parameter interpretation
    ------------------------
    Input order is ``(w,s0,ell_short,ell_long,kappa,ah,delta0)``; ``ah`` remains unconstrained.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension seven; duration scales use seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor with identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite ``ah``, ``delta0>epsilon``, and valid common parameters.

    Numerical stability
    -------------------
    Inverse softplus uses ``x + log(-expm1(-x))`` to avoid overflow/cancellation.

    Causal restrictions
    -------------------
    Initialization is independent of trials, outcomes, blocks, and feedback.

    Important edge cases
    --------------------
    The exact zero-contrast boundary is rejected because its raw value is infinite.

    Related tests
    -------------
    ``test_parameter_round_trip_and_static_symmetric_prototypes`` and the frozen
    eight-start registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 7:
        raise ValueError("M2 constrained_parameters must have final dimension 7")
    ah = constrained_parameters[..., 5]
    delta0 = constrained_parameters[..., 6]
    if not bool(torch.all(torch.isfinite(ah)).detach().cpu()):
        raise ValueError("M2 hazard logits must be finite")
    if not bool(torch.all(torch.isfinite(delta0) & (delta0 > epsilon)).detach().cpu()):
        raise ValueError("M2 delta0 must be finite and greater than epsilon")
    common_raw = encode_constrained_parameters(
        constrained_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    shifted = delta0 - epsilon
    raw_delta0 = shifted + torch.log(-torch.expm1(-shifted))
    return torch.cat((common_raw, ah.unsqueeze(-1), raw_delta0.unsqueeze(-1)), dim=-1)
