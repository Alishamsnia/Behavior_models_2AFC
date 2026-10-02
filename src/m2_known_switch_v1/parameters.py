"""Pooled parameter transformations for M2_known_switch_v1.

M2_known_switch_v1 keeps M2's fitted, symmetric, static subjective rates
``(0.5, sigmoid(-delta0), sigmoid(delta0))`` but, like M1_known_switch_v1,
replaces hazard-inferred block-switch timing with the true (oracle) switch
index. As in M1_known_switch_v1, no hazard parameter is fitted: the free
vector is M0's five common coordinates plus ``delta0``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from m0_sensory_v1.model.parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M2_KNOWN_SWITCH_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa", "delta0")


@dataclass(frozen=True)
class M2KnownSwitchParameters:
    """M0's five common coordinates plus the fitted static contrast ``delta0``."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    delta0: torch.Tensor

    def common(self) -> DecodedParameters:
        """Return the unchanged five-parameter Bayesian sensory/lapse readout."""

        return DecodedParameters(
            w=self.w, s0=self.s0, ell_short=self.ell_short,
            ell_long=self.ell_long, kappa=self.kappa,
        )

    def prototypes(self) -> torch.Tensor:
        r"""Return static subjective rates in registered ``(N,S,L)`` order.

        Purpose
        -------
        Construct this model's one pooled, session-invariant prototype
        vector, identical in form to real M2's.

        Mathematical definition
        -----------------------
        .. math::

           \boldsymbol\theta(\delta_0)=(0.5,\sigma(-\delta_0),\sigma(\delta_0)).

        Parameter interpretation
        ------------------------
        ``delta0`` is the mouse-level subjective block contrast;
        ``delta0=log(4)`` matches M1_known_switch_v1's fixed rates exactly,
        and ``delta0=0`` is the functional boundary at which the prior is
        always 0.5 regardless of switch timing.

        Input units and shapes
        ----------------------
        ``delta0`` is dimensionless and may have any leading batch shape.

        Returned quantities and shapes
        ------------------------------
        Returns the leading batch shape followed by a three-state axis.

        Constraints
        -----------
        Freely decoded values have ``delta0>0``; direct boundary tests may
        explicitly construct ``delta0=0`` or ``delta0=log(4)``.

        Numerical stability
        -------------------
        Torch's stable sigmoid is used without clipping the prototypes.

        Causal restrictions
        -------------------
        The function receives no trials, feedback, or switch information.

        Important edge cases
        --------------------
        At ``delta0=0`` all entries equal 0.5 and the prior becomes constant
        regardless of the oracle switch schedule.

        Related tests
        -------------
        ``test_m2_known_switch_zero_contrast_gives_constant_half_prior``.
        """

        theta_n = torch.full_like(self.delta0, 0.5)
        theta_s = torch.sigmoid(-self.delta0)
        theta_l = torch.sigmoid(self.delta0)
        return torch.stack((theta_n, theta_s, theta_l), dim=-1)

    def as_tensor(self) -> torch.Tensor:
        """Stack values as ``(w,s0,ell_short,ell_long,kappa,delta0)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.delta0),
            dim=-1,
        )

    def with_explicit_delta0(self, value: float | torch.Tensor) -> "M2KnownSwitchParameters":
        """Return a copy with ``delta0`` pinned to an explicit nonnegative scalar.

        Used for the ``delta0=0`` boundary test (constant 0.5 prior,
        matching M0's own fixed neutral prior regardless of switch timing)
        and the ``delta0=log(4)`` boundary test against
        ``m1_known_switch_v1`` -- both unreachable through the free positive
        decoder.
        """

        contrast = torch.as_tensor(value, dtype=self.w.dtype, device=self.w.device)
        if contrast.ndim != 0 or not bool(torch.isfinite(contrast) & (contrast >= 0)):
            raise ValueError("Explicit delta0 must be a finite nonnegative scalar")
        return replace(self, delta0=contrast)


def decode_m2_known_switch_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M2KnownSwitchParameters:
    r"""Decode M2_known_switch's six unconstrained pooled coordinates.

    Purpose
    -------
    Reuse M0's five-coordinate decoder unchanged and add one positive static
    contrast, with no hazard coordinate because switch timing is supplied.

    Mathematical definition
    -----------------------
    .. math::

       \delta_0=\operatorname{softplus}(\widetilde\delta_0)+\epsilon.

    Parameter interpretation
    ------------------------
    The first five values are M0's unchanged sensory/lapse parameters;
    ``delta0`` is the static subjective block contrast.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension six; ``s0`` is in seconds.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply positive ``w``, ``s0``, and ``delta0`` plus
    bounded category-specific lapse rates.

    Numerical stability
    -------------------
    Stable Torch softplus and sigmoid operations are used with the
    registered epsilon.

    Causal restrictions
    -------------------
    Decoding accepts parameters only and cannot inspect behavioral history
    or switch timing.

    Important edge cases
    --------------------
    Exact ``delta0=0`` is intentionally unavailable through the free decoder
    and must be constructed via :meth:`M2KnownSwitchParameters.with_explicit_delta0`.

    Related tests
    -------------
    ``test_m2_known_switch_parameter_round_trip``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 6:
        raise ValueError("M2_known_switch raw_parameters must have final dimension 6")
    common = decode_raw_parameters(
        raw_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    delta0 = F.softplus(raw_parameters[..., 5]) + epsilon
    return M2KnownSwitchParameters(
        w=common.w, s0=common.s0, ell_short=common.ell_short,
        ell_long=common.ell_long, kappa=common.kappa, delta0=delta0,
    )


def encode_m2_known_switch_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M2_known_switch values to raw optimizer coordinates.

    Purpose
    -------
    Initialize this model from scientific values while reusing M0's exact
    five-coordinate encoder.

    Mathematical definition
    -----------------------
    .. math::

       \widetilde\delta_0=\operatorname{softplus}^{-1}(\delta_0-\epsilon).

    Parameter interpretation
    ------------------------
    Input order is ``(w,s0,ell_short,ell_long,kappa,delta0)``.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension six.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite common parameters and ``delta0>epsilon``.

    Numerical stability
    -------------------
    Inverse softplus uses ``x + log(-expm1(-x))`` to avoid overflow/cancellation.

    Causal restrictions
    -------------------
    Initialization is independent of trials, outcomes, blocks, and switches.

    Important edge cases
    --------------------
    The exact ``delta0=0`` boundary is rejected because its raw value is
    infinite; use :meth:`M2KnownSwitchParameters.with_explicit_delta0` for
    boundary tests instead.

    Related tests
    -------------
    ``test_m2_known_switch_parameter_round_trip`` and the frozen eight-start
    registry test.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 6:
        raise ValueError("M2_known_switch constrained_parameters must have final dimension 6")
    delta0 = constrained_parameters[..., 5]
    if not bool(torch.all(torch.isfinite(delta0) & (delta0 > epsilon)).detach().cpu()):
        raise ValueError("M2_known_switch delta0 must be finite and greater than epsilon")
    common_raw = encode_constrained_parameters(
        constrained_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    shifted = delta0 - epsilon
    raw_delta0 = shifted + torch.log(-torch.expm1(-shifted))
    return torch.cat((common_raw, raw_delta0.unsqueeze(-1)), dim=-1)
