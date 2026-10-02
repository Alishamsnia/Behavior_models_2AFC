"""Pooled parameter transformations for M1_stimulus_kernel_v1.

M1_stimulus_kernel_v1 keeps M0's unchanged five-parameter sensory/lapse
readout and adds exactly one new pooled coordinate, a logit ``a`` whose
sigmoid is the scalar delta-rule learning rate ``alpha`` (see
``session.py``). This mirrors, coordinate-for-coordinate, how
``m0_sensory_v1.model.m1_parameters`` extends M0 with the hazard logit
``ah``: the first five coordinates decode/encode through M0's own
registered transform (reused, not reimplemented) and the sixth is stored
raw (the logit itself, not the sigmoid-transformed rate) with a derived
property for the interpretable value -- exactly the shape M1 already uses
for ``ah``/``hazard``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from m0_sensory_v1.model.parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M1_STIMULUS_KERNEL_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa", "a")


@dataclass(frozen=True)
class M1StimulusKernelDecodedParameters:
    """M0's five pooled values plus the fitted stimulus-kernel learning-rate logit."""

    w: torch.Tensor
    s0: torch.Tensor
    ell_short: torch.Tensor
    ell_long: torch.Tensor
    kappa: torch.Tensor
    a: torch.Tensor
    alpha: torch.Tensor

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
        """Stack parameters as ``(w,s0,ell_short,ell_long,kappa,a)``."""

        return torch.stack(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.a),
            dim=-1,
        )


def decode_m1_stimulus_kernel_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> M1StimulusKernelDecodedParameters:
    r"""Decode M1_stimulus_kernel's unconstrained pooled parameter vector.

    Purpose
    -------
    Extend the unchanged M0 sensory/lapse decoder with one learning-rate
    logit shared by every session of a mouse.

    Mathematical definition
    -----------------------
    The first five coordinates use M0's transforms and the sixth obeys

    .. math::

       a=\widetilde a,\qquad \alpha=\operatorname{sigmoid}(a).

    Parameter interpretation
    ------------------------
    ``a`` is the fitted log learning-rate odds and ``alpha`` is its
    probability-scale representation, the per-trial weight the scalar
    prior ``pi_t`` places on the newly observed category (see
    :func:`m1_stimulus_kernel_v1.session.evaluate_m1_stimulus_kernel_session`).

    Input units and shapes
    ----------------------
    ``raw_parameters`` is a floating tensor with final dimension six.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors with the input's leading batch shape.

    Constraints
    -----------
    Finite raw values guarantee ``0 < alpha < 1``; common M0 constraints are
    delegated to :func:`decode_raw_parameters`.

    Numerical stability
    -------------------
    Torch's stable sigmoid is used in the input dtype and on the input device.

    Causal restrictions
    -------------------
    Decoding receives no trial, outcome, block, history, or session data.

    Important edge cases
    --------------------
    Very large finite ``|a|`` can saturate ``alpha`` to a value that rounds
    to exactly ``0.0`` or ``1.0`` in floating point; the sigmoid itself
    never raises and the downstream prior is still clamped safely by
    :func:`m0_sensory_v1.model.m1_bayesian.m1_bayesian_choice_probability`.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_parameter_round_trip_and_alpha_constraints``.
    """

    if not isinstance(raw_parameters, torch.Tensor):
        raise TypeError("raw_parameters must be a torch.Tensor")
    if not raw_parameters.is_floating_point():
        raise TypeError("raw_parameters must use a floating dtype")
    if raw_parameters.ndim == 0 or raw_parameters.shape[-1] != 6:
        raise ValueError("M1_stimulus_kernel raw_parameters must have final dimension 6")
    common = decode_raw_parameters(
        raw_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    a = raw_parameters[..., 5]
    return M1StimulusKernelDecodedParameters(
        w=common.w,
        s0=common.s0,
        ell_short=common.ell_short,
        ell_long=common.ell_long,
        kappa=common.kappa,
        a=a,
        alpha=torch.sigmoid(a),
    )


def encode_m1_stimulus_kernel_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M1_stimulus_kernel values ``(w,s0,ell_short,ell_long,kappa,a)`` to raw coordinates.

    Purpose
    -------
    Initialize M1_stimulus_kernel from scientific values while reusing M0's
    exact common parameter transforms.

    Mathematical definition
    -----------------------
    The common coordinates are inverse-transformed as for M0 and

    .. math::

       \widetilde a=a.

    Parameter interpretation
    ------------------------
    The sixth value is a learning-rate logit, not a probability. Its
    interpretable probability-scale rate is ``sigmoid(a)``. This matches
    ``m0_sensory_v1.model.m1_parameters.encode_m1_constrained_parameters``,
    which stores the raw hazard logit ``ah`` rather than the sigmoid-
    transformed hazard.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension six; ``s0`` is seconds.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Common values obey M0's open bounds and ``a`` must be finite.

    Numerical stability
    -------------------
    Stable inverse softplus/logit operations are delegated to M0's encoder.

    Causal restrictions
    -------------------
    No behavioral data are accepted.

    Important edge cases
    --------------------
    Infinite learning-rate logits are rejected.

    Related tests
    -------------
    ``test_m1_stimulus_kernel_parameter_round_trip_and_alpha_constraints``.
    """

    if not isinstance(constrained_parameters, torch.Tensor):
        raise TypeError("constrained_parameters must be a torch.Tensor")
    if not constrained_parameters.is_floating_point():
        raise TypeError("constrained_parameters must use a floating dtype")
    if constrained_parameters.ndim == 0 or constrained_parameters.shape[-1] != 6:
        raise ValueError("M1_stimulus_kernel constrained_parameters must have final dimension 6")
    a = constrained_parameters[..., 5]
    if not bool(torch.all(torch.isfinite(a)).detach().cpu()):
        raise ValueError("M1_stimulus_kernel learning-rate logits must be finite")
    common_raw = encode_constrained_parameters(
        constrained_parameters[..., :5], lapse_max=lapse_max, epsilon=epsilon
    )
    return torch.cat((common_raw, a.unsqueeze(-1)), dim=-1)
