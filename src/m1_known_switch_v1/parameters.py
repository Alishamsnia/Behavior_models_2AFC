"""Pooled parameter transformations for M1_known_switch_v1.

M1_known_switch_v1 keeps M1's fixed true-rate prototypes ``(0.5,0.2,0.8)`` and
its unchanged five-parameter sensory/lapse readout, but replaces M1's
hazard-inferred block-switch timing with the true (oracle) switch index (see
``session.py``). Because the switch timing is supplied rather than inferred,
no hazard parameter is fitted: the free parameter vector is identical in
shape and meaning to ``m0_sensory_v1``'s own five coordinates. This module
does not invent a new decoder -- it re-exports M0's registered decoder under
this model's own name, matching the naming convention every other model in
this family follows, so callers never need to know the coincidence in shape.
"""

from __future__ import annotations

import torch

from m0_sensory_v1.model.parameters import (
    DecodedParameters,
    decode_raw_parameters,
    encode_constrained_parameters,
)


M1_KNOWN_SWITCH_PARAMETER_ORDER = ("w", "s0", "ell_short", "ell_long", "kappa")

M1_KNOWN_SWITCH_FIXED_PROTOTYPES = (0.5, 0.2, 0.8)
"""Fixed ``(theta_N, theta_S, theta_L)``, identical to real M1. Not fitted."""

M1_KNOWN_SWITCH_LOG_LIKELIHOOD_RATIO = float(torch.log(torch.tensor(4.0, dtype=torch.float64)))
"""``log(theta_L/theta_S) = log(0.8/0.2) = log(4)``, the fixed per-category
feedback increment applied to the biased-subspace log-odds belief."""


def decode_m1_known_switch_raw_parameters(
    raw_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> DecodedParameters:
    r"""Decode M1_known_switch's five unconstrained pooled coordinates.

    Purpose
    -------
    Give this model its own named decoder entry point while reusing M0's
    registered transform exactly -- the free parameter set is unchanged
    because switch timing is supplied, not inferred, so no hazard coordinate
    is fitted.

    Mathematical definition
    -----------------------
    Identical to M0's registered decoder:

    .. math::

       w=\operatorname{softplus}(\widetilde w)+\epsilon,\quad
       s_0=\operatorname{softplus}(\widetilde s_0)+\epsilon,\quad
       \ell_{\mathrm{short}}=\ell_{\max}\sigma(\widetilde\ell_{\mathrm{short}}),\\
       \ell_{\mathrm{long}}=\ell_{\max}\sigma(\widetilde\ell_{\mathrm{long}}),\quad
       \kappa=\exp(\widetilde{\log\kappa}).

    Parameter interpretation
    ------------------------
    Identical to M0: ``w`` and ``s0`` set sensory variance, ``ell_short`` and
    ``ell_long`` are category-specific lapse rates, ``kappa`` is the decision
    cost-ratio bias. There is no hazard or contrast coordinate in this model.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension five.

    Returned quantities and shapes
    ------------------------------
    Returns named tensors preserving leading shape, dtype, device, and graph.

    Constraints
    -----------
    Finite raw values imply ``w>0``, ``s0>0``, and both lapse rates in
    ``(0, lapse_max)``. ``kappa`` has no registered upper bound.

    Numerical stability
    -------------------
    Delegates to M0's stable softplus/sigmoid/exp decoder.

    Causal restrictions
    -------------------
    Decoding accepts parameters only and cannot inspect behavioral history,
    switch timing, or block identity.

    Important edge cases
    --------------------
    None beyond M0's own registered decoder boundary behavior.

    Related tests
    -------------
    ``test_m1_known_switch_parameter_round_trip_matches_m0``.
    """

    return decode_raw_parameters(raw_parameters, lapse_max=lapse_max, epsilon=epsilon)


def encode_m1_known_switch_constrained_parameters(
    constrained_parameters: torch.Tensor,
    *,
    lapse_max: float = 1.0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    r"""Encode readable M1_known_switch values to raw optimizer coordinates.

    Purpose
    -------
    Named counterpart to :func:`decode_m1_known_switch_raw_parameters`, reused
    from M0's registered encoder without modification.

    Mathematical definition
    -----------------------
    Identical to M0's registered encoder (inverse softplus/sigmoid/log).

    Parameter interpretation
    ------------------------
    Input order is ``(w,s0,ell_short,ell_long,kappa)``, identical to M0.

    Input units and shapes
    ----------------------
    A floating tensor with final dimension five.

    Returned quantities and shapes
    ------------------------------
    Returns a raw tensor of identical shape, dtype, and device.

    Constraints
    -----------
    Requires finite, in-range constrained values as M0's encoder does.

    Numerical stability
    -------------------
    Delegates to M0's stable inverse-softplus/logit/log encoder.

    Causal restrictions
    -------------------
    Initialization is independent of trials, outcomes, blocks, and switches.

    Important edge cases
    --------------------
    None beyond M0's own registered encoder boundary behavior.

    Related tests
    -------------
    ``test_m1_known_switch_parameter_round_trip_matches_m0``.
    """

    return encode_constrained_parameters(
        constrained_parameters, lapse_max=lapse_max, epsilon=epsilon
    )
