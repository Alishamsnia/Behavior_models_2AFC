"""Post-fit hazard/contrast identifiability analysis for M2."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .m2_parameters import decode_m2_raw_parameters, encode_m2_constrained_parameters
from .m2_session import M2SessionTrials, evaluate_m2_mouse_objective


def evaluate_m2_hazard_contrast_surface(
    sessions: Iterable[M2SessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    grid_points: int = 15,
    ah_half_width: float = 2.0,
    delta0_half_width: float = 1.25,
) -> pd.DataFrame:
    r"""Evaluate a fixed-other-parameters post-fit ``(ah,delta0)`` loss surface.

    Purpose
    -------
    Diagnose the hazard/contrast ridge after fitting without changing the model,
    training objective, checkpoint, or selected parameters.

    Mathematical definition
    -----------------------
    .. math::

       S(a_h,\delta_0)=J\left(
       \widehat w,\widehat s_0,\widehat\ell_short,\widehat ell_long,
       \widehat\kappa,a_h,\delta_0
       \right).

    Parameter interpretation
    ------------------------
    ``ah`` and ``delta0`` vary on a grid while the other five coordinates remain
    at the fitted checkpoint. This is a local slice, not a profile likelihood.

    Input units and shapes
    ----------------------
    Sessions share one train/validation role; ``best_raw_parameters`` has shape
    ``(7,)`` and grid widths are dimensionless scientific-coordinate widths.

    Returned quantities and shapes
    ------------------------------
    Returns one table row per grid point with hazard, prototypes, mean NLL, and
    NLL elevation above the grid minimum.

    Constraints
    -----------
    At least three grid points, positive widths, non-test sessions, and one role
    are required.

    Numerical stability
    -------------------
    Every grid point round-trips through registered transforms and is evaluated
    in float64 without gradients.

    Causal restrictions
    -------------------
    The existing causal M2 choice-only objective is evaluated unchanged; no true
    block, future category, or validation-driven parameter update is introduced.

    Important edge cases
    --------------------
    The lower contrast grid is clipped above zero because the free M2 decoder
    cannot represent the exact boundary. A broad valley indicates weak separate
    identification even when the fitted optimum is unique.

    Related tests
    -------------
    ``test_identifiability_surface_is_post_fit_and_has_optimum_grid_point``.
    """

    records = tuple(sessions)
    if not records:
        raise ValueError("Identifiability analysis requires at least one session")
    if best_raw_parameters.shape != (7,):
        raise ValueError("best_raw_parameters must have shape (7,)")
    if grid_points < 3 or ah_half_width <= 0.0 or delta0_half_width <= 0.0:
        raise ValueError("Grid size and half widths must be positive")
    roles = {session.split_role for session in records}
    if len(roles) != 1 or "test" in roles:
        raise ValueError("Identifiability sessions must share one non-test role")
    role = next(iter(roles))
    best = decode_m2_raw_parameters(
        best_raw_parameters,
        lapse_max=registry.lapse_max,
        epsilon=registry.parameter_epsilon,
    ).as_tensor().detach()
    ah_values = np.linspace(
        float(best[5]) - ah_half_width,
        float(best[5]) + ah_half_width,
        grid_points,
    )
    delta_values = np.linspace(
        max(registry.parameter_epsilon * 10.0, float(best[6]) - delta0_half_width),
        float(best[6]) + delta0_half_width,
        grid_points,
    )
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for delta0 in delta_values:
            for ah in ah_values:
                constrained = best.clone()
                constrained[5] = ah
                constrained[6] = delta0
                raw = encode_m2_constrained_parameters(
                    constrained,
                    lapse_max=registry.lapse_max,
                    epsilon=registry.parameter_epsilon,
                )
                objective = evaluate_m2_mouse_objective(
                    records,
                    raw,
                    registry.task_moments,
                    objective_weighting=registry.objective_weighting,
                    required_split_role=role,
                    lapse_max=registry.lapse_max,
                    parameter_epsilon=registry.parameter_epsilon,
                    log_probability_epsilon=registry.log_probability_epsilon,
                    prior_probability_epsilon=registry.prior_probability_epsilon,
                    quadratic_tolerance=registry.quadratic_tolerance,
                    discriminant_tolerance=registry.discriminant_tolerance,
                    first_long_probability=registry.first_long_probability,
                )
                theta_long = float(
                    torch.sigmoid(torch.tensor(delta0, dtype=torch.float64))
                )
                rows.append({
                    "ah": float(ah),
                    "hazard": float(
                        torch.sigmoid(torch.tensor(ah, dtype=torch.float64))
                    ),
                    "delta0": float(delta0),
                    "theta_short": 1.0 - theta_long,
                    "theta_long": theta_long,
                    "mean_nll": float(objective.loss),
                })
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = (
        result["mean_nll"] - result["mean_nll"].min()
    )
    result["fitted_ah"] = float(best[5])
    result["fitted_delta0"] = float(best[6])
    result["split_role"] = role
    result["surface_type"] = "fixed_other_four_post_fit_objective_slice"
    return result
