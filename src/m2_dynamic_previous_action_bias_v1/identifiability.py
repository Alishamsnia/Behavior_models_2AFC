"""Fixed and profiled post-fit identifiability diagnostics for
M2_dynamic_previous_action_bias_v1.

Extends M2_dynamic_block_length_v1's own fixed-other-N grid/profile
machinery with ``eta_logit`` and ``a`` as addressable axes -- the concrete
tool for empirically investigating the eta-vs-lapse, eta-vs-kappa, and
eta/alpha-vs-hazard-parameter identifiability-ridge concerns any two-prior
mixture model of this shape can have (see
``docs/models/M2_dynamic_previous_action_bias_v1.md``, "Identifiability
considerations").
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import (
    M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER,
    decode_m2_dynamic_previous_action_bias_raw_parameters,
    encode_m2_dynamic_previous_action_bias_constrained_parameters,
)
from .session import M2DynamicPreviousActionBiasSessionTrials, evaluate_m2_dynamic_previous_action_bias_mouse_objective


_DYNAMIC_INDEX = {"ah": 5, "delta0": 6, "bh": 7, "eta_logit": 8}


def _validated(sessions, raw, grid_points):
    records = tuple(sessions)
    if not records or raw.shape != (9,):
        raise ValueError("M2_dynamic_previous_action_bias surface requires sessions and a raw nine-vector")
    if grid_points < 3:
        raise ValueError("M2_dynamic_previous_action_bias surface requires at least three grid points per axis")
    roles = {item.split_role for item in records}
    if len(roles) != 1 or "test" in roles:
        raise ValueError("M2_dynamic_previous_action_bias surfaces may use only one non-test split")
    return records, next(iter(roles))


def _objective(records, raw, registry, role):
    return evaluate_m2_dynamic_previous_action_bias_mouse_objective(
        records, raw, registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role, lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        first_long_probability=registry.first_long_probability,
        initial_pi=registry.initial_pi,
        prior_probability_epsilon=registry.prior_probability_epsilon,
        log_probability_epsilon=registry.log_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
    )


def _axes(best, registry, points, widths):
    epsilon = max(float(registry.parameter_epsilon), 1e-8)
    return {
        "ah": np.linspace(best[5] - widths["ah"], best[5] + widths["ah"], points),
        "delta0": np.linspace(max(epsilon, best[6] - widths["delta0"]), best[6] + widths["delta0"], points),
        "bh": np.linspace(max(epsilon, best[7] - widths["bh"]), best[7] + widths["bh"], points),
        "eta_logit": np.linspace(best[8] - widths["eta_logit"], best[8] + widths["eta_logit"], points),
    }


def evaluate_m2_dynamic_previous_action_bias_dynamic_surface(
    sessions: Iterable[M2DynamicPreviousActionBiasSessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("eta_logit", "bh"),
    grid_points: int = 15,
    ah_half_width: float = 2.0,
    delta0_half_width: float = 1.0,
    bh_half_width: float = 0.20,
    eta_logit_half_width: float = 2.0,
) -> pd.DataFrame:
    """Evaluate a fixed-other-seven loss slice on training or validation data."""

    records, role = _validated(sessions, best_raw_parameters, grid_points)
    if len(pair) != 2 or pair[0] == pair[1] or any(name not in _DYNAMIC_INDEX for name in pair):
        raise ValueError("pair must contain two distinct M2_dynamic_previous_action_bias dynamic coordinates")
    widths = {
        "ah": ah_half_width, "delta0": delta0_half_width, "bh": bh_half_width,
        "eta_logit": eta_logit_half_width,
    }
    if any(float(value) <= 0 for value in widths.values()):
        raise ValueError("M2_dynamic_previous_action_bias surface half-widths must be positive")
    best = decode_m2_dynamic_previous_action_bias_raw_parameters(
        best_raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor().detach().cpu().numpy()
    axes = _axes(best, registry, grid_points, widths)
    rows = []
    with torch.no_grad():
        for first in axes[pair[0]]:
            for second in axes[pair[1]]:
                constrained = torch.tensor(best, dtype=best_raw_parameters.dtype, device=best_raw_parameters.device)
                constrained[_DYNAMIC_INDEX[pair[0]]] = float(first)
                constrained[_DYNAMIC_INDEX[pair[1]]] = float(second)
                raw = encode_m2_dynamic_previous_action_bias_constrained_parameters(
                    constrained, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
                )
                objective = _objective(records, raw, registry, role)
                rows.append({
                    pair[0]: float(first), pair[1]: float(second), "mean_nll": float(objective.loss),
                    "mean_tail_probability_pre": objective.mean_tail_probability_pre, "split_role": role,
                })
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result.mean_nll - result.mean_nll.min()
    for name, index in _DYNAMIC_INDEX.items():
        result[f"fitted_{name}"] = float(best[index])
    result["surface_type"] = f"fixed_other_seven_{pair[0]}_by_{pair[1]}"
    return result


def profile_m2_dynamic_previous_action_bias_dynamic_surface(
    sessions: Iterable[M2DynamicPreviousActionBiasSessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("eta_logit", "bh"),
    grid_points: int = 7,
    maximum_lbfgs_steps: int = 40,
    **surface_widths,
) -> pd.DataFrame:
    """Profile a dynamic pair by L-BFGS optimization of the other six values."""

    records, role = _validated(sessions, best_raw_parameters, grid_points)
    if maximum_lbfgs_steps <= 0:
        raise ValueError("maximum_lbfgs_steps must be positive")
    surface = evaluate_m2_dynamic_previous_action_bias_dynamic_surface(
        records, best_raw_parameters, registry, pair=pair, grid_points=grid_points, **surface_widths,
    )
    fixed_indices = [_DYNAMIC_INDEX[name] for name in pair]
    mask = best_raw_parameters.new_ones(9)
    mask[fixed_indices] = 0.0
    fitted = decode_m2_dynamic_previous_action_bias_raw_parameters(
        best_raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor().detach()
    rows = []
    for point in surface.itertuples(index=False):
        constrained = fitted.clone()
        for name in pair:
            constrained[_DYNAMIC_INDEX[name]] = float(getattr(point, name))
        fixed_raw = encode_m2_dynamic_previous_action_bias_constrained_parameters(
            constrained, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
        ).detach()
        free_raw = best_raw_parameters.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [free_raw], lr=1.0, max_iter=maximum_lbfgs_steps, history_size=10,
            line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            candidate = free_raw * mask + fixed_raw * (1.0 - mask)
            loss = _objective(records, candidate, registry, role).loss
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            profiled_raw = free_raw * mask + fixed_raw * (1.0 - mask)
            objective = _objective(records, profiled_raw, registry, role)
            decoded = decode_m2_dynamic_previous_action_bias_raw_parameters(
                profiled_raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ).as_tensor()
        row = {name: float(getattr(point, name)) for name in pair}
        row.update({
            "mean_nll": float(objective.loss),
            "mean_tail_probability_pre": objective.mean_tail_probability_pre,
            "split_role": role, "profile_optimizer": "torch_full_batch_lbfgs_strong_wolfe",
        })
        for index, name in enumerate(M2_DYNAMIC_PREVIOUS_ACTION_BIAS_PARAMETER_ORDER):
            row[f"profiled_{name}"] = float(decoded[index])
        rows.append(row)
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result.mean_nll - result.mean_nll.min()
    result["surface_type"] = f"profiled_{pair[0]}_by_{pair[1]}"
    return result
