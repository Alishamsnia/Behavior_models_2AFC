"""Fixed and profiled post-fit identifiability diagnostics for
M1_dynamic_block_length_v1."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import (
    M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER,
    decode_m1_dynamic_block_length_raw_parameters,
    encode_m1_dynamic_block_length_constrained_parameters,
)
from .session import M1DynamicBlockLengthSessionTrials, evaluate_m1_dynamic_block_length_mouse_objective


_DYNAMIC_INDEX = {"ah": 5, "bh": 6}


def _validated(sessions, raw, grid_points):
    records = tuple(sessions)
    if not records or raw.shape != (7,):
        raise ValueError("M1_dynamic_block_length surface requires sessions and a raw seven-vector")
    if grid_points < 3:
        raise ValueError("M1_dynamic_block_length surface requires at least three grid points per axis")
    roles = {item.split_role for item in records}
    if len(roles) != 1 or "test" in roles:
        raise ValueError("M1_dynamic_block_length surfaces may use only one non-test split")
    return records, next(iter(roles))


def _objective(records, raw, registry, role):
    return evaluate_m1_dynamic_block_length_mouse_objective(
        records, raw, registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role, lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        run_length_center=registry.run_length_center,
        maximum_run_length=registry.maximum_represented_run_length,
        first_long_probability=registry.first_long_probability,
        prior_probability_epsilon=registry.prior_probability_epsilon,
        log_probability_epsilon=registry.log_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
    )


def _axes(best, registry, points, widths):
    epsilon = max(float(registry.parameter_epsilon), 1e-8)
    return {
        "ah": np.linspace(best[5] - widths["ah"], best[5] + widths["ah"], points),
        "bh": np.linspace(max(epsilon, best[6] - widths["bh"]), best[6] + widths["bh"], points),
    }


def evaluate_m1_dynamic_block_length_dynamic_surface(
    sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("ah", "bh"),
    grid_points: int = 15,
    ah_half_width: float = 2.0,
    bh_half_width: float = 0.20,
) -> pd.DataFrame:
    """Evaluate a fixed-other-five loss slice on training or validation data."""

    records, role = _validated(sessions, best_raw_parameters, grid_points)
    if len(pair) != 2 or pair[0] == pair[1] or any(name not in _DYNAMIC_INDEX for name in pair):
        raise ValueError("pair must contain two distinct M1_dynamic_block_length dynamic coordinates")
    widths = {"ah": ah_half_width, "bh": bh_half_width}
    if any(float(value) <= 0 for value in widths.values()):
        raise ValueError("M1_dynamic_block_length surface half-widths must be positive")
    best = decode_m1_dynamic_block_length_raw_parameters(
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
                raw = encode_m1_dynamic_block_length_constrained_parameters(
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
    result["surface_type"] = f"fixed_other_five_{pair[0]}_by_{pair[1]}"
    return result


def profile_m1_dynamic_block_length_dynamic_surface(
    sessions: Iterable[M1DynamicBlockLengthSessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("ah", "bh"),
    grid_points: int = 7,
    maximum_lbfgs_steps: int = 40,
    **surface_widths,
) -> pd.DataFrame:
    """Profile a dynamic pair by L-BFGS optimization of the other four values."""

    records, role = _validated(sessions, best_raw_parameters, grid_points)
    if maximum_lbfgs_steps <= 0:
        raise ValueError("maximum_lbfgs_steps must be positive")
    surface = evaluate_m1_dynamic_block_length_dynamic_surface(
        records, best_raw_parameters, registry, pair=pair, grid_points=grid_points, **surface_widths,
    )
    fixed_indices = [_DYNAMIC_INDEX[name] for name in pair]
    mask = best_raw_parameters.new_ones(7)
    mask[fixed_indices] = 0.0
    fitted = decode_m1_dynamic_block_length_raw_parameters(
        best_raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor().detach()
    rows = []
    for point in surface.itertuples(index=False):
        constrained = fitted.clone()
        for name in pair:
            constrained[_DYNAMIC_INDEX[name]] = float(getattr(point, name))
        fixed_raw = encode_m1_dynamic_block_length_constrained_parameters(
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
            decoded = decode_m1_dynamic_block_length_raw_parameters(
                profiled_raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ).as_tensor()
        row = {name: float(getattr(point, name)) for name in pair}
        row.update({
            "mean_nll": float(objective.loss),
            "mean_tail_probability_pre": objective.mean_tail_probability_pre,
            "split_role": role, "profile_optimizer": "torch_full_batch_lbfgs_strong_wolfe",
        })
        for index, name in enumerate(M1_DYNAMIC_BLOCK_LENGTH_PARAMETER_ORDER):
            row[f"profiled_{name}"] = float(decoded[index])
        rows.append(row)
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result.mean_nll - result.mean_nll.min()
    result["surface_type"] = f"profiled_{pair[0]}_by_{pair[1]}"
    return result
