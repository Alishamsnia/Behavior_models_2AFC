"""Fixed and profiled post-fit identifiability diagnostics for M3 dynamics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .parameters import (
    M3_PARAMETER_ORDER,
    decode_m3_raw_parameters,
    encode_m3_constrained_parameters,
)
from .session import M3SessionTrials, evaluate_m3_mouse_objective


_DYNAMIC_INDEX = {"ah": 5, "delta0": 6, "alpha_theta": 7}


def _validated_inputs(
    sessions: Iterable[M3SessionTrials],
    best_raw_parameters: torch.Tensor,
    *,
    grid_points: int,
) -> tuple[tuple[M3SessionTrials, ...], str]:
    records = tuple(sessions)
    if not records or best_raw_parameters.shape != (8,):
        raise ValueError("M3 surface requires sessions and a raw eight-vector")
    if grid_points < 3:
        raise ValueError("M3 surface requires at least three grid points per axis")
    roles = {item.split_role for item in records}
    if len(roles) != 1 or "test" in roles:
        raise ValueError("M3 surface sessions must share one non-test role")
    return records, next(iter(roles))


def _objective(records, raw, registry, role):
    return evaluate_m3_mouse_objective(
        records,
        raw,
        registry.task_moments,
        objective_weighting=registry.objective_weighting,
        required_split_role=role,
        lapse_max=registry.lapse_max,
        parameter_epsilon=registry.parameter_epsilon,
        delta_min=registry.delta_min,
        first_long_probability=registry.first_long_probability,
        prior_probability_epsilon=registry.prior_probability_epsilon,
        log_probability_epsilon=registry.log_probability_epsilon,
        quadratic_tolerance=registry.quadratic_tolerance,
        discriminant_tolerance=registry.discriminant_tolerance,
    )


def _dynamic_axes(best: torch.Tensor, registry, grid_points: int, widths):
    alpha_epsilon = max(float(registry.parameter_epsilon), 1e-8)
    # Encoding/decoding uses a softplus offset.  Stay one registered transform
    # epsilon inside the admissible region so round-off cannot decode below the
    # evaluator's inclusive hard floor.
    safe_delta_floor = float(registry.delta_min) + max(
        float(registry.parameter_epsilon), float(np.spacing(registry.delta_min)) * 8.0
    )
    return {
        "ah": np.linspace(float(best[5]) - widths["ah"], float(best[5]) + widths["ah"], grid_points),
        # The evaluator requires delta >= delta_min.  This exact registered floor
        # fixes the former 1e-8-grid crash for weak-contrast fits.
        "delta0": np.linspace(
            max(safe_delta_floor, float(best[6]) - widths["delta0"]),
            float(best[6]) + widths["delta0"],
            grid_points,
        ),
        "alpha_theta": np.linspace(
            max(alpha_epsilon, float(best[7]) - widths["alpha_theta"]),
            min(1.0 - alpha_epsilon, float(best[7]) + widths["alpha_theta"]),
            grid_points,
        ),
    }


def evaluate_m3_dynamic_surface(
    sessions: Iterable[M3SessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("delta0", "alpha_theta"),
    grid_points: int = 15,
    ah_half_width: float = 2.0,
    delta0_half_width: float = 1.0,
    alpha_half_width: float = 0.15,
) -> pd.DataFrame:
    """Evaluate a two-dimensional loss slice with the other six values fixed."""

    records, role = _validated_inputs(
        sessions, best_raw_parameters, grid_points=grid_points
    )
    if len(pair) != 2 or pair[0] == pair[1] or any(name not in _DYNAMIC_INDEX for name in pair):
        raise ValueError("pair must contain two distinct names from ah, delta0, alpha_theta")
    widths = {
        "ah": float(ah_half_width),
        "delta0": float(delta0_half_width),
        "alpha_theta": float(alpha_half_width),
    }
    if any(value <= 0.0 for value in widths.values()):
        raise ValueError("M3 surface half-widths must be positive")
    best = decode_m3_raw_parameters(
        best_raw_parameters,
        lapse_max=registry.lapse_max,
        epsilon=registry.parameter_epsilon,
    ).as_tensor().detach()
    axes = _dynamic_axes(best, registry, grid_points, widths)
    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for first in axes[pair[0]]:
            for second in axes[pair[1]]:
                constrained = best.clone()
                constrained[_DYNAMIC_INDEX[pair[0]]] = float(first)
                constrained[_DYNAMIC_INDEX[pair[1]]] = float(second)
                raw = encode_m3_constrained_parameters(
                    constrained,
                    lapse_max=registry.lapse_max,
                    epsilon=registry.parameter_epsilon,
                )
                objective = _objective(records, raw, registry, role)
                row = {
                    pair[0]: float(first),
                    pair[1]: float(second),
                    "mean_nll": float(objective.loss),
                    "projection_fraction": objective.projection_fraction,
                    "split_role": role,
                }
                rows.append(row)
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result["mean_nll"] - result["mean_nll"].min()
    for name, index in _DYNAMIC_INDEX.items():
        result[f"fitted_{name}"] = float(best[index])
    result["surface_type"] = f"fixed_other_six_{pair[0]}_by_{pair[1]}"
    return result


def evaluate_m3_delta_alpha_surface(
    sessions: Iterable[M3SessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    grid_points: int = 15,
    delta0_half_width: float = 1.0,
    alpha_half_width: float = 0.15,
) -> pd.DataFrame:
    """Evaluate the registered delta0/alpha slice without crossing delta_min."""

    return evaluate_m3_dynamic_surface(
        sessions,
        best_raw_parameters,
        registry,
        pair=("delta0", "alpha_theta"),
        grid_points=grid_points,
        delta0_half_width=delta0_half_width,
        alpha_half_width=alpha_half_width,
    )


def evaluate_m3_hazard_alpha_surface(*args, **kwargs) -> pd.DataFrame:
    """Evaluate the complementary hazard-logit/learning-rate loss slice."""

    return evaluate_m3_dynamic_surface(*args, pair=("ah", "alpha_theta"), **kwargs)


def evaluate_m3_hazard_delta_surface(*args, **kwargs) -> pd.DataFrame:
    """Evaluate the complementary hazard-logit/initial-contrast loss slice."""

    return evaluate_m3_dynamic_surface(*args, pair=("ah", "delta0"), **kwargs)


def evaluate_m3_dynamic_volume(
    sessions: Iterable[M3SessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    grid_points: int = 7,
    ah_half_width: float = 2.0,
    delta0_half_width: float = 1.0,
    alpha_half_width: float = 0.15,
) -> pd.DataFrame:
    """Evaluate the joint fixed-common `(ah, delta0, alpha_theta)` loss volume."""

    records, role = _validated_inputs(sessions, best_raw_parameters, grid_points=grid_points)
    best = decode_m3_raw_parameters(
        best_raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor().detach()
    axes = _dynamic_axes(best, registry, grid_points, {
        "ah": ah_half_width, "delta0": delta0_half_width, "alpha_theta": alpha_half_width
    })
    rows = []
    with torch.no_grad():
        for ah in axes["ah"]:
            for delta0 in axes["delta0"]:
                for alpha in axes["alpha_theta"]:
                    constrained = best.clone()
                    constrained[5:] = constrained.new_tensor((ah, delta0, alpha))
                    raw = encode_m3_constrained_parameters(
                        constrained, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
                    )
                    objective = _objective(records, raw, registry, role)
                    rows.append({
                        "ah": float(ah), "delta0": float(delta0),
                        "alpha_theta": float(alpha), "mean_nll": float(objective.loss),
                        "projection_fraction": objective.projection_fraction, "split_role": role,
                    })
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result.mean_nll - result.mean_nll.min()
    result["surface_type"] = "fixed_common_four_joint_dynamic_volume"
    return result


def profile_m3_dynamic_surface(
    sessions: Iterable[M3SessionTrials],
    best_raw_parameters: torch.Tensor,
    registry,
    *,
    pair: tuple[str, str] = ("delta0", "alpha_theta"),
    grid_points: int = 7,
    maximum_lbfgs_steps: int = 40,
    ah_half_width: float = 2.0,
    delta0_half_width: float = 1.0,
    alpha_half_width: float = 0.15,
) -> pd.DataFrame:
    """Profile a dynamic pair by reoptimizing the other five raw coordinates.

    Every grid point uses PyTorch's full-batch L-BFGS with strong-Wolfe line
    search. The two selected coordinates are held exactly fixed through a raw
    coordinate mask; validation or locked-test sessions are never used to fit.
    """

    records, role = _validated_inputs(sessions, best_raw_parameters, grid_points=grid_points)
    if maximum_lbfgs_steps <= 0:
        raise ValueError("maximum_lbfgs_steps must be positive")
    fixed_surface = evaluate_m3_dynamic_surface(
        records, best_raw_parameters, registry, pair=pair, grid_points=grid_points,
        ah_half_width=ah_half_width, delta0_half_width=delta0_half_width,
        alpha_half_width=alpha_half_width,
    )
    fixed_indices = [_DYNAMIC_INDEX[name] for name in pair]
    mask = best_raw_parameters.new_ones(7)
    mask[fixed_indices] = 0.0
    rows = []
    best_constrained = decode_m3_raw_parameters(
        best_raw_parameters, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
    ).as_tensor().detach()
    for point in fixed_surface.itertuples(index=False):
        constrained = best_constrained.clone()
        for name in pair:
            constrained[_DYNAMIC_INDEX[name]] = float(getattr(point, name))
        fixed_raw = encode_m3_constrained_parameters(
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
            decoded = decode_m3_raw_parameters(
                profiled_raw, lapse_max=registry.lapse_max, epsilon=registry.parameter_epsilon
            ).as_tensor()
        row = {name: float(getattr(point, name)) for name in pair}
        row.update({
            "mean_nll": float(objective.loss),
            "projection_fraction": objective.projection_fraction,
            "split_role": role,
            "profile_optimizer": "torch_full_batch_lbfgs_strong_wolfe",
        })
        for index, name in enumerate(M3_PARAMETER_ORDER):
            row[f"profiled_{name}"] = float(decoded[index])
        rows.append(row)
    result = pd.DataFrame(rows)
    result["delta_mean_nll_from_grid_minimum"] = result.mean_nll - result.mean_nll.min()
    result["surface_type"] = f"profiled_{pair[0]}_by_{pair[1]}"
    return result
