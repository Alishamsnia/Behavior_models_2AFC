"""Frozen initialization records for the eight-parameter M3 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_m3_constrained_parameters


@dataclass(frozen=True)
class M3InitializationSpec:
    """One readable M3 start and its source M2 anchor identifier."""

    start_id: int
    source_m2_start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    delta0: float
    alpha_theta: float
    purpose: str

    @property
    def hazard(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.ah, dtype=torch.float64)))

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    def constrained_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                self.w,
                self.s0,
                self.ell_short,
                self.ell_long,
                self.kappa,
                self.ah,
                self.delta0,
                self.alpha_theta,
            ],
            dtype=dtype,
            device=device,
        )

    def raw_tensor(
        self,
        *,
        lapse_max: float = 1.0,
        epsilon: float = 1e-10,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return encode_m3_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m3_initialization_table(
    initializations: Iterable[M3InitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M3InitializationSpec, ...]:
    """Validate count, ordering, bounds, spread, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M3 starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M3 start IDs must be consecutive from one")
    vectors = [tuple(float(x) for x in item.constrained_tensor()) for item in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M3 initialization vectors must be unique")
    for item in records:
        if not 1 <= item.source_m2_start_id <= 8:
            raise ValueError("M3 source_m2_start_id must lie in 1..8")
        if not item.purpose.strip():
            raise ValueError("Every M3 initialization requires a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0 or item.delta0 <= 0.0:
            raise ValueError("M3 starts require positive w, s0, and delta0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M3 lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M3 kappa starts must be finite and positive")
        if not 0.0 < item.alpha_theta < 1.0:
            raise ValueError("M3 learning-rate starts must lie in (0,1)")
        if not bool(torch.all(torch.isfinite(item.raw_tensor(lapse_max=lapse_max)))):
            raise ValueError("Every M3 start must map to finite raw coordinates")
    if [item.source_m2_start_id for item in records] != list(range(1, 9)):
        raise ValueError("M3 starts must preserve M2 start IDs row-for-row")
    expected_alphas = (0.02, 0.001, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1)
    if tuple(item.alpha_theta for item in records) != expected_alphas:
        raise ValueError("M3 alpha_theta assignments must match the frozen eight-row table")
    return records
