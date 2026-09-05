"""Frozen eight-start initialization records for M4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_m4_constrained_parameters


@dataclass(frozen=True)
class M4InitializationSpec:
    start_id: int
    source_m3_start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    delta0: float
    alpha_theta: float
    bh: float
    purpose: str

    def as_dict(self) -> dict:
        return asdict(self)

    def constrained_tensor(self, *, dtype=torch.float64, device=None) -> torch.Tensor:
        return torch.tensor(
            (self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah, self.delta0,
             self.alpha_theta, self.bh),
            dtype=dtype,
            device=device,
        )

    def raw_tensor(
        self, *, lapse_max: float = 1.0, epsilon: float = 1e-10,
        dtype=torch.float64, device=None,
    ) -> torch.Tensor:
        return encode_m4_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m4_initialization_table(
    initializations: Iterable[M4InitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M4InitializationSpec, ...]:
    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M4 starts")
    if tuple(item.start_id for item in records) != tuple(range(1, expected_count + 1)):
        raise ValueError("M4 start IDs must be consecutive from one")
    if tuple(item.source_m3_start_id for item in records) != tuple(range(1, 9)):
        raise ValueError("M4 starts must preserve M3 start IDs row-for-row")
    vectors = [tuple(float(x) for x in item.constrained_tensor()) for item in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M4 initialization vectors must be unique")
    for item in records:
        if min(item.w, item.s0, item.delta0, item.bh) <= 0:
            raise ValueError("M4 starts require positive w, s0, delta0, and bh")
        if not 0 < item.ell_short < lapse_max or not 0 < item.ell_long < lapse_max:
            raise ValueError("M4 lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M4 kappa starts must be finite and positive")
        if not 0 < item.alpha_theta < 1:
            raise ValueError("M4 alpha_theta starts must lie in the M3 open domain (0,1)")
        if not item.purpose.strip():
            raise ValueError("Every M4 start requires a purpose")
        if not bool(torch.all(torch.isfinite(item.raw_tensor(lapse_max=lapse_max)))):
            raise ValueError("Every M4 start must map to finite raw coordinates")
    return records
