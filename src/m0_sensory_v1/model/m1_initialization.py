"""Frozen initialization records for M1's five pooled fitted parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .m1_parameters import encode_m1_constrained_parameters


@dataclass(frozen=True)
class M1InitializationSpec:
    """One M1 start in ``(w,s0,ell_short,ell_long,kappa,ah)`` order."""

    start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    purpose: str

    @property
    def hazard(self) -> float:
        """Return the probability-scale constant biased hazard for reporting."""

        return float(torch.sigmoid(torch.tensor(self.ah, dtype=torch.float64)))

    def as_dict(self) -> dict[str, int | float | str]:
        """Return JSON-safe constrained initialization values."""

        return asdict(self)

    def constrained_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``(w,s0,ell_short,ell_long,kappa,ah)`` coordinates."""

        return torch.tensor(
            [self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah],
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
        """Return the five unconstrained optimizer coordinates."""

        return encode_m1_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m1_initialization_table(
    initializations: Iterable[M1InitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M1InitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M1 starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M1 start IDs must be consecutive from one")
    vectors = [(x.w, x.s0, x.ell_short, x.ell_long, x.kappa, x.ah) for x in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M1 initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every M1 initialization needs a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0:
            raise ValueError("M1 starts require positive w and s0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M1 lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M1 kappa starts must be finite and positive")
        raw = item.raw_tensor(lapse_max=lapse_max)
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("Every M1 start must map to finite raw coordinates")
    if len({item.ah for item in records}) < 2:
        raise ValueError("M1 initialization hazard logits must vary")
    if len({item.kappa for item in records}) < 2:
        raise ValueError("M1 initialization kappa values must vary")
    return records
