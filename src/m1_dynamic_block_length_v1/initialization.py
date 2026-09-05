"""Frozen initialization records for M1_dynamic_block_length's seven pooled
fitted parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_m1_dynamic_block_length_constrained_parameters


@dataclass(frozen=True)
class M1DynamicBlockLengthInitializationSpec:
    """One start in ``(w,s0,ell_short,ell_long,kappa,ah,bh)`` order.

    ``source_m1_start_id`` names the M1 start (1-8) whose six shared
    coordinates this row must preserve exactly.
    """

    start_id: int
    source_m1_start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    bh: float
    purpose: str

    @property
    def hazard_at_center(self) -> float:
        """Return the probability-scale hazard at the registered center, for reporting."""

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
            [self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.ah, self.bh],
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
        return encode_m1_dynamic_block_length_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m1_dynamic_block_length_initialization_table(
    initializations: Iterable[M1DynamicBlockLengthInitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M1DynamicBlockLengthInitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M1_dynamic_block_length starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M1_dynamic_block_length start IDs must be consecutive from one")
    if sorted(item.source_m1_start_id for item in records) != list(range(1, expected_count + 1)):
        raise ValueError("M1_dynamic_block_length starts must preserve M1's eight source IDs exactly once")
    vectors = [
        (x.w, x.s0, x.ell_short, x.ell_long, x.kappa, x.ah, x.bh) for x in records
    ]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M1_dynamic_block_length initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every M1_dynamic_block_length initialization needs a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0:
            raise ValueError("M1_dynamic_block_length starts require positive w and s0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M1_dynamic_block_length lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M1_dynamic_block_length kappa starts must be finite and positive")
        if item.bh < 0.0:
            raise ValueError("M1_dynamic_block_length bh starts must be nonnegative")
        raw = item.raw_tensor(lapse_max=lapse_max)
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("Every M1_dynamic_block_length start must map to finite raw coordinates")
    if len({item.ah for item in records}) < 2:
        raise ValueError("M1_dynamic_block_length initialization hazard logits must vary")
    if len({item.bh for item in records}) < 2:
        raise ValueError("M1_dynamic_block_length initialization bh values must vary")
    return records
