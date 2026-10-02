"""Frozen initialization records for Pure_WSLS's two fitted lapse rates.

Reuses M0's own eight registered ``(ell_short, ell_long)`` rows unchanged,
by ``start_id`` -- the same eight rows ``m1_stimulus_kernel_v1`` extends
with its own learning-rate logit, and the same eight-point joint design M0
itself uses for multi-start coverage (see M0's own initialization table
note: "not a controlled one-parameter-at-a-time identifiability sweep").
Pure_WSLS has no sensory coordinate (``w``/``s0``/``kappa``), so only the
two lapse columns are carried over; ``registry.py`` verifies each start's
``(ell_short, ell_long)`` matches its M0 source row exactly, the same
provenance chain ``m1_stimulus_kernel_v1/registry.py`` enforces. Fitting
two parameters by L-BFGS is extremely cheap, so every start is run
sequentially in one process rather than dispatched across a process pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_pure_wsls_constrained_parameters


@dataclass(frozen=True)
class PureWslsInitializationSpec:
    """One Pure_WSLS start in ``(ell_short, ell_long)`` order."""

    start_id: int
    ell_short: float
    ell_long: float
    purpose: str

    def as_dict(self) -> dict[str, int | float | str]:
        """Return JSON-safe constrained initialization values."""

        return asdict(self)

    def constrained_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``(ell_short, ell_long)`` coordinates."""

        return torch.tensor([self.ell_short, self.ell_long], dtype=dtype, device=device)

    def raw_tensor(
        self,
        *,
        lapse_max: float = 1.0,
        epsilon: float = 1e-10,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the two unconstrained optimizer coordinates."""

        return encode_pure_wsls_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_pure_wsls_initialization_table(
    initializations: Iterable[PureWslsInitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[PureWslsInitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} Pure_WSLS starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("Pure_WSLS start IDs must be consecutive from one")
    vectors = [(x.ell_short, x.ell_long) for x in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("Pure_WSLS initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every Pure_WSLS initialization needs a purpose")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("Pure_WSLS lapse starts lie outside open bounds")
        raw = item.raw_tensor(lapse_max=lapse_max)
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("Every Pure_WSLS start must map to finite raw coordinates")
    if len({item.ell_short for item in records}) < 2:
        raise ValueError("Pure_WSLS ell_short starts must vary")
    if len({item.ell_long for item in records}) < 2:
        raise ValueError("Pure_WSLS ell_long starts must vary")
    return records
