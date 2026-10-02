"""Frozen initialization records for Extended_WSLS's three fitted parameters.

Reuses M0's own eight registered ``(ell_short, ell_long)`` rows unchanged,
by ``start_id``, each paired with one of the same eight learning rates
``m1_stimulus_kernel_v1`` already uses (``alpha`` in
``[0.02,0.05,0.1,0.2,0.35,0.5,0.7,0.9]``) -- identical in spirit to how
``m1_hmm_true_v1``'s own initialization table pairs those same eight rows
with eight varied hazards. This is the same deliberate eight-point joint
design M0 itself uses for multi-start coverage (see M0's own initialization
table note: "not a controlled one-parameter-at-a-time identifiability
sweep") -- lapse and learning rate are intentionally varied together
across the eight starts, a strict one-to-one pairing, not crossed
independently. ``registry.py`` verifies each start's ``(ell_short,
ell_long)`` matches its M0 source row exactly, the same provenance chain
``m1_stimulus_kernel_v1/registry.py`` enforces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_extended_wsls_constrained_parameters


@dataclass(frozen=True)
class ExtendedWslsInitializationSpec:
    """One Extended_WSLS start in ``(ell_short, ell_long, a)`` order."""

    start_id: int
    ell_short: float
    ell_long: float
    a: float
    purpose: str

    @property
    def alpha(self) -> float:
        """Return the probability-scale learning rate for reporting."""

        return float(torch.sigmoid(torch.tensor(self.a, dtype=torch.float64)))

    def as_dict(self) -> dict[str, int | float | str]:
        """Return JSON-safe constrained initialization values."""

        return asdict(self)

    def constrained_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``(ell_short, ell_long, a)`` coordinates."""

        return torch.tensor(
            [self.ell_short, self.ell_long, self.a], dtype=dtype, device=device
        )

    def raw_tensor(
        self,
        *,
        lapse_max: float = 1.0,
        epsilon: float = 1e-10,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the three unconstrained optimizer coordinates."""

        return encode_extended_wsls_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_extended_wsls_initialization_table(
    initializations: Iterable[ExtendedWslsInitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[ExtendedWslsInitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} Extended_WSLS starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("Extended_WSLS start IDs must be consecutive from one")
    vectors = [(x.ell_short, x.ell_long, x.a) for x in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("Extended_WSLS initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every Extended_WSLS initialization needs a purpose")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("Extended_WSLS lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.a, dtype=torch.float64)):
            raise ValueError("Extended_WSLS learning-rate logits must be finite")
        raw = item.raw_tensor(lapse_max=lapse_max)
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("Every Extended_WSLS start must map to finite raw coordinates")
    if len({item.a for item in records}) < 2:
        raise ValueError("Extended_WSLS initialization learning-rate logits must vary")
    if len({(item.ell_short, item.ell_long) for item in records}) < 2:
        raise ValueError("Extended_WSLS lapse configurations must vary across starts")
    return records
