"""Frozen initialization records for M2's six pooled fitted parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .m2_parameters import encode_m2_constrained_parameters


@dataclass(frozen=True)
class M2InitializationSpec:
    """One readable M2 optimizer start in registered parameter order."""

    start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    delta0: float
    purpose: str

    @property
    def hazard(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.ah, dtype=torch.float64)))

    @property
    def theta_short(self) -> float:
        return float(torch.sigmoid(torch.tensor(-self.delta0, dtype=torch.float64)))

    @property
    def theta_long(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.delta0, dtype=torch.float64)))

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
                self.w, self.s0, self.ell_short, self.ell_long, self.kappa,
                self.ah, self.delta0,
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
        return encode_m2_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m2_initialization_table(
    initializations: Iterable[M2InitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M2InitializationSpec, ...]:
    r"""Validate the frozen one-to-one extensions of M1's eight starts.

    Purpose
    -------
    Enforce identity, uniqueness, bounds, and transform validity of M2 starts.

    Mathematical definition
    -----------------------
    .. math::

       \Theta_r=(w_r,s_{0,r},\ell_r,g_r,a_{h,r},\delta_{0,r}),
       \qquad r=1,\ldots,8.

    Parameter interpretation
    ------------------------
    Every record is a pooled mouse-level starting point, not a session parameter.

    Input units and shapes
    ----------------------
    Accepts an iterable of readable scalar records; ``s0`` is seconds.

    Returned quantities and shapes
    ------------------------------
    Returns eight records sorted by consecutive ``start_id``.

    Constraints
    -----------
    Positive scales/contrast, open lapse bounds, unique vectors, and finite raw
    transforms are required.

    Numerical stability
    -------------------
    Every record is passed through the stable M2 encoder before acceptance.

    Causal restrictions
    -------------------
    Validation reads configuration values only and no behavioral data.

    Important edge cases
    --------------------
    Missing IDs, duplicate vectors, empty purposes, and boundary values raise.

    Related tests
    -------------
    ``test_registry_preserves_all_eight_m1_starts_and_adds_log4``.
    """

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M2 starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M2 start IDs must be consecutive from one")
    vectors = [
        (x.w, x.s0, x.ell_short, x.ell_long, x.kappa, x.ah, x.delta0)
        for x in records
    ]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M2 initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every M2 initialization needs a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0 or item.delta0 <= 0.0:
            raise ValueError("M2 starts require positive w, s0, and delta0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M2 lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M2 kappa starts must be finite and positive")
        if not bool(torch.all(torch.isfinite(item.raw_tensor(lapse_max=lapse_max)))):
            raise ValueError("Every M2 start must map to finite raw coordinates")
    return records
