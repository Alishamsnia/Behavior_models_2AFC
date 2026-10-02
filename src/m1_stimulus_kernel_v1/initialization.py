"""Frozen initialization records for M1_stimulus_kernel's six pooled fitted
parameters.

Mirrors ``m0_sensory_v1.model.m1_initialization.M1InitializationSpec``
exactly (``ah``/``hazard`` renamed to ``a``/``alpha``), because this model
adds exactly one new pooled logit beyond M0's five, the same shape M1
itself adds. Unlike ``m1_known_switch_v1`` (which fits no new parameter and
therefore reuses M0's own ``InitializationSpec`` directly), this model
needs its own six-field record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_m1_stimulus_kernel_constrained_parameters


@dataclass(frozen=True)
class M1StimulusKernelInitializationSpec:
    """One M1_stimulus_kernel start in ``(w,s0,ell_short,ell_long,kappa,a)`` order."""

    start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    a: float
    purpose: str

    @property
    def alpha(self) -> float:
        """Return the probability-scale constant learning rate for reporting."""

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
        """Return ``(w,s0,ell_short,ell_long,kappa,a)`` coordinates."""

        return torch.tensor(
            [self.w, self.s0, self.ell_short, self.ell_long, self.kappa, self.a],
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
        """Return the six unconstrained optimizer coordinates."""

        return encode_m1_stimulus_kernel_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m1_stimulus_kernel_initialization_table(
    initializations: Iterable[M1StimulusKernelInitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M1StimulusKernelInitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M1_stimulus_kernel starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M1_stimulus_kernel start IDs must be consecutive from one")
    vectors = [(x.w, x.s0, x.ell_short, x.ell_long, x.kappa, x.a) for x in records]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M1_stimulus_kernel initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every M1_stimulus_kernel initialization needs a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0:
            raise ValueError("M1_stimulus_kernel starts require positive w and s0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M1_stimulus_kernel lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M1_stimulus_kernel kappa starts must be finite and positive")
        if not torch.isfinite(torch.tensor(item.a, dtype=torch.float64)):
            raise ValueError("M1_stimulus_kernel learning-rate logits must be finite")
        raw = item.raw_tensor(lapse_max=lapse_max)
        if not bool(torch.all(torch.isfinite(raw))):
            raise ValueError("Every M1_stimulus_kernel start must map to finite raw coordinates")
    if len({item.a for item in records}) < 2:
        raise ValueError("M1_stimulus_kernel initialization learning-rate logits must vary")
    if len({item.kappa for item in records}) < 2:
        raise ValueError("M1_stimulus_kernel kappa values must vary")
    return records
