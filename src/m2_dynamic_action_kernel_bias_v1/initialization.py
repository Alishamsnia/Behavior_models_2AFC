"""Frozen initialization records for M2_dynamic_action_kernel_bias's ten
pooled fitted parameters: M2_dynamic_previous_action_bias_v1's nine plus the
action-history learning-rate logit ``a``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_m2_dynamic_action_kernel_bias_constrained_parameters


@dataclass(frozen=True)
class M2DynamicActionKernelBiasInitializationSpec:
    """One start in
    ``(w,s0,ell_short,ell_long,kappa,ah,delta0,bh,eta_logit,a)`` order.

    ``source_m2_dynamic_start_id`` names the M2_dynamic_block_length_v1
    start (1-8) whose eight shared coordinates this row must preserve
    exactly.
    """

    start_id: int
    source_m2_dynamic_start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    ah: float
    delta0: float
    bh: float
    eta_logit: float
    a: float
    purpose: str

    @property
    def eta(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.eta_logit, dtype=torch.float64)))

    @property
    def alpha(self) -> float:
        return float(torch.sigmoid(torch.tensor(self.a, dtype=torch.float64)))

    @property
    def hazard_at_center(self) -> float:
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
                self.ah, self.delta0, self.bh, self.eta_logit, self.a,
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
        return encode_m2_dynamic_action_kernel_bias_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_m2_dynamic_action_kernel_bias_initialization_table(
    initializations: Iterable[M2DynamicActionKernelBiasInitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[M2DynamicActionKernelBiasInitializationSpec, ...]:
    """Validate IDs, uniqueness, bounds, variation, and finite raw transforms."""

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} M2_dynamic_action_kernel_bias starts")
    if [item.start_id for item in records] != list(range(1, expected_count + 1)):
        raise ValueError("M2_dynamic_action_kernel_bias start IDs must be consecutive from one")
    if sorted(item.source_m2_dynamic_start_id for item in records) != list(range(1, expected_count + 1)):
        raise ValueError(
            "M2_dynamic_action_kernel_bias starts must preserve M2_dynamic_block_length_v1's "
            "eight source IDs exactly once"
        )
    vectors = [
        (x.w, x.s0, x.ell_short, x.ell_long, x.kappa, x.ah, x.delta0, x.bh, x.eta_logit, x.a)
        for x in records
    ]
    if len(set(vectors)) != len(vectors):
        raise ValueError("M2_dynamic_action_kernel_bias initialization vectors must be unique")
    for item in records:
        if not item.purpose.strip():
            raise ValueError("Every M2_dynamic_action_kernel_bias initialization needs a purpose")
        if item.w <= 0.0 or item.s0 <= 0.0 or item.delta0 <= 0.0:
            raise ValueError("M2_dynamic_action_kernel_bias starts require positive w, s0, and delta0")
        if not 0.0 < item.ell_short < lapse_max or not 0.0 < item.ell_long < lapse_max:
            raise ValueError("M2_dynamic_action_kernel_bias lapse starts lie outside open bounds")
        if not torch.isfinite(torch.tensor(item.kappa)) or item.kappa <= 0.0:
            raise ValueError("M2_dynamic_action_kernel_bias kappa starts must be finite and positive")
        if item.bh < 0.0:
            raise ValueError("M2_dynamic_action_kernel_bias bh starts must be nonnegative")
        if not torch.isfinite(torch.tensor(item.eta_logit, dtype=torch.float64)):
            raise ValueError("M2_dynamic_action_kernel_bias bias-mixing-weight logits must be finite")
        if not torch.isfinite(torch.tensor(item.a, dtype=torch.float64)):
            raise ValueError("M2_dynamic_action_kernel_bias learning-rate logits must be finite")
        if not bool(torch.all(torch.isfinite(item.raw_tensor(lapse_max=lapse_max)))):
            raise ValueError("Every M2_dynamic_action_kernel_bias start must map to finite raw coordinates")
    if len({item.bh for item in records}) < 2:
        raise ValueError("M2_dynamic_action_kernel_bias initialization bh values must vary")
    if len({item.eta_logit for item in records}) < 2:
        raise ValueError("M2_dynamic_action_kernel_bias initialization bias-mixing-weight logits must vary")
    if len({item.a for item in records}) < 2:
        raise ValueError("M2_dynamic_action_kernel_bias initialization learning-rate logits must vary")
    return records
