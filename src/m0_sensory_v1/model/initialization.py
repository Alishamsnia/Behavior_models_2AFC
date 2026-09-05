"""Frozen constrained initialization records for M0 multi-start fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .parameters import encode_constrained_parameters


@dataclass(frozen=True)
class InitializationSpec:
    """One readable, non-Cartesian M0 optimizer starting point."""

    start_id: int
    w: float
    s0: float
    ell_short: float
    ell_long: float
    kappa: float
    purpose: str

    def as_dict(self) -> dict[str, int | float | str]:
        """Return the exact constrained record for manifests and result files."""

        return asdict(self)

    def constrained_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``(w,s0,ell_short,ell_long,kappa)`` in fitted coordinates."""

        return torch.tensor(
            [self.w, self.s0, self.ell_short, self.ell_long, self.kappa],
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
        r"""Convert the constrained start to differentiable raw coordinates.

        Purpose
        -------
        Supply the optimizer with an unconstrained raw vector while keeping
        the frozen initialization table scientifically readable.

        Mathematical definition
        -----------------------
        For the stored :math:`(w,s_0,\ell_short,ell_long,\kappa)`, return

        .. math::

           \left(
           \operatorname{softplus}^{-1}(w-\epsilon),
           \operatorname{softplus}^{-1}(s_0-\epsilon),
           \operatorname{logit}(\ell_short/\ell_{\max}),
           \operatorname{logit}(ell_long/\ell_{\max})
           \right).

        Parameter interpretation
        ------------------------
        Stored values are constrained scientific values; returned values are
        optimizer coordinates in the registered parameter order.

        Input units and shapes
        ----------------------
        ``s0`` and ``epsilon`` use seconds where applicable. The result has
        shape ``(5,)`` and defaults to ``torch.float64``.

        Returned quantities and shapes
        ------------------------------
        One raw Torch tensor on the requested device.

        Constraints
        -----------
        This delegates all transforms to the single registered encoder paired
        with the parameter decoder.

        Numerical stability
        -------------------
        Starts lie strictly inside open bounds, so inverse logits remain finite.

        Causal restrictions
        -------------------
        Initializations are fixed before validation comparisons and contain no
        trial, outcome, split-performance, or test-set information.

        Important edge cases
        --------------------
        Boundary values are rejected by the shared encoder.

        Related tests
        -------------
        ``test_all_frozen_starts_round_trip_and_have_finite_raw_values``.
        """

        return encode_constrained_parameters(
            self.constrained_tensor(dtype=dtype, device=device),
            lapse_max=lapse_max,
            epsilon=epsilon,
        )


def validate_initialization_table(
    initializations: Iterable[InitializationSpec],
    *,
    lapse_max: float = 1.0,
    expected_count: int = 8,
) -> tuple[InitializationSpec, ...]:
    r"""Validate the frozen spread table without generating a Cartesian grid.

    Purpose
    -------
    Enforce the exact structural and open-bound requirements of Part IV before
    the optimizer or any validation report can use the starts.

    Mathematical definition
    -----------------------
    Every start must satisfy

    .. math::

       w>0,\qquad s_0>0,\qquad 0<\ell_short<1,\qquad 0<ell_long<1.

    Parameter interpretation
    ------------------------
    The records are eight deliberately mixed points, not the
    :math:`8^4=4096` Cartesian product of marginal values.

    Input units and shapes
    ----------------------
    Input is an iterable of scalar initialization records; ``s0`` is seconds.

    Returned quantities and shapes
    ------------------------------
    Returns the validated records as an immutable tuple sorted by ``start_id``.

    Constraints
    -----------
    Requires exactly eight unique IDs ``1..8``, eight unique parameter vectors,
    nonempty purposes, strict bounds, and variation in every parameter.

    Numerical stability
    -------------------
    Every constrained record is converted to raw ``float64`` and required to be
    finite.

    Causal restrictions
    -------------------
    Validation uses only the frozen table and parameter bounds, never observed
    likelihoods or validation/test performance.

    Important edge cases
    --------------------
    Duplicate vectors, boundary lapse values, or a generated Cartesian table
    with the wrong count raise immediately.

    Related tests
    -------------
    ``test_frozen_table_has_exact_values_and_is_not_cartesian`` and
    ``test_invalid_initialization_tables_raise``.
    """

    records = tuple(sorted(initializations, key=lambda item: item.start_id))
    if len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} initialization records")
    if [record.start_id for record in records] != list(range(1, expected_count + 1)):
        raise ValueError("Initialization IDs must be consecutive integers 1..8")
    if any(not record.purpose.strip() for record in records):
        raise ValueError("Every initialization requires a nonempty purpose")

    vectors = [
        (item.w, item.s0, item.ell_short, item.ell_long, item.kappa)
        for item in records
    ]
    if len(set(vectors)) != len(vectors):
        raise ValueError("Initialization parameter vectors must be unique")
    for record in records:
        if not record.w > 0.0 or not record.s0 > 0.0:
            raise ValueError("Every initialization requires positive w and s0")
        if not 0.0 < record.ell_short < lapse_max:
            raise ValueError("Every initialization ell_short must lie strictly in (0, lapse_max)")
        if not 0.0 < record.ell_long < lapse_max:
            raise ValueError("Every initialization ell_long must lie strictly in (0, lapse_max)")
        if not torch.isfinite(torch.tensor(record.kappa)) or record.kappa <= 0.0:
            raise ValueError("Every initialization kappa must be finite and positive")
        if not bool(torch.all(torch.isfinite(record.raw_tensor(lapse_max=lapse_max)))):
            raise ValueError("Every initialization must map to finite raw coordinates")

    for name in ("w", "s0", "ell_short", "ell_long", "kappa"):
        if len({getattr(record, name) for record in records}) < 2:
            raise ValueError(f"Initialization parameter {name} does not span values")
    return records
