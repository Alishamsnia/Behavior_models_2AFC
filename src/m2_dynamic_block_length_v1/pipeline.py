"""Data preparation for M2_dynamic_block_length_v1.

Delegates entirely to M2's own (model-agnostic) data-prep function via the
proxied ``source_m2_registry`` -- the session-data contract is identical to
M2's, so there is nothing model-specific to prepare here. Mirrors
``m4_hsmm_delta_v1/pipeline.py``'s delegation pattern to M3, pointed at M2
instead.
"""

from __future__ import annotations

from pathlib import Path

from m0_sensory_v1.pipeline import PreparedFitData, prepare_registered_m2_fit_data

from .registry import M2DynamicBlockLengthRegistry


def prepare_registered_m2_dynamic_block_length_fit_data(
    raw_data_root: str | Path,
    registry: M2DynamicBlockLengthRegistry,
) -> PreparedFitData:
    """Prepare train/validation timelines via M2's own data-prep function."""

    if registry.model_id != "M2_dynamic_block_length_v1":
        raise ValueError(
            "prepare_registered_m2_dynamic_block_length_fit_data requires an "
            "M2_dynamic_block_length_v1 registry"
        )
    return prepare_registered_m2_fit_data(raw_data_root, registry.source_m2_registry)
