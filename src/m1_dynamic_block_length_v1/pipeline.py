"""Data preparation for M1_dynamic_block_length_v1.

Delegates entirely to M1's own (model-agnostic) data-prep function via the
proxied ``source_m1_registry`` -- the session-data contract is identical to
M1's, so there is nothing model-specific to prepare here. Mirrors
``m4_hsmm_delta_v1/pipeline.py``'s delegation pattern to M3, pointed at M1
instead.
"""

from __future__ import annotations

from pathlib import Path

from m0_sensory_v1.pipeline import PreparedFitData, prepare_registered_m1_fit_data

from .registry import M1DynamicBlockLengthRegistry


def prepare_registered_m1_dynamic_block_length_fit_data(
    raw_data_root: str | Path,
    registry: M1DynamicBlockLengthRegistry,
) -> PreparedFitData:
    """Prepare train/validation timelines via M1's own data-prep function."""

    if registry.model_id != "M1_dynamic_block_length_v1":
        raise ValueError(
            "prepare_registered_m1_dynamic_block_length_fit_data requires an "
            "M1_dynamic_block_length_v1 registry"
        )
    return prepare_registered_m1_fit_data(raw_data_root, registry.source_m1_registry)
