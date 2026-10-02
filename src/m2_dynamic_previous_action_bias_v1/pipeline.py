"""Registered data preparation for M2_dynamic_previous_action_bias_v1.

No model-specific data preparation exists here because the session-data
contract is identical to M2_dynamic_block_length_v1's own (``choice_long``,
the field this model's action-history update reads, is already carried by
every session). This module purely delegates to
M2_dynamic_block_length_v1's own registered pipeline.
"""

from __future__ import annotations

from pathlib import Path

from m2_dynamic_block_length_v1.pipeline import prepare_registered_m2_dynamic_block_length_fit_data
from m0_sensory_v1.pipeline import PreparedFitData

from .registry import M2DynamicPreviousActionBiasRegistry


def prepare_registered_m2_dynamic_previous_action_bias_fit_data(
    raw_data_root: str | Path,
    registry: M2DynamicPreviousActionBiasRegistry,
) -> PreparedFitData:
    r"""Prepare M2_dynamic_previous_action_bias fit data via the unchanged
    M2_dynamic_block_length_v1 registered pipeline.

    Purpose
    -------
    Reuse M2_dynamic_block_length_v1's own raw-file discovery, canonical
    trial preparation, and "never open locked test" guard unmodified.

    Mathematical definition
    -----------------------
    No transformation; this is a pure delegation.

    Parameter interpretation
    ------------------------
    ``registry.source_m2_dynamic_block_length_registry`` supplies the
    discovery/inclusion contract this model shares with its parent.

    Input units and shapes
    ----------------------
    ``raw_data_root`` names a directory of raw MATLAB session files.

    Returned quantities and shapes
    ------------------------------
    Returns M0's own ``PreparedFitData`` (train/validation session tuples,
    canonical timeline, audit).

    Constraints
    -----------
    ``registry.model_id`` must be ``M2_dynamic_previous_action_bias_v1``.

    Numerical stability
    -------------------
    Not applicable; no numeric transforms happen here.

    Causal restrictions
    -------------------
    Locked test files are never opened, exactly as in the parent pipeline.

    Important edge cases
    --------------------
    Delegates every edge case (missing files, inconsistent split roles) to
    M2_dynamic_block_length_v1's own pipeline.

    Related tests
    -------------
    ``test_m2_dynamic_previous_action_bias_pipeline_matches_m2_dynamic``.
    """

    if registry.model_id != "M2_dynamic_previous_action_bias_v1":
        raise ValueError("prepare_registered_m2_dynamic_previous_action_bias_fit_data requires the matching registry")
    return prepare_registered_m2_dynamic_block_length_fit_data(
        raw_data_root, registry.source_m2_dynamic_block_length_registry
    )
