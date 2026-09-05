"""Shared --resume staleness check for the sliding-window/single-session
runner scripts.

Every model's fit result already records a ``registry_config_sha256`` per
run (a hash of the exact registry config file used, which for M1/M2 also
folds in the subject overlay and, for the split/initialization identity,
the same bytes the model's own richer registry-snapshot check uses -- see
``m0_sensory_v1.model.fit._registry_snapshot``). ``--resume`` previously
reused a saved fit purely on file existence and a "complete" status flag,
with no check that the registry, config, initialization table, or split
assignment used to produce it still matches the one that would be used
today. This module makes that comparison a one-line call so every runner
script performs it the same way.
"""

from __future__ import annotations

from hashlib import sha256


def registry_config_sha256(registry) -> str:
    """Hash of the exact registry config file the given registry loaded."""

    return sha256(registry.config_path.read_bytes()).hexdigest()


def saved_result_registry_checksum(result: dict) -> str | None:
    """The ``registry_config_sha256`` recorded in a saved fit result, if any.

    Handles both schemas used across this repo's runners: a multi-start
    result (``{"runs": [{"registry_config_sha256": ...}, ...]}``) and a
    single fit-run result with the field at the top level (as saved by
    single-session fine-tune jobs). Returns ``None`` for a result saved
    before this field existed (older schema) rather than treating that as
    evidence of staleness -- there is nothing to compare against.
    """

    top_level = result.get("registry_config_sha256")
    if top_level:
        return top_level
    for run in result.get("runs", []) or []:
        checksum = run.get("registry_config_sha256")
        if checksum:
            return checksum
    return None


def result_registry_is_stale(result: dict | None, registry) -> bool:
    """True if a saved fit result was produced under a different registry
    config/initialization/split identity than ``registry`` reflects now --
    i.e. ``--resume`` should refit rather than reuse it."""

    if result is None:
        return False
    saved_checksum = saved_result_registry_checksum(result)
    if saved_checksum is None:
        return False
    return saved_checksum != registry_config_sha256(registry)
