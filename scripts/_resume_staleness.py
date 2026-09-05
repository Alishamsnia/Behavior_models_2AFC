"""Shared --resume staleness check for the sliding-window/single-session
runner scripts.

Two independent freshness signals are checked, because they have different
retroactive guarantees:

1. **Registry identity** (`result_registry_is_stale`): every model's fit
   result already records a ``registry_config_sha256`` per run, and M0/M1/M2
   additionally embed the exact ``initialization_table_sha256`` and
   ``split_assignment_sha256`` already used by their own registry-snapshot
   check (see ``m0_sensory_v1.model.fit._registry_snapshot``). This data
   already exists in every real saved result, so a saved result's identity
   can be verified (or found wanting) with full confidence, and a result
   that cannot be verified at all -- an older schema without even the
   top-level checksum -- is treated as stale rather than silently reused.

2. **Manifest fingerprint** (`manifest_fingerprint_is_stale`): a newer,
   broader check (currently: a hash of the model's own source code tree)
   that was never recorded by any result produced before this check
   existed. Unlike (1), there is no way to retroactively know whether an
   old result's source code matched today's -- no per-fit code snapshot
   was ever taken, and this repository has no commit history predating a
   recent `git init`. Treating that absence as "stale" would force a full
   refit of every already-computed result the next time anyone resumes,
   as a side effect of adding a new check, which is a materially different
   and much larger action than the check itself warrants. A saved fingerprint
   is therefore only used when it exists to be compared against; its absence
   is "not yet verifiable," not evidence of staleness. Every runner records a
   fingerprint into its own window/session manifest going forward, so this
   protection strengthens over time as fits are (re)run for other reasons.

``--resume`` previously reused a saved fit purely on file existence and a
"complete" status flag, with no check that the registry, source code,
initialization table, or split assignment used to produce it still matches
the one that would be used today. This module makes both checks a one-line
call so every runner script performs them the same way.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def registry_config_sha256(registry) -> str:
    """Hash of the exact registry config file the given registry loaded."""

    return sha256(registry.config_path.read_bytes()).hexdigest()


def source_code_sha256(source_root: Path) -> str:
    """Deterministic hash of every ``.py`` file under ``source_root``
    (relative path plus contents, sorted for reproducibility).

    Mirrors the scheme ``scripts/audit_m0_release.py`` already uses for the
    frozen M0 release manifest's ``source_tree_sha256``, reused here rather
    than reinvented so the two checks agree on what "the same source code"
    means.
    """

    digest = sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _registry_artifact_hashes(registry) -> dict[str, str]:
    """Best-effort extra identity hashes exposed by some registry families
    (M0/M1/M2: initialization table, split assignment). Absence here means
    "this model family has no such concept" -- M3/M4 and the dynamic
    variants only ever expose the plain config checksum.
    """

    hashes: dict[str, str] = {}
    table_hash = getattr(registry, "initialization_table_sha256", None)
    if table_hash:
        hashes["initialization_table_sha256"] = str(table_hash)
    split = getattr(registry, "split", None)
    split_hash = getattr(split, "assignment_sha256", None) if split is not None else None
    if split_hash:
        hashes["split_assignment_sha256"] = str(split_hash)
    return hashes


def current_registry_identity(registry) -> dict[str, str]:
    """Everything about ``registry``'s identity that can be verified against
    a saved result today: the config checksum, plus initialization-table and
    split-assignment hashes where the registry family has them."""

    identity = {"registry_config_sha256": registry_config_sha256(registry)}
    identity.update(_registry_artifact_hashes(registry))
    return identity


def _all_run_records(result: dict) -> list[dict]:
    runs = result.get("runs")
    if isinstance(runs, list) and runs:
        return runs
    return [result]


def saved_registry_identity(result: dict) -> dict[str, str] | None:
    """Extract the registry-identity fields already recorded in a saved fit
    result (see module docstring), verified for internal agreement.

    A multi-start result records the same registry per run; if two runs of
    the *same* saved result disagree (which should never legitimately
    happen), that is treated as unverifiable rather than picking one value
    arbitrarily. Returns ``None`` when nothing is recorded at all (older
    schema, or a genuinely empty/malformed result) -- there is nothing to
    compare against, and a caller should treat that as stale.
    """

    all_runs = _all_run_records(result)
    per_run: list[dict[str, str]] = []
    for run in all_runs:
        checksum = run.get("registry_config_sha256")
        if not checksum:
            # A run with no checksum at all is exactly as unverifiable as
            # one whose checksum disagrees with the others -- skipping it
            # instead would let a single verifiable run (out of many
            # unverifiable ones) make the whole result look confirmed.
            return None
        identity = {"registry_config_sha256": str(checksum)}
        snapshot = run.get("model_registry_snapshot") or {}
        artifacts = snapshot.get("resolved_artifacts") or {}
        if artifacts.get("initialization_table_sha256"):
            identity["initialization_table_sha256"] = str(artifacts["initialization_table_sha256"])
        if artifacts.get("split_assignment_sha256"):
            identity["split_assignment_sha256"] = str(artifacts["split_assignment_sha256"])
        per_run.append(identity)
    if not per_run:
        return None
    first = per_run[0]
    if any(identity != first for identity in per_run[1:]):
        return None
    return first


def result_registry_is_stale(result: dict | None, registry) -> bool:
    """True if a saved fit result was produced under a different registry
    config/initialization/split identity than ``registry`` reflects now --
    i.e. ``--resume`` should refit rather than reuse it.

    A saved result with no verifiable identity at all (older schema, or
    internally disagreeing runs) is treated as stale: unlike the manifest
    fingerprint below, this data already exists in every properly-produced
    result, so its absence is itself evidence something is wrong, not
    merely unrecorded.
    """

    if result is None:
        return False
    saved = saved_registry_identity(result)
    if saved is None:
        return True
    current = current_registry_identity(registry)
    for key, saved_value in saved.items():
        current_value = current.get(key)
        if current_value is not None and current_value != saved_value:
            return True
    return False


def manifest_fingerprint(registry, source_root: Path) -> dict[str, str]:
    """The newer, go-forward freshness fingerprint to record in a window's
    or session's own manifest (not the model's schema-validated fit
    result) when its status is written as complete."""

    return {"source_code_sha256": source_code_sha256(source_root)}


def manifest_fingerprint_for_write(
    resumed_from_saved_fit: bool,
    previous_manifest: dict | None,
    registry,
    source_root: Path,
) -> dict[str, str] | None:
    """The ``resume_fingerprint`` value a runner should write into a
    manifest it is marking complete.

    If this run actually (re)computed the fit from scratch, a fresh
    fingerprint correctly describes what produced it. But if this run only
    *resumed* an already-saved fit (``resumed_from_saved_fit=True`` --
    reusing an old ``fit_result.json`` and only redoing post-processing
    like predictions or figures), writing a fingerprint computed from
    TODAY's source code would misrepresent the underlying fit's actual
    provenance as verified against current code, when it was never
    (re)run under it. In that case the previous manifest's own recorded
    fingerprint (which may itself be absent, for a legacy manifest) is
    carried over unchanged instead.
    """

    if resumed_from_saved_fit:
        return (previous_manifest or {}).get("resume_fingerprint")
    return manifest_fingerprint(registry, source_root)


def manifest_fingerprint_is_stale(
    previous_manifest: dict | None, registry, source_root: Path
) -> bool:
    """True if a *recorded* manifest fingerprint disagrees with the current
    one. A manifest that predates this fingerprint (or never completed)
    records nothing here, which is "not yet verifiable" rather than stale --
    see the module docstring for why that is not the same as unknown
    registry identity."""

    if not previous_manifest:
        return False
    saved = previous_manifest.get("resume_fingerprint")
    if not saved:
        return False
    current = manifest_fingerprint(registry, source_root)
    for key, current_value in current.items():
        saved_value = saved.get(key)
        if saved_value is not None and saved_value != current_value:
            return True
    return False
