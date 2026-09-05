# Goal
Second review pass. A prior draft proposed delegating the new models' per-trial
computation to M4's own `evaluate_m4_session` (with `alpha_theta`/`delta0`
pinned via `dataclasses.replace`). You (Codex) reviewed that and found a real
dtype bug and flagged the M4 dependency as fragile coupling for a "frozen,
independently verified" model family. The user explicitly decided against
that tradeoff: **the two new models must be genuinely separate, independent
implementations that do not import anything from `m4_hsmm_delta_v1` in
production code**, reusing only the exact existing logic of M1/M2 (which they
already share with M3/M4 today) plus a freshly-authored duration-dependent
HSMM kernel. Please review this REVISED architecture — confirm whether it
actually resolves your prior findings, and look for new problems it might
introduce.

# Constraints
Same as before: Python/PyTorch float64, unittest-based suite, frozen/checksummed
model-family conventions, nothing under existing model directories may be
edited (only new files, plus additive appends to two shared registration
tables and one audit script).

# Subject (revised plan, no code written yet)
Full plan file: /home/ali/.claude/plans/keen-discovering-teacup.md — please
read it in full; it has been substantially rewritten since your last review
(the "Revision: independent implementations, no dependency on M4" section
near the top replaces the old delegation architecture).

Summary of the new design:
- New shared primitive `src/m0_sensory_v1/model/capped_tail_hsmm.py`: an
  independently-authored (not copied, not imported from M4) implementation of
  the capped-tail duration-dependent HSMM kernel — `capped_tail_hazard`,
  `initialize_capped_tail_state`, `block_marginal`, `run_length_marginal`,
  `feedback_update`, `capped_tail_hsmm_transition` — living in the same
  shared-primitives location as `m1_session.py`/`m2_session.py`. Used by BOTH
  new models (written once, not duplicated twice). Gets its own unit tests
  plus a test-only numerical cross-check against `m4_hsmm_delta_v1`'s own
  kernel (test-suite dependency on M4 for verification confidence only, no
  production dependency).
- Each new model's own `session.py` is a fresh, independent per-trial loop
  mirroring `src/m0_sensory_v1/model/m2_session.py`'s exact structure (predict
  -> feedback update -> transition), generalized from a flat `(3,)` belief to
  a joint `(3,Rmax)` belief, using the new shared kernel for the transition
  step. M2_dynamic computes `theta` once per session from decoded `delta0`
  (M2's own formula); M1_dynamic hardcodes `theta=(0.5,0.2,0.8)` (M1's own
  convention). Neither model has any delta-learning/projection code at all.
- Each new model's `parameters.py` reuses the true parent's own decode
  function directly (`decode_m1_raw_parameters` from
  `m0_sensory_v1.model.m1_parameters`, or `decode_m2_raw_parameters` from
  `m0_sensory_v1.model.m2_parameters`) for the inherited coordinates, adding
  its own `bh` decode — this mirrors the confirmed existing pattern where
  M3's `parameters.py` imports `decode_m2_raw_parameters`/
  `encode_m2_constrained_parameters` from `m0_sensory_v1.model.m2_parameters`.
- Each new `registry.py` proxies only genuinely shared fields (task_moments,
  split, optimizer, scheduler, seeds, precision/epsilons) from the TRUE
  parent's own registry object, and declares its OWN `required_checks` tuple
  (9 checks: the parent's ~7 base checks + `no_optimizer_runtime_errors` +
  `tail_occupancy_acceptable`; no projection check, since there is no
  delta-update code to guard) rather than forwarding the parent's convergence
  object. M2_dynamic keeps M2's own `delta0` domain (no artificial floor),
  since there's no delta-update rule to protect from going negative.
- Full "Everything"-tier scaffolding otherwise unchanged: fit/analysis/
  figures/m0_style_figures/diagnostics/identifiability/recovery/simulation
  per model, sliding-window variant package, config+starts+schema+release,
  docs, tests. Family-wide registration (`post_analysis/common.py`,
  `raw_behavior_by_winner/session_mapping.py`) is sequenced to happen only
  after real fits exist. `scripts/audit_scientific_documentation.py` gets an
  additive `.update()` (not `src/m0_sensory_v1/documentation.py`, which is an
  existing model file and is never touched).

# Files worth reading
- /home/ali/.claude/plans/keen-discovering-teacup.md (the full revised plan)
- src/m0_sensory_v1/model/m2_session.py (the exact template the new session.py
  files will mirror)
- src/m0_sensory_v1/model/m1_session.py (constant_hazard_transition_matrix,
  for comparison to what capped_tail_hsmm_transition needs to generalize)
- src/m4_hsmm_delta_v1/prior_state.py (the HSMM kernel being independently
  reimplemented, NOT imported — read to judge whether a fresh reimplementation
  is likely to preserve its exact conservation/routing properties, and what a
  cross-check test should verify)
- src/m3_hmm_delta_v1/parameters.py lines 1-15 (confirms the decode-reuse
  pattern this design relies on)
- src/m0_sensory_v1/m2_registry.py (M2's own registry, to check the "proxy
  only genuinely shared fields, declare own required_checks" claim is
  actually coherent against M2's real contract)

# Questions
1. Does this revised design actually resolve your prior findings (dtype bug,
   M2/M4 delta_min domain mismatch, "not independent" coupling, ambiguous
   convergence-check ownership, the documentation.py edit contradiction)? Say
   explicitly, per finding, whether it's now moot/resolved or still applies.
2. Is "freshly reimplement the capped-tail HSMM kernel independently, verify
   via a cross-check test against M4's kernel" actually a safe way to get an
   independent implementation, or does it just relocate the coupling risk
   (e.g., if the fresh kernel has a subtle bug the cross-check test doesn't
   happen to exercise, would production code silently diverge from the
   verified-correct M4 behavior with no test catching it)? What specific
   properties should that cross-check test assert to make this safe (beyond
   what's already listed in the plan)?
3. Is reusing `decode_m1_raw_parameters`/`decode_m2_raw_parameters` directly
   (an existing, real production dependency on M1's/M2's own modules) an
   acceptable amount of coupling, given the user wants "separate codes" — or
   should even this be duplicated/re-derived independently? Where's the line?
4. Any new problems introduced by this design that weren't present in the
   delegation draft (e.g., risk of the two independently-written `session.py`
   files silently drifting from each other's HSMM bookkeeping conventions
   since they no longer share one runtime engine; risk in the "own
   required_checks tuple" design; anything about the shared
   `capped_tail_hsmm.py` module's location/API design)?
