# M0 full quadratic region: implementation and MC08 verification

M0 now integrates the same complete Gaussian decision region as the context models.
The fixed prior 0.5, five parameters, sensory variance, category-specific lapses,
choice mask, independent sessions, choice likelihood, frozen starts, optimizer
settings, convergence criteria, and rolling/fine-tuning logic are unchanged.

## MC08 failure and controlled rerun

Historical `full_session_matched_v1` windows 001–007 each had eight failed starts.
All 56 saved exceptions were exactly:

```text
ValueError: Quadratic boundary is not an upper long-choice threshold
```

The old solver rejects downward quadratics (`a2 < 0`), even though the region
between their two roots is a valid long-choice region. For MC08 task moments,
curvature changes sign at `w ≈ 0.0002748394336`. The original window-001/start-01
fit was replayed from the original source and reproduced this error at step 55.
The saved best checkpoint is not the rejected line-search proposal; the replay
and exception establish the failure rather than an assumption about that checkpoint.

Each of the first seven windows was then fitted from all eight frozen starts
with the saved L-BFGS settings (250 maximum outer steps, strong-Wolfe search,
unchanged tolerances). Only the training/validation union (session ranks 1–18)
was loaded. Every start converged, with no exceptions; every window has a winner.
Winning starts were selected by the unchanged training-NLL criterion.

| Window | Converged starts | Winner | Training mean NLL | Validation mean NLL |
|---|---:|---:|---:|---:|
| 001 | 8/8 | 4 | 0.494373559 | 0.438307556 |
| 002 | 8/8 | 4 | 0.495430960 | 0.419503378 |
| 003 | 8/8 | 4 | 0.488742702 | 0.459746704 |
| 004 | 8/8 | 8 | 0.486799420 | 0.440228207 |
| 005 | 8/8 | 6 | 0.474860357 | 0.443368727 |
| 006 | 8/8 | 6 | 0.492711669 | 0.405174152 |
| 007 | 8/8 | 7 | 0.486249373 | 0.409543944 |

All 56 saved checkpoints were reevaluated after the batched-readout review fixes:
maximum training/validation NLL difference = 0; maximum training-gradient-norm
difference = 0. The final mixed-dtype fix only affects the batched compatibility
path, which pooled M0 does not enter.

Existing result directories were not overwritten. Validation fits and detailed
logs are under `.codex-consult/mc08_full_region/`, with summaries in
`.codex-consult/mc08_validation.json` and `final_checkpoint_replay.json`.
These are isolated verification fits, not regenerated production figures or
rolling-test/single-session outputs.

## Diagnostics and result identity

M0 exports both region boundaries and the regime. The historical
`model_positive_threshold_s` column remains an upper-boundary alias, matching
context-model exports; it does not represent a single-threshold action rule.
Linear regions can have an infinite endpoint, and constant regions have NaN
boundaries. Omissions retain NaN numeric fields and an explicit skipped regime.

All three M0 registries declare the full-region rule and integration of negative
measurement tails. Updated config hashes invalidate old M0 fits under the existing
`--resume` checks. Historical release manifests and recovery reports were retained;
they are not certifications of the revised implementation. The existing shared
source fingerprint hashes the entire `src` tree, so other model runners with saved
fingerprints may also detect source staleness; that resume policy was not changed.

## Verification

- Nine targeted tests cover independent numerical Gaussian integrals, both curvature
  signs, linear/constant/no-root/double-root regimes, scalar and batched gradients,
  mixed dtypes, omission handling, registry policy, and bounded vectorized calls.
- Scalar probabilities and gradients compared against original source are bitwise
  identical across 11 regimes, protecting the existing context-model readout.
- M0 registry audit, scientific documentation audit (54 functions), and the existing
  M0/M1/M2 reference prediction audit passed.
- Full suite: 292 tests passed in 219.4 seconds. After the final mixed-dtype
  compatibility fix, all nine targeted full-region tests passed (the full suite
  had included eight of these; the ninth was added for that fix).

## Consult: plan and review — shared full-region readout

**Independent local assessment:** Reuse the context-model solver while preserving
M0’s neutral prior and vectorized session evaluation. Inspect saved exceptions and
replay fitting before claiming the MC08 failures are resolved.

| Finding | Raised by | Severity | Agree? | Action |
|---|---|---|---|---|
| Trial-shaped priors and constant-region output shapes need explicit support (`model/m0.py`, `model/m1_bayesian.py`). | Both | Medium | Yes | Scalar M0 coefficient set, broadcast constant outputs, tested batched compatibility. |
| New readout needs full boundary diagnostics (`model/bayesian.py`, `analysis.py`). | Both | Medium | Yes | Export both boundaries/regime and document compatibility aliases. |
| Per-trial batching causes a public-readout slowdown (`model/m1_bayesian.py`). | Independent review | Medium | Yes | Group into at most ten regimes and reuse the same vector equations; test call count and gradients. |
| Mixed-dtype batches fail at `index_copy` (`model/m1_bayesian.py`). | Second independent review | Medium | Yes | Promote batched inputs before allocating outputs; dedicated probability/boundary/gradient regression passed. |
| M0 config still labels negative roots as discarded (`config/m0_sensory_v1*.json`). | Independent review | Low | Yes | Correct policy in all three configs and validate it on load. |
| Removing an exception does not itself prove convergence. | Both | Medium | Yes | Replay original failure, then verify all 56 starts against unchanged convergence checks. |

**Where we split:** No disagreements. Both review rounds are complete; the final
mixed-dtype correction was checked locally with a dedicated regression rather than
launching a third independent review.

**Verdict:** ship. The requested readout change is implemented, the recorded
MC08 failure is reproduced and resolved in all seven validation reruns, and
no review finding remains unaddressed.
