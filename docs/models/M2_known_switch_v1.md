# M2_known_switch_v1

`M2_known_switch_v1` is the `M2_hmm_static_v1` counterpart to
`M1_known_switch_v1`: it keeps M2's fitted, symmetric, static subjective
rates `(0.5, sigmoid(-delta0), sigmoid(delta0))` and unchanged Bayesian
sensory readout, but replaces M2's hazard-inferred block-switch timing with
the true (oracle) trial-by-trial switch index, exactly as
`M1_known_switch_v1` does for M1. This is an independent implementation --
it has no runtime dependency on `M1_known_switch_v1` (mirroring how
`m1_dynamic_block_length_v1` has no runtime dependency on
`m4_hsmm_delta_v1` in this codebase); the two `known_switch` models are
cross-verified against each other only in the test suite.

## Parameters

Six parameters, ordered `(w, s0, ell_short, ell_long, kappa, delta0)`. The
first five are M0's unchanged sensory/lapse parameters. `delta0` is decoded
as `softplus(raw) + epsilon` (identical formula to M2's own), giving

```text
theta_N = 0.5
theta_S = sigmoid(-delta0)
theta_L = sigmoid(delta0) = 1 - theta_S
```

There is no hazard parameter: as in M1_known_switch_v1, conditioning on the
exactly-given switch/no-switch outcome removes the hazard-weighted mixture
from the recursion entirely (see `M1_known_switch_v1.md` for the full
argument). The free vector is therefore M0's five coordinates plus
`delta0` -- one fewer than M2's own seven.

## Causal recursion

Identical in structure to M1_known_switch_v1, with `theta_S`/`theta_L`
computed from the fitted `delta0` instead of fixed at 0.2/0.8, and the
per-observation feedback increment equal to `delta0` itself (which is
exactly `log(theta_L/theta_S)` under the symmetric decoding above, in place
of M1_known_switch_v1's fixed `log(4)`):

```text
r_t^+ = r_t^- + (2*C_t - 1) * delta0
```

## Two exact boundaries (this model's correctness anchors)

Unlike M1_known_switch_v1, this model has two genuine boundary checks
available, both used as primary correctness anchors in
`tests/test_m2_known_switch_model.py`:

- **`delta0 = 0`** (constructed directly via `with_explicit_delta0`, since
  the free positive decoder cannot reach it): `theta_S = theta_L = 0.5`, so
  the prior is a constant 0.5 regardless of switch timing -- matching M0's
  own fixed neutral prior.
- **`delta0 = log(4)`** (also constructed directly): the prototypes become
  exactly M1_known_switch_v1's fixed `(0.5,0.2,0.8)`, and this model then
  matches M1_known_switch_v1 trial-by-trial in prior, choice probability,
  and NLL -- verified in the test suite over sessions exercising both
  Neutral and multiple biased-block switches, with and without missing
  feedback.

## The oracle switch signal

Identical contract to M1_known_switch_v1:
`m2_known_switch_session_trials_from_frame` derives `is_switch` from the
canonical `true_block` column, validates the same real-block-schedule
invariants (starts `N`, every label in `{N,S,L}`, no re-entry into `N`),
and discards block identity before returning session tensors. The
validation logic is duplicated rather than imported from
`m1_known_switch_v1`, consistent with this model having no runtime
dependency on its sibling.

## Registry and fitting

Eight frozen starts: each of M0's own registered eight starts extended with
`delta0=log(4)` (`config/initializations/m2_known_switch_v1_starts.json`),
mirroring exactly how `M2_hmm_static_v1`'s own table extends each M1 start.
No fitted M0/M1/M2 winner is used as an initialization.

## Verification

- Full-vector finite-difference gradient agreement.
- An independently-authored brute-force reference filter (raw probability
  space) cross-checked against the closed-form recursion across many
  random valid schedules, at a non-boundary `delta0`.
- The `delta0=0` and `delta0=log(4)` exact boundaries above.
- Frame-adapter rejection of malformed real-block-schedule inputs.

See `tests/test_m2_known_switch_model.py`.

## Registry, data preparation, and fitting

Mirrors `M1_known_switch_v1.md`'s registry/pipeline/fit architecture exactly
(proxy the real subject-specific M0 registry; reuse M0's own
`prepare_registered_fit_data` for discovery/inclusion-rule/leak-safety, then
rebuild session tensors via this model's own adapter; sequential
multi-start Adam/L-BFGS/hybrid with full per-start failure containment).

Subject registries: `config/m2_known_switch_v1.json` (MC08 default),
`config/m2_known_switch_v1_mc06.json`, `config/m2_known_switch_v1_mc07.json`.
CLI: `scripts/fit_m2_known_switch.py`; registry audit:
`scripts/audit_m2_known_switch_registry.py`.

## Implementation status

The core scientific engine, registry, data preparation, and fitting CLI are
implemented and tested (unit tests on synthetic data; read-only structural
checks against real MC06/MC07 MATLAB data, including a cross-model check
confirming M1_known_switch_v1 and M2_known_switch_v1 derive identical
switch timing from the same real sessions). No real fit has been executed
against real behavioral data. Figure generation, recovery, and
sliding-window pipeline parity with M0-M4 are not yet built.
