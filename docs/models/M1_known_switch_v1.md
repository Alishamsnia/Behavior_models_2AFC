# M1_known_switch_v1

`M1_known_switch_v1` keeps `M1_hmm_true_v1`'s fixed true-rate prototypes
`(0.5,0.2,0.8)` and its unchanged Bayesian sensory readout, but replaces
M1's hazard-inferred block-switch timing with the true (oracle) trial-by-
trial switch index. The model is told exactly which trials start a new
block; it is never told which category (short/long) the new block is. This
is an independent implementation living in its own package, not a
modification of `M0_sensory_v1` or `M1_hmm_true_v1` and not a runtime
dependency of either.

## Scientific definition

Real block schedules for MC06/MC07/MC08 were checked (across all 128
available sessions in the sliding-window and single-session result trees,
before this model was written) and, in every session, `true_block` starts
`N` and, once left, strictly alternates `S`/`L` with no repeats and no
re-entry into `N` -- exactly the structural assumption M1's own transition
matrix already encodes. Given exact knowledge of switch timing, the model
tracks a scalar biased-subspace log-odds belief

```text
r_t = log( P(Z_t = L) / P(Z_t = S) ),
```

restricted to the two biased states; Neutral is handled separately (a
constant prior of 0.5 until the first switch). Let `e_t` be True exactly
when trial `t` is the first trial of a new block:

```text
r_t^- = 0            if e_t is the first switch (leaving Neutral)
r_t^- = -r_{t-1}^+    if e_t is a later switch
r_t^- = r_{t-1}^+     if e_t is False (no switch)

pi_t = 0.5                                     while still Neutral
pi_t = theta_S + (theta_L - theta_S)*sigmoid(r_t^-)   once biased
```

After a valid category observation `C_t in {0,1}` while biased,

```text
r_t^+ = r_t^- + (2*C_t - 1) * log(4)
```

where `log(4) = log(theta_L/theta_S) = log(0.8/0.2)`, the fixed
log-likelihood-ratio implied by M1's true rates. The sensory
readout/decision-region solver and category-dependent lapse mixture are
byte-for-byte the same functions M1 uses (`m0_sensory_v1.model.m1_bayesian`,
`m0_sensory_v1.model.bayesian.lapse_mixture`); only the source of the prior
`pi_t` changes.

### Why the hazard disappears

Because block identity is known to alternate deterministically and the
realized switch/no-switch outcome is now given rather than inferred, the
biased-block hazard never enters the recursion: conditioning on the
realized transition branch removes the hazard-weighted mixture entirely.
The fitted pooled vector is therefore `(w, s0, ell_short, ell_long, kappa)`
-- five parameters, identical in shape and meaning to `M0_sensory_v1`, and
one fewer than M1's own six. This is a deliberate design consequence, not
an oversight: the model answers "how much of M1's residual error is
switch-timing uncertainty, versus irreducible sensory/lapse noise", under
an oracle-timing assumption -- it does not claim to be an upper bound on
M1's achievable likelihood, since the mouse itself may not know switch
timing and refit sensory/lapse parameters can partially compensate for any
gap either way.

## Missing choices and session boundaries

Identical policy to M1: an invalid or omitted choice contributes no
Bernoulli term but the switch/feedback state update still applies unless
the trial is the session's last. Feedback observed while still Neutral has
no effect on belief (a one-hot Neutral state is invariant to any emission
reweighting), matching M1's own behavior exactly. Dynamic state is never
carried between sessions.

## The oracle switch signal

`m1_known_switch_session_trials_from_frame` derives `is_switch` from the
canonical `true_block` column and then discards block identity entirely --
only the boolean switch flag reaches the session tensors. The adapter is a
hard gate, not a best-effort parser: it raises if any trial's `true_block`
is missing or outside `{N,S,L}`, if the session does not start `N`, or if
`N` is re-entered after being left. `is_switch[0]` is always False by
construction (trial 0 is always a session's true start); a caller handing
this model a non-session-initial slice is rejected rather than silently
defaulting to a guessed initial state, since a schedule fragment with no
switches observed could equally be "still Neutral" or "already biased" and
there is no principled way to disambiguate that from local data alone.

## Registry and fitting

The free parameter vector `(w,s0,ell_short,ell_long,kappa)` is identical in
shape to M0's, so this model's eight frozen starts are M0's own registered
eight starts, reused verbatim (`config/initializations/m1_known_switch_v1_starts.json`,
which records the exact M0 source file and checksum it was copied from)
rather than re-derived. No fitted M0/M1 winner is used as an initialization.

## Verification

Because this model's recursion mechanism (deterministic switch-conditioned
belief) is structurally different from every hazard-driven model in this
family, it has no exact-nesting boundary to an already-verified parent --
unlike M2->M1, M3->M2, and M4->M3, which the family's usual correctness
methodology relies on. Correctness is instead established by:

- an independently-authored brute-force reference filter operating in raw
  probability space (not the log-odds formula under test), cross-checked
  against the closed-form recursion across many random valid schedules;
- direct hand-checkable cases (zero switches gives a constant 0.5 prior;
  feedback during Neutral has no effect; missing feedback retains belief
  exactly);
- full-vector finite-difference gradient agreement;
- pooled-objective session-order invariance;
- frame-adapter rejection of every malformed real-block-schedule case
  (missing labels, non-Neutral start, re-entry into Neutral, an
  out-of-alphabet label); and
- the alternation assumption itself checked directly against all 128
  available real MC06/MC07/MC08 sessions before implementation, not merely
  assumed from other models' documentation.

See `tests/test_m1_known_switch_model.py`.

## Registry, data preparation, and fitting

`registry.py` proxies genuinely shared fields (task moments, split,
optimizer, scheduler, seeds, epsilons) from the real, subject-specific M0
registry via `load_model_registry()`, exactly as `m1_dynamic_block_length_v1`
proxies from M1. `pipeline.py` reuses M0's own registered
`prepare_registered_fit_data` (called with the real M0 registry) for
discovery, the legacy inclusion rule, and the "never open locked test"
leak guard, then rebuilds this model's own session tensors (with
`is_switch`) from the returned canonical timeline -- it does not
reimplement discovery itself. `fit.py` is a fully independent multi-start
Adam/L-BFGS/hybrid fitter: sequential rather than parallel-worker-based
(each start still gets a fresh optimizer/scheduler and is independent;
this only affects wall-clock, not correctness), with every per-start
failure -- including the initial parameter encode and the restored-best
checkpoint's gradient recomputation -- contained so a single bad start
cannot abort the remaining seven. `analysis.py` reuses M0's fully generic
`calculate_evaluation_metrics` directly.

Subject registries: `config/m1_known_switch_v1.json` (MC08 default),
`config/m1_known_switch_v1_mc06.json`, `config/m1_known_switch_v1_mc07.json`.
CLI: `scripts/fit_m1_known_switch.py` (see repository root README for
example invocations); registry audit: `scripts/audit_m1_known_switch_registry.py`.

## Implementation status

The core scientific engine, registry, data preparation, and fitting CLI are
implemented and tested (unit tests on synthetic data for all three optimizer
strategies, failure containment, winner selection, and metrics agreement;
read-only structural checks against real MC06/MC07 MATLAB data confirming
correct train/validation session counts and zero locked-test files opened).
No real fit (i.e. an actual multi-epoch optimization run) has been executed
against real behavioral data -- only data loading and adapter conversion
have been checked against real files. Figure generation, recovery, and
sliding-window pipeline parity with M0-M4 are not yet built.
