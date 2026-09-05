# M1_dynamic_block_length_v1

`M1_dynamic_block_length_v1` adds duration-dependent ("dynamic block length")
hazard estimation to M1. It keeps M1's fixed true-rate prototypes
`(0.5,0.2,0.8)` (no fitted prototype parameter, no learning) and replaces only
the constant-hazard block transition with a capped-tail joint state x
run-length HSMM transition whose hazard depends on how long the current
block has run.

This is an **independent implementation**, not a modification of any existing
model and not a runtime dependency on `M4_hsmm_delta_v1`. It reuses only
genuinely shared, pre-existing primitives: M1's own parameter decoder
(`m0_sensory_v1.model.m1_parameters`), the shared quadratic decision-region
solver (`m0_sensory_v1.model.m1_bayesian`), and a freshly-authored capped-tail
HSMM kernel (`m0_sensory_v1.model.capped_tail_hsmm`) that lives alongside
M1's/M2's own building blocks in the project's shared-primitives namespace.
`m0_sensory_v1.model.capped_tail_hsmm` is not part of M0's own scientific
contract; it is cross-verified against `m4_hsmm_delta_v1`'s independently
authored kernel only in the test suite, never at runtime.

## Parameters

Seven parameters are ordered as:

```text
(w, s0, ell_short, ell_long, kappa, ah, bh)
```

The first six are decoded by M1's own unchanged decoder
(`decode_m1_raw_parameters`): `w, s0, ell_short, ell_long, kappa` are the
common sensory/lapse parameters and `ah` anchors the biased hazard logit at
the registered center run length. There is no fitted prototype coordinate at
all; the prototypes are the hardcoded constant `theta=(0.5,0.2,0.8)`,
identical to M1's own `evaluate_m1_session`.

`bh=softplus(raw_bh)+epsilon` is the new nonnegative run-length hazard slope.
The biased-state hazard is

```text
h_B(r) = sigmoid(ah + bh * (r - 23))
```

with the fixed task-design tie `h_N(r) = h_B(r) / 2`. The run-length center is
fixed at 23 trials and the largest represented run length is capped at 65
(`r=65` represents `R >= 65`, a capped tail, not a forced switch).

## Joint state and duration hazard

The causal belief is a 3x65 joint distribution over block state
`z in (N,S,L)` and represented run length `r in (1,...,65)`. Every session
resets to `P(N,r=1)=1`. On a switch, `S -> L` and `L -> S`; a neutral switch
enters short or long according to the fixed first-long probability (0.5).
Neutral is not re-entered. On survival, run length increments; survival from
`r=65` stays in the tail bin, using the corresponding hazard at `r=65`.

## Causal trial order

For each trial:

1. Marginalize the joint pre-choice belief over run length to get the block
   prior; compute the choice prior as `dot(block_prior, theta)`.
2. Predict choice from duration and the pre-choice prior via M1's general
   quadratic decision-region solver, with the shared category-dependent
   lapse mixture.
3. Score a valid choice.
4. Use valid category feedback to update the joint belief with the fixed
   `theta` (Bayesian emission, generalized to every run-length column of the
   fixed block).
5. Apply the duration-dependent HSMM transition.

Invalid choice rows contribute no likelihood. Missing feedback skips emission
update but does not skip the subsequent HSMM transition. True block labels
and true run lengths never enter predictions, optimization, or simulated
choices.

## Exact nesting

At `bh=0` (constructed via `M1DynamicBlockLengthParameters.with_explicit_bh(0.0)`,
since the free `softplus`-decoded `bh` can approach but never exactly reach
zero), every biased hazard entry becomes `sigmoid(ah)` for every represented
run length, and summing the joint state over run length reproduces exactly
M1's own constant-hazard block recursion -- verified trial-by-trial (prior,
block belief, choice probability, NLL) on sessions longer than 65 trials to
exercise the tail-state code path.

A second, equally strong nesting fact: this model is numerically identical to
`M2_dynamic_block_length_v1` evaluated at `delta0=log(4)` (giving prototypes
exactly `(0.5,0.2,0.8)`) at the same `(w,s0,ell_short,ell_long,kappa,ah,bh)`
for any `bh` -- this is verified directly against real
`M2_dynamic_block_length_v1` predictions and is the primary guard against the
two independently-written session loops drifting apart.

## Frozen initialization and fitting

Eight starts. Start `i` copies all six coordinates from M1's registered start
`i` exactly and adds `bh` from the registered set
`{0.02,0.05,0.10,0.20,0.40,0.08,0.30,0.01}` (the same `bh` values M4 uses,
reused as literal registered numbers, not as shared code). No fitted M1
winner is ever used to initialize this model. The shared fitter supports
full-batch Adam, strong-Wolfe L-BFGS, and Adam warm-up followed by L-BFGS.

Convergence requires nine checks: M1's own seven
(`finite_objective, finite_parameters, finite_gradients, small_gradient_norm,
stable_fresh_training_nll, no_parameter_transform_errors,
no_probability_normalization_errors`) plus two duration-specific checks
(`no_optimizer_runtime_errors, tail_occupancy_acceptable`). There is no
projection-related check, because there is no delta-update rule to guard.
`fit.py` asserts the evaluated check-name set exactly equals the registry's
declared `required_checks` before computing `converged`, closing a gap
present in the upstream M4 fit script that this model does not repeat. A
failed tail-occupancy gate (mean pre-choice tail probability over 0.01) is a
documented scientific-rejection reason, not silent non-convergence; its
`diagnostics.py` truncation-sensitivity check assesses whether a larger
`Rmax` is warranted before drawing conclusions from such a fit.

## Registry design

`registry.py` proxies genuinely shared fields (precision, epsilons,
objective weighting, state order, true-rate prototypes,
`first_long_probability`, `neutral_hazard_scale`, task moments, split,
optimizer, scheduler, seeds) from the real M1 registry
(`load_m1_model_registry`, unmodified) via read-only forwarding properties --
the same architectural pattern `m4_hsmm_delta_v1/registry.py` uses for its
`source_m3_registry`, applied here to M1 instead. `convergence` is its own
object, not forwarded, since this model's convergence contract differs from
M1's. There is no Python reference to `m4_hsmm_delta_v1` anywhere in this
model's registry or production code.

## Outputs and figures

`scripts/fit_m1_dynamic_block_length.py` saves complete start histories,
validation metrics, trial-level causal trajectories (including duration
diagnostics: expected/MAP run length, run-length entropy, expected/MAP
hazard, switch/survival mass, tail probability), the fitted hazard curve,
implied duration distribution, and per-session performance. It generates
native duration/diagnostic figures and the five-family M0-style figure suite.

`scripts/generate_m1_dynamic_block_length_figures.py` and
`scripts/generate_m1_dynamic_block_length_m0_style_figures.py` rebuild
analysis without refitting.

## Recovery

`scripts/run_m1_dynamic_block_length_recovery.py` grids over the empirical
median-duration hazard-at-center `{0.02, 1/23, 0.08}` and `bh in
{0, 0.05, 0.10, 0.20, 0.40}` (including the exact `bh=0` boundary), simulating
on the real session design (lengths, omissions, feedback masks) and refitting
all seven parameters from all eight registered starts.

## Comparison to the true parent

`scripts/compare_m1_m1_dynamic_block_length.py` produces paired held-out
comparisons against real M1 predictions -- this is the scientific test of
whether duration-dependent hazard estimation improves over M1's constant
hazard.
