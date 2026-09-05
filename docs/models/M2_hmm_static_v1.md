# M2_hmm_static_v1

`M2_hmm_static_v1` is the Phase-I guide's `M_HMM,static` model. It preserves
the complete M1 constant-hazard HMM and Bayesian choice readout while replacing
M1's fixed true task rates with fitted, symmetric, static subjective rates.

## Parameters

The seven parameters are ordered as:

```text
(w, s0, ell_short, ell_long, kappa, ah, delta0)
```

The common sensory/lapse parameters and fitted biased hazard
`h_B=sigmoid(ah)` are unchanged from M1. Neutral uses the fixed derived hazard
`h_N=h_B/2`; this adds no parameter.
The new positive contrast is decoded with `delta0=softplus(raw_delta0)+epsilon`
and defines:

```text
theta_N = 0.5
theta_S = sigmoid(-delta0)
theta_L = sigmoid(delta0) = 1 - theta_S
```

These prototypes never change within or between analyzed sessions. There is no
learning rate or prototype update in M2.

## Causal recursion

Every session resets `xi_pre=(1,0,0)` in state order `(N,S,L)`. On trial `t`:

1. Form `prior_long = dot(xi_pre, theta)`.
2. Predict the current choice using the full quadratic Bayesian decision-region
   solver and the shared category-dependent random-choice lapse mixture.
3. Add a Bernoulli term only for a valid completed choice.
4. If category feedback is valid, update `xi_post` using the same subjective
   `theta` values as the category emission probabilities.
5. Apply M1's constant-hazard transition before the next task trial.

Category, reward, true block, and true run length cannot affect the current
prediction. Category emissions update latent belief but are not added to the
outer fitted objective.

## Exact M1 restriction

At `delta0=log(4)`, the prototypes are exactly `(0.5,0.2,0.8)`. M2 must then
match M1 trial by trial in prior, block belief, choice probability, and NLL.
This restriction is protected by automated tests.

## Frozen initialization and fitting

M2 uses eight starts. Start `i` copies all six coordinates from M1 start `i`
and adds `delta0=log(4)`. Starts do not use any fitted M1 winner. The shared
fitter supports:

- full-batch Adam;
- strong-Wolfe L-BFGS;
- Adam warm-up followed by L-BFGS.

All methods optimize the same raw seven-vector and pooled choice objective.
Validation does not enter gradients, checkpoint selection, or start selection.

## Outputs and figures

`scripts/fit_m2.py` saves complete start histories, validation metrics,
trial-level causal trajectories, static prototype columns, hazard diagnostics,
and per-session performance. It generates:

- M2 belief, calibration, switch, prototype/hazard, and optional M1/M0
  comparison figures;
- a standalone fitted-`delta0` versus `log(4)` reference figure;
- when M1/M0 prediction tables are supplied, M1/M2 prior and block-belief
  overlays, M0/M1/M2 validation NLL/Brier, and M1/M2 validation ELPD;
- every M0/M1 behavioral and optimization figure family, updated for the six
  M2 coordinates.

`scripts/generate_m2_figures.py` rebuilds analysis without refitting and keeps
test access behind an explicit `--unlock-test` flag.

Its optional `--identifiability-grid-points N` analysis evaluates

```text
J(w_hat, s0_hat, ell_hat, g_hat, ah, delta0)
```

on an `(ah, delta0)` grid and saves a loss-surface CSV and PNG/PDF contour. This
does not change M2's loss function or fitted result. It is a post-fit
fixed-other-four slice of the same choice-only mean NLL, not a profile
likelihood and not an optimization step.

A completed non-smoke `scripts/fit_m2.py` run generates this diagnostic
automatically on a 15-by-15 training-objective grid after the winner and saved
checkpoint have already been selected. Smoke fits skip it.

## Recovery

`scripts/run_m2_recovery.py` uses the single registered generating hazard
`h=1/23`, because the observed median block duration is 23 task trials. It
evaluates near-zero (`delta0=0.1`), M1-like (`delta0=log(4)`), and strong
(`delta0=2.2`) contrasts. For every condition it preserves the real session
lengths, durations, omission positions, feedback masks, and true-block
schedules, generates categories/choices offline, and refits all seven parameters
from all eight registered starts. True block is available only to the simulator
and never enters the fitted evaluator. The script exports complete start files
and generating-versus-recovered hazard and `delta0` figures. Scientific evidence
for subjective prototypes requires both held-out improvement over M1 and
recoverable `delta0`.

## Scientific documentation governance

The Part VI audit registers M2 parameter transforms, prototypes, initialization
validation, frame adaptation, session and pooled objectives, simulations,
recovery helpers, registry loading, and identifiability analysis. Each follows
the same ten-section scientific contract and `.. math::` requirement as M0/M1;
`scripts/audit_scientific_documentation.py` therefore cannot report green while
silently checking zero M2 functions.
