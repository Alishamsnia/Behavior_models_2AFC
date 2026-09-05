# M4_hsmm_delta_v1: capped-tail duration-aware prototype learner

`M4_hsmm_delta_v1` extends Model 3 additively. M0--M3 source behavior remains
unchanged. M4 keeps M3's sensory model, choice rule, subjective prototypes,
responsibility-weighted delta learning, session split, objective, optimizers,
convergence rules, projection audit, and winner-selection rule. Its only new
fitted coordinate is a nonnegative run-length slope `bh`.

## Parameters and registered constants

One nine-vector is pooled per mouse:

`(w, s0, ell_short, ell_long, kappa, ah, delta0, alpha_theta, bh)`.

The first eight coordinates have precisely their M3 meanings. The ninth is
decoded as `bh = softplus(raw_bh) + epsilon`, so a freely fitted M4 has a
positive duration slope. The explicit `with_explicit_bh(0)` helper represents
the exact M3 boundary for nesting and recovery tests.

The run-length center is fixed at 23 trials. This corrects the conflicting
value in the Phase-I document and is not fitted. The largest represented run
length is fixed at 65. State `r=65` means `R >= 65`; it is a capped tail, not a
forced switch and not an assertion that a block is exactly 65 trials long.

## Joint state and duration hazard

The causal belief is a 3 by 65 joint distribution over block state
`z in (N,S,L)` and represented run length `r in (1,...,65)`. Every session
resets to `P(N,r=1)=1`. The biased-state hazard is

`h_B(r) = sigmoid(ah + bh * (r - 23))`.

Neutral uses the fixed task-design tie `h_N(r)=h_B(r)/2`; this adds no fitted
parameter. On a switch, `S -> L` and `L -> S`. A neutral switch enters short
or long according to the unchanged fixed first-long probability (registered as 0.5). The sensory
Weber parameter `w` does not enter this transition. Neutral is not re-entered.
On survival, run length increments; survival from `r=65` stays in the tail bin and continues using
the corresponding state-specific hazard at `r=65`. Thus all mass is retained
and normalized for arbitrarily long blocks.

## Causal trial order

For each trial, M4:

1. marginalizes the joint pre-choice belief over run length;
2. builds M3's prototypes from the current `delta`;
3. predicts choice from duration and the pre-choice prior;
4. scores a valid choice;
5. uses valid category feedback to update the joint belief;
6. marginalizes the joint posterior to obtain M3's three responsibilities;
7. applies M3's unchanged delta update and positivity projection; and
8. applies the duration-dependent HSMM transition.

Invalid choice rows contribute no likelihood. Valid feedback on an omission
still updates belief and learning. Missing feedback skips emission and learning
but does not skip the subsequent HSMM transition. True block labels and true
run lengths never enter predictions, optimization, or simulated choices; block
labels remain available only for retrospective diagnostics.

## Exact M3 nesting

When `bh=0`, every biased hazard entry is `sigmoid(ah)` and every neutral
hazard entry is half that value. Summing the M4 joint state over run length
then gives exactly M3's tied-hazard block recursion. Acceptance
tests establish trialwise agreement in choice probabilities, priors, block
beliefs, delta trajectories, and NLL across sessions longer than 65 trials.
The tail-state implementation is therefore included in the nesting test rather
than tested only on short sessions.

## Starts and fitting

There are exactly eight independent M4 starts. For start ID `i`, its first seven
coordinates equal M3 start `i` exactly, and only `bh` is appended. A fitted M3
winner is never used to initialize M4. M3 and M4 use the same requested learning
rate starts in ID order:

`{0.020, 0.001, 0.005, 0.010, 0.030, 0.050, 0.075, 0.100}`.

These are strictly positive because the free sigmoid parameterization cannot
represent zero at a finite raw value. Exact zero is reserved for explicit
boundary tests. The eight M4 `bh` starts are

`{0.020, 0.050, 0.100, 0.200, 0.400, 0.080, 0.300, 0.010}`.

Full-batch Adam, L-BFGS, and hybrid optimization retain M3's settings. Training
likelihood alone controls optimization, convergence, checkpointing, and start
selection. Validation is reported but never used to select a start. Projection
activation must remain within M3's registered acceptance limit. The optimizer
is maximum-likelihood only: no MAP penalty or prior enters fitting. The
trialwise MAP run length is an exported descriptive diagnostic and never enters
the choice likelihood, state recursion, checkpointing, or model selection.

Mean pre-choice probability in the `R>=65` tail is also an automated training
acceptance gate. The registered maximum is 0.01. A start above that threshold
cannot be marked converged and cannot win, even if its optimizer checks pass.
The fit result records training/validation mean and maximum tail occupancy.

Run the registry audit with:

```bash
PYTHONPATH=src python scripts/audit_m4_registry.py
```

Run an end-to-end smoke fit with:

```bash
PYTHONPATH=src python scripts/fit_m4.py \
  --registry config/m4_hsmm_delta_v1.json \
  --raw-data-root /path/to/raw/mat/files \
  --output-dir results/m4_smoke \
  --optimizer lbfgs --smoke-test --smoke-epochs 2
```

Run parameter recovery with the eight registered parallel CPU starts:

```bash
PYTHONPATH=src python scripts/run_m4_recovery.py \
  --raw-data-root /path/to/raw/mat/files --output-dir results/m4_recovery \
  --registered-grid-base 0.03,0.55,0.03,0.5 \
  --max-steps 250 --parallel-workers 8
```

The grid includes `bh={0,.05,.10,.20,.40}`, `alpha_theta=0`, centered hazard
`1/23`, and multiple initial contrasts. `bh=0` is generated exactly, not by a
small positive surrogate. The separate `run_m3_m4_model_recovery.py` command
fits both models with their frozen eight-start CPU pipelines and uses held-out
validation evidence with the one-standard-error preference for M3.

After the model and analysis plan are frozen, generate post-fit diagnostics:

```bash
PYTHONPATH=src python scripts/generate_m4_figures.py \
  --raw-data-root /path/to/raw/mat/files \
  --fit-result results/m4/fit_result.json \
  --output-dir results/m4/postfit
```

The same command also writes `postfit/m0_style_figures/`, containing the five
established M0-style figure families rebuilt with M4 predictions: the 4-by-4
psychometric grid, rare-trial panels, block-transition panels, all eight fitted
parameter trajectories, and training/validation NLL histories. To rebuild only
those panels from saved artifacts, use `scripts/generate_m4_m0_style_figures.py`.

The locked test is not opened by default. Only this post-fit command supports
`--unlock-test`; fitting and identifiability never do. Test predictions are
descriptive final evaluation only and cannot select M3 or M4.

## Outputs and scientific checks

The fit exports ordinary choice metrics, all M3 learning fields, expected/MAP
run length, run-length entropy, expected/MAP hazard, switch/survival mass,
block-specific switch mass, pre/post tail probability, reachable support, and
normalization errors. The 100 pre/post marginal columns are opt-in via
`--include-full-run-length-distributions` so routine CSVs stay usable. The
fitted hazard, implied duration distribution and quantiles, session diagnostics,
and Rmax sensitivity at 65/75/100 are written separately.

M4-versus-M3 comparison requires identical held-out timelines and observations.
It reports overall/per-session ELPD and bits, Brier/calibration, early/late
context, transition-aligned, and rare/common results. M4 is selected only when
its validation gain exceeds one session-level standard error; otherwise M3 is
retained. Fixed and profiled `(ah,bh)` and `(alpha_theta,bh)` loss surfaces use
training data and full-batch L-BFGS only. Design-preserving recovery reports all
nine parameter errors and recovery of delta, block belief, expected run length,
expected hazard, and tail probability.

The registered model always fits center 23 and Rmax 65. Sensitivity evaluation
holds parameters fixed while expanding only the numerical tail representation;
it does not silently create a different fitted model. The post-transition
normalization step remains in place as requested, while explicit pre/post/
transition mass errors are exported for auditing.

Interpret M4 only after checking mass conservation, causal timing, session
resets, all eight gradients, exact `bh=0` M3 nesting beyond 65 trials, parameter
and trajectory recovery, boundary proximity of `bh`, tail occupancy, and paired
held-out gain over M3. A positive `bh` by itself is not evidence that M4
generalizes better.
