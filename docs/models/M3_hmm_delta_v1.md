# M3_hmm_delta_v1: constant-step subjective-prototype learner

`M3_hmm_delta_v1` is an additive implementation of Phase-I Model 3
(`M_HMM,delta`). It imports the verified M0/M1/M2 numerical primitives and does
not modify their source files or behavior.

## Pooled parameters

One eight-vector is fitted per mouse across all training sessions:

`(w, s0, ell_short, ell_long, kappa, ah, delta0, alpha_theta)`.

The first five coordinates are the unchanged sensory/readout parameters. The
fitted biased hazard is `h_B = sigmoid(ah)` and the neutral hazard is the fixed
tie `h_N=h_B/2`. `delta0 > 0` is the session-start prototype
contrast and `alpha_theta = sigmoid(raw_alpha)` lies in `(0,1)`. Every session
resets both hidden-state belief and learning state:

`xi_pre[1] = (1,0,0)` and `delta[1] = delta0`.

## Exact causal trial update

At trial `t`, the subjective prototypes are computed once from the pre-update
contrast:

`theta_t = (0.5, sigmoid(-delta_t), sigmoid(delta_t))` in `(N,S,L)` order.

The current choice prior is `pi_t = xi_pre_t dot theta_t`. The unchanged M1/M2
full quadratic log-odds readout predicts the choice from duration and `pi_t`.
The current category is not an input to this prediction.

If valid category feedback is present, the same pre-update `theta_t` is used in
the categorical emission update:

`gamma_t = normalize(xi_pre_t * theta_t^C_t * (1-theta_t)^(1-C_t))`.

The responsibility-weighted local signal and registered update are:

`u_t = gamma_L,t (C_t-theta_L,t) - gamma_S,t (C_t-theta_S,t)`

`delta_proposal = delta_t + alpha_theta u_t`

`delta_(t+1) = max(1e-6, delta_proposal)`.

Afterward, the block posterior transitions with the unchanged constant-hazard
matrix. With missing feedback, belief and delta are retained before that
transition, and all learning diagnostics for the trial are marked missing. The
outer fitting objective remains choice likelihood only.

The evaluator deliberately retains the autograd graph through `gamma_t`. The
guide's “fixed responsibilities” wording describes the local update
interpretation; it does not authorize detaching responsibilities from the outer
likelihood gradient.

## Exact nesting and projection audit

`M3DecodedParameters.with_explicit_alpha(0)` bypasses the open sigmoid decoder.
At this exact boundary, every delta remains `delta0` and M3 reproduces M2
trial-by-trial. This equality covers choice probabilities, priors, beliefs,
emission normalizers, and NLL—not only aggregate fit.

The positivity projection is part of the registered model. Every evaluation
records the proposal, projected delta, activation flag, one-based activation
locations, and a structured event record containing category, responsibilities,
prototypes, signal, proposal, and projected value. Trial exports also include a
within-session cumulative activation count. A fitted run is eligible for
selection only when the training activation fraction is at most `0.001`.

The free `delta0` transform is inherited unchanged from M2: `softplus(raw) +
epsilon`. The trial evaluator separately enforces the registered scientific
floor `delta_min=1e-6`. An optimizer proposal below that floor terminates that
start as a numerical/transform failure; this hard-wall behavior is intentional
in the current implementation. Identifiability grids begin one transform
epsilon above `delta_min`, preventing round-off from creating an invalid cell.

## Data and fitting pipeline

The M3 data entry point preserves complete timelines and separately carries
choice-valid and feedback-valid masks. Frozen test files are never opened by
fitting. True block labels are retained only for retrospective diagnostics.

M3 exposes the same three optimizer strategies and command-line controls as M2:
registered full-batch Adam, pure full-batch L-BFGS, and a full-batch Adam-to-L-BFGS
hybrid. Adam is the default for pipeline consistency; the Phase-I guide's
recommended primary analysis remains pure L-BFGS. L-BFGS is implemented with
`torch.optim.LBFGS`, PyTorch autograd, and strong-Wolfe line search. There are
exactly eight registered starts. Rows 1–8 preserve the corresponding M2 table
row exactly and add only `alpha_theta =
{0.020,0.001,0.005,0.010,0.030,0.050,0.075,0.100}` in row order. No fitted M2
winner is used as an M3 initialization. Each closure resets every session and
sums the complete training objective before one optimizer step. The final loss
and gradient are freshly reevaluated. Validation never controls optimization,
checkpointing, convergence, or winner selection.

The eight registered starts execute concurrently in up to eight isolated CPU
processes, with one Torch thread assigned to each worker. Every worker calls the
unchanged single-start fitter with the same start-specific seed. Completed runs
are sorted back into start-ID order before the unchanged convergence filter,
ridge check, and minimum-training-NLL winner selection. Setting
`parallel_workers=1` provides the exact sequential reference path.
Workers send initial and per-step fit records through a process-safe progress
queue to the parent, so full parallel runs retain live start-tagged output.
`--print-every N` controls its frequency. CLI options `--parallel-workers` and
`--cpu-threads-per-worker` override the defaults of eight and one respectively.

Run a registry-only audit with:

```bash
PYTHONPATH=src python scripts/audit_m3_registry.py
```

Run a short end-to-end pipeline check with:

```bash
PYTHONPATH=src python scripts/fit_m3.py \
  --registry config/m3_hmm_delta_v1.json \
  --raw-data-root /path/to/raw/mat/files \
  --output-dir results/m3_smoke \
  --optimizer lbfgs --smoke-test --smoke-epochs 2
```

For a full parallel run, omit `--smoke-test` and optionally specify
`--parallel-workers 8 --cpu-threads-per-worker 1 --print-every 1`.

Select a strategy with `--optimizer adam`, `--optimizer lbfgs`, or
`--optimizer hybrid`. Omit `--smoke-test` for the full eight-start run. Outputs include all
trialwise delta/prototype/responsibility fields, projection locations, predictive
metrics, the three pairwise dynamic identifiability surfaces, and a joint
`(ah, delta0, alpha_theta)` objective volume. A separate profiled-surface API
reoptimizes the other six coordinates with full-batch PyTorch L-BFGS.

The observed median block length is 23 trials. Its inverse `1/23` is registered
only as a recovery-grid and diagnostic reference. The fitted hazard remains free,
and the exact inherited M2 initialization table is not rewritten (so its first
row's `1/30` initializer remains an initializer, not an empirical block-length
claim).

Recovery covers all 60 combinations of alpha
`{0,.02,.05,.10,.20}`, delta0 `{.20,log(4),2.20}`, and hazard
`{.02,1/23,.08,.20}`. The exact alpha-zero generating condition is constructed
explicitly because a finite sigmoid raw coordinate cannot equal zero. Recovery
reports all eight parameter errors plus delta, prototype, update-signal, and
projection-trajectory agreement.

`scripts/generate_m3_figures.py` creates the M3 learning, belief,
responsibility, projection, switch-aligned, rare-trial, early/late, calibration,
cross-model validation, M2/M3 prior, and block-conditioned psychometric figures.
`scripts/generate_m3_m0_style_figures.py` additionally recreates every established
M0 figure family for M3.

## Scientific acceptance checks

Before interpreting learning, verify exact alpha-zero M2 nesting, all four
one-hot update cases, causal category timing, finite-difference agreement for all
seven gradients, session resets and session-order invariance, negligible
projection, full parameter and trajectory recovery on the observed design, and
paired held-out M3-versus-M2 likelihood gains. Validation comparison checks the
timeline and observation masks exactly and reports total ELPD, bits/trial,
session-level uncertainty, calibration/Brier, common versus rare trials,
early versus late trials, switch-aligned effects, and block-conditioned
psychometrics. A positive fitted learning rate alone is not evidence for learning.
