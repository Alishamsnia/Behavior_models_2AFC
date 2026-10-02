# M0_sensory_v1 — Parts I-X specification

M0 now uses the full quadratic decision region shared with M1–M4. This
supersedes the historical positive-root-only restriction. Its fixed prior,
five parameters, category-specific lapse mixture, omission mask, objective,
session independence, initialization table, and fitting/window logic are unchanged.
The updated registry checksum distinguishes new fits from historical results;
old fitted parameters and predictions must not be treated as full-region fits.

## Observations and valid decisions

For session `d` and task trial `t`, `s[d,t]` is physical duration in seconds.
The response is

```text
y[d,t] = 1  long-associated choice
y[d,t] = 0  short-associated choice
```

An undefined response is stored as `NaN`. Such a row remains in the canonical
table but is indexed out before any model or gradient calculation. RewardNaive
and PunishNaive are treated as completed rewarded/punished choices while their
original outcome names remain stored.

## Fixed M0 prior

M0 is stimulus-only:

```text
pi[d,t] = 0.5
logit(pi[d,t]) = 0
```

Block, trial history, reward, session identity, and trial position do not enter
a prediction. The current physical short/long stimulus category selects the
corresponding lapse rate; it never changes the fixed M0 prior.

## Trial-level measurement model

```text
X[d,t] | s[d,t] ~ Normal(s[d,t], w*s[d,t]² + s0²)
sigma[d,t] = sqrt(w*s[d,t]² + s0²)
```

Here `w > 0` is the duration-dependent variance coefficient and `s0 > 0` is
baseline sensory standard deviation in seconds.

## Real truncated category distributions

The fixed moments are empirical population moments of all presented real
stimuli, including presented stimuli on trials without a completed choice:

```text
mu_k   = (1/N_k) sum_i s_i
tau_k² = (1/N_k) sum_i (s_i - mu_k)²
```

Outcomes and choices are not used. The current verified MC08 values are:

| Quantity | Short (`k=0`) | Long (`k=1`) |
|---|---:|---:|
| `N_k` | 6,845 | 6,875 |
| `mu_k` (s) | 0.5002191380569759 | 1.99688 |
| `tau_k²` (s²) | 0.04492106227800066 | 0.04389390196363635 |
| observed support (s) | [0.2, 0.8] | [1.7, 2.3] |

These real distributions have strictly positive, truncated, non-overlapping
support. Variance uses `ddof=0` because the required quantity is the task
distribution variance.

## Gaussian category approximation

```text
f_k(x) ~= Normal(x; mu_k, v_k)
v_k = tau_k² + w*mu_k² + s0²
```

The moments are fixed; only `w` and `s0` are fitted.

## Posterior log odds

```text
Lambda(x) = log(f_1(x)/f_0(x)) + logit(pi)
          = a2*x² + a1*x + a0

a2 = 1/(2*v0) - 1/(2*v1)
a1 = mu1/v1 - mu0/v0
a0 = mu0²/(2*v0) - mu1²/(2*v1)
     + 0.5*log(v0/v1) + logit(pi) - log(kappa)
```

`kappa > 0` is an internal decision-cost ratio. `kappa=1` is neutral,
`kappa>1` favors short choices, and `kappa<1` favors long choices.

## Full quadratic decision rule

The shared solver in `model/m1_bayesian.py` integrates the complete region
`a2*X² + a1*X + a0 > 0`, using the same tolerances and Gaussian CDF expressions
as the context models. M0 still supplies the exact prior `0.5`.

For two distinct roots `lower < upper`:

- `a2 > 0`: choose long outside the roots, including the negative measurement tail.
- `a2 < 0`: choose long between the roots.

Linear limits use the appropriate upper or lower tail. Constant-sign,
no-real-root, and double-root cases give the corresponding all-long or
all-short probability (constant ties choose short). Valid regions are not
rejected because a boundary is negative or because curvature changes sign.

## Analytical choice probability

For a valid trial:

```text
X[d,t] | s[d,t] ~ Normal(s[d,t], w*s[d,t]² + s0²)
q[d,t] = P(a2*X[d,t]² + a1*X[d,t] + a0 > 0 | s[d,t]).
```

The calculation is analytic. A single scalar coefficient set and vector CDFs
serve all valid trials in an M0 session. Legacy positive-root helper functions
remain available for historical checks but do not enter M0 prediction or fitting.

## Lapse mixture

```text
epsilon[d,t] = ell_long*I(category_long[d,t]=1)
             + ell_short*I(category_long[d,t]=0)
p[d,t] = (1 - epsilon[d,t])*q[d,t] + epsilon[d,t]/2
```

`ell_short` and `ell_long` are the lapse probabilities on short- and
long-category trials. Conditional on a lapse, the response is a discrete
uniform draw over the short and long alternatives. The registered domain uses
the complete natural probability range: `0 < ell_short < 1`,
`0 < ell_long < 1`.

## Raw Torch parameterization

Adam will eventually optimize unconstrained coordinates:

```text
w   = softplus(raw_w) + epsilon
s0  = softplus(raw_s0) + epsilon
ell_short = sigmoid(raw_ell_short)
ell_long   = sigmoid(raw_ell_long)
kappa      = exp(raw_log_kappa)
epsilon = 1e-10
```

Part I exposes both forward decoding and constrained-initialization encoding.
All scientific operations use Torch tensors and preserve autograd.

## Numerical policy

- Registered precision: `torch.float64`.
- Quadratic tolerance: `1e-12`.
- Discriminant tolerance: `1e-12`.
- Do not clamp diagnostic `q` or `p`.
- The Bernoulli likelihood clamps only the probability passed to logs.
- Missing/non-decision rows are skipped before the scientific forward graph.

## Part II — independent session likelihood

For session `d`, define the valid completed-decision set

```text
V_d = {t : s[d,t] is finite and positive, y[d,t] is 0 or 1}.
```

RewardNaive and PunishNaive trials belong to this set after the same binary
choice reconstruction as Reward and Punish. Other outcomes retain undefined
choices and do not enter the forward or gradient graph.

The session likelihood is

```text
J_d = -sum_{t in V_d} [y[d,t]*log(p[d,t])
                       + (1-y[d,t])*log(1-p[d,t])]
N_d = |V_d|.
```

Only the probability supplied to logarithms is clamped to
`[1e-12, 1-1e-12]`; diagnostic probabilities remain unclamped.

Every session is evaluated independently. No dynamic or trial-history state
from one session may enter another session. The pooled mouse parameters are
stable across sessions, while M0's session-local state is empty and discarded
after every evaluation.

### Registered pooled objective

One raw `(w,s0,ell_short,ell_long,log_kappa)` vector is decoded once. All training sessions are then
evaluated before the Part VII fitter performs one `backward()` call:

```text
J_pooled = sum_d J_d / sum_d N_d.
```

This `pooled_trials` objective is registered as primary. The supported
sensitivity objective is

```text
J_equal = (1/D) * sum_d (J_d/N_d).
```

The objective does not call autograd or an optimizer internally. A zero-valid
session contributes zero to the pooled numerator/count but makes the
equal-session objective undefined. A dataset with no valid choices raises.

Test-tagged sessions are rejected unconditionally.

## Part III — frozen chronological splits

Splits are produced independently for each mouse using session metadata only:

1. Sort by recording timestamp, then stable session ID.
2. Calculate quotas `0.70*D`, `0.15*D`, and `0.15*D`.
3. Floor every quota.
4. Allocate remaining sessions by largest fractional remainder.
5. Resolve exact ties using `test -> validation -> train`.
6. Assign earliest sessions to train, then validation, and latest sessions to
   locked test.

The split version is `chronological_70_15_15_v1`. For MC08's 42 sessions the
allocation is 30 train, 6 validation, and 6 locked test sessions. The frozen
assignments are stored in
`config/splits/chronological_70_15_15_v1.csv`.

The source-path selector blocks locked-test paths before MATLAB trial data are
loaded. Test sessions remain locked until all planned models are implemented
and frozen.

## Part IV — frozen eight spread initializations

The table contains exactly eight scientific-coordinate vectors. It is not an `8^5`
Cartesian grid.

| Start | `w` | `s0` (s) | `ell_short` | `ell_long` | `kappa` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0271 | 0.555 | 0.030 | 0.030 | 1.0 |
| 2 | 0.0050 | 0.700 | 0.135 | 0.015 | 0.5 |
| 3 | 0.0100 | 1.200 | 0.0105 | 0.0195 | 2.0 |
| 4 | 0.0200 | 0.250 | 0.224 | 0.056 | 0.75 |
| 5 | 0.0400 | 0.900 | 0.002 | 0.008 | 1.3333333333333333 |
| 6 | 0.0700 | 0.150 | 0.130 | 0.070 | 0.9 |
| 7 | 0.1200 | 0.450 | 0.036 | 0.324 | 1.1111111111111112 |
| 8 | 0.2000 | 0.350 | 0.098 | 0.002 | 1.0 |

The active artifact is
`config/initializations/m0_sensory_v1_starts_ell100.json`, version
`m0_directional_lapse_kappa_eight_starts_v3`. It retains the eight constrained start values while
encoding them under the full lapse-probability domain. Registry
loading verifies its checksum and requires:

```text
w > 0
s0 > 0
0 < ell_short < 1
0 < ell_long < 1
kappa > 0
```

All eight vectors must be unique, every parameter must span multiple values,
every constrained-to-raw-to-constrained round trip must agree in `float64`, and
every start must yield finite probabilities in [0,1] with the real task moments.

## Part V — software component registry

The typed registry loaded from `config/m0_sensory_v1.json` is the single entry
point for the complete component contract:

| Component | Responsibility | Implementation |
|---|---|---|
| Model registry | Identity, versions, moments, tolerances, starts, split, seeds, convergence, metrics | `registry.py` |
| Parameter decoder | Raw `(w,s0,ell_short,ell_long,log_kappa)` to scientific values | `model/parameters.py` |
| Fixed-neutral prior | Return exact `pi=0.5` without history inputs | `model/m0.py` |
| Category likelihood | Calculate `v0`, `v1`, and `(a2,a1,a0)` | `model/bayesian.py` |
| Decision solver | Shared full quadratic region | `model/m1_bayesian.py` |
| Bayesian probability | Analytical `Phi((s-T)/sigma(s))` | `model/bayesian.py` |
| Lapse mixture | Unclamped `(1-epsilon_t)q + epsilon_t/2` | `model/bayesian.py` |
| Session evaluator | Independent session NLL and valid count | `model/session.py` |
| Pooled objective | Decode once, sum session NLLs, normalize once | `model/session.py` |

### Structured full region

Predictions expose `decision_lower_boundary_s`, `decision_upper_boundary_s`,
and `decision_regime`. Trial CSVs prefix these names with `model_`.
The historical `positive_threshold_s` / `model_positive_threshold_s` field is
retained as an alias for the upper boundary, as in the context models. It does
not imply an upper-tail action rule or a positive finite threshold. Linear
regions may have an infinite endpoint; constant regions have NaN boundaries.
Omitted rows retain NaN numerical diagnostics and `not_evaluated_invalid_choice`.

## Part VI — scientific documentation contract

Every public scientific function is registered in `documentation.py` and must
contain Purpose, Mathematical definition, Parameter interpretation, Input units
and shapes, Returned quantities and shapes, Constraints, Numerical stability,
Causal restrictions, Important edge cases, and Related tests. Every docstring
also includes a Sphinx `.. math::` block. The executable audit is:

```bash
python scripts/audit_scientific_documentation.py
```

The active decision-function documentation describes the shared full-region
calculation. Positive-root helper documentation describes legacy utilities only.

## Part VII — registered Adam and opt-in alternative optimization

The frozen settings are Adam in `float64`, learning rate `1e-2`, at most 2,000
epochs, no clipping, no regularization, and ReduceLROnPlateau with factor 0.5,
patience 25, and minimum learning rate `1e-6`.

The command-line runner may explicitly override maximum epochs and initial
learning rate using `--max-epochs` and `--learning-rate`. Both registered and
executed values, plus flags identifying each override, are serialized in every
start result. This keeps an extended convergence run auditable without silently
editing the frozen registry.

One epoch builds the pooled objective across every training session, calls
`backward()` once, and takes one Adam step. It then freshly recalculates complete
training and validation objectives. The scheduler and checkpoint both use fresh
training mean NLL. Validation is evaluated under `torch.no_grad()` and cannot
affect updates, scheduling, or checkpoint selection.

Frozen convergence values are gradient norm `<=1e-5`, meaningful NLL change
`1e-8`, no-improvement patience 100 epochs, and a 25-epoch stability window. A
run may terminate at maximum epochs or after a minimum-learning-rate plateau,
but it is accepted as converged only when all registered finite-value, gradient,
stability, transform, and probability checks pass. Maximum epochs alone is not
evidence of convergence.

The registered Adam path remains the default and is not altered by the optional
optimizer strategies. Pure L-BFGS starts directly from each frozen raw
initialization. The hybrid path runs the unchanged Adam procedure for an
explicit warm-up duration, restores its best training checkpoint, and then
starts L-BFGS. L-BFGS uses one recorded quasi-Newton iteration per outer step,
history size 10, learning rate 1.0, at most 20 closure evaluations per step,
and strong-Wolfe line search by default. It has no learning-rate scheduler.

Every closure and line-search decision uses the complete pooled training
objective only. Validation remains a no-gradient post-step report. Alternative
runs retain the same checkpoint improvement tolerance, final convergence
checks, training-only winner selection, parameter transforms, and test lock.
Their records additionally identify the optimizer phase and phase-local step,
count closure/objective evaluations, record the first-closure and accepted-step
gradient diagnostics, and report per-phase update counts. These before-step
diagnostics are captured from L-BFGS's first closure instead of performing a
redundant forward/backward pass. An invalid or nonfinite L-BFGS candidate
restores the raw vector from before that outer step and records the failure
rather than substituting a penalty loss.

Pure L-BFGS stability is evaluated over its L-BFGS records. Hybrid stability is
evaluated over the combined chronological Adam plus L-BFGS history, so a short
polishing tail can complete a stability window established across both phases.
A full alternative run is rejected when its planned combined history cannot
fill the registered 25-record window. Bounded smoke tests may explicitly waive
that budget check, but an actually short history remains ineligible for
convergence. Alternative and Adam results are documented by the strict
`m0_fit_result_v2` schema; optimizer-specific fields are typed explicitly.

## Part VIII — multi-start records and selection

All eight frozen starts run independently. Each result serializes initial and
best raw/constrained parameters, fresh and recalculated training NLL, restored
validation NLL, gradient norm, epochs, learning rate, optimizer-step count,
termination/convergence details, combined and separate train/validation
histories, boundary proximity, nonfinite count, and deterministic seed.

Only converged runs are eligible. The winner minimizes freshly recalculated
training mean NLL after restoring the best checkpoint. Validation never enters
the argmin. Converged solutions within `1e-6` training NLL but at scaled
parameter distance at least `0.25` are reported as an identifiability ridge.
The software recommends additional starts in that case but does not generate
them automatically.

## Part IX — evaluation metrics

For valid completed choices only:

```text
ELPD = sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]
MeanNLL = -ELPD/N
Brier = mean((y_i-p_i)^2)
Accuracy = mean(1[(p_i>0.5) = y_i])
```

The metric bundle also contains ten-bin calibration, observed and predicted
psychometric curves grouped by duration, per-session metrics, probability
quantiles/distribution, and observed versus predicted long-choice rates.
Optional block-stratified psychometrics use block labels only after prediction.

Training and validation are the only roles permitted by the fitting-time Part
IX metric API, so an accidental test evaluation there raises unconditionally.
After all model and analysis choices are frozen, the separate post-fit analysis
command requires an explicit `--unlock-test` flag and records that unlock in its
output manifest.

## Part X — mandatory verification and recovery

The test suite covers the reference category variances, log-odds coefficients,
all raw-parameter transforms, full decision regions, numerical quadrature,
pooled-objective finite differences for all five raw parameters, neutral-prior
restrictions, session-order invariance, omission masking, optimizer-step count,
live NLL callbacks, leak-safe raw-file discovery, and synthetic recovery.

Full-region regression tests cover both curvature signs, linear and constant
limits, batched probabilities and gradients, masked rows, and agreement with
the context-model readout at prior 0.5. Legacy positive-root rejection tests
remain scoped to that historical helper, not the active M0 model.

The historical positive-root four-case recovery report is
[`M0_sensory_v1_recovery.md`](M0_sensory_v1_recovery.md). It demonstrates low
prediction-probability RMSE across low/high sensory noise and lapse conditions.
Parameter ridges can exist even when the psychometric prediction is accurately recovered;
the recovery report now evaluates both category-specific lapse rates directly.

## Preliminary post-fit diagnostics and durable outputs

The full runner saves one row per original trial to `trial_predictions.csv`.
Non-decisions remain present with NaN model/NLL fields. Valid rows contain the
mouse decision, deterministic M0 decision, `P(long)`, expected probability of a
correct decision, per-trial NLL, sensory SD, full-region boundaries/regime, the compatibility threshold alias, and the five
winning parameters. Original `RewardNaive` and `PunishNaive` labels remain even
though they are modeled as reward and punishment.

M0's fitted parameter inventory is entirely static: `(w, s0, ell_short, ell_long, kappa)`. It has
no adaptive parameters or latent state. `model_output_summary.json` records
this explicitly, while every start's fit JSON now stores raw and constrained
parameter values after every optimizer step.

The preliminary figures include notebook-compatible four-bin psychometrics
with all/train/validation/test rows and combined/short/long/neutral block columns,
performance around rare trials and directional block transitions, constrained
parameter trajectories, and fresh training/validation NLL trajectories. Block
and trial category are post-prediction diagnostics only. The test split remains
a visibly locked row unless `scripts/generate_m0_figures.py --unlock-test` is
invoked explicitly after analysis decisions are frozen.
