# M1_hmm_true_v1

`M1_hmm_true_v1` implements the Phase-I model denoted `M_HMM,true`: a causal
three-state hidden-block model with fixed category-rate prototypes and one
pooled biased-block hazard. The existing `M0_sensory_v1` implementation remains
available and unchanged at its public entry points.

## Scientific definition

States are ordered `(N, S, L)`, with fixed long-category probabilities
`theta=(0.5, 0.2, 0.8)`. Every session and every objective closure starts from
the pre-choice belief `xi_pre=(1,0,0)`. For fitted biased hazard
`h_B=sigmoid(ah)`, derived neutral hazard `h_N=h_B/2`, and fixed
first-biased-block design probability `omega=0.5`,

```text
T = [[1-h_N, h_N*(1-omega), h_N*omega],
     [0,     1-h_B,         h_B],
     [0,     h_B,           1-h_B]].
```

The factor `1/2` is fixed by the two-period neutral warm-up design and is not
an additional fitted parameter. Thus the geometric mean duration is `2/h_B`
for neutral and `1/h_B` for biased blocks. Neutral cannot be re-entered.

On trial `t`, the pre-choice category prior is `pi_t=xi_pre @ theta`. The
Bayesian sensory readout receives only the current duration and this prior.
After prediction, feedback-valid category `C_t` updates the belief with
`theta**C_t * (1-theta)**(1-C_t)`, then the hazard transition advances the task
timeline. Current category, reward, true block, and true run length never enter
the current choice prediction. Category-emission likelihood is not part of the
outer behavioral objective.

The fitted pooled vector is `(w, s0, ell_short, ell_long, kappa, ah)`. M1 uses the full quadratic
log-odds decision region required by the Phase-I guide, including linear,
two-root, double-root, no-root, and constant regimes. M0 retains its previously
registered positive-root-only solver.

## Missing choices and session boundaries

An invalid or omitted choice contributes no Bernoulli term. If
`feedback_valid=1`, the authoritative category can still update the hidden
state; otherwise belief is retained. In either case the hazard transition is
applied unless the trial is final. Dynamic state is never carried between
sessions, validation calls, or optimizer closures.

## Registry and fitting

The base MC08 registry is
[`config/m1_hmm_true_v1.json`](../../config/m1_hmm_true_v1.json). Subject
overlays provide the frozen MC06 and MC07 task moments and chronological
splits. Every subject uses the same eight-start table. M1 start `i` preserves
the exact `(w,s0,ell_short,ell_long,kappa)` values of shared M0 start `i` and adds one hazard from
the ordered list `{1/30,0.01,0.02,0.05,0.08,0.12,0.20,0.35}`. No fitted M0
winner or subject-specific fitted parameter enters M1 initialization. These are
initializations, not fixed parameters; all six coordinates are jointly fit.

A bounded end-to-end run is:

```bash
python -u scripts/fit_m1.py \
  --raw-data-root /path/to/SCHR_MC08 \
  --output-dir results/M1_hmm_true_v1/MC08_smoke \
  --smoke-test --smoke-epochs 2
```

For MC06 or MC07, pass `--registry config/m1_hmm_true_v1_mc06.json` or the
corresponding MC07 file. Full runs support the same Adam, L-BFGS, and hybrid
optimizer controls as M0. The M1 trial export includes pre-choice and
post-feedback state probabilities, trialwise prior, both decision boundaries,
decision regime, emission normalizer, feedback mask, NLL, fitted hazard,
implied subjective mean block duration, and probability assigned to the
diagnostic true block. M1 fit records additionally report hazard proximity to
zero and one.

To compare already-fitted development predictions without refitting:

```bash
python scripts/compare_m0_m1.py \
  --m0-predictions results/M0/trial_predictions.csv \
  --m1-predictions results/M1/trial_predictions.csv \
  --split validation --output results/M1_vs_M0_validation.json
```

This reports paired ELPD and `delta_bits_per_trial_m1_minus_m0`. Test
comparison remains explicitly locked unless `--unlock-test` is supplied.

Synthetic acceptance workflows and their figures are available without real
data:

```bash
python scripts/run_m1_recovery.py \
  --output-dir results/M1_hmm_true_v1/recovery

python scripts/run_m0_m1_model_recovery.py \
  --output-dir results/M1_hmm_true_v1/model_recovery
```

Generate belief trajectories, switch-aligned summaries, calibration, fitted
hazard, and optional paired early/late and rare/common M0 comparisons with:

```bash
python scripts/generate_m1_figures.py \
  --m1-predictions results/M1/trial_predictions.csv \
  --m0-predictions results/M0/trial_predictions.csv \
  --output-dir results/M1/figures
```

The complete M0 behavioral/optimization figure set can also be generated from
an existing M1 result without refitting or reopening raw data:

```bash
python scripts/generate_m1_m0_style_figures.py \
  --m1-predictions results/M1/trial_predictions.csv \
  --fit-summary results/M1/starts/multistart_summary.json \
  --output-dir results/M1/m1_m0_style_figures
```

This produces M1 versions of the mouse-versus-model psychometric grid,
rare-trial and block-transition performance, all-start parameter trajectories,
and training/validation NLL histories. The parameter figure includes all six
M1 coordinates `(w,s0,ell_short,ell_long,kappa,ah)`.

## Verification status

No full real-data M1 fit was run during implementation. The test suite covers
all analytic decision regimes, state conservation, direct `h=0` and `h=1`
kernels, causal no-leakage, feedback-valid omissions, no-feedback omissions,
session resets, session-order invariance, full-vector autograd, seeded
design-preserving simulations, subject registries, optimizer smoke execution,
and state-rich trial exports. Hazard recovery, M0/M1 model recovery, validation
comparison, and diagnostic plotting are implemented as runnable workflows.
They have not been used to launch a full M1 real-data fit or score locked test
data.
