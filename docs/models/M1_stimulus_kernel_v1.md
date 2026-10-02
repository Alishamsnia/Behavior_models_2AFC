# M1_stimulus_kernel_v1

`M1_stimulus_kernel_v1` keeps `M1_hmm_true_v1`'s unchanged Bayesian
sensory/lapse readout, but replaces M1's entire hazard-driven three-state
`(N,S,L)` hidden-block belief with a single scalar prior `pi_t`, updated by
a Rescorla-Wagner-style delta rule directly on the observed category. There
is no hidden state, hazard, transition matrix, or fixed category-rate
prototype anywhere in this model -- `pi_t` is itself the fitted belief.
This is an independent implementation living in its own package, not a
modification of `M0_sensory_v1` or `M1_hmm_true_v1` and not a runtime
dependency of either.

## Scientific definition

The learning rate is estimated in unconstrained (logit) space and mapped
to a probability by the logistic sigmoid:

```text
alpha = sigmoid(a) = 1 / (1 + exp(-a)),   a in R
```

so that `0 < alpha < 1` always holds, with no separate clamping needed. The
scalar prior then follows the complete update

```text
pi_1 = 0.5
pi_{t+1} = pi_t + alpha * (C_t - pi_t)     if feedback_valid[t] is True
pi_{t+1} = pi_t                            otherwise
```

where `C_t in {0,1}` is the observed category (`category_long`) on trial
`t`. The prediction at trial `t` always uses `pi_t` (the value carried in
from the previous trial); the update to `pi_{t+1}` happens only after that
trial's prediction has been made, so the current trial's category or
choice can never leak into its own prior. `pi` resets to `0.5` at the start
of every session; dynamic state never crosses a session boundary.

Repeatedly unrolling the recursion shows the "exponential kernel" the
update rule implies without computing one explicitly:

```text
pi_t = (1-alpha)^(t-1) * 0.5 + alpha * sum_{k=1}^{t-1} (1-alpha)^(t-1-k) * C_k
```

(the sum running only over feedback-valid trials before `t`) -- an
exponentially-weighted moving average of past categories, with `alpha`
controlling how quickly the weight on older trials decays. No exponential
function is coded anywhere in `session.py`; it is a mathematical consequence
of the recursion above, exactly as intended.

The sensory readout/decision-region solver and category-dependent lapse
mixture are byte-for-byte the same functions M1 uses
(`m0_sensory_v1.model.m1_bayesian.m1_bayesian_choice_probability`,
`m0_sensory_v1.model.bayesian.lapse_mixture`); only the source of the prior
`pi_t` changes. The fitted pooled vector is `(w, s0, ell_short, ell_long,
kappa, a)` -- M0's own five coordinates plus one new learning-rate logit,
the same shape M1 itself uses for its hazard logit `ah`.

### Why the hidden state disappears

M1's three-state belief and hazard exist to answer "which latent block is
the mouse in, and when might it switch". This model asks a different
question: "does a simple, history-only integrator of the observed category
sequence -- with no notion of discrete blocks or switch timing at all --
explain the mouse's choices as well as, or differently from, M1's causal
block-inference process?" `pi_t` here is not a projection of a discrete
belief through fixed category-rate prototypes (there are no prototypes);
it is the fitted scalar quantity itself.

## Numerical properties

Because `pi_{t+1}` is a convex combination of `pi_t in [0,1]` and `C_t in
{0,1}` with weight `alpha` in the open interval `(0,1)` (`sigmoid` never
reaches its endpoints for finite input), `pi_t` remains in `[0,1]` by
construction at every trial -- there is no renormalization step comparable
to the HMM belief vector's emission-normalizer division, so no
`no_probability_normalization_errors` convergence check applies (see
`registry.py`/config `convergence.note`). The evaluator additionally raises
`FloatingPointError` immediately if `pi_t` is ever non-finite, rather than
silently propagating it. Very large finite `|a|` can still saturate `alpha`
to a value that rounds to exactly `0.0` or `1.0` in floating point -- the
same accepted corner case M1's own hazard logit `ah` already has; the
downstream sensory readout clamps only its own internal logit and never
crashes on a saturated prior.

## Missing choices and session boundaries

Identical policy to M1: an invalid or omitted choice contributes no
Bernoulli term but the feedback-gated prior update still applies (a
feedback-valid omission still updates `pi`). Feedback observed while
`feedback_valid` is False leaves `pi` unchanged. Dynamic state is never
carried between sessions.

## Reused session-tensor contract

Unlike `m1_known_switch_v1` (which needs an oracle `is_switch` field this
model has no use for), `M1_stimulus_kernel_v1` needs no field beyond what
`m0_sensory_v1.model.m1_session.M1SessionTrials` already carries (duration,
choice, category, feedback validity). `session.py` therefore reuses
`M1SessionTrials` and `m1_session_trials_from_frame` unchanged rather than
defining a duplicate dataclass. Trial order within each session is
guaranteed by the shared M0 pipeline (`m0_sensory_v1/data/preparation.py`
and `m0_sensory_v1/pipeline.py` sort every session's canonical trial frame
by `trial_index` before this adapter ever sees it) -- the same precondition
M1's own evaluator relies on; this model does not re-validate it.

## Registry and fitting

The free parameter vector `(w,s0,ell_short,ell_long,kappa,a)` adds one new
coordinate beyond M0's five, so this model has its own
`M1StimulusKernelInitializationSpec` (mirroring
`m0_sensory_v1.model.m1_initialization.M1InitializationSpec`'s `ah` field,
renamed `a`). Its eight frozen starts pair M0's own eight registered
`(w,s0,ell_short,ell_long,kappa)` starts with eight learning rates `alpha
in {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9}` (stored as the logit `a`),
spanning a slow-integrating to a near-instantaneous, noisy-tracking kernel
-- one-to-one, identical in spirit to how `m1_hmm_true_v1`'s own table
pairs those same eight rows with eight varied hazards
(`config/initializations/m1_stimulus_kernel_v1_starts.json`, which records
the exact M0 source file and checksum each row's shared coordinates were
copied from). This is a deliberate eight-point joint design for multi-start
coverage, not a controlled one-parameter-at-a-time identifiability sweep.
No fitted M0/M1 winner is used as an initialization.

`registry.py` proxies genuinely shared fields (task moments, split,
optimizer, scheduler, seeds, epsilons) from the real, subject-specific M0
registry via `load_model_registry()`, exactly as `m1_known_switch_v1`
proxies. `pipeline.py` reuses M0's own registered
`prepare_registered_fit_data` (called with the real M0 registry) for
discovery, the legacy inclusion rule, and the "never open locked test" leak
guard, then builds session tensors with M1's own frame adapter
(`m1_session_trials_from_frame`) from the returned canonical timeline --
it does not reimplement discovery itself. `fit.py` is a fully independent
multi-start Adam/L-BFGS/hybrid fitter, with every per-start failure --
including the initial parameter encode and the restored-best checkpoint's
gradient recomputation -- contained so a single bad start cannot abort the
remaining seven. It runs sequentially in one process (a single
`torch.set_num_threads` call) when `parallel_workers<=1`, or dispatches
across a `spawn`-context `ProcessPoolExecutor` (one process per start, each
pinned to `cpu_threads_per_worker` Torch threads) when `parallel_workers>1`
-- the same `parallel_workers`/`cpu_threads_per_worker`/`output_directory`
contract `m1_known_switch_v1/fit.py` added for its own sliding-window CLI,
added here for the same reason. The plain CLI
(`scripts/fit_m1_stimulus_kernel.py`) exposes this as `--parallel-workers`
(default 8) / `--cpu-threads-per-worker` (default 1), matching the majority
family convention used by `fit_m0.py`/`fit_m1.py`/`fit_m2.py`/`fit_m3.py`/
`fit_m4.py`/the dynamic-block-length fit scripts -- unlike
`fit_m1_known_switch.py`/`fit_m2_known_switch.py`, which expose only a
single-process `--cpu-threads` flag despite their own `fit.py` retaining
the same parallel capability. Because this model's forward pass is a
genuinely sequential per-trial Python loop (not batched, unlike M0's
readout), the CLI explicitly caps Torch's intra-op thread count in every
code path, including the smoke-test path -- leaving it at Torch's default
(one thread per visible core) was measured to slow a 3-epoch smoke run down
by an order of magnitude on a 40-core machine, since thread-pool overhead
dominates trial-sized scalar tensor ops with no parallel work to divide.
`analysis.py` reuses M0's fully generic `calculate_evaluation_metrics` for
scalar metrics, plus its own `build_m1_stimulus_kernel_trial_predictions`/
`save_m1_stimulus_kernel_analysis_outputs` (a native implementation, not a
reuse of `known_switch_analysis`, whose equivalent helpers hard-require
oracle-switch-specific prediction fields this model does not have) for the
per-trial predictions table the sliding-window CLI and figures consume.

`fit.py`'s per-start optimizer loop records the full constrained parameter
vector (plus derived `alpha`) after every step in each `EpochRecord`
(`constrained_parameters_after_step`), and each `FitRunResult` also carries
`initial_constrained_parameters`; `fit_m1_stimulus_kernel_all_initializations`
uses these to flag an identifiability ridge exactly as
`m1_dynamic_block_length_v1` does (a converged start within
`near_equal_training_nll_tolerance` of the winner's training NLL but at
least `identifiability_parameter_distance_threshold` away in scaled
constrained-parameter space), reporting `identifiability_ridge_detected`/
`ridge_start_ids`/`additional_starts_recommended` and writing one
`starts/multistart_summary.json` per completed multi-start fit alongside
the per-start `start_<id>/fit_result.json` files.

Subject registries: `config/m1_stimulus_kernel_v1.json` (MC08 default),
`config/m1_stimulus_kernel_v1_mc06.json`, `config/m1_stimulus_kernel_v1_mc07.json`.
CLI: `scripts/fit_m1_stimulus_kernel.py` (full-split multistart fit) and
`scripts/run_m1_stimulus_kernel_sliding_window.py` (chronological
9-train/3-validation/1-test rolling-window fits plus per-session
fine-tuning, adapted from `run_m1_known_switch_sliding_window.py`'s
generic orchestration -- same `--workflow`/`--resume`/CPU-budget
(`CPU_BUDGET=32`) contract, built on the shared `m0_sliding_window_v1`
engine via the new `src/m1_stimulus_kernel_sliding_window_v1/` package);
registry audit: `scripts/audit_m1_stimulus_kernel_registry.py`. The sliding
package's own `aggregate_m1_stimulus_kernel_session_initializations`
(used only by the `legacy_partitioned` workflow) averages the five shared
sensory/lapse coordinates exactly as `m1_known_switch_sliding_window_v1`
does (arithmetic mean, geometric mean for `kappa`) and averages the
learning-rate logit `a` in probability space (mean `alpha=sigmoid(a)`,
then `logit` back) -- the same convention
`m1_dynamic_block_length_sliding_window_v1` uses for its hazard logit `ah`,
rather than a naive arithmetic mean of the raw logit. The sliding CLI's own
figures reuse the shared, model-agnostic `_plot_event_grid` engine for the
rare-trial/block-transition panels (mouse-vs-model performance around
task-defined events is a property of the timeline and the fitted choice
probability, not of whether the model itself represents a block or switch,
so these apply here exactly as for every other model in the family), plus
the model-agnostic psychometric grid and a native NLL/parameter-trajectory
plot (`parameter_history.csv`/`parameter_trajectories.png`, one line per
start, winner emphasized) built from the per-step
`constrained_parameters_after_step` values `fit.py` now records.

## Verification

Because this model's recursion mechanism (a scalar delta rule with no
discrete state) is structurally different from every hazard-driven model
in this family, it has no exact-nesting boundary to an already-verified
parent -- unlike M2->M1, M3->M2, and M4->M3. Correctness is instead
established by:

- an independently-derived closed-form exponential-moving-average
  reference, cross-checked against the recursive implementation across
  many synthetic category sequences and learning rates;
- direct hand-checkable cases (`alpha` near zero freezes `pi_t` near 0.5;
  `alpha` near one makes `pi_t` closely track the previous category;
  feedback-invalid trials leave `pi` exactly unchanged);
- a causal no-leakage check (the current trial's category/choice never
  affects its own `prior_long`);
- full-vector finite-difference gradient agreement over all six raw
  parameters;
- pooled-objective session-order invariance; and
- session-boundary reset to `pi_1=0.5`.

See `tests/test_m1_stimulus_kernel_model.py`.

## Implementation status

The core scientific engine, registry, data preparation, plain fitting CLI,
and sliding-window/single-session fitting CLI are implemented and tested
(unit tests on synthetic data for all three optimizer strategies, sequential
and process-pool dispatch, failure containment, winner selection, metrics
agreement, and the probability-space learning-rate aggregation; a real-data
structural pipeline test; real end-to-end smoke runs of both CLIs against
actual MC06 MATLAB data, including a real one-window rolling fit exercising
the `ProcessPoolExecutor` dispatch). No full real multi-epoch multistart fit
across all three subjects has been executed as part of implementing this
model -- only bounded smoke runs. Recovery (parameter-recovery simulation)
is not yet built, matching the fact that `m1_known_switch_v1` itself has
none either.
