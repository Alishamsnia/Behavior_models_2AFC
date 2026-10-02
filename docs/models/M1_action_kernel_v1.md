# M1_action_kernel_v1

`M1_action_kernel_v1` keeps `M1_hmm_true_v1`'s unchanged Bayesian
sensory/lapse readout, but replaces M1's entire hazard-driven three-state
`(N,S,L)` hidden-block belief with a single scalar prior `pi_t`, updated by
a Rescorla-Wagner-style delta rule directly on the mouse's **own previous
action** -- not the true stimulus category. There is no hidden state,
hazard, transition matrix, or fixed category-rate prototype anywhere in
this model -- `pi_t` is itself the fitted belief. This is an independent
implementation living in its own package, not a modification of
`M0_sensory_v1` or `M1_hmm_true_v1` and not a runtime dependency of either.

This is the direct sibling of `M1_stimulus_kernel_v1`
(`docs/models/M1_stimulus_kernel_v1.md`), which this document assumes as a
point of comparison throughout. Every scientific and software detail is
identical between the two models with exactly one exception: the signal
that drives the `pi_t` recursion. The stimulus kernel updates from the true
observed category, gated on `feedback_valid`. The action kernel updates
from the mouse's own observed choice, gated on whether that choice itself
is valid. Neither model modifies the other, and neither is a runtime
dependency of the other.

## Definition of the action

All variables are oriented toward the long response:

```text
A_t = 1   if the mouse chose the side associated with "long"
A_t = 0   if the mouse chose the side associated with "short"
```

Because each subject's side-to-response mapping is fixed across all of
that subject's sessions, `A_t` is exactly the canonical `choice_long`
field already produced by data preparation -- no new field or oracle
signal is added, and the physical left/right side is never represented
directly. Two subjects with opposite physical mappings but the same
long/short choice history therefore receive identical `A_t` trajectories.

## Scientific definition

The learning rate is estimated in unconstrained (logit) space and mapped
to a probability by the logistic sigmoid, exactly as in the stimulus
kernel:

```text
alpha = sigmoid(a) = 1 / (1 + exp(-a)),   a in R
```

so that `0 < alpha < 1` always holds, with no separate clamping needed. The
scalar prior then follows the complete update

```text
pi_1 = 0.5
pi_{t+1} = pi_t + alpha * (A_t - pi_t)     if choice_long[t] is a valid 0/1 action
pi_{t+1} = pi_t                            otherwise (missing/invalid choice)
```

where `A_t = choice_long[t] in {0,1}` is the mouse's own observed action on
trial `t`. The prediction at trial `t` always uses `pi_t` (the value
carried in from the previous trial); the update to `pi_{t+1}` happens only
after that trial's prediction has been made, so trial `t`'s own action or
category can never leak into its own prior. `pi` resets to `0.5` at the
start of every session; dynamic state never crosses a session boundary.

**The update gate is choice validity, not `feedback_valid`.** This is the
one place a naive copy of the stimulus kernel would get the causal contract
wrong: `feedback_valid` records whether *category feedback* was available,
which is irrelevant to whether the mouse's *own action* was observed. The
action kernel's gate (`choice_long[t]` finite and in `{0,1}`) is
deliberately independent of `feedback_valid` and also independent of
whether that trial's `duration_s` is valid enough to support a likelihood
term -- a trial with a perfectly valid recorded action but a corrupted
duration still updates `pi`, even though it contributes no Bernoulli term.
`session.py` therefore computes two masks rather than one: `valid_mask`
(duration-and-choice validity, gating the likelihood, identical in spirit
to the stimulus kernel's own mask) and `action_valid_mask` (choice validity
alone, gating the history update).

### The exponential kernel, and why the naive closed form is wrong when trials are gapped

Repeatedly unrolling the recursion shows the "exponential kernel" the
update rule implies without computing one explicitly. Writing `alpha`'s
complement as `beta = 1 - alpha`, and letting `k_1 < k_2 < ... < k_m` be the
indices of every *action-valid* trial strictly before trial `t` (so `m` is
the number of updates actually applied before trial `t`, not the number of
elapsed trials):

```text
pi_t = beta^m * 0.5 + alpha * sum_{i=1}^{m} beta^(m-i) * A_{k_i}
```

Note the exponent is indexed by *how many prior valid-action updates have
occurred*, not by elapsed trial count `t-1`: a trial with an invalid/missing
choice contributes no decay factor at all, because `pi` is carried forward
completely unchanged across it (not merely "not updated with new
information" -- literally the same floating-point value). Only when every
trial before `t` has a valid action does this collapse to the simpler
`pi_t = beta^(t-1) * 0.5 + alpha * sum_{k=1}^{t-1} beta^(t-1-k) * A_k` form.
No exponential function is coded anywhere in `session.py`; it is a
mathematical consequence of the recursion above, exactly as intended, and
is verified against an independent recursive reference implementation
rather than against this closed form directly (see Verification).

The sensory readout/decision-region solver and category-dependent lapse
mixture are byte-for-byte the same functions M1 uses
(`m0_sensory_v1.model.m1_bayesian.m1_bayesian_choice_probability`,
`m0_sensory_v1.model.bayesian.lapse_mixture`) and are **still driven by the
true current category**, exactly as in every other model in this family --
only the source of the history state `pi_t` changes. `ell_short`/`ell_long`
are still selected by the true current category, never by the mouse's
action. The fitted pooled vector is `(w, s0, ell_short, ell_long, kappa,
a)` -- M0's own five coordinates plus one new learning-rate logit, the same
shape M1 itself uses for its hazard logit `ah` and the stimulus kernel uses
for its own `a`.

### Meaning of the history state

In the stimulus kernel, `pi_t` is interpreted as an estimated probability
that the next stimulus category is long -- a (mis-specified, delta-rule)
attempt at environmental inference. In the action kernel, that
interpretation no longer applies: `pi_t` is a **history-dependent tendency
to choose the long-associated action**, i.e. a choice-history/
perseveration state, not an inferred belief about the environment. It
should be described as an "action-history state" or "choice-history
tendency" in any output or figure, never as a "block belief". When this
same state is fed into the unchanged Bayesian sensory readout as the long
prior, the model asks: *does the mouse behave as though its own previous
choices create a bias for its next choice, independent of whether those
choices were correct or rewarded?* That is a valid choice-perseveration
comparison model, but it is explicitly not a model of correct
environmental inference, and it is not a win-stay/lose-switch model either
(reward/correctness never appear in the update at all).

If the mouse chooses long (`A_t=1`), `pi_{t+1} > pi_t` and the model
becomes more likely to predict long on the next trial; if it chooses short
(`A_t=0`), `pi_{t+1} < pi_t`. This happens regardless of whether the action
was correct, whether the stimulus was actually short or long, whether the
choice was rewarded, or which physical side the long response maps to for
this subject. A mouse that is presented a short stimulus but incorrectly
chooses long still moves the state toward long, because the update reads
the mouse's action, not the true category.

### Why the hidden state disappears

M1's three-state belief and hazard exist to answer "which latent block is
the mouse in, and when might it switch". The stimulus kernel instead asks
"does a simple, history-only integrator of the observed category sequence
explain the mouse's choices?" This model asks a third, distinct question:
"does a simple, history-only integrator of the mouse's *own previous
choices* -- with no notion of environmental correctness, reward, or
discrete blocks at all -- explain the mouse's choices?" `pi_t` here is not
a projection of a discrete belief through fixed category-rate prototypes
(there are no prototypes); it is the fitted scalar quantity itself.

## Numerical properties

Because `pi_{t+1}` is a convex combination of `pi_t in [0,1]` and `A_t in
{0,1}` with weight `alpha` in the open interval `(0,1)` (`sigmoid` never
reaches its endpoints for finite input), `pi_t` remains in `[0,1]` by
construction at every trial -- there is no renormalization step comparable
to the HMM belief vector's emission-normalizer division, so no
`no_probability_normalization_errors` convergence check applies (see
`registry.py`/config `convergence.note`). The evaluator additionally raises
`FloatingPointError` immediately if `pi_t` is ever non-finite, rather than
silently propagating it. Very large finite `|a|` can still saturate `alpha`
to a value that rounds to exactly `0.0` or `1.0` in floating point -- the
same accepted corner case M1's own hazard logit `ah` and the stimulus
kernel's own `a` already have; the downstream sensory readout clamps only
its own internal logit and never crashes on a saturated prior.

## Missing choices and session boundaries

An invalid or omitted choice contributes no Bernoulli term to the
likelihood **and** leaves `pi` exactly unchanged -- the opposite of the
stimulus kernel, where a feedback-valid omission can still update the
prior from the true category. `feedback_valid` is carried on the session
tensor (it is still validated by the shared `M1SessionTrials` dataclass)
but plays no role at all in this model's update: a valid action updates
`pi` even when `feedback_valid` is False for that trial (feedback
independence), and an invalid/omitted action leaves `pi` unchanged even
when `feedback_valid` is True. Dynamic state is never carried between
sessions. In the real MATLAB data loaders currently registered
(`m0_sensory_v1/data/preparation.py`), `feedback_valid` is in fact written
identically to choice validity for every trial, so this action-vs-feedback
distinction currently has no observable effect on real data -- it matters
for the model's formal causal contract, for synthetic/edge-case testing,
and for any future data source where the two could diverge (e.g. omitted
feedback delivery despite a completed choice).

## Reused session-tensor contract

Unlike `m1_known_switch_v1` (which needs an oracle `is_switch` field this
model has no use for), `M1_action_kernel_v1` needs no field beyond what
`m0_sensory_v1.model.m1_session.M1SessionTrials` already carries (duration,
choice, category, feedback validity). `session.py` therefore reuses
`M1SessionTrials` and `m1_session_trials_from_frame` unchanged rather than
defining a duplicate dataclass -- `choice_long` (the field this model's
update actually reads) is already present, and `feedback_valid` is simply
carried through unused for gating purposes. Trial order within each
session is guaranteed by the shared M0 pipeline
(`m0_sensory_v1/data/preparation.py` and `m0_sensory_v1/pipeline.py` sort
every session's canonical trial frame by `trial_index` before this adapter
ever sees it) -- the same precondition M1's own evaluator relies on; this
model does not re-validate it.

## Registry and fitting

The free parameter vector `(w,s0,ell_short,ell_long,kappa,a)` adds one new
coordinate beyond M0's five, so this model has its own
`M1ActionKernelInitializationSpec` (structurally identical to the stimulus
kernel's own spec). Its eight frozen starts pair M0's own eight registered
`(w,s0,ell_short,ell_long,kappa)` starts with the same eight learning rates
`alpha in {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9}` (stored as the logit
`a`) the stimulus kernel uses, copied verbatim so the two sibling models
share identical multi-start coverage of `alpha`
(`config/initializations/m1_action_kernel_v1_starts.json`, which records
the exact M0 source file and checksum each row's shared coordinates were
copied from). This is a deliberate eight-point joint design for multi-start
coverage, not a controlled one-parameter-at-a-time identifiability sweep.
No fitted M0/M1/stimulus-kernel winner is used as an initialization.

`registry.py` proxies genuinely shared fields (task moments, split,
optimizer, scheduler, seeds, epsilons) from the real, subject-specific M0
registry via `load_model_registry()`, exactly as `m1_stimulus_kernel_v1`
and `m1_known_switch_v1` proxy. `pipeline.py` reuses M0's own registered
`prepare_registered_fit_data` (called with the real M0 registry) for
discovery, the registered model-trial inclusion rule, and the "never open locked test" leak
guard, then builds session tensors with M1's own frame adapter
(`m1_session_trials_from_frame`) from the returned canonical timeline --
it does not reimplement discovery itself. `fit.py` is a fully independent
multi-start Adam/L-BFGS/hybrid fitter (a structural copy of the stimulus
kernel's own `fit.py` -- the objective function it calls is model-specific,
but the optimizer loop, convergence bookkeeping, and parallel dispatch are
not), with every per-start failure -- including the initial parameter
encode and the restored-best checkpoint's gradient recomputation --
contained so a single bad start cannot abort the remaining seven. It runs
sequentially in one process (a single `torch.set_num_threads` call) when
`parallel_workers<=1`, or dispatches across a `spawn`-context
`ProcessPoolExecutor` (one process per start, each pinned to
`cpu_threads_per_worker` Torch threads) when `parallel_workers>1`. The
plain CLI (`scripts/fit_m1_action_kernel.py`) exposes this as
`--parallel-workers` (default 8) / `--cpu-threads-per-worker` (default 1).
Because this model's forward pass is a genuinely sequential per-trial
Python loop (not batched, unlike M0's readout), the CLI explicitly caps
Torch's intra-op thread count in every code path, including the smoke-test
path, matching the stimulus kernel's own measured mitigation for the same
issue. `analysis.py` reuses M0's fully generic `calculate_evaluation_metrics`
for scalar metrics, plus its own `build_m1_action_kernel_trial_predictions`/
`save_m1_action_kernel_analysis_outputs` for the per-trial predictions
table the sliding-window CLI and figures consume; the exported
`model_action_updated` column reflects `action_valid_mask` (choice
validity), a distinct diagnostic from `model_evaluated`
(`valid_choice_mask`, duration-and-choice validity) even though the two
happen to coincide whenever every trial has a valid duration.

`fit.py`'s per-start optimizer loop records the full constrained parameter
vector (plus derived `alpha`) after every step in each `EpochRecord`
(`constrained_parameters_after_step`), and each `FitRunResult` also carries
`initial_constrained_parameters`; `fit_m1_action_kernel_all_initializations`
uses these to flag an identifiability ridge exactly as
`m1_stimulus_kernel_v1`/`m1_dynamic_block_length_v1` do (a converged start
within `near_equal_training_nll_tolerance` of the winner's training NLL but
at least `identifiability_parameter_distance_threshold` away in scaled
constrained-parameter space), reporting `identifiability_ridge_detected`/
`ridge_start_ids`/`additional_starts_recommended` and writing one
`starts/multistart_summary.json` per completed multi-start fit alongside
the per-start `start_<id>/fit_result.json` files.

Subject registries: `config/m1_action_kernel_v1.json` (MC08 default),
`config/m1_action_kernel_v1_mc06.json`, `config/m1_action_kernel_v1_mc07.json`.
CLI: `scripts/fit_m1_action_kernel.py` (full-split multistart fit) and
`scripts/run_m1_action_kernel_sliding_window.py` (chronological
9-train/3-validation/1-test rolling-window fits plus per-session
fine-tuning, structurally identical to
`run_m1_stimulus_kernel_sliding_window.py`, except that the action-kernel
runner intentionally supports only the `full_session_matched` workflow. It
has no `--workflow` option and no legacy block-partition or window-aggregation
path. Rolling train, validation, and test roles use all blocks in each session.
Each rolling winner initializes only that window's test session; fine-tuning
uses all blocks before the literal final two and evaluates those final two,
including the terminal block. Outputs always live under the
`full_session_matched_v1` subdirectory, so historical unsuffixed action-kernel
results are neither resumed nor overwritten. The `--resume` and CPU-budget
(`CPU_BUDGET=32`) contracts remain in force. The implementation is built on
the shared `m0_sliding_window_v1` engine through
`src/m1_action_kernel_sliding_window_v1/`; registry audit:
`scripts/audit_m1_action_kernel_registry.py`.

MC08, four concurrent windows with eight one-thread optimizer processes per
window (32 fitting Torch threads total):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -u scripts/run_m1_action_kernel_sliding_window.py \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --registry config/m1_action_kernel_v1.json \
  --optimizer lbfgs \
  --window-parallel-workers 4 \
  --rolling-workers 8 \
  --rolling-threads-per-worker 1 \
  --single-session-workers 32 \
  --single-session-threads-per-worker 1
```

The sliding CLI's own figures reuse the shared, model-agnostic
`_plot_event_grid` engine for the rare-trial/block-transition panels
(mouse-vs-model performance around task-defined events is a property of
the timeline and the fitted choice probability, not of whether the model
itself represents a block or switch, so these apply here exactly as for
every other model in the family), plus the model-agnostic psychometric
grid and a native NLL/parameter-trajectory plot
(`parameter_history.csv`/`parameter_trajectories.png`, one line per start,
winner emphasized) built from the per-step
`constrained_parameters_after_step` values `fit.py` records.

## Real-data evaluation

Evaluation is causal and one-step-ahead, exactly as fitting is. Within a
validation or test session, the prediction for each trial may condition on
the mouse's actual observed choices from earlier trials in that same
session (through the recursively-updated `pi_t`), never on the current or
future choice. During fitting and evaluation on real behavioral data, the
model always uses the mouse's actual observed action to update `pi` --
never a sampled model action, which would create a random internal
trajectory that could disagree with the observed behavioral history and
make the likelihood dependent on random samples. A model-generated action
sampled from the predicted probability is appropriate only for a
fully-generative free-running simulation (predict -> sample -> use the
sampled action to update `pi` -> repeat), which is not currently
implemented by this package (see Implementation status).

## Session boundaries

`pi` resets to `0.5` at the start of every physical session
(`initial_pi=0.5` in every registered config, validated at registry-load
time). The fixed response-side mapping needed to interpret `choice_long`
as "chose long" is read from the subject's own upstream data preparation
(`choice_long` is already computed there per-subject); it is never
re-estimated or represented directly by this model.

## Verification

Because this model's recursion mechanism (a scalar delta rule with no
discrete state) is structurally different from every hazard-driven model
in this family, it has no exact-nesting boundary to an already-verified
parent -- unlike M2->M1, M3->M2, and M4->M3, and like its sibling
`M1_stimulus_kernel_v1`. Correctness is instead established by:

- an independent, non-vectorized recursive reference implementation,
  cross-checked against the evaluator across many random action/validity
  sequences and learning rates (not a literal closed-form comparison,
  since the correct general closed form must index its decay exponent by
  the count of prior *valid* updates, not elapsed trial count -- see
  "The exponential kernel" above);
- direct hand-checkable cases (`alpha` near zero freezes `pi_t` near 0.5;
  `alpha` near one makes `pi_t` closely track the immediately preceding
  action);
- a causal no-leakage check: flipping trial `t`'s own action changes only
  trial `t+1`'s prior onward, never trial `t`'s own prediction;
- source separation: two sessions with identical stimuli but different
  mouse actions produce different `pi_t` trajectories, and conversely,
  changing a trial's true category while holding its action fixed leaves
  the `pi_t` trajectory completely unchanged (only the current trial's
  likelihood can differ);
- feedback independence: a valid action updates `pi` even when
  `feedback_valid` is False for that trial, and even when `duration_s` is
  invalid for that trial (so the trial contributes no likelihood term);
- incorrect-choice updates: choosing long on a short-category trial (or
  vice versa) still moves `pi` toward the chosen action, not the true
  category -- this is pure perseveration, not win-stay/lose-switch;
- missing/invalid choices leave `pi` exactly unchanged;
- session-boundary reset to `pi_1=0.5`, independent of the preceding
  session's final state;
- full-vector finite-difference gradient agreement over all six raw
  parameters;
- pooled-objective session-order invariance; and
- a direct cross-model consistency check against `M1_stimulus_kernel_v1`:
  when a session's `choice_long` is identical to its `category_long` for
  every trial and `feedback_valid` mirrors choice validity (as it does in
  every real dataset currently registered), the two sibling models must
  produce byte-for-byte identical `pi_t` trajectories and likelihoods for
  the same parameters, since their update rules then coincide exactly.

See `tests/test_m1_action_kernel_model.py`.

## Implementation status

The core scientific engine (`session.py`, `parameters.py`,
`initialization.py`), registry, data preparation, plain fitting CLI, and
sliding-window/single-session fitting CLI are implemented and covered by
unit tests on synthetic data (all three optimizer strategies, sequential
and process-pool dispatch, failure containment, winner selection, metrics
agreement, and matched-workflow scope/CPU-budget wiring) plus a
real-MATLAB-data structural pipeline test that is skipped automatically
when the raw data directory is not present on the current machine. No
multi-epoch multistart fit against real subject data, and no end-to-end
smoke run of either CLI against real MATLAB data, has been executed as
part of implementing this model -- unlike `M1_stimulus_kernel_v1`, whose
own implementation history includes such runs. Recovery (parameter-recovery
simulation) and a fully-generative free-running simulation mode (sample
-> update, as opposed to fit/evaluate on observed actions) are not yet
built, matching the fact that `m1_stimulus_kernel_v1`/`m1_known_switch_v1`
themselves have neither either.
