# M0_sensory_v1 through M4_hsmm_delta_v1

`M0_sensory_v1` is a pooled, stimulus-only Bayesian model for binary short/long
choices. Parts I-X are implemented with PyTorch `float64` autograd, including
full-batch Adam, L-BFGS, and Adam-to-L-BFGS fitting, frozen multi-start
selection, and predictive metrics.

The pipeline now also includes `M1_hmm_true_v1` (`M_HMM,true` in the Phase-I
guide), while retaining all M0 entry points and behavior. M1 adds a causal
`(N,S,L)` hidden-block belief, fixed true category-rate prototypes
`(0.5,0.2,0.8)`, the fitted biased hazard `h_B=sigmoid(ah)`, and the fixed
neutral tie `h_N=h_B/2`. Its complete
specification and commands are in
[`docs/models/M1_hmm_true_v1.md`](docs/models/M1_hmm_true_v1.md).

Run a bounded M1 pipeline check with:

```bash
python -u scripts/fit_m1.py \
  --raw-data-root /path/to/SCHR_MC08 \
  --output-dir results/M1_hmm_true_v1/MC08_smoke \
  --smoke-test --smoke-epochs 2
```

No full real-data M1 fit was performed as part of the implementation.

The pipeline additionally implements `M2_hmm_static_v1`
(`M_HMM,static` in the Phase-I guide). M2 leaves M0 and M1 intact and extends
M1 with one positive static contrast parameter `delta0`. Its subjective rates
are `(theta_N,theta_S,theta_L)=(0.5,sigmoid(-delta0),sigmoid(delta0))`, and the
same rates are used in both the pre-choice prior and post-choice category
emission. All eight M2 starts preserve the corresponding five M1 coordinates
exactly and add `delta0=log(4)`, so initialization begins at the exact M1
restriction. See
[`docs/models/M2_hmm_static_v1.md`](docs/models/M2_hmm_static_v1.md).

Run a bounded M2 check with any of the three optimizer strategies:

```bash
python -u scripts/fit_m2.py \
  --raw-data-root /path/to/SCHR_MC08 \
  --output-dir results/M2_hmm_static_v1/MC08_smoke \
  --optimizer adam \
  --smoke-test --smoke-epochs 2
```

Replace `adam` with `lbfgs` or `hybrid` to exercise the other methods. No full
real-data M2 fit is launched by the implementation itself.

## Current implementation status

Implemented:

- MATLAB behavioral-data loading and canonical trial preparation;
- explicit retention and masking of trials without completed choices;
- empirical task moments from the real truncated short/long distributions;
- constrained/raw parameter transformations;
- Gaussian category approximation and posterior log-odds coefficients;
- the registered positive-root-only decision threshold;
- analytical Gaussian-CDF sensory integration;
- category-dependent random-choice lapse mixing;
- independent per-session Bernoulli likelihoods;
- pooled-trial and equal-session mouse objectives;
- frozen per-mouse chronological 70/15/15 splits;
- a frozen, checksummed eight-start non-Cartesian initialization table;
- a typed, integrity-checked model registry;
- an enforceable ten-section scientific-docstring audit;
- unchanged registered Adam fitting plus opt-in L-BFGS and Adam-to-L-BFGS;
- complete eight-start records and training-only winner selection;
- ELPD, mean NLL, Brier, accuracy, calibration, psychometric, per-session, and
  prediction-distribution metrics; and
- mathematical, masking, objective, split, autograd, optimizer, selection, and
  metric tests.

The implementation has been synthetic-smoke-tested, but no full 8 x 2,000-epoch
real-data fit was launched during Parts VI-IX. The test split remains locked and
is rejected by both objective and metric entry points.

## Binary-decision policy

The canonical table keeps every task-trial row so the original trial indices
and session timeline remain auditable. Only trials with a finite duration and a
completed choice encoded as `choice_long` equal to `0` or `1` are evaluated.
Other outcomes have undefined (`NaN`) choices and are removed by boolean
indexing before threshold, CDF, lapse, likelihood, or gradient calculations.

`RewardNaive` and `PunishNaive` are completed decisions. Their original outcome
labels are preserved in saved data, while their reward/punishment status is
handled identically to `Reward` and `Punish` by the choice reconstruction.

## Part I equations

For physical duration `s`, the measurement model is

```text
X | s ~ Normal(s, w*s² + s0²).
```

The empirical category approximation uses

```text
v_k = tau_k² + w*mu_k² + s0².
```

The posterior log odds are `a2*x² + a1*x + a0`. The implementation calculates
the unique positive real root `T`, discards the negative root, and uses

```text
q(s) = P(X > T | s) = Phi((s - T) / sqrt(w*s² + s0²)).
epsilon_t = ell_long*I(category_long=1) + ell_short*I(category_long=0).
p_t = (1 - epsilon_t)*q_t + epsilon_t/2.
```

The active parameter domain removes the former artificial lapse ceiling and
uses the full natural probability range: `w > 0`, `s0 > 0`, `0 < ell_short < 1`, and
`0 < ell_long < 1`. These remaining open domains are required for positive sensory
variance and valid probabilities; all optimizers still operate on unconstrained
raw coordinates.

The exact M0 prior is always `prior_long = 0.5`.

## Real task moments

The registered moments in [`config/m0_sensory_v1.json`](config/m0_sensory_v1.json)
were calculated from all 13,720 presented category/duration pairs in the 42
MC08 sessions listed by the previous notebook. Using all presented stimuli is
appropriate here because the moments describe the task generator and do not
use mouse choices or outcomes.

The observed supports are truncated and non-overlapping:

| Category | Count | Mean (s) | Population variance (s²) | Support (s) |
|---|---:|---:|---:|---:|
| Short | 6,845 | 0.5002191381 | 0.0449210623 | 0.2–0.8 |
| Long | 6,875 | 1.9968800000 | 0.0438939020 | 1.7–2.3 |

## Prepare behavioral data

```bash
python scripts/prepare_m0_data.py \
  /path/to/session_1.mat /path/to/session_2.mat \
  --output-dir data_processed/M0_sensory_v1
```

This writes canonical trials, the legacy-compatible valid-choice view, a
session audit, a preparation manifest, and `task_moments.json`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The authoritative modified Parts I-X specification is
[`docs/models/M0_sensory_v1.md`](docs/models/M0_sensory_v1.md).

## Part II session objective

Every session is evaluated independently. One raw parameter vector is decoded
once per mouse objective and shared by all of that mouse's sessions. M0 has no
dynamic state, and no information crosses session boundaries.

The registered training loss is

```text
pooled_trials = sum(session NLL) / sum(valid choices).
```

`equal_sessions = mean(session NLL / session valid choices)` is available only
as a sensitivity analysis. The objective function builds the complete loss but
does not call `backward()` or perform an optimizer step.

## Part III frozen split

Sessions are sorted separately for each mouse by recording timestamp and stable
session ID. Integer 70/15/15 counts use largest remainder with tie priority
`test -> validation -> train`.

The frozen MC08 assignment is
[`config/splits/chronological_70_15_15_v1.csv`](config/splits/chronological_70_15_15_v1.csv):

| Role | Sessions | Chronological ranks |
|---|---:|---:|
| Train | 30 | 1–30 |
| Validation | 6 | 31–36 |
| Locked test | 6 | 37–42 |

The objective rejects test-tagged sessions, and the source selector refuses to
return test paths during development. The locked test trials have not been
evaluated.

To regenerate assignments from a session audit table:

```bash
python scripts/create_chronological_splits.py \
  data_processed/M0_sensory_v1/sessions_audit.csv \
  --output config/splits/chronological_70_15_15_v1.csv
```

## Part IV frozen initializations

The eight starts are stored as scientific `(w, s0, ell_short, ell_long, kappa)` values in
[`config/initializations/m0_sensory_v1_starts_ell100.json`](config/initializations/m0_sensory_v1_starts_ell100.json).
They are eight explicit mixed vectors—not the 4,096-point Cartesian product of
their marginal values. The registry verifies their SHA-256 checksum, open
bounds, uniqueness, finite raw transforms, and positive decision thresholds.

## Part V component registry

[`config/m0_sensory_v1.json`](config/m0_sensory_v1.json) is loaded through an
immutable typed registry. It validates model identity, real task moments,
numerical tolerances, parameter order, prior, objective, initialization table,
split artifact, random seeds, convergence rules, metric names, and software
version.

The positive-root solver now returns a structured region containing the
positive threshold, solver regime, `x > threshold` rule, discriminant, and the
explicit `negative_root_policy="discard"`. It never returns a negative-root
value.

Audit the complete registry without loading behavioral trials:

```bash
python scripts/audit_m0_registry.py
```

## Parts VI-IX: documentation, fitting, selection, and metrics

Every registered scientific function has the ten required docstring sections
and a Sphinx-compatible equation. Audit them with:

```bash
python scripts/audit_scientific_documentation.py
```

`fit_single_initialization` remains the original registered Adam implementation:
one update per complete pooled training epoch, fresh train/validation reporting,
training-NLL scheduling, and training-only checkpointing.
`fit_single_initialization_with_strategy` adds opt-in pure L-BFGS and
Adam-to-L-BFGS paths over the same raw parameters and pooled objective.
Validation never changes gradients, line search, scheduling, checkpoints, or
multi-start selection under any strategy.

`fit_all_initializations` runs the frozen eight starts and optionally writes
eight detailed run JSON files plus `multistart_summary.json`. A winner is
declared only among runs passing every convergence check, using recalculated
training mean NLL. Near-equal training solutions with distant parameters are
reported as an identifiability ridge; extra starts are recommended but never
created automatically.

```python
import torch

from m0_sensory_v1 import (
    evaluate_split_metrics,
    fit_all_initializations,
    load_model_registry,
)

registry = load_model_registry()
fit = fit_all_initializations(
    training_sessions,
    validation_sessions,
    registry,
    output_directory="results/M0_sensory_v1/MC08",
)

if fit.winner_start_id is not None:
    winner = next(run for run in fit.runs if run.start_id == fit.winner_start_id)
    validation_metrics = evaluate_split_metrics(
        validation_sessions,
        torch.tensor(winner.best_raw_parameters, dtype=torch.float64),
        registry,
        required_split_role="validation",
    )
```

Omission/non-decision rows stay in session tensors but remain `NaN`, receive no
likelihood term, and contribute no gradient or metric denominator. Optional
block-stratified psychometrics are post-prediction diagnostics only; block never
enters M0.

## Run the complete real-data model

The runner discovers the frozen session IDs beneath `--raw-data-root`, opens
only the 30 training and 6 validation MATLAB files, and leaves all 6 test files
unopened. By default it executes all eight starts for up to 2,000 epochs.

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model
python -m pip install -e .

mkdir -p results/M0_sensory_v1/MC08
python -u scripts/fit_m0.py \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --output-dir results/M0_sensory_v1/MC08 \
  --max-epochs 5000 \
  --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC08/full_fit.log
```

`--optimizer adam` is the default and preserves the original code path and
output record. It may be written explicitly:

```bash
mkdir -p results/M0_sensory_v1/MC08_adam
python -u scripts/fit_m0.py \
  --optimizer adam \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --output-dir results/M0_sensory_v1/MC08_adam \
  --max-epochs 5000 \
  --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC08_adam/full_fit.log
```

Pure L-BFGS starts each of the same eight frozen initializations directly and
uses a strong-Wolfe line search. One recorded outer step contains one
quasi-Newton iteration but may evaluate the complete training objective several
times inside the line search:

```bash
mkdir -p results/M0_sensory_v1/MC08_lbfgs
python -u scripts/fit_m0.py \
  --optimizer lbfgs \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --output-dir results/M0_sensory_v1/MC08_lbfgs \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC08_lbfgs/full_fit.log
```

The hybrid strategy runs the unchanged Adam implementation for the requested
warm-up duration, restores its best training checkpoint, and then performs the
same L-BFGS refinement:

```bash
mkdir -p results/M0_sensory_v1/MC08_hybrid
python -u scripts/fit_m0.py \
  --optimizer hybrid \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --output-dir results/M0_sensory_v1/MC08_hybrid \
  --adam-warmup-epochs 1000 \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC08_hybrid/full_fit.log
```

For hybrid runs, `--learning-rate` optionally overrides only the Adam warm-up
learning rate. Pure L-BFGS uses `--lbfgs-learning-rate`. `--max-epochs` retains
its original Adam-only meaning; use `--adam-warmup-epochs` and
`--lbfgs-max-steps` for hybrid runs. Keep each strategy in a separate output
directory so fit summaries and figures cannot overwrite one another.

A full alternative run must budget at least the registered 25-record stability
window. For pure L-BFGS this means `--lbfgs-max-steps >= 25`; for hybrid the
Adam warm-up and L-BFGS outer-step budgets are added. The CLI rejects a shorter
full-run budget before loading behavioral data. `--smoke-test` intentionally
waives this requirement for bounded pipeline checks, but a history shorter than
the stability window cannot be marked converged.

The alternative-strategy JSON records add optimizer phase, phase-local step,
per-step closure counts, before/after gradient diagnostics, total objective
evaluation counts, per-phase optimizer-step counts, phase switch NLL, and the
phase containing the best checkpoint. Hybrid stability is assessed over the
combined Adam plus L-BFGS history; pure L-BFGS uses its own history. Version 2
of the fit-result schema explicitly types these additions and rejects unknown
top-level fields. The records retain all existing raw/constrained parameters,
histories, convergence checks, boundary diagnostics, validation reporting, and
training-only winner selection.

`--max-epochs` and `--learning-rate` are optional runtime overrides. Their
registered and executed values are both written to every Adam start result. For the
5,000-epoch boundary-seeking fit, retain the registered learning rate by
omitting `--learning-rate`; increasing it can delay the strict stability check.

Each epoch prints live output in this form:

```text
start=01 epoch=0001 train_mean_nll=... validation_mean_nll=... grad_norm=... lr=... checkpoint=...
```

After a converged winner is selected, the command also writes
`trial_predictions.csv` (every original timeline row),
`performance_summary.csv`, `model_output_summary.json`, and a
`preliminary_figures/` directory containing PNG/PDF figures and their source
CSV tables. The trial table contains mouse and model decisions, probabilities,
correctness, per-trial NLL, threshold, sensory SD, split/block/category labels,
original outcomes, and fitted parameters. Omissions remain as rows with NaN
model fields. M0 has no adaptive parameters; that empty inventory is recorded
explicitly in the output summary.

By default, test panels are marked `TEST LOCKED`; “all available” therefore
means train+validation. After the model and analysis specification are frozen,
generate final all/train/validation/test outputs without refitting:

```bash
python -u scripts/generate_m0_figures.py \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --results-dir results/M0_sensory_v1/MC08 \
  --unlock-test
```

Omit `--unlock-test` to regenerate development figures while leaving test files
unopened. The psychometric grid has rows for all available data, training,
validation, and test, and columns for combined short+long blocks, short blocks,
long blocks, and neutral blocks. Psychometrics use four quantile bins separately
for short and long trials with seed-42 `N(0, 1e-6)` jitter and equal session
weight. Rare-trial panels use ±3 retained decisions; transition panels use ±10.
Mouse performance is observed accuracy and M0 performance is analytical
expected accuracy, not a randomly sampled model action.

## Adding another subject

Use one frozen registry/split and one output directory per subject. The raw root
may be the subject directory or a common parent tree because files are resolved
by exact frozen session ID:

```text
Data/Behavior/SCHR_MC08/*.mat  -> results/M0_sensory_v1/MC08/
Data/Behavior/SCHR_MC09/*.mat  -> results/M0_sensory_v1/MC09/
```

Do not reuse MC08's split CSV or task-moment registry for MC09. First create and
freeze the new subject's chronological split and real-data task moments, then
point a subject-specific registry at those artifacts:

```bash
python -u scripts/fit_m0.py \
  --registry config/m0_sensory_v1_MC09.json \
  --raw-data-root /path/to/Data/Behavior/SCHR_MC09 \
  --output-dir results/M0_sensory_v1/MC09 \
  --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC09/full_fit.log
```

Fit subjects independently unless a later hierarchical model explicitly pools
them, and never reuse one results directory for multiple subjects.

### Notebook-registered MC06 and MC07

The exact commented `DATA_PATHS` from `Model_v24_validation.ipynb` are frozen in
`config/subjects/`: 45 MC06 and 43 MC07 input sessions. Reproducing the
notebook's five-short/five-long-level rule excludes one session per subject
before moment estimation and splitting:

- MC06 `20251222_134238`: 6 short and 6 long levels; 44 sessions remain
  (31 train, 6 validation, 7 locked test).
- MC07 `20251210_105817`: 9 short and 9 long levels; 42 sessions remain
  (30 train, 6 validation, 6 locked test).

Their registries are `config/m0_sensory_v1_mc06.json` and
`config/m0_sensory_v1_mc07.json`. Subject moments use retained sessions only;
both have nonoverlapping truncated supports: short `[0.2, 0.8]` seconds and
long `[1.7, 2.3]` seconds.

MC06:

```bash
mkdir -p results/M0_sensory_v1/MC06_5000
python -u scripts/fit_m0.py \
  --registry config/m0_sensory_v1_mc06.json \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06 \
  --output-dir results/M0_sensory_v1/MC06_5000 \
  --max-epochs 5000 --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC06_5000/full_fit.log
```

MC07:

```bash
mkdir -p results/M0_sensory_v1/MC07_5000
python -u scripts/fit_m0.py \
  --registry config/m0_sensory_v1_mc07.json \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC07 \
  --output-dir results/M0_sensory_v1/MC07_5000 \
  --max-epochs 5000 --print-every 1 \
  2>&1 | tee results/M0_sensory_v1/MC07_5000/full_fit.log
```

### MC06 three-optimizer comparison workflow

Use the following six-step workflow to compare Adam, pure L-BFGS, and hybrid
Adam-to-L-BFGS on the same frozen MC06 sessions. All three methods use the same
eight initializations, pooled training objective, parameter transformations,
convergence checks, and training-only winner rule. They must use separate
output directories so one method cannot overwrite another method's results.

#### Step 1 — verify the implementation

Run the unit tests and integrity audits before starting the real fits:

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

BASE="results/M0_sensory_v1/MC06_optimizer_comparison"
mkdir -p "$BASE"

python -m pip install -e .
python -m unittest discover -s tests -v \
  2>&1 | tee "$BASE/code_tests.log"
python scripts/audit_m0_registry.py \
  2>&1 | tee "$BASE/registry_audit.log"
python scripts/audit_scientific_documentation.py \
  2>&1 | tee "$BASE/documentation_audit.log"
python scripts/audit_m0_release.py \
  2>&1 | tee "$BASE/release_audit.log"
```

#### Step 2 — fit MC06 with all three optimizers

Define the common paths and create one result directory per optimizer:

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

RAW="/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06"
REGISTRY="config/m0_sensory_v1_mc06.json"
BASE="results/M0_sensory_v1/MC06_optimizer_comparison"

mkdir -p "$BASE/adam" "$BASE/lbfgs" "$BASE/hybrid"
```

Run the unchanged Adam implementation with initial learning rate `0.01`. The
5,000-epoch override matches the extended MC06 command above and is recorded in
the results. Remove `--max-epochs 5000` to use the registered 2,000-epoch limit.
Adam uses ReduceLROnPlateau with factor `0.5`, patience 25, and minimum learning
rate `1e-6`.

```bash
python -u scripts/fit_m0.py \
  --optimizer adam \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/adam" \
  --max-epochs 5000 \
  --print-every 10 \
  2>&1 | tee "$BASE/adam/full_fit.log"
```

Run pure full-batch L-BFGS from each of the same eight initializations:

```bash
python -u scripts/fit_m0.py \
  --optimizer lbfgs \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/lbfgs" \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 10 \
  2>&1 | tee "$BASE/lbfgs/full_fit.log"
```

Run the hybrid method. It first executes 1,000 epochs of the unchanged Adam
implementation, restores Adam's best training checkpoint, and then applies the
same L-BFGS refinement:

```bash
python -u scripts/fit_m0.py \
  --optimizer hybrid \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/hybrid" \
  --adam-warmup-epochs 1000 \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 10 \
  2>&1 | tee "$BASE/hybrid/full_fit.log"
```

`--print-every 10` only reduces terminal output. Complete step histories remain
serialized. These phase budgets are intentionally method-specific and are not
equal-compute budgets; objective-evaluation counts should therefore be reported
alongside fit quality when comparing computational efficiency.

#### Step 3 — inspect saved results and preliminary figures

A method with a converged winning start writes:

```text
starts/start_01.json ... start_08.json
starts/multistart_summary.json
validation_metrics.json
trial_predictions.csv
performance_summary.csv
model_output_summary.json
preliminary_figures/
full_fit.log
```

The preliminary directory contains PNG and PDF figures plus their source CSV
tables for mouse-versus-model psychometrics (including a validation row and a
neutral-block column), rare-trial performance,
block-transition performance, all-start parameter trajectories, and training
and validation NLL histories. During fitting, test files remain unopened and
test panels are marked `TEST LOCKED`. If no start passes every convergence
check, all eight start records and the multistart summary are still saved, but
winner-dependent metrics and figures are not generated.

#### Step 4 — compare training and validation results

This creates `optimizer_fit_comparison.csv`. A negative NLL delta means lower
NLL than Adam. Convergence status and restored gradient norm must be checked
before interpreting small NLL differences.

```bash
BASE="results/M0_sensory_v1/MC06_optimizer_comparison"

python - "$BASE" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1])
records = []

for optimizer in ("adam", "lbfgs", "hybrid"):
    directory = base / optimizer
    summary = json.loads(
        (directory / "starts" / "multistart_summary.json").read_text()
    )
    winner_id = summary["winner_start_id"]
    row = {
        "optimizer": optimizer,
        "selection_status": summary["selection_status"],
        "winner_start_id": winner_id,
        "ridge_detected": summary["identifiability_ridge_detected"],
    }
    if winner_id is not None:
        winner = next(
            run for run in summary["runs"] if run["start_id"] == winner_id
        )
        w, s0, ell_short, ell_long, kappa = winner["best_constrained_parameters"]
        row.update({
            "converged": winner["converged"],
            "termination_reason": winner["termination_reason"],
            "optimizer_steps": winner["optimizer_step_count"],
            "adam_steps": winner.get("adam_step_count"),
            "lbfgs_steps": winner.get("lbfgs_step_count"),
            "training_objective_evaluations": winner.get(
                "training_objective_evaluation_count",
                2 + 2 * len(winner["history"]),
            ),
            "training_mean_nll": winner[
                "recalculated_best_training_mean_nll"
            ],
            "validation_mean_nll": winner["final_validation_mean_nll"],
            "gradient_norm": winner["gradient_norm_at_restored_best"],
            "w": w,
            "s0": s0,
            "ell_short": ell_short,
            "ell_long": ell_long,
        })
    records.append(row)

comparison = pd.DataFrame(records)
for metric in ("training_mean_nll", "validation_mean_nll"):
    adam_value = comparison.loc[
        comparison["optimizer"].eq("adam"), metric
    ].iloc[0]
    comparison[f"delta_{metric}_vs_adam"] = comparison[metric] - adam_value

output = base / "optimizer_fit_comparison.csv"
comparison.to_csv(output, index=False)
print(comparison.to_string(index=False))
print(f"\nSaved: {output}")
PY
```

Use validation, not the locked test set, to make any optimizer choice. The
primary fit comparison should include convergence, training NLL, validation
NLL, gradient norm, fitted parameters, optimizer steps, and training-objective
evaluation counts. A method with lower NLL but failed convergence is not an
eligible winner.

#### Step 5 — perform the one-time locked-test evaluation

Only unlock the MC06 test sessions after the optimizer settings and comparison
plan are frozen. Using test results to change settings or choose a new optimizer
would leak the final evaluation set. The following preserves the original
train/validation-only tables and figures before regenerating the main analysis
outputs with test sessions included:

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

RAW="/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06"
REGISTRY="config/m0_sensory_v1_mc06.json"
BASE="results/M0_sensory_v1/MC06_optimizer_comparison"

for OPTIMIZER in adam lbfgs hybrid; do
  RESULT_DIR="$BASE/$OPTIMIZER"

  cp "$RESULT_DIR/trial_predictions.csv" \
     "$RESULT_DIR/trial_predictions_train_validation.csv"
  cp "$RESULT_DIR/performance_summary.csv" \
     "$RESULT_DIR/performance_summary_train_validation.csv"
  cp -a "$RESULT_DIR/preliminary_figures" \
        "$RESULT_DIR/preliminary_figures_train_validation"

  python -u scripts/generate_m0_figures.py \
    --registry "$REGISTRY" \
    --raw-data-root "$RAW" \
    --results-dir "$RESULT_DIR" \
    --unlock-test \
    2>&1 | tee "$RESULT_DIR/test_analysis.log"
done
```

After this step, each main `trial_predictions.csv`, `performance_summary.csv`,
and `preliminary_figures/` contains train, validation, and test. The copied
files preserve the development-only version. The fitting procedure is not
rerun and test data cannot change parameters or winning starts.

#### Step 6 — calculate split and test differences

This derives consistent mean NLL, ELPD, Brier score, and threshold accuracy for
all three splits and saves `optimizer_split_metrics_comparison.csv`:

```bash
BASE="results/M0_sensory_v1/MC06_optimizer_comparison"

python - "$BASE" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(sys.argv[1])
records = []

for optimizer in ("adam", "lbfgs", "hybrid"):
    predictions = pd.read_csv(base / optimizer / "trial_predictions.csv")
    evaluated = (
        predictions["model_evaluated"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    for split in ("train", "validation", "test"):
        subset = predictions.loc[
            evaluated & predictions["split_role"].eq(split)
        ].copy()
        y = pd.to_numeric(subset["choice_long"]).to_numpy(float)
        p = pd.to_numeric(subset["model_probability_long"]).to_numpy(float)
        nll = pd.to_numeric(
            subset["per_trial_negative_log_likelihood"]
        ).to_numpy(float)
        records.append({
            "optimizer": optimizer,
            "split": split,
            "n_valid_decisions": len(subset),
            "mean_nll": np.mean(nll),
            "elpd": -np.sum(nll),
            "brier_score": np.mean((p - y) ** 2),
            "classification_accuracy": np.mean((p >= 0.5) == y),
        })

comparison = pd.DataFrame(records)
adam = comparison.loc[
    comparison["optimizer"].eq("adam")
].set_index("split")

for metric in (
    "mean_nll",
    "elpd",
    "brier_score",
    "classification_accuracy",
):
    comparison[f"delta_{metric}_vs_adam"] = comparison.apply(
        lambda row: row[metric] - adam.loc[row["split"], metric],
        axis=1,
    )

output = base / "optimizer_split_metrics_comparison.csv"
comparison.to_csv(output, index=False)
print(comparison.to_string(index=False))
print(f"\nSaved: {output}")
PY
```

For mean NLL and Brier score, lower values are better. For ELPD and accuracy,
higher values are better. Test differences are final descriptive results and
must not become a new optimization or hyperparameter signal.

### M1 MC06 three-optimizer comparison workflow

This is the M1 counterpart of the M0 workflow above. It preserves the complete
M0 pipeline and fits `M1_hmm_true_v1` with Adam, pure L-BFGS, and hybrid
Adam-to-L-BFGS. Every method automatically uses exactly the same eight
registered starts shared by all subjects. M1 start `i` keeps the exact
`(w,s0,ell_short,ell_long,kappa)` values of shared M0 start `i` and adds its registered hazard; no
subject's fitted M0 winner is used. Use separate result directories so the
methods cannot overwrite each other.

| Start | `w` | `s0` | `ell_short` | `ell_long` | `kappa` | `h` | `ah = logit(h)` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0271 | 0.555 | 0.030 | 0.030 | 1.0 | 1/30 | -3.367295829986932 |
| 2 | 0.0050 | 0.700 | 0.135 | 0.015 | 0.5 | 0.01 | -4.595119850134590 |
| 3 | 0.0100 | 1.200 | 0.0105 | 0.0195 | 2.0 | 0.02 | -3.891820298110627 |
| 4 | 0.0200 | 0.250 | 0.224 | 0.056 | 0.75 | 0.05 | -2.944438979166440 |
| 5 | 0.0400 | 0.900 | 0.002 | 0.008 | 1.3333333333333333 | 0.08 | -2.442347035369204 |
| 6 | 0.0700 | 0.150 | 0.130 | 0.070 | 0.9 | 0.12 | -1.992430164690206 |
| 7 | 0.1200 | 0.450 | 0.036 | 0.324 | 1.1111111111111112 | 0.20 | -1.386294361119891 |
| 8 | 0.2000 | 0.350 | 0.098 | 0.002 | 1.0 | 0.35 | -0.619039208406224 |

#### Step 1 — verify M0 preservation and M1 integrity

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

BASE="results/M1_hmm_true_v1/MC06_optimizer_comparison"
mkdir -p "$BASE"

python -m pip install -e .
python -m unittest discover -s tests -v \
  2>&1 | tee "$BASE/code_tests.log"
python scripts/audit_m0_registry.py \
  2>&1 | tee "$BASE/m0_registry_audit.log"
python scripts/audit_m1_registry.py \
  2>&1 | tee "$BASE/m1_registry_audit.log"
python scripts/audit_scientific_documentation.py \
  2>&1 | tee "$BASE/documentation_audit.log"
python scripts/audit_m0_release.py \
  2>&1 | tee "$BASE/release_audit.log"
```

#### Step 2 — train M1 with all three optimizers

Define the shared MC06 paths:

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

RAW="/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06"
REGISTRY="config/m1_hmm_true_v1_mc06.json"
BASE="results/M1_hmm_true_v1/MC06_optimizer_comparison"

mkdir -p "$BASE/adam" "$BASE/lbfgs" "$BASE/hybrid"
```

Adam, using the registered learning rate and a recorded 5,000-epoch override:

```bash
python -u scripts/fit_m1.py \
  --optimizer adam \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/adam" \
  --max-epochs 5000 \
  --print-every 10 \
  2>&1 | tee "$BASE/adam/full_fit.log"
```

Pure full-batch L-BFGS from each of the same eight starts:

```bash
python -u scripts/fit_m1.py \
  --optimizer lbfgs \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/lbfgs" \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 10 \
  2>&1 | tee "$BASE/lbfgs/full_fit.log"
```

Hybrid Adam-to-L-BFGS, with 1,000 Adam warm-up epochs followed by L-BFGS:

```bash
python -u scripts/fit_m1.py \
  --optimizer hybrid \
  --registry "$REGISTRY" \
  --raw-data-root "$RAW" \
  --output-dir "$BASE/hybrid" \
  --adam-warmup-epochs 1000 \
  --lbfgs-max-steps 250 \
  --lbfgs-learning-rate 1.0 \
  --lbfgs-history-size 10 \
  --lbfgs-max-evaluations 20 \
  --lbfgs-tolerance-grad 1e-9 \
  --lbfgs-tolerance-change 1e-12 \
  --print-every 10 \
  2>&1 | tee "$BASE/hybrid/full_fit.log"
```

The three phase budgets are intentionally method-specific, so include
training-objective evaluation counts when comparing computational efficiency.
`--print-every 10` affects console output only; complete optimization histories
remain serialized.

#### Step 3 — inspect fit outputs and training/validation figures

A method with a converged winner writes:

```text
starts/start_01.json ... start_08.json
starts/multistart_summary.json
validation_metrics.json
trial_predictions.csv
performance_summary.csv
model_output_summary.json
m1_figures/
m1_m0_style_figures/
full_fit.log
```

The `m1_figures/` directory contains PNG/PDF belief trajectories, calibration,
switch-aligned analyses, and the fitted hazard/implied-duration plot, together
with source CSV files and `m1_figure_manifest.json`. The separate
`m1_m0_style_figures/` directory contains every M0 figure family rebuilt with
M1 predictions: the 4×4 psychometric grid, rare-trial performance,
block-transition performance, all-start parameter trajectories (including
`ah`), and train/validation NLL histories. At this stage, only train and
validation sessions have been opened. If no start passes every convergence
check, the eight start records and multistart summary remain available, but
winner-dependent outputs are not generated.

#### Step 4 — create the optimizer fit comparison CSV

This creates `optimizer_fit_comparison.csv` with convergence, objective counts,
train/validation NLL, all five M1 coordinates, fitted hazard, implied mean block
duration, and hazard-boundary diagnostics. Negative NLL deltas are improvements
relative to Adam.

```bash
BASE="results/M1_hmm_true_v1/MC06_optimizer_comparison"

python - "$BASE" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1])
records = []

for optimizer in ("adam", "lbfgs", "hybrid"):
    directory = base / optimizer
    summary = json.loads(
        (directory / "starts" / "multistart_summary.json").read_text()
    )
    winner_id = summary["winner_start_id"]
    row = {
        "optimizer": optimizer,
        "selection_status": summary["selection_status"],
        "winner_start_id": winner_id,
        "ridge_detected": summary["identifiability_ridge_detected"],
    }
    if winner_id is not None:
        winner = next(
            run for run in summary["runs"] if run["start_id"] == winner_id
        )
        boundary = winner["boundary_proximity"]
        w, s0, ell_short, ell_long, kappa, ah = winner["best_constrained_parameters"]
        hazard = boundary["hazard_from_zero"]
        row.update({
            "converged": winner["converged"],
            "termination_reason": winner["termination_reason"],
            "optimizer_steps": winner["optimizer_step_count"],
            "adam_steps": winner.get("adam_step_count"),
            "lbfgs_steps": winner.get("lbfgs_step_count"),
            "training_objective_evaluations": winner.get(
                "training_objective_evaluation_count",
                2 + 2 * len(winner["history"]),
            ),
            "training_mean_nll": winner[
                "recalculated_best_training_mean_nll"
            ],
            "validation_mean_nll": winner["final_validation_mean_nll"],
            "gradient_norm": winner["gradient_norm_at_restored_best"],
            "w": w,
            "s0": s0,
            "ell_short": ell_short,
            "ell_long": ell_long,
            "ah": ah,
            "hazard": hazard,
            "subjective_mean_block_duration_trials": 1.0 / hazard,
            "hazard_fraction_to_nearest_boundary": boundary[
                "hazard_fraction_to_nearest_boundary"
            ],
        })
    records.append(row)

comparison = pd.DataFrame(records)
for metric in ("training_mean_nll", "validation_mean_nll"):
    adam_value = comparison.loc[
        comparison["optimizer"].eq("adam"), metric
    ].iloc[0]
    comparison[f"delta_{metric}_vs_adam"] = comparison[metric] - adam_value

output = base / "optimizer_fit_comparison.csv"
comparison.to_csv(output, index=False)
print(comparison.to_string(index=False))
print(f"\nSaved: {output}")
PY
```

Choose and freeze the optimizer using validation results, never test results. A
lower NLL from a run that failed a registered convergence check is not an
eligible result.

#### Step 5 — perform the one-time locked-test analysis and regenerate figures

Run this only after the M1 specification, optimizer settings, and comparison
plan are frozen. It preserves each train/validation-only output, then rebuilds
predictions and figures with the frozen test sessions included. It does not
refit M1 and cannot alter the fitted parameters or winning start.

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

RAW="/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06"
REGISTRY="config/m1_hmm_true_v1_mc06.json"
BASE="results/M1_hmm_true_v1/MC06_optimizer_comparison"

for OPTIMIZER in adam lbfgs hybrid; do
  RESULT_DIR="$BASE/$OPTIMIZER"

  cp "$RESULT_DIR/trial_predictions.csv" \
     "$RESULT_DIR/trial_predictions_train_validation.csv"
  cp "$RESULT_DIR/performance_summary.csv" \
     "$RESULT_DIR/performance_summary_train_validation.csv"
  cp -a "$RESULT_DIR/m1_figures" \
        "$RESULT_DIR/m1_figures_train_validation"

  python -u scripts/generate_m1_figures.py \
    --results-dir "$RESULT_DIR" \
    --registry "$REGISTRY" \
    --raw-data-root "$RAW" \
    --unlock-test \
    2>&1 | tee "$RESULT_DIR/test_analysis.log"
done
```

Afterward, each main `trial_predictions.csv`, `performance_summary.csv`, and
`m1_figures/` includes train, validation, and test. The corresponding
`*_train_validation` copies preserve the development-only analyses.

#### Step 6 — create the split and test comparison CSV

This calculates the same split-level metrics used for M0 and writes
`optimizer_split_metrics_comparison.csv`:

```bash
BASE="results/M1_hmm_true_v1/MC06_optimizer_comparison"

python - "$BASE" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(sys.argv[1])
records = []

for optimizer in ("adam", "lbfgs", "hybrid"):
    predictions = pd.read_csv(base / optimizer / "trial_predictions.csv")
    evaluated = (
        predictions["model_evaluated"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    for split in ("train", "validation", "test"):
        subset = predictions.loc[
            evaluated & predictions["split_role"].eq(split)
        ].copy()
        y = pd.to_numeric(subset["choice_long"]).to_numpy(float)
        p = pd.to_numeric(subset["model_probability_long"]).to_numpy(float)
        nll = pd.to_numeric(
            subset["per_trial_negative_log_likelihood"]
        ).to_numpy(float)
        records.append({
            "optimizer": optimizer,
            "split": split,
            "n_valid_decisions": len(subset),
            "mean_nll": np.mean(nll),
            "elpd": -np.sum(nll),
            "brier_score": np.mean((p - y) ** 2),
            "classification_accuracy": np.mean((p >= 0.5) == y),
        })

comparison = pd.DataFrame(records)
adam = comparison.loc[
    comparison["optimizer"].eq("adam")
].set_index("split")

for metric in (
    "mean_nll",
    "elpd",
    "brier_score",
    "classification_accuracy",
):
    comparison[f"delta_{metric}_vs_adam"] = comparison.apply(
        lambda row: row[metric] - adam.loc[row["split"], metric],
        axis=1,
    )

output = base / "optimizer_split_metrics_comparison.csv"
comparison.to_csv(output, index=False)
print(comparison.to_string(index=False))
print(f"\nSaved: {output}")
PY
```

For mean NLL and Brier score, lower is better; for ELPD and classification
accuracy, higher is better. The test comparison is final reporting only and
must not be used to choose an optimizer or revise settings.

#### Generate the complete M0-style figure set for an existing M1 result

If test predictions have already been generated, this command reads the saved
M1 table and fit history directly. It does not reload raw data, unlock anything,
or refit the model. It retains `m1_figures/` and writes the additional figures
to `m1_m0_style_figures/`:

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model

RESULT_DIR="results/M1_hmm_true_v1/MC06_optimizer_comparison/lbfgs"

python -u scripts/generate_m1_m0_style_figures.py \
  --m1-predictions "$RESULT_DIR/trial_predictions.csv" \
  --fit-summary "$RESULT_DIR/starts/multistart_summary.json" \
  --output-dir "$RESULT_DIR/m1_m0_style_figures" \
  2>&1 | tee "$RESULT_DIR/m1_m0_style_figures.log"
```

The output directory contains PNG/PDF versions and source CSV tables:

```text
psychometrics_mouse_vs_model.{png,pdf}
psychometric_summary.csv
rare_trial_performance.{png,pdf}
rare_trial_performance_summary.csv
block_transition_performance.{png,pdf}
block_transition_performance_summary.csv
parameter_trajectories.{png,pdf}
parameter_history.csv
training_validation_nll.{png,pdf}
nll_history.csv
m1_m0_style_figure_manifest.json
```

For a bounded end-to-end check using the same real train/validation files:

```bash
python -u scripts/fit_m0.py \
  --raw-data-root /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC08 \
  --output-dir results/M0_sensory_v1/MC08_smoke \
  --smoke-test --smoke-epochs 2 --print-every 1
```

Synthetic recovery can be rerun with:

```bash
python scripts/run_m0_recovery.py \
  --output results/M0_sensory_v1/recovery.json
```

The implementation freeze is recorded in
`config/releases/m0_sensory_v1_release.json`. Verify all frozen artifact, source,
test, and locked-test declarations with:

```bash
python scripts/audit_m0_release.py
```

## M2 static-prototype workflow

M2 uses the same frozen chronological splits, task moments, choice-only pooled
objective, causal HMM timing, convergence checks, and three optimizers as M1.
Subject registries are:

- `config/m2_hmm_static_v1_mc06.json`
- `config/m2_hmm_static_v1_mc07.json`
- `config/m2_hmm_static_v1.json` for MC08

Audit all three without opening behavioral files:

```bash
python scripts/audit_m2_registry.py
```

Run all eight registered starts with Adam, pure L-BFGS, or hybrid
Adam-to-L-BFGS. Keep each method in a separate directory:

```bash
RAW=/path/to/SCHR_MC06
REG=config/m2_hmm_static_v1_mc06.json
BASE=results/M2_hmm_static_v1/MC06_optimizer_comparison

python -u scripts/fit_m2.py --raw-data-root "$RAW" --registry "$REG" \
  --optimizer adam --max-epochs 5000 --output-dir "$BASE/adam"

python -u scripts/fit_m2.py --raw-data-root "$RAW" --registry "$REG" \
  --optimizer lbfgs --lbfgs-max-steps 250 --output-dir "$BASE/lbfgs"

python -u scripts/fit_m2.py --raw-data-root "$RAW" --registry "$REG" \
  --optimizer hybrid --adam-warmup-epochs 1000 --lbfgs-max-steps 250 \
  --output-dir "$BASE/hybrid"
```

Each completed fit writes trial predictions, per-session performance,
validation metrics, complete start histories, and three figure directories:

- `m2_figures/`: belief/prior trajectories, calibration, switch-aligned
  behavior, fitted static prototypes/hazard, fitted `delta0` against the
  `log(4)` restriction, plus optional paired M2–M1 and M2–M0 figures. Supplying
  both reference tables also produces M1/M2 prior and block-belief overlays,
  consolidated M0/M1/M2 validation NLL and Brier panels, and M1/M2 validation
  ELPD;
- `m2_m0_style_figures/`: the psychometric grid, rare-trial panels,
  block-transition panels, seven-parameter trajectories, and train/validation NLL;
- `m2_identifiability/`: the automatic 15-by-15 post-fit `(ah, delta0)` loss
  surface, its source CSV, and a manifest declaring that no fit was changed.

Regenerate those outputs without refitting, optionally adding aligned M1 and M0
comparisons:

```bash
python -u scripts/generate_m2_figures.py \
  --results-dir "$BASE/lbfgs" \
  --raw-data-root "$RAW" \
  --registry "$REG" \
  --m1-predictions /path/to/M1/trial_predictions.csv \
  --m0-predictions /path/to/M0/trial_predictions.csv
```

Add `--identifiability-grid-points 15` in completed-fit mode to create
`m2_identifiability/m2_ah_delta0_loss_surface.{png,pdf}` and its source CSV.
This is strictly a post-fit diagnostic: it reevaluates the existing causal,
choice-only mean NLL over an `(ah, delta0)` grid while holding `(w, s0, ell_short, ell_long, kappa)`
at the saved winner. It does not refit, replace the checkpoint, select a start,
or add a new term to the loss. It is a fixed-parameter slice rather than a
profile likelihood.

The locked test remains closed unless `--unlock-test` is explicitly supplied.
Optimizer choice and model development must use train/validation outputs only.
For a standalone paired M2-versus-M1 report and session figure:

```bash
python scripts/compare_m1_m2.py \
  --m1-predictions /path/to/M1/trial_predictions.csv \
  --m2-predictions /path/to/M2/trial_predictions.csv \
  --split validation \
  --output /path/to/m2_vs_m1_validation.json
```

Run real-design recovery at the single hazard `h=1/23`, the inverse of the
observed median block duration of 23 trials. The three contrast conditions are
near-zero (`delta0=0.1`), the exact M1 restriction (`delta0=log(4)`), and strong
(`delta0=2.2`). Each condition preserves real session lengths, durations,
omissions, feedback masks, and true-block schedules, then refits all six
parameters from all eight registered starts:

```bash
python scripts/run_m2_recovery.py \
  --raw-data-root "$RAW" \
  --registry config/m2_hmm_static_v1_mc06.json \
  --output-dir results/M2_hmm_static_v1/MC06_recovery \
  --epochs 500
```

## Model 4: duration-aware capped-tail HSMM

M4 extends the unchanged M3 choice and learning model with one nonnegative
duration-hazard slope `bh`. Its registered hazard is centered at the observed
median block length, 23 trials, and uses a 65-state capped tail (`r65` means
`R>=65`). It inherits M3's eight starts exactly and appends one `bh` value;
neither a fitted M3 winner nor a MAP objective is used. M3 and M4 learning-rate
starts span the requested positive range through 0.1.

Fit all eight starts in parallel on CPU:

```bash
PYTHONPATH=src python scripts/fit_m4.py \
  --raw-data-root "$RAW" --registry config/m4_hsmm_delta_v1_mc06.json \
  --output-dir results/M4_hsmm_delta_v1/MC06 --optimizer lbfgs \
  --parallel-workers 8
```

An M4 start is winner-eligible only after the existing convergence/projection
checks and the new mean training tail-occupancy gate (`<=0.01`) pass. Routine
trial CSVs contain compact duration summaries; add
`--include-full-run-length-distributions` only when all 100 pre/post marginal
columns are needed.

Recovery and M3/M4 model recovery are executable through
`scripts/run_m4_recovery.py` and `scripts/run_m3_m4_model_recovery.py`. Complete
psychometric, calibration, rare-trial, block-transition, hazard/duration,
truncation-sensitivity, and `(ah,bh)`/`(alpha_theta,bh)` identifiability outputs
come from `scripts/generate_m4_figures.py`. The command also recreates the five
M0-style families under `m0_style_figures/`, including all eight M4 parameter
trajectories. That post-fit command is the only M4 entry point with
`--unlock-test`; test data never enter fitting or selection. Use
`scripts/generate_m4_m0_style_figures.py` when only the M0-style panels need to
be rebuilt from saved prediction and fit files.

The full scientific contract and command examples are in
`docs/models/M4_hsmm_delta_v1.md`.

## Sliding-window and single-session fine-tuning experiment (M0-M4)

`scripts/run_m{0,1,2,3,4}_sliding_window.py` support two isolated chronological
9 training / 3 validation / 1 rolling-test workflows, both sliding by one
session at a time. The default `--workflow legacy_partitioned` preserves the
original behavior exactly: the terminal block is ignored, the preceding two
complete blocks are held out, and all sessions are fine-tuned from
window-aggregated initializations.

The additive `--workflow full_session_matched` fits and evaluates every rolling
window on complete sessions, without removing any blocks. After every rolling
window finishes, only its test session is fine-tuned: window 1 directly
initializes chronological session 13, window 2 initializes session 14, and so
on. For that second stage, all blocks before the literal final two blocks form
the fine-tuning prefix and the final two blocks (including the terminal block)
are test-only. No window averaging is used, and held-out choices never update
parameters. Results are isolated below a `full_session_matched_v1` subdirectory
so they cannot overwrite legacy outputs.

M4 single-session fits must pass the same registered acceptance gates used for
rolling-window winner selection: convergence, acceptable projection activity,
and acceptable run-length tail occupancy. An ineligible session no longer
terminates the parallel run. Its fit, trial predictions, split metrics, hazard
curve, and provenance are retained under a manifest with explicit exclusion
reasons; duration-dependent optional diagnostics are skipped, and pooled
single-session figures include eligible sessions only. Automatic alternate-start
retry is intentionally disabled so the matched-window initialization contract
is not silently changed.

Each script accepts `--window-parallel-workers` to fit multiple rolling
windows at once instead of one at a time. Windows are fully independent, so a
finished window immediately frees its slot for the next queued window rather
than waiting on slower siblings, and `--resume` behaves identically regardless
of how many windows run concurrently. `--window-parallel-workers` defaults to
`1` (unchanged sequential behavior) unless set explicitly.

M0/M1 size their per-window start-parallelism with `--parallel-workers`
(default 8) and `--cpu-threads-per-worker` (default 1); total CPU usage is
roughly `window-parallel-workers x parallel-workers x cpu-threads-per-worker`.
M2/M3/M4 instead use `--rolling-workers`/`--rolling-threads-per-worker` and
enforce a hard 32-CPU budget: `window-parallel-workers x rolling-workers x
rolling-threads-per-worker` must not exceed 32, or the CLI rejects the run
before loading any data.

Run each model separately for MC06 (one at a time; each command is sized to
use the full 32-CPU budget by itself):

```bash
cd /home/ihsan/Desktop/data/Georgia_Tech/2AFC/Codes/Model_behavior/Bayesian_model
RAW="/home/ihsan/Desktop/data/Georgia_Tech/2AFC/Data/Behavior/SCHR_MC06"

# ---------- M0 ----------
mkdir -p results_sliding_window/M0_sensory_v1/MC06
python -u scripts/run_m0_sliding_window.py \
  --raw-data-root "$RAW" \
  --registry config/m0_sensory_v1_mc06.json \
  --window-parallel-workers 4 \
  --parallel-workers 8 \
  --cpu-threads-per-worker 1 \
  --single-session-workers 4 \
  --optimizer lbfgs \
  --resume \
  2>&1 | tee results_sliding_window/M0_sensory_v1/MC06/full_run.log

# ---------- M1 ----------
mkdir -p results_sliding_window/M1_hmm_true_v1/MC06
python -u scripts/run_m1_sliding_window.py \
  --raw-data-root "$RAW" \
  --registry config/m1_hmm_true_v1_mc06.json \
  --window-parallel-workers 4 \
  --parallel-workers 8 \
  --cpu-threads-per-worker 1 \
  --single-session-workers 4 \
  --optimizer lbfgs \
  --resume \
  2>&1 | tee results_sliding_window/M1_hmm_true_v1/MC06/full_run.log

# ---------- M2 ----------
mkdir -p results_sliding_window/M2_hmm_static_v1/MC06
python -u scripts/run_m2_sliding_window.py \
  --raw-data-root "$RAW" \
  --registry config/m2_hmm_static_v1_mc06.json \
  --window-parallel-workers 4 \
  --rolling-workers 8 \
  --rolling-threads-per-worker 1 \
  --single-session-workers 32 \
  --single-session-threads-per-worker 1 \
  --optimizer lbfgs \
  --resume \
  2>&1 | tee results_sliding_window/M2_hmm_static_v1/MC06/full_run.log

# ---------- M3 ----------
mkdir -p results_sliding_window/M3_hmm_delta_v1/MC06
python -u scripts/run_m3_sliding_window.py \
  --raw-data-root "$RAW" \
  --registry config/m3_hmm_delta_v1_mc06.json \
  --window-parallel-workers 4 \
  --rolling-workers 8 \
  --rolling-threads-per-worker 1 \
  --single-session-workers 32 \
  --single-session-threads-per-worker 1 \
  --optimizer lbfgs \
  --resume \
  2>&1 | tee results_sliding_window/M3_hmm_delta_v1/MC06/full_run.log

# ---------- M4 ----------
mkdir -p results_sliding_window/M4_hsmm_delta_v1/MC06
python -u scripts/run_m4_sliding_window.py \
  --raw-data-root "$RAW" \
  --registry config/m4_hsmm_delta_v1_mc06.json \
  --window-parallel-workers 4 \
  --rolling-workers 8 \
  --rolling-threads-per-worker 1 \
  --single-session-workers 32 \
  --single-session-threads-per-worker 1 \
  --optimizer lbfgs \
  --resume \
  2>&1 | tee results_sliding_window/M4_hsmm_delta_v1/MC06/full_run.log
```

M2/M3/M4's `--rolling-threads-per-worker` must be lowered from its default of
4 to 1 when `--window-parallel-workers` is raised above 1, or the CLI's
32-CPU budget guard rejects the run (`4 windows x 8 workers x 4 threads = 128
> 32`). M0/M1 have no such hard guard; keep the product of all three
multipliers within your machine's thread budget manually.
