# M0 data preparation

This is the first implemented layer of `M0_sensory_v1`. It contains no model
probability, Torch parameter, objective, optimizer, split, or test-set scoring
code.

## Source-compatible extraction rules

The loader intentionally preserves the behavioral rules from
`Model_v24_validation.ipynb`:

1. Load the top-level MATLAB `SessionData` structure.
2. Select an outcome using the same ordered Bpod-state priority.
3. Default missing contingency metadata to normal contingency (`1`).
4. For reversed contingency (`2`), swap trial and block codes `1 <-> 2`.
5. Read duration from
   `ProcessedSessionData[i].trial_isi.PostMeanISI` in seconds.
6. Treat `Reward`, `Punish`, `RewardNaive`, and `PunishNaive` as valid binary
   choice trials.
7. Reconstruct the canonical long-associated choice from category and outcome:
   rewarded trials chose the category-associated action; punished trials chose
   the opposite action.
8. Optionally require exactly five unique short and five unique long duration
   levels after rounding to three decimals.

## Deliberate improvements over the legacy dataframe

The legacy notebook removed invalid trials before fitting. This project keeps
two data views:

- `canonical_trials`: one row per raw task trial, including omissions;
- `m0_model_trials`: the valid-choice likelihood view.

The complete table is retained for stable indexing and auditing. Non-decision
rows have undefined choices and are boolean-indexed out before model and
gradient calculations. They must not update M0 or any later dynamic state.

Missing trial and block codes remain missing. They are not silently converted
to long trials or long blocks. File failures are raised rather than printed and
skipped.

`feedback_valid` currently follows the legacy valid-choice rule. The task audit
must revise this field if authoritative protocol information shows that the
mouse receives category feedback after an omission or another invalid choice.

## Canonical encodings

- `category_long`: `0=short`, `1=long`.
- `choice_long`: `0=short-associated action`, `1=long-associated action`.
- `true_block`: `N`, `S`, or `L`; diagnostics only.
- `trial_index`, `block_index`, and `true_run_length`: one-based.
- `source_trial_index`: zero-based raw MATLAB index.
- `duration_s`: seconds.

M0 evaluates canonical rows only where `duration_s` is finite and positive and
`choice_long` is binary. True block, category, reward, trial position, and
session identity must never enter its prediction.

## Canonical trial schema

| Column | Definition |
|---|---|
| `mouse_id` | Mouse parsed from the first filename token. |
| `session_id` | Full filename stem; stable tie-breaker for chronological ordering. |
| `session_index` | One-based chronological rank within mouse. |
| `session_date` | Timestamp parsed from the filename. |
| `trial_index` | One-based task-trial number within session. |
| `source_trial_index` | Zero-based index into the raw MATLAB trial arrays. |
| `block_index` | One-based ID of a contiguous run of the same canonical block. |
| `true_block` | Canonical task block: `N`, `S`, or `L`; diagnostics only. |
| `true_run_length` | One-based position within the contiguous block run. |
| `category_long` | Presented category: `0=short`, `1=long`. |
| `duration_s` | `PostMeanISI`, in seconds. |
| `choice_long` | Reconstructed response: `0=short-associated`, `1=long-associated`. |
| `rewarded` | `1` for reward states and `0` for punish states; missing otherwise. |
| `omission` | `1` only when the selected outcome is `DidNotChoose`. |
| `choice_valid` | `1` for a recognized reward/punish outcome with known category. |
| `feedback_valid` | Currently identical to `choice_valid`; see audit note above. |
| `outcome` | First finite Bpod state under the frozen priority rule. |
| `outcome_time_s` | First timestamp of the selected outcome state, in seconds. |
| `warm_up` | `1` for raw indices below `NumNaiveWarmup`. |
| `jitter_flag` | Canonical binary copy of `JitterFlag`. |
| `opto_tag` | Raw per-trial `OptoType` value; diagnostics only. |
| `task_version` | Version token parsed from the filename, such as `V_1`. |
| `contingency_index` | Session contingency; `1=normal`, `2=reversed`. |
| `source_file` | Absolute raw MATLAB path. |
| `source_sha256` | SHA-256 checksum of the raw MATLAB file. |

The model view adds `model_trial_index`, a one-based count over retained
choice-likelihood rows within each session. It does not replace either raw
trial index.

`RewardNaive` and `PunishNaive` remain visible in `outcome`. They are completed
decisions, so response reconstruction handles them as reward and punishment,
respectively.

## Session audit schema

Each input file produces one audit row containing its stable identifiers,
source checksum, raw/valid/model-eligible counts, omission and feedback counts,
separate Reward/Punish/RewardNaive/PunishNaive counts, observed rounded
short/long duration levels, the legacy-bin-filter decision, and any exclusion
reason. Files that fail to load or violate the required schema raise an
exception and do not silently disappear.

## Verified compatibility baseline

Against the 42 files explicitly listed in `Model_v24_validation.ipynb`, this
implementation produces 13,720 canonical trials and 13,057 model trials. All
42 sessions pass the exact five-short/five-long ISI-level rule. A row-by-row
comparison of source trial index, duration, category, block, choice, reward,
and outcome found no differences across the 13,057 retained trials.

## Command-line use

From the repository root:

```bash
python scripts/prepare_m0_data.py \
  /path/to/session_1.mat /path/to/session_2.mat \
  --output-dir data_processed/M0_sensory_v1
```

The command writes the canonical table, M0 likelihood table, session audit,
and preparation manifest. It does not create or evaluate chronological splits.

## Notebook-derived MC06 and MC07 cohorts

The exact commented `DATA_PATHS` from `Model_v24_validation.ipynb` are retained
in `config/subjects/`. As in the notebook, every listed file is audited first,
then sessions without exactly five rounded short and five rounded long ISI
levels are excluded before task-moment estimation and splitting.

This retains 44 of 45 MC06 sessions and 42 of 43 MC07 sessions. The two excluded
source sessions remain visible in the manifests with their observed level
counts. Both retained cohorts have short support `[0.2, 0.8]` seconds and long
support `[1.7, 2.3]` seconds, preserving the subject-specific truncated,
nonoverlapping distributions.
