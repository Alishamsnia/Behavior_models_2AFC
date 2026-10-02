# post_analysis

Post-hoc analysis code, kept separate from `src/` and `scripts/`. Everything
here only **reads** artifacts already written under `results_sliding_window/`
and `results_single_session/`; nothing is re-fit and nothing outside
`results_post_analysis/` is ever written.

## Analysis 0: expanded sliding-window comparison (MC08)

This comparison registers 12 requested models, including the two new M2 bias
variants. It rejects any workflow other than `full_session_matched`, excludes
an incomplete model wholesale from scientific outputs, and prints
`<model>: not completely run` for every incomplete configured model.

```bash
python -u -m post_analysis.run_expanded_model_comparison --mouse MC08
```

Outputs are written to
`results_post_analysis/expanded_model_comparison/MC08/`. Figure 0 ranks fit
quality separately for train, validation, and test. Its PDF adds paired 95%
basic moving-block-bootstrap confidence intervals and Holm-adjusted block-bootstrap
TOST results for the eligible deployable models, grouped into WSLS, M0, M1,
and M2 families. The practical-equivalence margin is +/-0.005 mean-window NLL.
The tables also contain
all pairwise comparisons, all-model and non-oracle rankings, completion
diagnostics, win counts, and AIC/BIC. The known-switch models are identified
as oracle benchmarks. Until all 12 models finish, every figure is visibly
marked provisional and partial model windows are never included.

To recompute only the Figure 0 CI/TOST tables and rerender its PNG/PDF from the
already collected tables (without auditing models or rendering Figures 1-23):

```bash
python -u -m post_analysis.expanded_model_comparison.figures --mouse MC08 --figure fig00
```

The expanded output contains Figure 0 plus a complete sliding-window Figure
1–23 suite. Existing Figure 1–6 artifacts are retained; Figures 7–23 add
trial-subset NLL, rank stability, parameter trajectories/distributions,
availability, train/test agreement, IC weights, event-aligned NLL and
correctness, held-out psychometrics, transition/rare-trial performance, prior
samples, parameter mean/SD, and hazard curves. No single-session artifacts are
read by this analysis.

Expanded Figure 7–23 mapping:

- Figure 7: train/validation/test NLL for seven trial subsets.
- Figure 8: per-window model-rank stability by split.
- Figures 9, 11, 12, and 21: shared/model-specific parameter trajectories,
  distributions, and across-window mean ± SD.
- Figure 10: completion availability for all 12 configured models; this is the
  only suite panel allowed to show incomplete models.
- Figures 13–14: training winner timeline and train-versus-test agreement.
- Figure 15: AIC/BIC model weights on identical training trials.
- Figures 16–17: event-aligned NLL and model-minus-mouse correctness
  (transitions −5…+5; rare trials −2…+2).
- Figure 18: held-out mouse-versus-model psychometrics.
- Figure 19: transition performance by stimulus type over trials −15…+20.
- Figure 20: causal prior trajectories for applicable eligible models.
- Figure 22: switch-hazard curves for applicable eligible hazard models.
- Figure 23: rare-trial performance by stimulus type over trials −2…+2.

## Analysis 1: `model_comparison` (M0 vs M1 vs M2 vs M3 vs M4)

Run both stages for one mouse:

```bash
cd Bayesian_model_v2
python -u -m post_analysis.run_model_comparison --mouse MC06
```

or the three stages separately:

```bash
python -u -m post_analysis.model_comparison.collect --mouse MC06   # tables only
python -u -m post_analysis.model_comparison.figures --mouse MC06   # figures only (needs the tables)
python -u -m post_analysis.model_comparison.report  --mouse MC06   # summary.md only (needs the tables)
```

### Render figures independently

Once the tables have been collected, use `--figure` to regenerate only the
figure you are editing:

```bash
python -u -m post_analysis.model_comparison.figures --mouse MC06 --figure fig03
```

Selectors may be written as `3`, `03`, `fig3`, or `fig03`. Repeat the option
to render several figures; they are deduplicated and rendered in numeric order:

```bash
python -u -m post_analysis.model_comparison.figures \
  --mouse MC06 --figure fig03 --figure fig11 --figure fig17
```

List the available IDs without reading any tables or creating outputs:

```bash
python -u -m post_analysis.model_comparison.figures --list-figures
```

Omitting `--figure` preserves the previous behavior and renders all 21. A
selective run regenerates only the requested PNG/PDF files; existing outputs
for unselected figures remain untouched. The full
`post_analysis.run_model_comparison` command still performs collection, renders
all figures, and rebuilds the report.

Outputs land in `results_post_analysis/model_comparison/<mouse>/`:

```text
tables/
  sliding_window_split_session_metrics.csv    # per (model, window, split, session): detailed (per-session) + global (ALL_SESSIONS row) NLL/ELPD/Brier/accuracy
  sliding_window_global_pooled_metrics.csv    # per (model, split): pooled over the common windows every model reached; test split = true walk-forward evaluation
  sliding_window_parameters.csv               # long format: per (model, window, parameter) fitted value
  sliding_window_win_counts_per_window.csv    # per window (intersection of all 5 models): each model's test NLL + the winner
  sliding_window_win_counts_summary.csv       # per model: how many rolling windows it wins
  sliding_window_train_win_counts_*.csv       # in-sample (train-split) counterparts of the two win-count tables above
  single_session_metrics.csv                  # per (model, session, fine_tune_prefix/heldout): NLL etc., plus convergence/gate diagnostics
  single_session_availability.csv             # per model: attempted / valid / flagged-not-converged / flagged-by-pipeline-gate / never-attempted counts
  single_session_parameters.csv               # long format: per (model, session, parameter) initial + fitted value
  single_session_win_counts_per_session.csv   # per session (intersection of all 5 models): each model's heldout NLL + the winner
  single_session_win_counts_summary.csv       # per model: how many sessions it wins
  single_session_train_win_counts_*.csv       # in-sample (fine_tune_prefix) counterparts of the two win-count tables above
  chronological_winner_timeline*.csv          # per chronological rank: which model wins, held-out and (train.csv) in-sample
  chronological_winner_phase_summary*.csv     # win counts/shares by early/mid/late phase, held-out and (train.csv) in-sample
  chronological_train_vs_test_winner_agreement_*.csv  # does the in-sample winner also win held-out, per rank / summarized
  sliding_window_paired_pairwise.csv          # four adjacent paired contrasts per split, exact common-window population
  single_session_paired_pairwise.csv          # four adjacent paired contrasts per role, exact common-session population
  sliding_window_trial_subset_metrics.csv     # per (model, window, split, trial_subset): mean_nll/nll_sem/n_trials (SEM across finite individual-trial NLLs)
  single_session_trial_subset_metrics.csv     # per (model, session, role, trial_subset): mean_nll/n_trials for the 7 fig03 columns
  sliding_window_trial_subset_pooled_diamonds.csv  # trial-weighted pooled NLL per (model, split, trial_subset) -- fig01's diamonds
  sliding_window_trial_subset_paired_pairwise.csv  # four adjacent contrasts per (split, trial_subset) subplot
  single_session_trial_subset_paired_pairwise.csv  # four adjacent contrasts per (role, trial_subset) subplot
  sliding_window_event_aligned_nll.csv        # per (model, window, split_role, event_category, relative_position): mean_nll + nominal/per-position event counts; split_role in {all,train,validation,test} -- fig16's 4 pages
  single_session_event_aligned_nll.csv        # per (model, session, event_category, relative_position): mean_nll + nominal/per-position event counts
  sliding_window_event_aligned_paired_pairwise.csv  # four adjacent contrasts per (split_role, event_category), using each window's whole-range mean
  sliding_window_event_aligned_trial0_paired_pairwise.csv  # four adjacent contrasts at trial 0 (the anchor trial), per (split_role, event_category)
  single_session_event_aligned_paired_pairwise.csv  # four adjacent contrasts per event_category, using each session's whole-range mean
  information_criteria_sliding_window.csv     # per (model, window): AIC/BIC/deltas/weights from that window's converged pooled training fit
  information_criteria_summary.csv            # per model: mean/median AIC/BIC, mean weight, win counts (common windows only)
  information_criteria_single_session.csv     # per (model, session): AIC/BIC on jointly numerically-sound fine-tuning fits with the same trial N
  information_criteria_single_session_summary.csv # per-model summary of the common numerically-sound single-session IC population
  information_criteria_paired_pairwise.csv    # four adjacent paired contrasts for AIC/BIC at both analysis levels
  psychometric_comparison_trials.csv          # per (model, source, session, trial): matched trial-level rows behind fig18
  psychometric_comparison_summary.csv         # per (model, source, block_subset, bin): four-quantile-bin psychometric summary for fig18
  sliding_window_block_transition_performance.csv  # per (model, window, split_role, event_category, trial_type, relative_position): window-level mean mouse/model correctness, trials -15 through +20 around block transitions, for fig19
figures/
  fig01_sliding_window_global_and_detailed_nll.{png,pdf}           # 3 rows (train/val/test) x 7 trial-subset columns; four adjacent paired tests per subplot
  fig02_sliding_window_nll_trajectory.{png,pdf}                    # per-window pooled mean NLL ± SEM across the contributing individual-trial NLLs; row 3 states the held-out test definition explicitly
  fig03_single_session_heldout_nll.{png,pdf}                       # 2 rows (training-prefix/held-out) x 7 trial-subset columns; four adjacent paired tests per subplot
  fig04_single_session_win_counts.{png,pdf}                        # "how many sessions does each model win"
  fig05_shared_parameter_trajectories.{png,pdf}
  fig06_sliding_window_win_counts.{png,pdf}                        # "how many rolling windows does each model win"
  fig07_win_counts_single_session_vs_sliding_window.{png,pdf}      # both win-count views side by side, per model
  fig08_chronological_winner_timeline.{png,pdf}                    # held-out winner per chronological rank, both analysis levels
  fig09_sliding_window_model_specific_parameters.{png,pdf}          # trajectories, a column explaining each parameter, plus a column plotting the explained derived quantity with a linear-trend overlay
  fig10_single_session_availability.{png,pdf}
  fig11_single_session_shared_parameter_trajectories.{png,pdf}
  fig12_single_session_model_specific_parameters.{png,pdf}
  fig13_chronological_winner_timeline_train.{png,pdf}              # in-sample counterpart of fig08
  fig14_train_vs_test_winner_agreement.{png,pdf}
  fig15_model_comparison_information_criteria.{png,pdf}            # AIC/BIC for common rolling windows + jointly numerically-sound fine-tuned sessions
  fig16_sliding_window_event_aligned_nll.{png,pdf}                 # 2x2 event trajectories plus 2 rows of 4 paired summary panels (block transitions, then rare trials); NOW 4 PAGES -- the .pdf has one page each for all/train/validation/test sessions (collect.EVENT_ALIGNED_SPLIT_ROLES), the .png mirrors page 1 (all), and fig16_..._page{1..4}_{split}.png give each page as its own flat PNG
  fig17_single_session_event_aligned_nll.{png,pdf}                 # same 2x2 layout, single-session fine-tuning
  fig18_psychometric_comparison.{png,pdf}                          # 5 model rows x 4 block-type columns; sliding-window (zero-shot) vs. single-session (fine-tuned) prediction on the identical trial population
  fig19_sliding_window_block_transition_performance_by_trial_type.pdf  # PDF ONLY (no PNG), 6 pages (one per model plus a final all-model overlay); each page is a plain 4x4 grid (rows=all/train/validation/test, columns=short/long trials x short-to-long/long-to-short transitions), mouse/model P(correct) from trial -15 through +20 around the transition; the overlay uses all-model matched windows at each relative position
  fig20_sliding_window_long_block_prior_causal_trajectory_samples.pdf  # PDF ONLY; M1-M4 parameter trajectories plus an all-model overlay
  fig21_sliding_window_parameter_mean_sd.{png,pdf}                    # 3x3 constrained-parameter summary; dots=across-window means, whiskers=±1 sample SD, with 5/4/3/2/1 models across shared/specific panels
summary.md
```

## Trial-subset columns (Figures 01/03) and event-aligned figures (16/17)

Figures 01 and 03 break the same train/validation/test (or training-prefix/
held-out) NLL down by 7 trial subsets, derived from each trial's stimulus
category and block bias: **all trials**, **short block**, **long block**,
**common short** (short stimulus in a short block), **common long**, **rare
short** (short stimulus in a *long* block), **rare long**. Every subplot
recomputes its own matched window/session population (a window/session can
have zero qualifying trials of a given subset, e.g. no rare-long trials in a
short test session) rather than inheriting the "all trials" population, and
runs the same four adjacent-model Holm-Wilcoxon tests independently within
that subplot.

Figures 16/17 show mean NLL at each relative trial position around two kinds
of events, matched by exact window/session identity across all 5 models:
a **block transition** (relative trial -5..+5, trial 0 = the first trial of
the new block) and a **rare trial** (relative trial -2..+2, trial 0 = the
rare trial itself). Multiple qualifying events within one window/session are
averaged together first, so a window/session contributes one number per
relative position regardless of how many events it contains. An event is
excluded if any requested relative position is absent from the evaluated
trial sequence; saved nominal and per-position event counts are required to
match before testing. The four
adjacent-model paired tests per subplot use each window/session's mean NLL
over the *whole* displayed range collapsed to one scalar -- one set of 4
Holm-corrected tests per subplot, not one test per relative position.

For the NLL plots and win-count analyses, every model is counted the same way: a session/window counts as
soon as it has a computed NLL, regardless of that run's own convergence flag.
`converged` (all models) and `pipeline_gate_pass` (M4's extra HSMM
projection/tail-occupancy gate) are kept only as diagnostic columns in
`single_session_metrics.csv` -- they never silently drop one model's rows
while keeping another's.

`fig01`, `fig03`, and their paired-test tables use a **fixed common
population** across the 5 models (`collect.common_metric_window_ids` /
`collect.common_session_ids`) rather than each model's own larger, unequal-N
availability. Every box, point, and statistical contrast therefore refers to
the same session/window IDs for all models. The four scientifically specified
nested comparisons are M0–M1, M1–M2, M2–M3, and M3–M4. They use paired
Wilcoxon signed-rank tests with Holm correction across exactly those four
tests within each panel; paired t-tests are retained in the CSVs as a
parametric cross-check. No omnibus-test gate is used. Rolling-window p-values
are labeled exploratory because neighboring windows reuse data and fitted
history.

The rolling-window row of `fig15` uses that same common rolling population.
Its single-session row is intentionally stricter: information criteria are
only valid for the common sessions where all five fine-tuned likelihood fits
were numerically sound (finite, stable, no transform/normalization/runtime
error -- deliberately looser than each model's own absolute-tolerance
`converged` flag, which is miscalibrated for single-session trial counts;
see the report's §7 reliability note) and used the same number of training
trials. This can be a smaller population than `fig03` and is labeled as a
reduced-N sensitivity analysis. See `model_comparison/stats.py`. `summary.md` §3b, §7, and §8
walk through the paired-test results, the AIC/BIC model-selection analysis,
and each model's average fitted parameters with a plain-language meaning for
every parameter (`w`, `s0`, `ell_short`, `ell_long`, `kappa`, `ah`, `delta0`,
`alpha_theta`, `bh`).

## Adding another analysis

Create `post_analysis/<analysis_name>/` with its own `collect.py` /
`figures.py`, reuse `post_analysis/common.py` for paths and model metadata,
and write outputs to `results_post_analysis/<analysis_name>/<mouse>/`.
