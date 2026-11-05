# Current exploratory workflow: all sessions, all-model reference

The current goal is intuitive model-defined session patterns for behavioral and
physiological follow-up. Use all 92 sessions; no calendar or forced-fit filtering.
The original all-model result in `sliding_window_final/` remains the reference.

```bash
python -m post_analysis.session_clustering.pattern_exploration
python -m post_analysis.session_clustering.pattern_report
```

Outputs go to `sliding_window_pattern_exploration/`. The main exploratory refinement
is RBF similarity on the original 48 drivers: K3 sizes24/34/34, preserving the
original 34-session weak-context group exactly, and moving seven MC07 sessions
from C2 to C1. Richer functional kernels and performance-only groups are additional
comparisons. All 20 variant/K results are saved, including unsuccessful pruning.
Only two current PDFs are produced: eleven readable cluster-summary/session pages
(including a full-feature appendix) and eight observed behavior/RT pages. `findings.md` in that output folder explains the patterns and limitations;
`selected_session_clusters.csv` supplies labels for the plotted comparisons.

Both parameter transformations and nonlinear similarity reuse existing fits. The
original within-mouse scaling, block/model weights and nuisance/oracle exclusions
are retained. History lags count valid updates, not elapsed trials. Equal model
weight is divided across derived lag features. Test-NLL inputs and the inspection
of behavior make these exploratory descriptions, with pupil/motion follow-up still
separate. No source-session gaps or fit-status flags suppress any row.

The older compact/epoch analyses below remain available as historical sensitivities;
`python -m post_analysis.session_clustering` still invokes the compact experiment
for compatibility. Use the explicit two commands above for the current exploration.

## Earlier compact sliding-window clustering experiment

Run `python -m post_analysis.session_clustering --bootstrap 200`.
New outputs go to `results_post_analysis/expanded_model_comparison/pooled/clustering/sliding_window_compact/`.
The previous `sliding_window_final/` outputs are preserved for comparison.
No decision models are refitted and forced-fit status is not used for selection or weighting.

## Scientific unit and feature panel

Each row is a training/validation window associated with its next held-out test
session: nine training sessions supply fitted parameters, three validation
sessions supply performance, and one test session supplies behavioral checks.
These are historical-window profiles, not parameter estimates fitted to the test
session. Test choices/performance never enter the primary feature vectors.

The candidate drivers are:

- M0 effective sensory SD at 0.8 and 1.7 seconds, log transformed:
  `sqrt(w * duration**2 + s0**2)`. M0 is the common sensory reference.
- M1 own-action history update rate `sigmoid(a)`.
- M2 static subjective block contrast `sigmoid(delta0)-0.5` and biased-block
  switch hazard `sigmoid(ah)`. Static M2 still updates block beliefs every trial.
- Validation NLL gain of M0 over a fair coin, and M1 action/M2 static over M0.

Dynamic M2 contributes **annotations only**: static-minus-dynamic NLL on each
split, hazards at inferred ages 10/23/35, and their change. Its parameters and
performance do not enter the primary distance. Raw `w`/`s0`, lapse, fixed bias,
oracle models and other exploratory models are excluded from the compact panel.
The fair-coin anchor measures absolute predictability, not superiority to an
optimal fitted constant-bias model.

## Fixed feature screens and geometry

Before scaling, each feature must have a 10th–90th percentile spread exceeding
its threshold in at least two mice: log(1.05) for log noise, 0.02 for history rate
and prior contrast, 0.01 for switch hazard, 0.002 nats/trial for performance.
Then within-mouse constant inputs and same-block redundant inputs (`|r|>=0.95`
after within-mouse scaling) are removed in the declared feature order.
These are variation screens, **not parameter uncertainty/recovery tests**.
All eight candidate features survive on the current full dataset.

Within-mouse mean/SD normalization and clipping at +/-3 SD remove mouse levels
and scales; they cannot remove all mouse covariance or chronology effects.
Squared weights sum to one per block (sensory/history/prior/performance), with
equal model weight inside each block. Feature selection and scaling are refitted
inside every bootstrap. LOMO selects features and fits centroids using donor mice
only; excluded-mouse mean/SD normalization remains transductive. It is not a
prospective new-mouse prediction test.

K-means uses K=2/3, n_init=20, seed=20261025. Rastermap only orders rows. Diagnostic
thresholds flag failures without hiding requested figures. Moving-block bootstrap
uses 13 consecutive windows (the full train/validation/test span), without wrapping
chronology. This does not make overlapping windows independent. Full-cohort
clustering/scaling is retrospective; later windows' training data can contain an
earlier window's test session. Truly prospective validation would require fitting
all preprocessing/centroids on earlier, source-session-purged windows. No such
prospective claim is made here.

## Comparisons and current finding

Three predeclared panels use identical resampling settings, with no hyperparameter
search: previous all-model/test features, compact/test sensitivity, and the primary
compact/validation panel. `panel_comparison.csv` and matching labels retain every
result. Cluster IDs are local to each panel and K.

With 200 bootstrap draws and 13-window blocks:

| Panel | K | Median bootstrap ARI | Minimum LOMO ARI | Mouse NMI |
|---|---:|---:|---:|---:|
| Previous 48 features | 2 | 0.658 | 0.521 | 0.002 |
| Compact, test performance | 2 | 0.571 | 0.261 | 0.023 |
| Compact, validation performance | 2 | 0.514 | 0.141 | 0.027 |
| Previous 48 features | 3 | 0.436 | 0.502 | 0.027 |
| Compact, test performance | 3 | 0.378 | 0.628 | 0.050 |
| Compact, validation performance | 3 | 0.395 | 0.319 | 0.078 |

The compact representation is simpler and has higher silhouette, but is **not
more stable**. Its K=3 sizes are 34/3/55; the three-session group is MC06 windows
1–3 only. It fails the mouse-representation diagnostic and must not be called a
shared third strategy. K=2 contains all mice in both groups but has weak LOMO
agreement. The previous results are not automatically replaced or relabeled.

## Outputs and interpretation checks

Only two PDF figures are produced, retaining the user's existing visual layout:

- `clusters.pdf`: K=2 and K=3 compact heatmaps, mouse counts and diagnostics.
- `cluster_behavior_pooled.pdf`: four pages: K=2/K=3 mouse psychometrics,
  transitions and rare trials, then K=2/K=3 reaction-time curves by stimulus bin.
  All clusters are superimposed. Psychometric/rare axes are square; transition
  widths remain at 60% of the earlier slots; event y limits autoscale. No model
  behavior curves are plotted. Subject-confined groups carry a visible caution.

First-page PNG previews accompany the PDFs. `python -m
post_analysis.session_clustering.behavior` refreshes only the behavioral figures.

`validation_test_transfer.csv` compares predictive gains across the preceding
validation and target test sets, including within-mouse rank correlations.
This assesses transfer of model performance, not recovery of individual parameters.
`dynamic_m2_annotations.csv`, `dynamic_by_cluster_mouse.csv` and
`dynamic_age_by_mouse.csv` keep duration effects visible without changing labels.
True block age is an evaluation stratum, never a model input.

`test_model_category_metrics.csv` and `test_model_category_adequacy.csv` compare
M0/M1 action/M2 static/M2 dynamic with observed accuracy in common/rare x
short/long stimulus cells. Expected accuracy uses choice probabilities, not hard
threshold predictions. Warmup is excluded. Summary means weight mice equally,
then sessions equally; 95% descriptive intervals resample 13-window blocks within
mouse. An interval containing zero does not establish equivalence. No models are
selected by these test results. Cluster profiles are also saved per mouse in
`test_model_profiles_by_cluster_mouse.csv`. These checks do not establish a
mechanism or parameter identifiability.

Reaction-time extraction follows `Behavior/New_version/licking_analysis.ipynb`:
first lick at/after choice-window onset; exclude warmup, invalid/negative scalar
latencies and any negative value in the all-lick reaction/start-time arrays; no
upper cutoff. Unique date/source-trial joins verify stimulus, block and warmup.
Four rank bins per short/long stimulus type use within-session means followed by
session mean +/- SEM. Choice plots preserve pooled trial/event SEM. RT checks and
bin summaries are saved as before; 92 sessions match and 26,211 trials are eligible.

`feature_selection.csv` records exclusions; `clustering_features.csv` records
weights; `feature_values.csv`, `clustering_matrix.csv`, raw metrics, labels and
`summary.json` make the experiment reproducible.

## Checks

`python -m unittest discover -s tests -p 'test_*clustering*.py' -q`

Tests cover target-test/dynamic exclusion, effective-noise algebra, feature
screens, donor-only LOMO selection, mouse diagnostics, session joins and the
existing behavior/RT conventions. Saved predictions are also checked for exact
matching trial inputs across all four diagnostic models before comparison.

## Scale/method and recording-gap audit

The additional audit preserves both previous cohorts and their labels. Reproduce it with:

```bash
python -m post_analysis.session_clustering.scale_audit --bootstrap 200
python -m post_analysis.session_clustering.scale_supplement
python -m post_analysis.session_clustering.calendar_check
python -m post_analysis.session_clustering.scale_report
```

Raw sensitivity outputs are in `sliding_window_scale_audit/`, including the complete
24-configuration comparison, paired resampling draws, per-cluster Jaccard, LOMO
contingencies, clipping fractions, feature-block deletions, calendar/source-session
audit, synchrony-shift reference, and two Claude reviews of statistical briefs.
Claude did not read code or independently analyze data. `findings.md` documents
agreement, disagreements and the limits of that consultation.

Own-mouse SD, shared within-mouse SD and shared robust scaling were compared with
KMeans, Ward and regularized tied GMM. Exploratory Euclidean PAM k-medoids adds an
unsquared-distance robustness comparison, without trimming or balancing groups.
Shared scaling removes each mouse's mean and divides by the square root of the
equally weighted mean of donor mice's within-mouse variances. Donor-only feature
selection/denominator fitting is repeated in LOMO. Robust denominators are the
median per-mouse IQR/1.349, floored at half the shared SD. All use +/-3 clipping.

The original nonwrapping bootstrap under-samples chronology endpoints. A separate
circular-block sensitivity equalizes endpoint inclusion, but assumes stationarity
around the boundary; it never wraps model training histories. Both diagnostics
are retained, and better circular ARI is not treated as evidence that clusters
became better. Missing reference groups have NaN Jaccard, not fabricated recovery.
LOMO reports whether each held-out mouse retains multiple reference/predicted groups.
Ward transfer is a nearest-centroid surrogate. Tied GMM changes the metric.

The tiny three-row group is MC06's first three windows. The first two training
sets are ~104 days older than their January targets, and the third contains only
one January training session. A fixed >30-day source-recording-gap sensitivity
leaves 69 eligible windows (22/25/22 by mouse). All original 92 windows remain in
the main audit. Forced-fit flags never affect eligibility or weights.

`sliding_window_epoch_candidate/` holds a separate candidate: existing windows
whose full train9/validation3/test1 span lies within a recording epoch. The 23
other windows are explicitly unclassified in `cohort_membership.csv`, not deleted.
Shared-scale KMeans is used consistently for both K values. K2 has sizes23/46,
nonwrapping/circular ARI .691/.662, minimum LOMO ARI .600 and mouse NMI .053.
It passes the fixed descriptive gates, but one cluster has just 3 MC06 windows;
low NMI does not guarantee absence of mouse effects. Its agreement with own-scale
K2 on the same 69 rows is only ARI .540. K3 remains unstable and mouse-confounded.
This candidate changes the eligible population and was motivated by the calendar
audit, so it is exploratory rather than independent confirmation.

Only two candidate PDFs are generated. `clusters.pdf` has K2/K3 heatmaps followed
by complete method and calendar comparison pages (four pages total).
`cluster_behavior_pooled.pdf` retains the existing four-page observed behavior/RT
layout, now for the 69 eligible sessions (19,732 eligible RT trials). No method
was chosen using those test behavior curves. Behavior subset rendering is explicit;
default full-cohort rendering remains strict about missing/extra session labels.

Calendar clarification: gaps are measured between source sessions included in the
saved fits, not necessarily a complete absence of recordings. Raw behavior tables
contain some intervening dates (`raw_session_gap_context.csv`). The candidate's
epochs refer to available fitted sessions; they do not assert a biological reset.

## Readable cluster visualization

`pattern_report` now automatically finishes with the readable display. To refresh
only that display, without refitting clusters or changing labels:

```bash
python -m post_analysis.session_clustering.pattern_visuals
```

The first page shows cluster-level model benefits (percent reduction in mean NLL
relative to M0), behavioral effects in percentage points, mouse counts/accuracy,
and four physical-unit parameter dotplots. Means give each mouse equal weight,
then sessions equal weight within mouse. Small parameter markers identify each
mouse (circle/triangle/square); colored dots are equal-mouse means.

The second page shows all 92 sessions with 12 selected, horizontally labeled
readouts grouped into sensory/history/block-context/combined-model sections.
Blue means above that mouse's usual value; orange means below. These readouts
explain the frozen labels visually; they do not replace the full feature set.
Pages1–2 show the original all-model K3 reference. Pages3–4 show the RBF refinement.
The two-page structure then repeats for the remaining selected comparisons, with
a full original 48-driver appendix on page11. The former dense PDF is retained as
`clusters_dense_archive.pdf`. Behavior/RT PDFs and session labels are untouched.

`visual_readout_manifest.csv` records definitions/units, `visual_readouts.csv`
records session values, `visual_cluster_readouts.csv` records equal-mouse means,
and `visualization_audit.json` verifies the selected-label checksum is unchanged.
