# Independent zero-anchored four-feature Ward analysis

From the repository root:

```bash
python -m post_analysis.zero_anchored_ward
```

The default output is
`results_post_analysis/expanded_model_comparison/pooled/clustering/sliding_window_pattern_exploration_zero_anchored_colstd_rowl2_ward_four_features_star/`.
The implementation imports no previous `post_analysis.session_clustering` modules. Previous code and result files are hashed before and after a run to verify preservation. The previous four-feature star label CSV is read only for the requested historical comparison.

Use the repository environment plus `pip install -r post_analysis/zero_anchored_ward/requirements.txt` if the additional report/clustering dependencies are missing. For the scientific-invariant tests, install pytest and run:

```bash
python -m pytest tests/test_zero_anchored_ward.py -q
```

## Exact scientific specification

Four ordered coordinates, and no other clustering inputs:

1. `sensory`: L(const) - L(S), with S = `m0_sensory`.
2. `action_history_total`: L(S) - L(S+AH), with S+AH = `m1_action_kernel`.
3. `wsls_history_total`: L(S) - L(S+WH), with S+WH = `m1_stimulus_kernel` (the rewarded-side/WSLS history model, despite the registry name).
4. `dynamic_context_total`: L(S) - L(S+Bdynamic), with S+Bdynamic = `m2_dynamic`.

L is mean negative log-likelihood in nats/trial on evaluated held-out test trials. The Bernoulli constant baseline is fitted only on evaluated training choices. Complete trial identities, evaluation masks and observed choice/stimulus/context fields must agree across the four models; per-trial losses are verified from probabilities and aggregate means/counts are checked against upstream comparison tables. There are 92 unique mouse/session pairs in the current analysis; no models are refitted.

The geometry is `positive = maximum(gain, 0)`, `u = positive / positive.std(axis=0, ddof=1)`, `z = u / sqrt(sum(u**2, axis=1) + 1e-24)`. No mean centering, subject-specific scales or additional clipping are applied. Epsilon is 1e-12. All-zero columns use denominator 1 and remain zero; nonzero constant columns fail explicitly. Sessions with all-zero positive gains remain in exports with unresolved label 0 and are excluded from Ward. Small nonzero norms are recorded without imposing an arbitrary threshold or claiming reliable directions.

SciPy Ward receives normalized coordinates directly with Euclidean distance. All candidate labels K2..8 are exported; the descriptive selected K maximizes silhouette, with smaller K breaking ties. The six core report figures are retained for K2/K3/K4 and any additional selected K. Cluster labels are numbered by decreasing size; the earliest input session breaks ties. They are not aligned with historical labels.

Partition stability refits the whole normalization and clustering pipeline on 100 random 80% subsets; leave-one-mouse-out sensitivity also refits scaling. These are descriptive perturbation checks, not uncertainty estimates for model gains. Overlapping rolling training windows make sessions dependent. The three-mouse sample limits population-level interpretation. No mouse-composition constraint is imposed.

## Figures and tables

The generated README documents the numbered folders and every report family. `figure_coverage.csv` maps reference figure purposes to newly rendered figures. `completion.json` is written only after all figures and expected report page counts pass validation. Every core figure has PNG and editable-text PDF, and follow-ups also have SVG. The K3 report adds pooled RT mean/SEM, shared-duration RT and transition RT pages; mean/SD is separate. E/F sequence overview/details, matched-history follow-ups, normalization sensitivity and comparison to previous RBF labels are rebuilt using new labels.

Raw signed, clamped, column-scaled and row-normalized gains are separate exports. Raw-gain and normalized centroid tables are named separately. All fitted parameters/functional readouts are descriptive annotations. Behavioral summaries average sessions within mouse and then available mice; support counts identify missing mice. Subject-specific clusters are allowed and must be interpreted accordingly.

Observed choice plots retain pooled trial/event SEM. Core RT plots use SEM across within-session means. Supplementary RT figures reproduce pooled-trial SD/SEM and export Welch tests with Holm adjustment; these tests assume independent trials and are descriptive. Matched follow-up estimates use shared-support cells within sessions, with formal-block influence intervals only when support is adequate. Sequence E/F uses complete adjusted blocks, excludes the final incomplete block and requires entire trains inside seven-trial windows, step three, centers at most 25.

## Reproducibility and publication

Inputs are source prediction CSVs, upstream comparison metrics/parameters and original raw behavior CSVs. The latter default to the sibling `Codes/Behavior/New_version/` directory; override with `--raw-root PATH` if moved. Source SHA256 hashes and trial/baseline/raw-join audits are exported. No pickled cache is accepted as an analysis input.

The analysis first writes an isolated staging directory. An existing destination is refused by default. `--replace` replaces only a directory whose protocol identifies this new analysis and retains its previous version as a timestamped backup. Historical clustering directories cannot be used as destinations. The source code must remain unchanged during the run. A failed run leaves its staging directory available for diagnosis and never publishes a completion flag to the destination.

## K4 reference-style report extension

`python -m post_analysis.zero_anchored_ward.refresh_k4_report` updates only the published K4 report and its supporting tables/metadata, using existing audited trial observations and fixed K4 labels. It preserves pages 1-4 and the K2/K3 reports and retains a complete backup. Pages 5-8 follow the extended reference layout: choice behavior, shared-duration trial-pooled RT with binwise statistics, mean/SEM with all six pairwise tests per condition, and transition-aligned RT. A separate mean/SD figure is included. Page 6 now uses the reference's pooled-trial SEM; the original session-mean summaries remain in their CSV. Condition-level Holm correction covers 96 tests; binwise tests form a separate family. These trial-level tests are exploratory and assume independence of repeated trials, matching the reference procedure.

The original `source_manifest.json` describes the fixed clustering run. `K4_report_source_manifest.json` records the code used for this report-only refresh. A full future pipeline run also produces the extended eight-page K4 report directly.
