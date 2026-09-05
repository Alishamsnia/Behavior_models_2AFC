# M3 implementation verification

Date: 2026-08-11

## Frozen implementation status

- Model identifier: `M3_hmm_delta_v1` (existing project naming retained).
- Parameter order: `(w,s0,ell_short,ell_long,kappa,ah,delta0,alpha_theta)`.
- Lapse upper bound: `lapse_max=1.0` (unchanged by explicit decision).
- Initializations: eight rows, preserving M2 rows 1–8 exactly and adding only
  the registered alpha values.
- Empirical median block length: 23 trials. `1/23` is used only in recovery and
  diagnostics; hazard remains fitted.
- Optimizers: full-batch Adam, pure PyTorch L-BFGS with strong-Wolfe line
  search, and Adam-to-L-BFGS hybrid. The eight starts run in isolated CPU
  processes and are restored to start-ID order before unchanged winner selection.
  Worker progress is relayed live to the parent; worker/process thread counts are
  exposed as M3-only command-line options.
- Locked test data: not opened during fitting or implementation verification.

## Automated verification

The full repository suite passes: 124 tests, zero failures. M3-specific tests
cover the eight-parameter transform, exact alpha-zero M2 equivalence, the four
one-hot update signals, causal category timing, valid-feedback omissions,
invalid feedback, projection records, the weak-delta identifiability regression,
all eight finite-difference gradients, session resets/order invariance, fresh
full-batch objectives, all three optimizer strategies, and exact sequential versus
parallel multistart equivalence, live cross-process progress delivery, and M3
parallel CLI validation.

The deterministic M0/M1/M2 pre-M3 prediction artifact is
`config/releases/m0_m1_m2_pre_m3_reference_predictions_v1.json`; its independent
audit passes at absolute tolerance `1e-14`. The original 117-file pre-M3 hash
manifest also passes without a mismatch, confirming that M0–M2 source,
registries, scripts, documentation, and tests were not changed.

## Recovery and validation reports

The software needed to produce empirical reports is complete, but numerical
results are intentionally not fabricated here:

- `scripts/run_m3_recovery.py` runs either one specified case or all 60
  registered hazard/contrast/learning-rate cases, including exact alpha zero,
  and writes parameter plus latent-trajectory recovery reports.
- `run_m2_m3_model_confusion` fits both M2 and M3 to static- or dynamic-generated
  data and reports held-out model-selection evidence.
- `scripts/compare_m2_m3.py` writes aligned validation ELPD, bits/trial,
  session-clustered uncertainty, calibration, common/rare, early/late,
  transition, surprising-category, and psychometric tables.
- `scripts/generate_m3_figures.py` and
  `scripts/generate_m3_m0_style_figures.py` generate the complete requested
  scientific and legacy M0-style figure families.

Empirical recovery and validation values become valid only after running the
registered full fits on the chosen mouse; they are not implementation-time
constants.
