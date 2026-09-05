"""Post-analysis code for the Bayesian_model_v2 pipeline.

This package is intentionally separate from ``src/`` and ``scripts/``: it only
*reads* artifacts already written under ``results_sliding_window/`` and
``results_single_session/`` and writes derived tables/figures under
``results_post_analysis/``. It never re-fits a model and never edits any file
outside ``results_post_analysis/``.
"""
