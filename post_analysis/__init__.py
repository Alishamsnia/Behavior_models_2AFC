"""Post-analysis code for the Bayesian_model_v2 pipeline.

This package only *reads* artifacts already written under
``results_sliding_window/`` and ``results_single_session/`` and writes
derived tables/figures under ``results_post_analysis/``. It never re-fits a
model (no optimizer is ever invoked here) and never edits any file outside
``results_post_analysis/``.

``model_comparison/`` is intentionally separate from ``src/`` and
``scripts/``: it only reads tidy CSVs already on disk. ``identifiability/``
is the one exception -- computing a Hessian needs the actual differentiable
model code, so it imports from ``src/`` to RE-EVALUATE (never re-fit) the
negative log-likelihood at an already-fitted point. See
``identifiability/collect.py`` for a ``sys.path`` caveat that import relies
on (a stale sibling checkout is importable under the same package names on
this machine and must be shadowed).
"""
