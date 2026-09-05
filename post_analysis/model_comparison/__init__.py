"""Analysis 1: cross-model comparison (M0-M4) for a given mouse.

Two entry points:

- ``collect.py`` reads the raw sliding-window and single-session result trees
  and writes tidy CSV tables to ``results_post_analysis/model_comparison/<mouse>/tables``.
- ``figures.py`` reads those tables and writes paper-quality PNG/PDF figures
  to ``results_post_analysis/model_comparison/<mouse>/figures``.

Run both with ``python -m post_analysis.model_comparison.run --mouse MC06``
(see ``post_analysis/run_model_comparison.py``).
"""
