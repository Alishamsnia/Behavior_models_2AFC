"""Analysis 2: post-fit identifiability diagnostics for a given mouse.

Covers the same 13-model registry as the expanded full-session-matched
model comparison (``post_analysis.expanded_model_comparison.config``:
Pure_WSLS, Extended_WSLS, M0 sensory, the M1 family -- action kernel,
stimulus kernel, known switch, hmm-true, dynamic block length -- and the M2
family -- known switch, hmm-static, dynamic block length, and dynamic +
WSLS/repetition bias), reusing that registry's model names and colors,
rather than the older, coarser M0-M4 chain.

A single multi-page PDF (one page per topic) covering:

1. Multistart parameter spread -- how much the 8 independently-initialized
   sliding-window restarts disagree, per model/parameter (zero new model
   evaluations; reads ``starts/multistart_summary.json``, already on disk).
2. Fisher information / Hessian curvature (with Laplace-approximated
   parameter SEs) at one representative already-fitted point per model.
3. The same Hessian's eigenspectrum ("sloppy model" analysis): how many
   orders of magnitude separate the stiffest and sloppiest parameter
   combinations.
4. Profile likelihood -- placeholder (existing partial machinery in
   ``src/{m1_dynamic_block_length_v1,m2_dynamic_block_length_v1,
   m2_dynamic_previous_wsls_bias_v1,m2_dynamic_previous_action_bias_v1,
   m3_hmm_delta_v1,m4_hsmm_delta_v1}/identifiability.py`` is too slow to
   extend to every model/parameter within a reasonable per-mouse budget;
   see the page for the estimate).
5. Simulation-based parameter/model recovery -- placeholder (existing
   ``run_m2_recovery.py``/``run_m4_recovery.py`` require many full
   multistart refits; too slow for this report's budget).

Unlike ``post_analysis.model_comparison`` (which only reads tidy CSVs already
written under ``results_sliding_window/``/``results_single_session/``), pages
2-3 here import the actual differentiable model code under ``src/`` to
RE-EVALUATE (never re-fit -- no optimizer is ever invoked) the negative
log-likelihood at an already-fitted parameter point, so a Hessian can be
taken via autograd. See ``collect.py`` for the ``sys.path`` note this
requires (a stale sibling checkout is importable under the same package
names and must be shadowed).

Run with ``python -m post_analysis.identifiability.figures --mouse MC08``.
"""
