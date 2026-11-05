# Pure_action_v1 — equations (`actionSingle_lapse`)

No sensory evidence. The prediction is the mouse's own **last valid choice**,
mixed only with the category-dependent lapse. The action-driven sibling of
`Pure_WSLS_v1` (which uses the last feedback-valid *category* instead); the
`alpha = 1` boundary of `Extended_action_v1`.

## 1. Parameters (2 fitted coordinates)

```
(ell_short, ell_long)
```

$$
\ell_{\text{short}}=\ell_{\max}\,\sigma(\widetilde\ell_{\text{short}}),\qquad
\ell_{\text{long}}=\ell_{\max}\,\sigma(\widetilde\ell_{\text{long}}),\qquad
\alpha\equiv 1\ \text{(fixed, not fitted)} .
$$

## 2. Per-trial recursion (predict → score → update)

$$
\pi_1=0.5,\qquad
\epsilon_t=\ell_{\text{long}}\mathbf 1(C_t=1)+\ell_{\text{short}}\mathbf 1(C_t=0),\qquad
p_t=(1-\epsilon_t)\,\pi_t+\epsilon_t/2,
$$

$$
\pi_{t+1}=\begin{cases}A_t, & \text{choice valid at } t\\ \pi_t, & \text{otherwise}\end{cases}
$$

$A_t$ = `choice_long[t]`. The update is the exact assignment
`pi_next = A_t` (not floating-point arithmetic), so $\pi_t$ is **exactly**
the last valid choice before $t$ (0.5 before any). `pi` resets to 0.5 every
session.

## 3. Masks

- **Likelihood:** valid binary choice **and** finite binary current category
  (the lapse needs a category). Duration is never read.
- **Update:** valid choice alone — independent of `feedback_valid`, category
  validity and duration validity. An omitted choice adds no likelihood and
  leaves $\pi$ unchanged.

## 4. Category-dependent lapse (inherited, deliberate)

$\epsilon_t$ is selected by the *true current* category, exactly as in
`Pure_WSLS_v1` and every other model. It can only pull the prediction toward
0.5, never toward the correct side, but it is the one place current-trial
stimulus identity enters this otherwise stimulus-free model. Kept so the
action and WSLS families are directly comparable.

## 5. Expressiveness

$p_t$ always lies between $\pi_t$ and 0.5, so the model can express choice
repetition but never alternation — the same repetition-only limit as the
WSLS models and every $\eta$-mixing bias model.

## 6. Implementation

`src/action_core_v1/evaluator.py` (shared with `Extended_action_v1`; the
action-driven sibling of `wsls_core_v1`, which is untouched),
`src/pure_action_v1/`, `src/pure_action_sliding_window_v1/`,
`scripts/run_pure_action_sliding_window.py`. Configs:
`config/pure_action_v1{,_mc06,_mc07}.json`. Starts: Pure_WSLS_v1's eight lapse
rows, byte-identical (`config/initializations/pure_action_v1_starts.json`).
Convergence: the 7 scalar-history checks (no normalization check — there is
no renormalized belief). Tests: `tests/test_action_core.py`,
`tests/test_pure_action_*.py`.

The runner continues past a window with no converged start (saved summary,
`window_manifest.json` status `no_winner`); such windows can be resolved
without retraining by `extra_tools/forced_best_train_nll.py --runner
pure_action`. Its `--workflow` defaults to `full_session_matched` (the
workflow every comparison result uses).
