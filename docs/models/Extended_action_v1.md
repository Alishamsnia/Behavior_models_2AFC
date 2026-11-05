# Extended_action_v1 — equations (`actionHistory_lapse`)

No sensory evidence. A delta-rule (exponentially weighted) history of the
mouse's own valid choices, with a fitted learning rate, mixed only with the
category-dependent lapse. The action-driven sibling of `Extended_WSLS_v1`;
its `alpha = 1` boundary is `Pure_action_v1`.

## 1. Parameters (3 fitted coordinates)

```
(ell_short, ell_long, a)
```

$$
\ell_k=\ell_{\max}\,\sigma(\widetilde\ell_k),\qquad \alpha=\sigma(a)\in(0,1).
$$

## 2. Per-trial recursion (predict → score → update)

$$
\pi_1=0.5,\qquad p_t=(1-\epsilon_t)\,\pi_t+\epsilon_t/2,\qquad
\pi_{t+1}=\begin{cases}\pi_t+\alpha\,(A_t-\pi_t), & \text{choice valid at } t\\ \pi_t, & \text{otherwise}\end{cases}
$$

with $\epsilon_t$ the category-dependent lapse (see `Pure_action_v1.md` §4).
The decay is indexed by the number of *valid choices*, not elapsed trials.
Masks, session reset, the repetition-only limit, and the category-dependent
lapse are identical to `Pure_action_v1`.

## 3. Nesting and identifiability

- $\alpha\to1$ approaches `Pure_action_v1`; the literal Python `alpha = 1.0`
  passed to the shared core reproduces it exactly
  (`test_action_core_alpha_one_gives_last_valid_action_exactly`).
- $\alpha\to0$ freezes $\pi_t$ near 0.5, where both lapse rates become
  ineffective (flat likelihood) — the same region `Extended_WSLS_v1` has.

## 4. Implementation

`src/action_core_v1/`, `src/extended_action_v1/`,
`src/extended_action_sliding_window_v1/`,
`scripts/run_extended_action_sliding_window.py`. Configs:
`config/extended_action_v1{,_mc06,_mc07}.json`. Starts: Extended_WSLS_v1's
eight `(ell_short, ell_long, a)` rows, byte-identical. Convergence: the 7
scalar-history checks. Forced-best runner key: `extended_action`.
