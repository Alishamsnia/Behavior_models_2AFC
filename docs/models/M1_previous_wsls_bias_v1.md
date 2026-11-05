# M1_previous_wsls_bias_v1 — equations (`sensory_wslsSingle_lapse`)

M0's unchanged Bayesian sensory readout (fixed neutral prior 0.5), mixed
before lapse with the **last feedback-valid true category** (the rewarded
side: win-stay/lose-shift). Equivalently,
`M2_dynamic_previous_wsls_bias_v1` with the HSMM removed. Sibling of
`M1_previous_action_bias_v1`; every equation, mask, start and fitting detail
is identical except the bias state's driver and gate below. See that file
for why the bias is mixed into the output rather than the prior (the
`M1_stimulus_kernel_v1`-with-`alpha=1` construction saturates the readout in
exactly the same way).

## Parameters (6)

```
(w, s0, ell_short, ell_long, kappa, eta_logit)      alpha == 1 (fixed)
```

## The one difference: bias-state update

$$
\pi_1=0.5,\qquad
\pi_{t+1}=\begin{cases}C_t, & \texttt{feedback\_valid}[t]\\ \pi_t, & \text{otherwise,}\end{cases}
$$

gated by `feedback_valid` alone, independent of choice validity: a
feedback-valid omission still updates. The mouse's own choice never enters
$\pi_t$.

$$
m_t=(1-\eta)\,r_t+\eta\,\pi_t,\qquad p_t=(1-\epsilon_t)\,m_t+\epsilon_t/2 .
$$

## Nesting

- $\eta=0$ reproduces `M0_sensory_v1` exactly.
- Equals `M1_previous_action_bias_v1` exactly when choices equal categories
  and feedback mirrors choice validity; diverges otherwise (tested both ways).

## Implementation

`src/sensory_previous_bias_core_v1/` (`history_driver="category"`),
`src/m1_previous_wsls_bias_v1/`, `src/m1_previous_wsls_bias_sliding_window_v1/`,
`scripts/run_m1_previous_wsls_bias_sliding_window.py`. Configs:
`config/m1_previous_wsls_bias_v1{,_mc06,_mc07}.json`. Starts identical to
`M1_previous_action_bias_v1`. Exports `model_previous_wsls_state` and
`model_feedback_updated` (the names `M2_dynamic_previous_wsls_bias_v1` uses).
Forced-best runner key: `previous_wsls_bias`.
