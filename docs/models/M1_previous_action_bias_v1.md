# M1_previous_action_bias_v1 — equations (`sensory_actionSingle_lapse`)

M0's unchanged Bayesian sensory readout (fixed neutral prior 0.5), mixed
before lapse with the mouse's own **last valid choice**. Equivalently,
`M2_dynamic_previous_action_bias_v1` with the block/run-length HSMM removed.
Sibling: `M1_previous_wsls_bias_v1` (identical except the bias state is the
last feedback-valid *category*).

## 1. Why the bias is mixed into the output, not the prior

The obvious construction — `M1_action_kernel_v1` with `alpha` fixed at 1 —
feeds $\pi_t\in\{0,1\}$ into the Bayesian prior. The readout clamps the prior
to $[10^{-10},1-10^{-10}]$ before its logit, shifting the decision boundary
by about 23 log-odds: with fitted MC08 sensory parameters the readout becomes
exactly 0 or 1 at every duration and $\partial/\partial w,\ \partial/\partial s_0$
are exactly 0 (checked numerically). The sensory pathway would be switched
off. This model therefore uses the $\eta$-mixing architecture of the M2 bias
models, which keeps the sensory readout intact.

## 2. Parameters (6 fitted coordinates)

```
(w, s0, ell_short, ell_long, kappa, eta_logit)
```

M0's five transforms (unchanged) plus $\eta=\sigma(\widetilde\eta_{\text{logit}})$.
$\alpha\equiv1$ is fixed, not fitted.

## 3. Per-trial computation

$$
r_t=\mathrm{BayesianReadout}(s_t;\,w,s_0,\kappa,\ \text{prior}=0.5)\quad\text{(M0's own readout)}
$$

$$
m_t=(1-\eta)\,r_t+\eta\,\pi_t,\qquad
p_t=(1-\epsilon_t)\,m_t+\epsilon_t/2,
$$

$$
\pi_1=0.5,\qquad
\pi_{t+1}=\begin{cases}A_t, & \text{choice valid at } t\\ \pi_t, & \text{otherwise.}\end{cases}
$$

**Masks.** Likelihood: finite positive duration, valid binary choice, and a
finite binary current category. Update: choice validity alone (independent
of `feedback_valid`, category validity and duration validity).

## 4. Nesting

- $\eta=0$ reproduces `M0_sensory_v1` **exactly** (zero-tolerance test
  `test_eta_zero_reproduces_m0_exactly`).
- Not nested in `M1_action_kernel_v1` (different coupling: output mixing vs.
  prior). Relative to `M2_dynamic_previous_action_bias_v1` it replaces the
  HSMM block prior by the constant 0.5; the HSMM prior reaches 0.5 only in
  the limit `delta0 -> 0`, so that relation is a limit, not an exact
  parameter-boundary nesting.
- When every choice equals its category and feedback mirrors choice validity
  (as in every registered real dataset), it equals `M1_previous_wsls_bias_v1`
  bit-for-bit.

## 5. Implementation and fitting

Shared core: `src/sensory_previous_bias_core_v1/evaluator.py`
(`history_driver="action"`). Because $\alpha$ is fixed, $\pi_t$ depends only
on data and the readout prior is constant, so the session is evaluated in
one vectorized readout call — mathematically identical to a per-trial loop
(verified against an independent per-trial reference to 1e-12).

Package/runner: `src/m1_previous_action_bias_v1/`,
`src/m1_previous_action_bias_sliding_window_v1/`,
`scripts/run_m1_previous_action_bias_sliding_window.py` — a copy of the
`M1_action_kernel_v1` runner (same 8-start L-BFGS fitter, same convergence
tolerances and checks, same convergence escalation, same 32-CPU budget,
full-session-matched only), except that a window with no converged winner is
saved as `no_winner` and the run continues (the M2 bias runners' contract).
Configs: `config/m1_previous_action_bias_v1{,_mc06,_mc07}.json`. Starts:
M0's eight rows plus the M2 bias models' eight $\eta$ values
(0.05–0.60), copied by `start_id` and verified at registry load. Exports:
`model_prior_long` (= 0.5), `model_action_history_state`,
`model_bias_mixed_probability_long`, `model_non_bias_probability_long`,
`model_action_updated`. Forced-best runner key: `previous_action_bias`.
