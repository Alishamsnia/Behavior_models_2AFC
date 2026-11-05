# M2_dynamic_action_kernel_bias_v1 — equations (`sensory_learnRatio_learnDynamicBlockSwitch_actionHistory_lapse`)

Parent: `M2_dynamic_previous_action_bias_v1` (unchanged). This model is the
parent with the action-history learning rate **fitted** instead of fixed at
1: $\pi_t$ becomes an exponentially weighted history of the mouse's own
valid choices instead of the single last choice. (The name was previously
reserved for a retired model; the parent's config still describes itself as
"the alpha=1 boundary of m2_dynamic_action_kernel_bias_v1", which this
implementation now makes true.)

## 1. Parameters (10 fitted coordinates)

```
(w, s0, ell_short, ell_long, kappa, ah, delta0, bh, eta_logit, a)
```

The first nine decode exactly as in the parent (M2_dynamic_block_length_v1's
eight transforms plus $\eta=\sigma(\widetilde\eta_{\text{logit}})$). New:

$$
\alpha=\sigma(a)\in(0,1).
$$

## 2. Recursion (identical to the parent except the update)

Block/run-length HSMM belief, Bayesian readout $r_t$, mixing
$m_t=(1-\eta)r_t+\eta\pi_t$ and lapse are line-for-line the parent's
(see `M2_dynamic_previous_action_bias_v1.md`). The action-history update:

$$
\pi_1=0.5,\qquad
\pi_{t+1}=\begin{cases}\pi_t+\alpha\,(A_t-\pi_t), & \text{choice valid at } t\\ \pi_t, & \text{otherwise,}\end{cases}
$$

gated by choice validity alone (independent of `feedback_valid` and duration
validity), decaying once per valid choice.

## 3. Nesting boundaries (both constructed explicitly)

- $\alpha=1$ (`with_explicit_alpha(1.0)`): reproduces
  `M2_dynamic_previous_action_bias_v1` **bit-for-bit**
  (`test_m2_dynamic_action_kernel_bias_at_alpha_one_equals_previous_action_bias`).
- $\eta=0$ (`with_explicit_eta(0.0)`): reproduces
  `M2_dynamic_block_length_v1`.

## 4. Identifiability

$\alpha$ is weakly identified as $\eta\to0$ (the history barely enters), and
for small $\alpha$ the state $\pi_t$ drifts slowly, trading off with
$\kappa$. Mitigations, without adding starts: (i) the eight frozen starts
pair the parent's ascending $\eta$ grid (0.05…0.60) with
`M1_action_kernel_v1`'s learning rates in **reversed** order
($\alpha$ = 0.9, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05, 0.02), so low-$\eta$ starts
begin at fast learning rates; (ii) the existing identifiability-ridge flag in
the multistart summary; (iii) the opt-in `--identifiability-grid-points`
now also writes an $(\eta_{\text{logit}}, a)$ surface.

## 5. Implementation and fitting

`src/m2_dynamic_action_kernel_bias_v1/`,
`src/m2_dynamic_action_kernel_bias_sliding_window_v1/`,
`scripts/run_m2_dynamic_action_kernel_bias_sliding_window.py` — a copy of the
parent's runner (same 8-start L-BFGS fitter, the parent's 9 convergence
checks including `no_probability_normalization_errors` and
`tail_occupancy_acceptable`, same tolerances, save-and-continue on no-winner
windows, same 32-CPU budget). Configs:
`config/m2_dynamic_action_kernel_bias_v1{,_mc06,_mc07}.json`. Starts:
the parent's nine-coordinate rows byte-for-byte plus the reversed learning-rate
logits; both provenances are verified at registry load against the
checksummed source tables. Parameter-distance scale: the parent's nine values
plus 2.0 for `a` (the `M1_action_kernel_v1` value). Forced-best runner key:
`action_kernel_bias`.

Known inherited caveat: `recovery.py` (not used by fitting) keeps the
parent's grid convention of placing the eta grid values directly in the
`eta_logit` slot with a `0.0` exact-boundary sentinel, which collides with a
real `eta_logit = 0` point; `M2_dynamic_previous_wsls_bias_v1` fixed this in
its own copy, the action-bias parent did not.
