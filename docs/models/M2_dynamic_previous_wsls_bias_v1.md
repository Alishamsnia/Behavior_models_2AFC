# M2_dynamic_previous_wsls_bias_v1 — equations

Parent: `M2_dynamic_block_length_v1` (unchanged). Sibling:
`M2_dynamic_previous_action_bias_v1` (unchanged, also untouched by this
model's introduction). Same $\eta$-mixing architecture and fixed
$\alpha\equiv1$ as that sibling, but $\pi_t$ tracks the **true category**
(win-stay/lose-shift), not the mouse's own action, and is gated by
`feedback_valid` rather than choice validity — the same causal contract
`wsls_core_v1` (`pure_wsls_v1`/`extended_wsls_v1`) uses for its
category-driven update, reused here as a pattern alongside this model's own
sensory/HSMM state, which `wsls_core_v1` has no equivalent of.

## 1. Parameters (9 fitted coordinates)

```
(w, s0, ell_short, ell_long, kappa, ah, delta0, bh, eta_logit)
```

The first eight decode exactly as in `M2_dynamic_block_length_v1`:

$$
w=\operatorname{softplus}(\widetilde w)+\epsilon,\qquad
s_0=\operatorname{softplus}(\widetilde s_0)+\epsilon,\qquad
\ell_{\text{short}}=\ell_{\max}\,\sigma(\widetilde\ell_{\text{short}}),\qquad
\ell_{\text{long}}=\ell_{\max}\,\sigma(\widetilde\ell_{\text{long}}),
$$
$$
\kappa=\exp(\widetilde{\log\kappa}),\qquad
h_{\text{center}}=\sigma(a_h),\qquad
\delta_0=\operatorname{softplus}(\widetilde\delta_0)+\epsilon,\qquad
b_h=\operatorname{softplus}(\widetilde b_h)+\epsilon .
$$

New ninth coordinate:

$$
\eta=\sigma(\widetilde\eta_{\text{logit}}).
$$

The bias learning rate is **not fitted**; it is the fixed constant

$$
\alpha \equiv 1 ,
$$

matching `pure_wsls_v1`'s own fixed-$\alpha=1$ convention.

## 2. Static prototypes and hazard (unchanged from the parent)

$$
\boldsymbol\theta(\delta_0)=\big(0.5,\ \sigma(-\delta_0),\ \sigma(\delta_0)\big)
\qquad\text{(fixed for the whole session)}
$$

$$
h_B(r)=\sigma\big(a_h+b_h\,(r-23)\big),\qquad h_N(r)=h_B(r)/2,\qquad r=1,\dots,65 .
$$

## 3. Two separately-updated causal states

**(a) Joint block × run-length belief** $\xi_t(z,r)\in\Delta^{3\times 65}$,
driven only by true category/feedback — identical to
`M2_dynamic_block_length_v1`. Initialized $\xi_1(N,1)=1$.

**(b) Scalar previous-category (WSLS) bias state** $\pi_t\in[0,1]$, driven
only by the true category $C_t$, gated by `feedback_valid` — **never** by
the mouse's own action. Initialized $\pi_1=0.5$.

Both states are gated by the same `feedback_valid` signal (since both
represent what the animal can causally learn about the true category), but
neither state ever reads the other, and — critically — **neither state ever
depends on the mouse's own choice** `choice_long`, at any trial. This is
what lets `simulation.py` sample every valid trial's choice in one
vectorized pass rather than a sequential predict→sample→update loop (unlike
`M2_dynamic_previous_action_bias_v1`, whose $\pi_t$ is genuinely
choice-dependent).

## 4. Per-trial recursion

For trial $t$ (predict → mix → lapse → score → update, in that order):

**Prior from the block belief:**
$$
b_t=\sum_r \xi_t(\cdot,r)\quad\text{(block marginal)},\qquad
\text{prior}_t=b_t\cdot\boldsymbol\theta .
$$

**Sensory readout** (unchanged Gaussian ideal-observer solver):
$$
\sigma(s)=\sqrt{w s^2+s_0^2},\qquad
v_k=\tau_k^2+w\mu_k^2+s_0^2\ \ (k\in\{0,1\}),
$$
$$
a_2=\tfrac1{2v_0}-\tfrac1{2v_1},\quad
a_1=\tfrac{\mu_1}{v_1}-\tfrac{\mu_0}{v_0},\quad
a_0=\tfrac{\mu_0^2}{2v_0}-\tfrac{\mu_1^2}{2v_1}+\tfrac12\log\tfrac{v_0}{v_1}
+\operatorname{logit}(\text{prior}_t)-\log\kappa,
$$
$$
a_2 T^2+a_1 T+a_0=0\ \Rightarrow\ T_t\ \ (\text{unique positive root}),\qquad
r_t=P\big(X_t>T_t \mid s_t\big),\quad X_t\sim\mathcal N(s_t,\sigma(s_t)^2).
$$

**Bias mixing (identical form to the previous-action-bias model):**
$$
m_t=(1-\eta)\,r_t+\eta\,\pi_t .
$$

**Lapse (outermost, unchanged form, fed $m_t$):**
$$
\epsilon_t=\ell_{\text{long}}\mathbf 1(c_t=1)+\ell_{\text{short}}\mathbf 1(c_t=0),
\qquad
p_t=(1-\epsilon_t)\,m_t+\epsilon_t/2 .
$$

**Choice likelihood** (only on trials $t\in V_s$ with valid duration, valid
choice, **and** a finite binary category — the category check matters here
because the shared session contract allows an invalid category on a
feedback-invalid trial, and `lapse_mixture` requires a binary category):
$$
NLL_s=-\sum_{t\in V_s}\Big[y_t\log p_t+(1-y_t)\log(1-p_t)\Big].
$$

**Previous-category bias update — the one equation that differs from the
choice-driven sibling** (gated by `feedback_valid` alone, independent of
that trial's own choice validity):
$$
\pi_{t+1}=
\begin{cases}
C_t, & \text{feedback valid at }t\ \ (\alpha\equiv1\text{ collapses the delta rule to exact assignment, as in }wsls\_core\_v1)\\
\pi_t, & \text{feedback invalid at }t
\end{cases}
$$
where $C_t=$ the true category (0/1) on trial $t$. An omitted-choice trial
with valid feedback still updates $\pi_{t+1}$ — the animal can still learn
the correct side even when it didn't respond.

**Belief update** (unchanged, identical to the parent):
$$
\xi_t^{+}(z,r)\propto \xi_t(z,r)\,\theta_z^{\,c_t}(1-\theta_z)^{1-c_t}
\qquad\text{(only if feedback valid; else }\xi_t^{+}=\xi_t\text{)}
$$
$$
\xi_{t+1}=\text{CappedTailHSMMTransition}(\xi_t^{+};\,h_B,h_N,\ \omega=0.5)
$$
with survival aging run length by one (capped tail retains mass at $r=65$)
and switch mass reset to $r=1$, routed $N\!\to\!S/L$ by $\omega$ and
$S\!\leftrightarrow\!L$ deterministically.

## 5. Objective (pooled over sessions for one mouse)

$$
\mathcal L=\frac{\sum_s NLL_s}{\sum_s n_s} .
$$

## 6. Exact nesting boundaries

- $\eta=0$ (approached but not reached by the free logit; must be
  constructed explicitly via `with_explicit_eta`): reproduces
  `M2_dynamic_block_length_v1` exactly — verified in
  `test_eta_zero_reproduces_m2_dynamic_block_length_exactly`.
- There is **no** $\alpha$-boundary relationship to
  `M2_dynamic_previous_action_bias_v1`: the two models are not nested in
  each other at any parameter value, because their $\pi_t$ drivers
  (`category_long` vs. `choice_long`) and gates (`feedback_valid` vs. choice
  validity) are structurally different, not merely different learning
  rates on the same recurrence.

## 7. Relation to this repo's other WSLS models (naming note)

Unlike `M2_dynamic_action_kernel_bias_v1` (retired — an action-driven,
fitted-$\alpha$ model that was never a true WSLS analogue), this model's
$\pi_t$ recurrence is the exact sensory-augmented generalization of
`pure_wsls_v1`'s own recurrence:
$$
\pi_{t+1}=\pi_t+\alpha\,(C_t-\pi_t)\ \big|_{\alpha=1}\ \ (\text{wsls\_core\_v1, this model})
\quad\text{vs.}\quad
\pi_{t+1}=\pi_t+\alpha\,(A_t-\pi_t)\ \ (\text{M2\_dynamic\_previous\_action\_bias\_v1}).
$$
The difference from `pure_wsls_v1`/`extended_wsls_v1` themselves is that
those models have **no sensory readout at all** ($\pi_t$ alone, mixed only
with lapse, is the entire prediction); this model mixes the identical
$\pi_t$ recurrence into a full Bayesian sensory/HSMM readout via $\eta$.
