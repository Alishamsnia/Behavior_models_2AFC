# M2_dynamic_previous_action_bias_v1 — equations

Parent: `M2_dynamic_block_length_v1` (unchanged). Adds one fitted coordinate,
`eta_logit`, and one fixed constant, `alpha ≡ 1`, that mixes a "previous
valid action" scalar bias into the sensory readout before lapse.

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

The action-history learning rate is **not fitted**; it is the fixed
constant

$$
\alpha \equiv 1 .
$$

## 2. Static prototypes and hazard (unchanged from the parent)

$$
\boldsymbol\theta(\delta_0)=\big(0.5,\ \sigma(-\delta_0),\ \sigma(\delta_0)\big)
\qquad\text{(fixed for the whole session)}
$$

$$
h_B(r)=\sigma\big(a_h+b_h\,(r-23)\big),\qquad h_N(r)=h_B(r)/2,\qquad r=1,\dots,65 .
$$

## 3. Two independent causal states

**(a) Joint block × run-length belief** $\xi_t(z,r)\in\Delta^{3\times 65}$,
driven only by true category/feedback — identical to
`M2_dynamic_block_length_v1`. Initialized $\xi_1(N,1)=1$.

**(b) Scalar action-history state** $\pi_t\in[0,1]$, driven only by the
mouse's own previous valid choice. Initialized $\pi_1=0.5$.

Neither state ever reads the other.

## 4. Per-trial recursion

For trial $t$ (predict → mix → lapse → score → update, in that order):

**Prior from the block belief:**
$$
b_t=\sum_r \xi_t(\cdot,r)\quad\text{(block marginal)},\qquad
\text{prior}_t=b_t\cdot\boldsymbol\theta .
$$

**Sensory readout** (unchanged Gaussian ideal-observer solver, M1's shared
machinery):
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

**Bias mixing (new):**
$$
m_t=(1-\eta)\,r_t+\eta\,\pi_t .
$$

**Lapse (outermost, unchanged form, now fed $m_t$ instead of $r_t$):**
$$
\epsilon_t=\ell_{\text{long}}\mathbf 1(c_t=1)+\ell_{\text{short}}\mathbf 1(c_t=0),
\qquad
p_t=(1-\epsilon_t)\,m_t+\epsilon_t/2 .
$$

**Choice likelihood** (only on valid trials $t\in V_s$):
$$
NLL_s=-\sum_{t\in V_s}\Big[y_t\log p_t+(1-y_t)\log(1-p_t)\Big].
$$

**Action-history update** (gated by choice validity alone, independent of
feedback validity):
$$
\pi_{t+1}=
\begin{cases}
A_t, & \text{choice valid at }t\ \ (\alpha\equiv1\text{ collapses the delta rule to exact assignment})\\
\pi_t, & \text{choice invalid/omitted at }t
\end{cases}
$$
where $A_t=$ the mouse's own choice (0/1) on trial $t$. So $\pi_t$ is the
mouse's **last valid** action, not necessarily literally trial $t-1$'s
action if $t-1$ was omitted.

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

- $\eta=0$ (only approached, not reached, by the free logit; must be
  constructed explicitly): reproduces `M2_dynamic_block_length_v1` exactly.
- This model's own action-history bias term is the direct action-driven
  analogue of `M2_dynamic_previous_wsls_bias_v1`'s previous-category bias
  term (same $\eta$-mixing architecture and fixed $\alpha=1$; the two models
  differ only in whether $\pi_t$ tracks the mouse's own action, gated by
  choice validity, or the true category, gated by `feedback_valid` -- see
  that model's own equations file).
