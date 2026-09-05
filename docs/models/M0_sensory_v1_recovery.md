# M0_sensory_v1 synthetic recovery

The recovery workflow uses the current five-parameter vector
`(w,s0,ell_short,ell_long,kappa)` and the same category-dependent random-choice lapse
equation as production fitting:

```text
epsilon_t = ell_long*I(category_long=1) + ell_short*I(category_long=0)
p_t = (1-epsilon_t)*q_t + epsilon_t/2
```

Synthetic sessions contain balanced short- and long-category duration levels,
so each lapse rate influences its corresponding psychometric branch. Choices
are sampled with an isolated deterministic seed, validation choices do not
affect fitting, and all calculations use `torch.float64`.

The registered recovery cases cover a symmetric lapse, a high
short-category lapse, near-zero category-specific lapses, and combined high
sensory/lapse noise. A case is accepted when probability RMSE is at most 0.05
and training NLL does not worsen. Numerical results must be regenerated after
changes to model equations rather than copied from the removed asymmetric
lapse parameterization.

Run the current recovery suite with:

```bash
python scripts/run_m0_recovery.py \
  --output results/M0_sensory_v1/recovery.json
```
