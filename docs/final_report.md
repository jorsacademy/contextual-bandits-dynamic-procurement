# Final Research Protocol

## Objective

This repository evaluates whether contextual bandit policies improve repeated supplier selection under bandit feedback without silently converting the problem into a full-information learner.

## Frozen evaluation blocks

Two disjoint final blocks are used.

1. `nominal_final`: stationary supplier economics and operational failure probabilities.
2. `supplier_regime_shift`: the same contextual process with a midpoint change in hidden supplier utility plus selected reliability/defect deterioration or improvement.

Final seeds are never used to tune exploration coefficients, discount factors or the static comparator.

## Policies

The final comparison includes random selection, one static supplier chosen from training seeds, epsilon-greedy linear models, LinUCB, discounted LinUCB and linear Thompson Sampling.

Discounted LinUCB is included because ordinary contextual bandit estimators can become slow to adapt when supplier economics change. Its exponential forgetting acts only on accumulated sufficient statistics; the update still uses the selected supplier's reward only.

## Oracle and regret

The environment can compute the hidden expected reward for every supplier. These counterfactual expectations are used only after the action is selected to calculate pseudo-regret against a clairvoyant oracle. They are never passed to the policy update.

This distinction is mandatory. Giving a policy unchosen supplier outcomes would turn the benchmark into full-information online learning.

## Metrics

Primary metric: cumulative pseudo-regret.

Operational secondary metrics:

- realized procurement cost;
- service-failure rate;
- pre/post-shift regret;
- supplier-selection share;
- mean action-selection latency.

For each learned method, cumulative regret is paired by environment seed against `static_best_train`. Reports include mean paired difference, paired 95% bootstrap confidence interval, win rate and exact two-sided sign-test p-value.

## Interpretation rules

A contextual bandit is not promoted because it is more sophisticated. The preferred policy must improve the regret/reliability trade-off without unreasonable latency. A method that lowers pseudo-regret while materially worsening service failures must be reported as a trade-off rather than an unconditional improvement.

Discounted LinUCB is expected to have an advantage only when nonstationarity is material. If ordinary LinUCB or a static supplier performs better in the stationary block, that is a valid result rather than a failure of the benchmark.

## Scope boundary

The action is supplier selection only. There is no inventory state, order quantity, pipeline inventory, contract commitment or delayed reward. Those features create a multi-stage control problem and belong in the separate approximate-dynamic-programming benchmark.
